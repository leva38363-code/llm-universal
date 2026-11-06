//! Shared HTTP layer: client defaults, retries, size caps and error mapping.

use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::Value;

use crate::error::{self, ApiError};
use crate::{LlmError, Result};

/// Largest successful response body that is read (32 MiB).
pub(crate) const MAX_RESPONSE_BYTES: usize = 32 * 1024 * 1024;
/// Largest error response body that is kept (64 KiB); the rest is dropped.
const MAX_ERROR_BYTES: usize = 64 * 1024;
/// Extra bytes read past the error limit, so a secret cut at the limit is still masked.
const MAX_SECRET_SPAN: usize = 1024;
/// Shorter secrets are not masked in bodies: they would match ordinary text.
const MIN_SECRET_LEN: usize = 4;
/// Retries by default, like the official SDKs.
pub const DEFAULT_MAX_RETRIES: u32 = 2;
/// A `retry-after` longer than this is ignored in favor of normal backoff (SDK behavior).
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);
/// Total time limit for one non-streaming attempt (the official SDKs default to 10 minutes).
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
const INITIAL_RETRY_DELAY: f64 = 0.5;
const MAX_RETRY_DELAY: f64 = 8.0;

/// HTTP client plus retry policy, shared by every provider.
#[derive(Clone, Debug)]
pub(crate) struct Http {
    pub(crate) client: reqwest::Client,
    pub(crate) max_retries: u32,
    /// Also repeat a POST that the server may have already processed (5xx, 408, 409, timeouts after
    /// sending). Off by default: a repeat can generate and bill the answer twice.
    pub(crate) repeat_post: bool,
    /// Whole-request limit for non-streaming calls (each attempt).
    pub(crate) request_timeout: Duration,
    /// Overall limit for a stream, however much data arrives (keep-alives included).
    pub(crate) stream_deadline: Duration,
}

impl Default for Http {
    fn default() -> Self {
        Self {
            client: default_client(),
            max_retries: DEFAULT_MAX_RETRIES,
            repeat_post: false,
            request_timeout: REQUEST_TIMEOUT,
            stream_deadline: Duration::from_secs(3600),
        }
    }
}

/// Default client: timeouts, no redirects (so the key is never sent to another host),
/// and a bounded idle pool.
fn default_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        // Idle limit between bytes, not a total limit, so long streams are not cut off.
        .read_timeout(Duration::from_secs(600))
        .redirect(reqwest::redirect::Policy::none())
        .pool_max_idle_per_host(8)
        .build()
        // Building with these options practically never fails; if it does, keep the redirect
        // policy and the post-send host check rather than falling back to the default client.
        .unwrap_or_else(|_| {
            reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap_or_else(|_| reqwest::Client::new())
        })
}

impl Http {
    /// Sends the request, retrying like the official SDKs: on connection errors, timeouts,
    /// 408, 409, 429 and >= 500, honoring `retry-after` / `retry-after-ms` and `x-should-retry`,
    /// otherwise backing off exponentially (0.5 s doubling up to 8 s, with jitter).
    /// Returns the successful response; non-2xx responses become [`LlmError::Api`].
    pub(crate) async fn send(&self, request: reqwest::RequestBuilder) -> Result<reqwest::Response> {
        let mut attempt = 0;
        loop {
            let this_try = request.try_clone().ok_or_else(|| {
                LlmError::InvalidArgument("request body cannot be retried".into())
            })?;
            let built = this_try.build().map_err(LlmError::from_reqwest)?;
            let origin = built.url().host_str().map(str::to_owned);
            // Values of secret headers, so an error body that echoes them cannot leak the key.
            let secrets: Vec<String> = built
                .headers()
                .values()
                .filter(|v| v.is_sensitive())
                .filter_map(|v| v.to_str().ok())
                .map(|v| v.trim_start_matches("Bearer ").to_string())
                .filter(|v| v.len() >= MIN_SECRET_LEN)
                .collect();
            let sent = self.client.execute(built).await;
            if let Ok(response) = &sent {
                // A client that follows redirects may have sent the request elsewhere.
                if response.url().host_str().map(str::to_owned) != origin {
                    return Err(LlmError::InvalidResponse(
                        "the response came from another host (redirect); refusing it".into(),
                    ));
                }
            }
            let (error, retry_after, should_retry) = match sent {
                Ok(response) if response.status().is_success() => return Ok(response),
                Ok(response) => {
                    let headers = response.headers().clone();
                    let status = response.status().as_u16();
                    let text =
                        match read_body(response, MAX_ERROR_BYTES + MAX_SECRET_SPAN, true).await {
                            Ok(text) => text,
                            Err(error) => {
                                // The body broke off after the server answered: it may have processed the request.
                                let retry = self.repeat_post && (status >= 500 || status == 429);
                                if !retry || attempt >= self.max_retries {
                                    return Err(error);
                                }
                                tokio::time::sleep(retry_delay(attempt, None)).await;
                                attempt += 1;
                                continue;
                            }
                        };
                    let text = cut_at_char(&redact(&text, &secrets), MAX_ERROR_BYTES).to_string();
                    let mut api = ApiError::from_response(status, request_id(&headers), &text);
                    api.retry_after = retry_after(&headers);
                    let server_says =
                        match headers.get("x-should-retry").and_then(|v| v.to_str().ok()) {
                            Some("true") => Some(true),
                            Some("false") => Some(false),
                            _ => None,
                        };
                    // 429 means "not processed, try later": always safe to repeat.
                    // 408/409/5xx may mean the request was processed: repeat only when asked for.
                    let default_retry = status == 429
                        || (self.repeat_post && (matches!(status, 408 | 409) || status >= 500));
                    let should_retry = match server_says {
                        Some(false) => false,
                        Some(true) => status == 429 || self.repeat_post,
                        None => default_retry,
                    };
                    let retry_after = api.retry_after;
                    (LlmError::Api(Box::new(api)), retry_after, should_retry)
                }
                Err(e) => {
                    // Connection failed before the request was sent: always safe to repeat.
                    // Timeouts and other errors may come after sending: repeat only when asked for.
                    let safe = e.is_connect();
                    let error = LlmError::after_send(e);
                    let retryable = safe || (self.repeat_post && error.is_retryable());
                    (error, None, retryable)
                }
            };
            if !should_retry || attempt >= self.max_retries {
                return Err(error);
            }
            tokio::time::sleep(retry_delay(attempt, retry_after)).await;
            attempt += 1;
        }
    }

    /// Sends the request and parses a JSON body. Each attempt has a total time limit
    /// (streams only have the idle read timeout, so long answers are not cut off).
    pub(crate) async fn json(&self, request: reqwest::RequestBuilder) -> Result<Value> {
        let response = self.send(request.timeout(self.request_timeout)).await?;
        let text = read_body(response, MAX_RESPONSE_BYTES, false).await?;
        serde_json::from_str(&text).map_err(|source| LlmError::Deserialize {
            source,
            content: error::BodySnippet::new(&text),
        })
    }
}

fn request_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("request-id")
        .or_else(|| headers.get("x-request-id"))
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

/// `retry-after-ms` (milliseconds) or `retry-after` (seconds or an HTTP date — only the
/// seconds form is read), if within [`MAX_RETRY_AFTER`].
/// `sk-SUPERSECRET-123` -> `sk-****23`: recognisable, but not usable.
pub(crate) fn mask_secret(secret: &str) -> String {
    let chars: Vec<char> = secret.chars().collect();
    if chars.len() < 8 {
        // Showing any part of a short secret would reveal most of it.
        return "****".to_string();
    }
    let head: String = chars.iter().take(3).collect();
    let tail: String = chars[chars.len().saturating_sub(2)..].iter().collect();
    format!("{head}****{tail}")
}

/// Replaces every secret of at least [`MIN_SECRET_LEN`] bytes in `text`, longest first.
pub(crate) fn redact(text: &str, secrets: &[String]) -> String {
    let mut secrets: Vec<&String> = secrets
        .iter()
        .filter(|s| s.len() >= MIN_SECRET_LEN)
        .collect();
    secrets.sort_by_key(|s| std::cmp::Reverse(s.len()));
    secrets.into_iter().fold(text.to_string(), |t, s| {
        t.replace(s.as_str(), &mask_secret(s))
    })
}

/// The longest prefix of `text` that is at most `max` bytes and ends on a character boundary.
pub(crate) fn cut_at_char(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let header = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
    let delay = header("retry-after-ms")
        .and_then(|v| v.trim().parse::<f64>().ok())
        .map(|ms| ms / 1000.0)
        .or_else(|| header("retry-after").and_then(|v| v.trim().parse::<f64>().ok()))?;
    (delay.is_finite() && delay > 0.0 && delay <= MAX_RETRY_AFTER.as_secs_f64())
        .then(|| Duration::from_secs_f64(delay))
}

fn retry_delay(attempt: u32, retry_after: Option<Duration>) -> Duration {
    if let Some(delay) = retry_after {
        return delay;
    }
    let base = (INITIAL_RETRY_DELAY * 2f64.powi(attempt as i32)).min(MAX_RETRY_DELAY);
    // Up to 25% jitter so many clients do not retry in lockstep.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let jitter = 1.0 - 0.25 * (f64::from(nanos % 1000) / 1000.0);
    Duration::from_secs_f64(base * jitter)
}

/// Reads at most `cap` bytes of the body. Past the cap it fails, or, with `truncate`,
/// keeps the first `cap` bytes and drops the rest.
pub(crate) async fn read_body(
    mut response: reqwest::Response,
    cap: usize,
    truncate: bool,
) -> Result<String> {
    let too_large = || LlmError::InvalidResponse(format!("response body exceeds {cap} bytes"));
    if !truncate && response.content_length().is_some_and(|n| n > cap as u64) {
        return Err(too_large());
    }
    let mut buf = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(LlmError::after_send)? {
        if buf.len() + chunk.len() > cap {
            if !truncate {
                return Err(too_large());
            }
            buf.extend_from_slice(&chunk[..cap - buf.len()]);
            break;
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Header value for a secret, marked sensitive so it is never printed by `Debug`.
pub(crate) fn secret_header(provider: &str, value: &str) -> Result<HeaderValue> {
    let mut header = HeaderValue::from_str(value).map_err(|_| {
        LlmError::InvalidArgument(format!(
            "{provider}: api_key contains characters not allowed in an HTTP header"
        ))
    })?;
    header.set_sensitive(true);
    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_grows_and_is_capped() {
        let d0 = retry_delay(0, None).as_secs_f64();
        let d3 = retry_delay(3, None).as_secs_f64();
        let d10 = retry_delay(10, None).as_secs_f64();
        assert!((0.375..=0.5).contains(&d0), "{d0}");
        assert!((3.0..=4.0).contains(&d3), "{d3}");
        assert!(d10 <= 8.0, "{d10}");
    }

    #[test]
    fn retry_after_headers_are_honored_within_limit() {
        let mut h = HeaderMap::new();
        h.insert("retry-after", HeaderValue::from_static("3"));
        assert_eq!(retry_after(&h), Some(Duration::from_secs(3)));
        h.insert("retry-after-ms", HeaderValue::from_static("1500"));
        assert_eq!(retry_after(&h), Some(Duration::from_millis(1500)));
        let mut h = HeaderMap::new();
        h.insert("retry-after", HeaderValue::from_static("3600"));
        assert_eq!(retry_after(&h), None);
    }
}

#[cfg(test)]
mod redaction_tests {
    use super::*;

    #[test]
    fn echoed_keys_are_masked_including_short_ones() {
        let secrets = vec!["sk-SECRET-12345".to_string(), "abcd".to_string()];
        let out = redact("key sk-SECRET-12345 and abcd came back", &secrets);
        assert!(!out.contains("SECRET"), "{out}");
        assert!(!out.contains("abcd"), "{out}");
    }

    #[test]
    fn short_secrets_show_nothing_of_themselves() {
        assert_eq!(mask_secret("abcd"), "****");
        assert_eq!(mask_secret("sk-SECRET-12345"), "sk-****45");
    }

    #[test]
    fn cut_never_splits_a_character() {
        let s = "aé"; // 'é' is two bytes
        assert_eq!(cut_at_char(s, 2), "a");
        assert_eq!(cut_at_char(s, 3), "aé");
    }
}
