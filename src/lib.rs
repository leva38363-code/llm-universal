//! One async client for many LLM APIs: Anthropic, OpenAI, Google Gemini, Cohere, Azure OpenAI,
//! Amazon Bedrock, Vertex AI and every OpenAI-compatible server.
//!
//! The same code (`ChatRequest` -> `ChatResponse`) runs against any provider; switching
//! providers means switching the constructor:
//!
//! ```no_run
//! # use llm_universal::*;
//! # async fn demo() -> Result<()> {
//! let provider: Box<dyn LlmProvider> = Box::new(AnthropicProvider::new("sk-ant-..."));
//! // or: Box::new(OpenAiProvider::new("sk-...")),
//! // or: Box::new(GeminiProvider::new("AIza...")),
//! // or: Box::new(OpenAiProvider::ollama()),
//! let req = ChatRequest::new("claude-opus-5-5")
//!     .system("Be concise.")
//!     .max_tokens(512)
//!     .message(Message::user("Hello!"));
//! let resp = provider.chat(&req).await?;
//! println!("{}", resp.text());
//!
//! // Streaming:
//! let mut stream = provider.chat_stream(&req).await?;
//! while let Some(event) = stream.next().await {
//!     if let StreamEvent::TextDelta(text) = event? {
//!         print!("{text}");
//!     }
//! }
//! # Ok(()) }
//! ```
//!
//! Errors follow the official SDKs: see [`LlmError`] and [`ApiError`]. Failed requests are
//! retried like the SDKs do (2 retries by default; see `with_max_retries`).

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

mod anthropic;
mod budget;
mod cohere;
mod commands;
mod error;
mod fallback;
mod gemini;
mod http;
mod openai;
mod runner;
mod sigv4;
mod stream;
mod types;

pub use anthropic::AnthropicProvider;
pub use budget::{Budget, Limits};
pub use cohere::CohereProvider;
pub use commands::Commands;
pub use error::{ApiError, ApiErrorKind, LlmError};
pub use fallback::FallbackProvider;
pub use futures_util::StreamExt;
pub use gemini::GeminiProvider;
pub use http::DEFAULT_MAX_RETRIES;
pub use openai::OpenAiProvider;
pub use runner::{run_tools, MAX_CALLS_PER_ROUND};
#[cfg(feature = "fuzzing")]
#[doc(hidden)]
pub use stream::fuzz_sse;
pub use stream::{ChatStream, StreamEvent};
pub use types::*;

use async_trait::async_trait;

pub type Result<T> = std::result::Result<T, LlmError>;

/// Common contract implemented by every provider.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Sends the request and waits for the whole answer.
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse>;

    /// Streams the answer as it is generated. The last event is [`StreamEvent::Done`]
    /// with the full response.
    ///
    /// The default implementation (for custom providers) calls [`chat`](Self::chat) and
    /// yields the whole text at once.
    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let response = self.chat(request).await?;
        let events = vec![
            Ok(StreamEvent::TextDelta(response.text())),
            Ok(StreamEvent::Done(response)),
        ];
        Ok(Box::pin(futures_util::stream::iter(events)))
    }

    /// Models available to the key.
    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Err(LlmError::InvalidArgument(
            "list_models is not supported by this provider".into(),
        ))
    }
}

/// Providers that can turn text into vectors (OpenAI and compatible servers, Gemini, Cohere).
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, model: &str, inputs: &[&str]) -> Result<Embeddings>;
}

fn invalid(message: String) -> LlmError {
    LlmError::InvalidArgument(message)
}

/// Host part of `base_url`, or an error if it carries credentials or a query string.
fn checked_host<'a>(provider: &str, base_url: &'a str) -> Result<&'a str> {
    let rest = base_url
        .strip_prefix("https://")
        .or_else(|| base_url.strip_prefix("http://"))
        .ok_or_else(|| {
            invalid(format!(
                "{provider}: base_url must start with http:// or https://"
            ))
        })?;
    if base_url
        .chars()
        .any(|c| c == '\\' || c.is_whitespace() || c.is_control())
    {
        return Err(invalid(format!(
            "{provider}: base_url must not contain backslashes, spaces or control characters"
        )));
    }
    let authority = rest.split('/').next().unwrap_or_default();
    if authority.contains('@') || base_url.contains('?') || base_url.contains('#') {
        // Credentials in the URL would leak into logs and error chains.
        return Err(invalid(format!(
            "{provider}: base_url must not contain credentials, a query string or a fragment"
        )));
    }
    if authority.is_empty() {
        return Err(invalid(format!("{provider}: base_url has no host")));
    }
    Ok(authority)
}

/// True only when the URL's real host (as reqwest will resolve it) is loopback.
/// The host comes from the same parser reqwest uses, so tricks like `127.evil.com`
/// or `evil.com\.localhost` cannot pass as local.
fn is_loopback(base_url: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(base_url) else {
        return false;
    };
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        Err(_) => host.eq_ignore_ascii_case("localhost"),
    }
}

/// Checks `base_url` and the key's transport before any request.
pub(crate) fn validate_endpoint(
    provider: &'static str,
    api_key: &str,
    require_key: bool,
    base_url: &str,
) -> Result<()> {
    checked_host(provider, base_url)?;
    if base_url.starts_with("http://") && !api_key.is_empty() && !is_loopback(base_url) {
        return Err(invalid(format!(
            "{provider}: refusing to send an API key over plain http:// to a non-local host; use https://"
        )));
    }
    if require_key && api_key.trim().is_empty() {
        return Err(invalid(format!("{provider}: api_key must not be empty")));
    }
    Ok(())
}

/// A value that goes into a URL path (model, project, region) may only use these characters.
pub(crate) fn checked_path_segment<'a>(
    provider: &str,
    what: &str,
    value: &'a str,
    extra: &[char],
) -> Result<&'a str> {
    // A value made only of dots (`.`, `..`) would change the URL path.
    let ok = !value.is_empty()
        && !value.chars().all(|c| c == '.')
        && value.chars().all(|c| {
            c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') || extra.contains(&c)
        });
    if ok {
        Ok(value)
    } else {
        Err(invalid(format!(
            "{provider}: {what} `{value}` may only contain a-z, A-Z, 0-9, '-', '_' and '.'"
        )))
    }
}

const IMAGE_TYPES: [&str; 4] = ["image/png", "image/jpeg", "image/gif", "image/webp"];

/// Validates the request and provider settings before any network call.
pub(crate) fn validate(
    provider: &'static str,
    api_key: &str,
    require_key: bool,
    base_url: &str,
    max_tool_name: usize,
    req: &ChatRequest,
) -> Result<()> {
    validate_endpoint(provider, api_key, require_key, base_url)?;
    if req.messages.len() > MAX_MESSAGES {
        return Err(invalid(format!(
            "too many messages: {} (limit {MAX_MESSAGES})",
            req.messages.len()
        )));
    }
    let text_bytes: usize = req.system.as_deref().map_or(0, str::len)
        + req
            .messages
            .iter()
            .map(|m| m.text_content().len())
            .sum::<usize>();
    if text_bytes > MAX_TEXT_BYTES {
        return Err(invalid(format!(
            "request text is {text_bytes} bytes (limit {MAX_TEXT_BYTES})"
        )));
    }
    if req.model.trim().is_empty() {
        return Err(invalid("model must not be empty".to_string()));
    }
    if req.model.chars().any(char::is_control) {
        return Err(invalid(
            "model must not contain control characters".to_string(),
        ));
    }
    if req.messages.is_empty() {
        return Err(invalid("messages must not be empty".to_string()));
    }
    for (i, message) in req.messages.iter().enumerate() {
        let has_content = message.parts.iter().any(|p| match p {
            Part::Text { text } => !text.is_empty(),
            Part::Reasoning { .. } => false,
            _ => true,
        });
        if !has_content {
            return Err(invalid(format!("messages[{i}] has no content")));
        }
        if message.has_images() && message.role != Role::User {
            return Err(invalid(format!(
                "messages[{i}]: images are only allowed in user messages"
            )));
        }
        for part in &message.parts {
            if let Part::Image { source } = part {
                validate_image(i, source)?;
            }
        }
    }
    if let Some(t) = req.temperature {
        if !(0.0..=2.0).contains(&t) {
            return Err(invalid(format!(
                "temperature must be between 0.0 and 2.0, got {t}"
            )));
        }
    }
    if let Some(format) = &req.response_format {
        let valid_name = !format.name.is_empty()
            && format.name.len() <= 64
            && format
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        if !valid_name {
            return Err(invalid(format!(
                "response_format name `{}` must match ^[a-zA-Z0-9_-]{{1,64}}$",
                format.name
            )));
        }
        if !format.schema.is_object() {
            return Err(invalid(
                "response_format schema must be a JSON object".to_string(),
            ));
        }
    }
    let mut names = std::collections::HashSet::new();
    for tool in &req.tools {
        let valid_name = !tool.name.is_empty()
            && tool.name.len() <= max_tool_name
            && tool
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        if !valid_name {
            return Err(invalid(format!(
                "tool name `{}` must match ^[a-zA-Z0-9_-]{{1,{max_tool_name}}}$",
                tool.name
            )));
        }
        if !names.insert(tool.name.as_str()) {
            return Err(invalid(format!("duplicate tool name `{}`", tool.name)));
        }
        if tool.parameters["type"] != "object" {
            return Err(invalid(format!(
                "tool `{}`: parameters must be a JSON Schema with \"type\": \"object\"",
                tool.name
            )));
        }
    }
    Ok(())
}

/// True for addresses that point at this machine or at a private network, IPv4 or IPv6.
/// IPv6 forms that embed an IPv4 address (mapped `::ffff:a.b.c.d`, NAT64 `64:ff9b::/96`,
/// the deprecated IPv4-compatible `::a.b.c.d`) are checked as the IPv4 address they carry.
pub(crate) fn is_local_address(ip: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(v4) => {
            v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || (v4.octets()[0] == 100 && (v4.octets()[1] & 0xc0) == 64) // CGNAT 100.64.0.0/10
                || (v4.octets()[0] == 198 && (v4.octets()[1] & 0xfe) == 18) // benchmarking 198.18.0.0/15
                || v4.octets()[0] == 0
                || v4.octets()[0] >= 240
        }
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_local_address(IpAddr::V4(v4));
            }
            let seg = v6.segments();
            let nat64 = seg[0] == 0x64
                && seg[1] == 0xff9b
                && seg[2] == 0
                && seg[3] == 0
                && seg[4] == 0
                && seg[5] == 0;
            if nat64 {
                let [a, b] = [(seg[6] >> 8) as u8, seg[6] as u8];
                let [c, d] = [(seg[7] >> 8) as u8, seg[7] as u8];
                return is_local_address(IpAddr::V4(std::net::Ipv4Addr::new(a, b, c, d)));
            }
            let compat = seg[..6].iter().all(|&x| x == 0) && (seg[6] != 0 || seg[7] > 1);
            if compat {
                let [a, b] = [(seg[6] >> 8) as u8, seg[6] as u8];
                let [c, d] = [(seg[7] >> 8) as u8, seg[7] as u8];
                return is_local_address(IpAddr::V4(std::net::Ipv4Addr::new(a, b, c, d)));
            }
            v6.is_loopback()
                || v6.is_unspecified()
                || (seg[0] & 0xfe00) == 0xfc00 // unique local fc00::/7
                || (seg[0] & 0xffc0) == 0xfe80 // link-local fe80::/10
                || (seg[0] & 0xffc0) == 0xfec0 // site-local fec0::/10 (deprecated)
                || (seg[0] & 0xff00) == 0xff00 // multicast ff00::/8
        }
    }
}

/// Image hosts that are refused: Wikimedia projects and known Wikipedia mirrors.
/// Subdomains are matched too (`en.wikipedia.org`, `upload.wikimedia.org`).
pub const BLOCKED_IMAGE_HOSTS: &[&str] = &[
    "wikipedia.org",
    "wikimedia.org",
    "wikimediafoundation.org",
    "mediawiki.org",
    "wikidata.org",
    "wiktionary.org",
    "wikibooks.org",
    "wikiquote.org",
    "wikisource.org",
    "wikinews.org",
    "wikiversity.org",
    "wikivoyage.org",
    "wikifunctions.org",
    "wikiwand.com",
    "wikizero.com",
    "wiki2.org",
    "wikiless.org",
    "wikimirror.com",
];

/// Longest image URL accepted.
const MAX_IMAGE_URL_BYTES: usize = 2048;
/// Most messages in one request.
const MAX_MESSAGES: usize = 1000;
/// Most text in one request, in bytes (all messages and the system prompt together).
const MAX_TEXT_BYTES: usize = 8 * 1024 * 1024;

fn validate_image(i: usize, source: &ImageSource) -> Result<()> {
    match source {
        ImageSource::Url { url } => {
            // The provider fetches this URL itself. Only https, and never a literal private or
            // local address, so a user-supplied URL cannot point the provider at internal services.
            if !url.starts_with("https://") {
                return Err(invalid(format!(
                    "messages[{i}]: image url must start with https://"
                )));
            }
            if url.len() > MAX_IMAGE_URL_BYTES
                || url.chars().any(|c| c.is_control() || c == '\\' || c == ' ')
            {
                return Err(invalid(format!(
                    "messages[{i}]: image url is too long or malformed"
                )));
            }
            // Parse the URL the way the provider will, so IPv4 forms such as `2130706433`,
            // `0x7f.1` or `127.1` become the address they stand for before the check.
            let parsed = reqwest::Url::parse(url)
                .map_err(|_| invalid(format!("messages[{i}]: image url is not a valid URL")))?;
            if !parsed.username().is_empty() || parsed.password().is_some() {
                return Err(invalid(format!(
                    "messages[{i}]: image url must not contain credentials"
                )));
            }
            let raw_host = parsed.host_str().unwrap_or_default();
            // IPv6 literals come back in brackets.
            let bare = raw_host.trim_start_matches('[').trim_end_matches(']');
            // A trailing dot names the same host (`localhost.`, `en.wikipedia.org.`).
            let host = bare.trim_end_matches('.').to_ascii_lowercase();
            let local_name = host.is_empty()
                || host == "localhost"
                || host.ends_with(".localhost")
                || host.ends_with(".internal");
            let private_ip = bare
                .parse::<std::net::IpAddr>()
                .map(is_local_address)
                .unwrap_or(false);
            if local_name || private_ip {
                return Err(invalid(format!(
                    "messages[{i}]: image url points to a local or private address"
                )));
            }
            if BLOCKED_IMAGE_HOSTS
                .iter()
                .any(|d| host == *d || host.ends_with(&format!(".{d}")))
            {
                return Err(invalid(format!(
                    "messages[{i}]: image url host `{host}` is blocked (Wikipedia, Wikimedia and their mirrors)"
                )));
            }
        }
        ImageSource::Base64 { media_type, data } => {
            if !IMAGE_TYPES.contains(&media_type.as_str()) {
                return Err(invalid(format!(
                    "messages[{i}]: image media_type `{media_type}` must be one of {}",
                    IMAGE_TYPES.join(", ")
                )));
            }
            let base64 = |c: char| c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=');
            if data.is_empty() || !data.chars().all(base64) {
                return Err(invalid(format!(
                    "messages[{i}]: image data must be non-empty base64 without a `data:` prefix"
                )));
            }
        }
    }
    Ok(())
}

/// Compiles the README examples as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
