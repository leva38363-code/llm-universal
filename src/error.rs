//! Errors modeled on the official SDKs: `src/core/error.ts` in anthropic-sdk-typescript and
//! openai-node (an `APIError` hierarchy keyed by HTTP status, `requestID`, `type`/`code`/`param`)
//! and `error.rs` in async-openai (an enum with `InvalidArgument` and `JSONDeserialize`).

use serde_json::Value;
use thiserror::Error;

/// How many characters of a response body are kept in an error.
const MAX_BODY_CHARS: usize = 4096;

/// Error class by HTTP status, matching the `APIError` subclasses in the official SDKs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiErrorKind {
    /// 400 `BadRequestError`
    BadRequest,
    /// 401 `AuthenticationError`
    Authentication,
    /// 403 `PermissionDeniedError`
    PermissionDenied,
    /// 404 `NotFoundError`
    NotFound,
    /// 409 `ConflictError`
    Conflict,
    /// 422 `UnprocessableEntityError`
    UnprocessableEntity,
    /// 429 `RateLimitError`
    RateLimit,
    /// >= 500 `InternalServerError` (includes Anthropic's 529 `overloaded_error`)
    InternalServer,
    /// Any other non-2xx status (the base `APIError`)
    Other,
}

impl ApiErrorKind {
    pub fn from_status(status: u16) -> Self {
        match status {
            400 => Self::BadRequest,
            401 => Self::Authentication,
            403 => Self::PermissionDenied,
            404 => Self::NotFound,
            409 => Self::Conflict,
            422 => Self::UnprocessableEntity,
            429 => Self::RateLimit,
            s if s >= 500 => Self::InternalServer,
            _ => Self::Other,
        }
    }
}

/// A response with a non-2xx status (`APIStatusError` in the SDKs).
#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    pub status: u16,
    pub kind: ApiErrorKind,
    /// `error.message` from the body; otherwise the whole body (truncated).
    pub message: Option<String>,
    /// `error.type`, e.g. `rate_limit_error` (Anthropic) or `invalid_request_error` (OpenAI).
    pub error_type: Option<String>,
    /// `error.code` (OpenAI).
    pub code: Option<String>,
    /// `error.param` (OpenAI).
    pub param: Option<String>,
    /// `request-id` (Anthropic) or `x-request-id` (OpenAI) header — quote it to support.
    pub request_id: Option<String>,
    /// The response body, if it was JSON.
    pub body: Option<Value>,
    /// Delay the server asked for in `retry-after` / `retry-after-ms`, if any.
    pub retry_after: Option<std::time::Duration>,
}

impl ApiError {
    pub(crate) fn from_response(status: u16, request_id: Option<String>, text: &str) -> Self {
        let body: Option<Value> = serde_json::from_str(text).ok();
        // Anthropic: {"type":"error","error":{"type","message"}}; OpenAI: {"error":{"message","type","code","param"}}.
        // Gemini: {"error":{"code","message","status"}}; Cohere: {"message"}.
        let field = |name: &str| {
            body.as_ref()
                .and_then(|b| b["error"][name].as_str())
                .map(str::to_string)
        };
        let top_message = body
            .as_ref()
            .and_then(|b| b["message"].as_str())
            .map(str::to_string);
        let message = field("message")
            .or(top_message)
            .or_else(|| {
                let text = text.trim();
                (!text.is_empty()).then(|| text.to_string())
            })
            .map(|m| truncate(&m));
        Self {
            status,
            kind: ApiErrorKind::from_status(status),
            message,
            error_type: field("type").or_else(|| field("status")),
            code: field("code"),
            param: field("param"),
            request_id,
            body,
            retry_after: None,
        }
    }

    /// An error sent inside a stream (`event: error` / a `{"error": ...}` chunk), where there is
    /// no HTTP status of its own. The status is derived from the error type or code.
    pub(crate) fn from_stream_error(error: &Value) -> Self {
        let error_type = error["type"].as_str().or_else(|| error["status"].as_str());
        let status = error["code"]
            .as_u64()
            .and_then(|c| u16::try_from(c).ok())
            .filter(|c| (400..600).contains(c))
            .unwrap_or(match error_type {
                Some("overloaded_error") => 529,
                Some("rate_limit_error" | "RESOURCE_EXHAUSTED") => 429,
                Some("invalid_request_error" | "INVALID_ARGUMENT") => 400,
                Some("authentication_error" | "UNAUTHENTICATED") => 401,
                Some("permission_error" | "PERMISSION_DENIED") => 403,
                Some("not_found_error" | "NOT_FOUND") => 404,
                _ => 500,
            });
        Self {
            status,
            kind: ApiErrorKind::from_status(status),
            message: error["message"].as_str().map(truncate),
            error_type: error_type.map(str::to_string),
            code: error["code"].as_str().map(str::to_string),
            param: error["param"].as_str().map(str::to_string),
            request_id: None,
            body: Some(error.clone()),
            retry_after: None,
        }
    }
}

/// Same format as `APIError.makeMessage` in the SDKs: `"{status} {message}"`
/// or `"{status} status code (no body)"`.
impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.message {
            Some(message) => write!(f, "{} {}", self.status, message),
            None => write!(f, "{} status code (no body)", self.status),
        }
    }
}

impl std::error::Error for ApiError {}

/// Every error this crate returns.
#[derive(Debug, Error)]
pub enum LlmError {
    /// The server answered with a non-2xx status (`APIStatusError` and subclasses).
    #[error("{0}")]
    Api(Box<ApiError>),
    /// No response: DNS, TLS or a dropped connection (`APIConnectionError`).
    #[error("Connection error.")]
    Connection(#[source] reqwest::Error),
    /// The request timed out (`APIConnectionTimeoutError`).
    #[error("Request timed out.")]
    Timeout(#[source] reqwest::Error),
    /// The response body is not JSON (`JSONDeserialize` in async-openai).
    /// `content` holds the start of the body; it is not part of the message, so model
    /// output does not end up in logs by accident.
    #[error("failed to deserialize api response: {source}")]
    Deserialize {
        #[source]
        source: serde_json::Error,
        content: BodySnippet,
    },
    /// The JSON parsed but is missing expected fields.
    #[error("unexpected api response: {0}")]
    InvalidResponse(String),
    /// Client-side validation before the request is sent (`InvalidArgument` in async-openai).
    #[error("invalid args: {0}")]
    InvalidArgument(String),
    /// A per-user spending limit from [`crate::Budget`] was reached. Nothing was sent.
    #[error("{0}")]
    BudgetExceeded(String),
}

impl LlmError {
    pub(crate) fn from_reqwest(error: reqwest::Error) -> Self {
        if error.is_builder() {
            // The request could not be built (e.g. a bad header); retrying cannot help.
            return Self::InvalidArgument(format!(
                "could not build request: {}",
                error.without_url()
            ));
        }
        // Drop the URL so a key in it can never reach logs through the error chain.
        let error = error.without_url();
        if error.is_timeout() {
            Self::Timeout(error)
        } else {
            Self::Connection(error)
        }
    }

    /// Maps an error from reading or sending a request whose bytes may already have reached the
    /// server. Only a failure to connect is a `Connection` (safe to retry or fall back). Anything
    /// later may mean the request was processed, so it is `InvalidResponse`, which is neither
    /// retried nor passed to another provider.
    pub(crate) fn after_send(error: reqwest::Error) -> Self {
        if error.is_builder() || error.is_connect() {
            return Self::from_reqwest(error);
        }
        let error = error.without_url();
        if error.is_timeout() {
            return Self::Timeout(error);
        }
        Self::InvalidResponse(format!(
            "connection broke after the request was sent: {error}"
        ))
    }

    /// HTTP status, if the server answered.
    pub fn status(&self) -> Option<u16> {
        match self {
            Self::Api(e) => Some(e.status),
            _ => None,
        }
    }

    /// Request ID of the response, if the server sent one.
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Self::Api(e) => e.request_id.as_deref(),
            _ => None,
        }
    }

    /// Same retry rules as the SDKs: 408, 409, 429, >= 500, connection errors and timeouts.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Api(e) => matches!(e.status, 408 | 409 | 429) || e.status >= 500,
            Self::Connection(_) | Self::Timeout(_) => true,
            _ => false,
        }
    }
}

pub(crate) fn truncate(text: &str) -> String {
    text.chars().take(MAX_BODY_CHARS).collect()
}

/// Start of a response body that failed to parse. `Debug` shows only its length, so model output
/// does not reach logs through `{:?}`. Read the text with [`BodySnippet::as_str`].
#[derive(Clone, PartialEq, Eq)]
pub struct BodySnippet(String);

impl BodySnippet {
    pub(crate) fn new(text: &str) -> Self {
        Self(truncate(text))
    }

    /// The first characters of the body.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for BodySnippet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BodySnippet(<{} bytes withheld>)", self.0.len())
    }
}

/// Replaces each secret in an error's text fields and body with its masked form.
pub(crate) fn redact_api_error(error: &mut ApiError, secrets: &[String]) {
    let redact = |s: &str| crate::http::redact(s, secrets);
    error.message = error.message.as_deref().map(redact);
    error.error_type = error.error_type.as_deref().map(redact);
    error.code = error.code.as_deref().map(redact);
    error.param = error.param.as_deref().map(redact);
    error.body = error.body.take().map(|v| {
        let text = redact(&v.to_string());
        serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text))
    });
}

#[cfg(test)]
mod redaction_tests {
    use super::*;

    #[test]
    fn stream_error_text_and_body_are_masked() {
        let key = "sk-LIVE-KEY-999";
        let payload = serde_json::json!({"type": "error", "message": format!("bad key {key}")});
        let mut err = ApiError::from_stream_error(&payload);
        redact_api_error(&mut err, &[key.to_string()]);
        assert!(!err.message.unwrap().contains(key));
        assert!(!err.body.unwrap().to_string().contains(key));
    }

    #[test]
    fn deserialize_debug_withholds_the_body() {
        let content = BodySnippet::new("model output that must not be logged");
        let debug = format!("{content:?}");
        assert!(!debug.contains("model output"), "{debug}");
        assert_eq!(content.as_str(), "model output that must not be logged");
    }
}
