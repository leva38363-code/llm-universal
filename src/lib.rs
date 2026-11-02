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
mod cohere;
mod error;
mod fallback;
mod gemini;
mod http;
mod openai;
mod runner;
mod stream;
mod types;

pub use anthropic::AnthropicProvider;
pub use cohere::CohereProvider;
pub use error::{ApiError, ApiErrorKind, LlmError};
pub use fallback::FallbackProvider;
pub use futures_util::StreamExt;
pub use gemini::GeminiProvider;
pub use http::DEFAULT_MAX_RETRIES;
pub use openai::OpenAiProvider;
pub use runner::run_tools;
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

fn is_loopback(authority: &str) -> bool {
    let host = if let Some(rest) = authority.strip_prefix('[') {
        rest.split(']').next().unwrap_or_default()
    } else {
        authority.split(':').next().unwrap_or_default()
    };
    host == "localhost" || host.ends_with(".localhost") || host == "::1" || host.starts_with("127.")
}

/// Checks `base_url` and the key's transport before any request.
pub(crate) fn validate_endpoint(
    provider: &'static str,
    api_key: &str,
    require_key: bool,
    base_url: &str,
) -> Result<()> {
    let authority = checked_host(provider, base_url)?;
    if base_url.starts_with("http://") && !api_key.is_empty() && !is_loopback(authority) {
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

fn validate_image(i: usize, source: &ImageSource) -> Result<()> {
    match source {
        ImageSource::Url { url } => {
            if !url.starts_with("https://") && !url.starts_with("http://") {
                return Err(invalid(format!(
                    "messages[{i}]: image url must start with http:// or https://"
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
