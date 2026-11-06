use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;

use crate::{LlmError, Result};

/// Author of a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    /// Tool results. Sent as role `tool` to OpenAI and Cohere, as `tool_result` blocks in a
    /// `user` turn to Anthropic, and as `functionResponse` parts to Gemini.
    Tool,
}

/// Where an image comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageSource {
    /// A public `http(s)://` URL. Not supported by Gemini; use [`ImageSource::Base64`] there.
    Url { url: String },
    /// Inline image data. `media_type` is `image/png`, `image/jpeg`, `image/gif` or `image/webp`.
    Base64 { media_type: String, data: String },
}

/// One piece of a message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Part {
    Text {
        text: String,
    },
    /// An image in a user message.
    Image {
        source: ImageSource,
    },
    /// Model reasoning ("thinking"). `raw` is the provider's original block; it is sent back
    /// unchanged on the next turn (Anthropic and Gemini require that) and skipped where the
    /// provider does not accept it back. `text` may be empty when the provider hides reasoning.
    Reasoning {
        text: String,
        #[serde(default)]
        raw: Value,
    },
    ToolCall {
        id: String,
        name: String,
        /// Parsed arguments. If a provider returns arguments that are not valid JSON
        /// (for example cut off by `max_tokens`), the raw string is kept as `Value::String`.
        arguments: Value,
        /// Gemini's `thoughtSignature` for this call; it must be sent back with the call.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    ToolResult {
        call_id: String,
        content: String,
        is_error: bool,
    },
    /// A provider block this crate does not model (e.g. Anthropic server-tool blocks or Gemini
    /// code-execution parts). Kept as received and sent back unchanged to the same provider.
    Other {
        raw: Value,
    },
}

impl Part {
    pub fn text(text: impl Into<String>) -> Self {
        Part::Text { text: text.into() }
    }

    /// An image from a public URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        Part::Image {
            source: ImageSource::Url { url: url.into() },
        }
    }

    /// An inline image. `data` is base64 without the `data:` prefix.
    pub fn image_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Part::Image {
            source: ImageSource::Base64 {
                media_type: media_type.into(),
                data: data.into(),
            },
        }
    }

    /// A tool result to send back for `call_id`.
    pub fn tool_result(
        call_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Part::ToolResult {
            call_id: call_id.into(),
            content: content.into(),
            is_error,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub parts: Vec<Part>,
}

impl Message {
    pub fn new(role: Role, parts: Vec<Part>) -> Self {
        Self { role, parts }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self::text(Role::User, text)
    }

    /// A user message with text and one image (see [`Part::image_url`] and [`Part::image_base64`]).
    pub fn user_with_image(text: impl Into<String>, image: Part) -> Self {
        Self::new(Role::User, vec![Part::text(text), image])
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self::text(Role::Assistant, text)
    }

    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self::new(role, vec![Part::text(text)])
    }

    /// A single tool result. To return several results from one step, put them in one
    /// message with `Message::new(Role::Tool, parts)`.
    pub fn tool_result(
        call_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self::new(
            Role::Tool,
            vec![Part::tool_result(call_id, content, is_error)],
        )
    }

    /// All text parts joined together (reasoning is not included).
    pub fn text_content(&self) -> String {
        self.parts
            .iter()
            .filter_map(|p| match p {
                Part::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub(crate) fn has_images(&self) -> bool {
        self.parts.iter().any(|p| matches!(p, Part::Image { .. }))
    }
}

/// A function the model may call. `parameters` is a JSON Schema with `"type": "object"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl Tool {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// A tool call taken out of a response.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Asks the model to answer with JSON matching `schema`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResponseFormat {
    /// Schema name (required by OpenAI; letters, digits, `_` and `-`).
    pub name: String,
    /// JSON Schema of the answer.
    pub schema: Value,
}

/// How hard the model should think, on models that support reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
        }
    }
}

/// Provider-neutral chat request.
#[derive(Debug, Clone, Default)]
pub struct ChatRequest {
    pub model: String,
    /// System prompt. Sent as a `system` message to OpenAI/Cohere, as the top-level
    /// `system` field to Anthropic and as `systemInstruction` to Gemini.
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<Tool>,
    /// Required by Anthropic; [`DEFAULT_MAX_TOKENS`] is used when unset.
    pub max_tokens: Option<u32>,
    /// Anthropic accepts 0.0–1.0, and newer Anthropic models accept only 1.0.
    /// Leave unset unless the model supports it.
    pub temperature: Option<f32>,
    /// Structured output: the answer is JSON matching this schema. Read it with [`ChatResponse::json`].
    pub response_format: Option<ResponseFormat>,
    /// Reasoning effort / thinking. Reasoning comes back as [`Part::Reasoning`].
    pub reasoning: Option<ReasoningEffort>,
    /// Anthropic only: mark the system prompt for prompt caching (`cache_control: ephemeral`).
    /// Ignored by other providers. Off by default.
    pub cache_system: bool,
}

/// `max_tokens` sent to Anthropic when the request does not set one.
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

impl ChatRequest {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    /// Anthropic: cache the system prompt between requests (cheaper for long, repeated prompts).
    pub fn cache_system(mut self) -> Self {
        self.cache_system = true;
        self
    }

    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Asks for JSON matching `schema`.
    pub fn json_schema(mut self, name: impl Into<String>, schema: Value) -> Self {
        self.response_format = Some(ResponseFormat {
            name: name.into(),
            schema,
        });
        self
    }

    pub fn reasoning(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning = Some(effort);
        self
    }
}

/// Why generation stopped, normalized across providers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// Natural end of the turn (`stop` / `end_turn` / `stop_sequence` / `STOP` / `COMPLETE`).
    EndTurn,
    /// The model wants tools to be called.
    ToolUse,
    /// Token limit reached.
    MaxTokens,
    /// A long-running turn was paused (`pause_turn`). Send the response back as-is to continue.
    Paused,
    /// The model declined or the output was blocked for policy reasons.
    Refusal,
    /// Any other value returned by the provider.
    Other(String),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    /// Input tokens that were not read from or written to a prompt cache.
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Input tokens read from a prompt cache (Anthropic `cache_read_input_tokens`).
    /// Billed at 0.1x the input price.
    pub cache_read_tokens: u32,
    /// Input tokens written to a prompt cache (Anthropic `cache_creation_input_tokens`).
    /// Billed at 1.25x the input price.
    pub cache_write_tokens: u32,
}

impl std::ops::AddAssign for Usage {
    /// Sums usage over several responses (for example, one chat session).
    fn add_assign(&mut self, other: Usage) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.cache_read_tokens = self
            .cache_read_tokens
            .saturating_add(other.cache_read_tokens);
        self.cache_write_tokens = self
            .cache_write_tokens
            .saturating_add(other.cache_write_tokens);
    }
}

impl Usage {
    /// All tokens that count against a token limit: input (cached or not), output.
    pub fn total_tokens(&self) -> u64 {
        u64::from(self.input_tokens)
            + u64::from(self.cache_read_tokens)
            + u64::from(self.cache_write_tokens)
            + u64::from(self.output_tokens)
    }

    /// Cost in your currency, given prices per million input and output tokens.
    /// Cache reads are priced at 0.1x and cache writes at 1.25x the input price (Anthropic's
    /// multipliers; OpenAI cached tokens are already part of `input_tokens`, so leave the cache
    /// fields at 0 for them). Prices change and differ per model, so they are passed in rather
    /// than built in. A NaN or infinite result means the prices were not usable.
    pub fn cost(&self, input_per_million: f64, output_per_million: f64) -> f64 {
        (f64::from(self.input_tokens)
            + f64::from(self.cache_read_tokens) * 0.1
            + f64::from(self.cache_write_tokens) * 1.25)
            * input_per_million
            / 1_000_000.0
            + f64::from(self.output_tokens) * output_per_million / 1_000_000.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatResponse {
    pub model: String,
    pub message: Message,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

impl ChatResponse {
    /// All text of the response.
    pub fn text(&self) -> String {
        self.message.text_content()
    }

    /// All reasoning text of the response.
    pub fn reasoning(&self) -> String {
        self.message
            .parts
            .iter()
            .filter_map(|p| match p {
                Part::Reasoning { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Tool calls the model asked for.
    pub fn tool_calls(&self) -> Vec<ToolCall> {
        self.message
            .parts
            .iter()
            .filter_map(|p| match p {
                Part::ToolCall {
                    id,
                    name,
                    arguments,
                    ..
                } => Some(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                }),
                _ => None,
            })
            .collect()
    }

    /// Parses the text as JSON into `T` (use with [`ChatRequest::json_schema`]).
    pub fn json<T: DeserializeOwned>(&self) -> Result<T> {
        let text = self.text();
        serde_json::from_str(&text).map_err(|source| LlmError::Deserialize {
            source,
            content: crate::error::BodySnippet::new(&text),
        })
    }
}

/// A model available to the key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    pub display_name: Option<String>,
}

/// Result of an embeddings call: one vector per input, in input order.
#[derive(Debug, Clone, PartialEq)]
pub struct Embeddings {
    pub vectors: Vec<Vec<f32>>,
    pub usage: Usage,
}

#[cfg(test)]
mod usage_tests {
    use super::Usage;

    #[test]
    fn usage_sums_across_responses() {
        let mut total = Usage::default();
        total += Usage {
            input_tokens: 100,
            output_tokens: 20,
            ..Default::default()
        };
        total += Usage {
            input_tokens: 50,
            output_tokens: 5,
            ..Default::default()
        };
        assert_eq!(total.input_tokens, 150);
        assert_eq!(total.output_tokens, 25);
    }

    #[test]
    fn cost_is_per_million() {
        let u = Usage {
            input_tokens: 1_000_000,
            output_tokens: 500_000,
            ..Default::default()
        };
        assert!((u.cost(3.0, 15.0) - 10.5).abs() < 1e-9);
    }

    #[test]
    fn cache_tokens_are_priced_by_anthropic_multipliers() {
        let u = Usage {
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: 1_000_000,
            cache_write_tokens: 1_000_000,
        };
        // 1.0 * 0.1 + 1.0 * 1.25 = 1.35 million input-price units
        assert!((u.cost(10.0, 0.0) - 13.5).abs() < 1e-9);
        assert_eq!(u.total_tokens(), 2_000_000);
    }

    #[test]
    fn sum_saturates_instead_of_overflowing() {
        let mut u = Usage {
            input_tokens: u32::MAX,
            ..Default::default()
        };
        u += Usage {
            input_tokens: 10,
            ..Default::default()
        };
        assert_eq!(u.input_tokens, u32::MAX);
    }
}
