use std::collections::BTreeMap;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::http::{secret_header, Http};
use crate::stream::{self, parse_data, Accumulator, ChatStream, StreamEvent};
use crate::{
    ChatRequest, ChatResponse, EmbeddingProvider, Embeddings, ImageSource, LlmError, LlmProvider,
    Message, ModelInfo, Part, Result, Role, StopReason, Usage,
};

/// Client for the OpenAI Chat Completions API (`POST {base_url}/chat/completions`).
///
/// The same wire format is served by many other providers; use the presets
/// ([`azure`](Self::azure), [`bedrock`](Self::bedrock), [`groq`](Self::groq), [`mistral`](Self::mistral),
/// [`deepseek`](Self::deepseek), [`xai`](Self::xai), [`openrouter`](Self::openrouter),
/// [`together`](Self::together), [`fireworks`](Self::fireworks), [`perplexity`](Self::perplexity),
/// [`ollama`](Self::ollama), [`lm_studio`](Self::lm_studio)) or [`with_base_url`](Self::with_base_url)
/// for anything else (vLLM, LiteLLM, …).
#[derive(Clone)]
pub struct OpenAiProvider {
    http: Http,
    api_key: String,
    base_url: String,
    auth: Auth,
    /// Set by a preset whose resource or region is invalid; returned by every call.
    config_error: Option<String>,
}

/// Default OpenAI endpoint.
const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

/// How the key is sent.
#[derive(Clone, Copy, Debug)]
enum Auth {
    /// `Authorization: Bearer <key>` (OpenAI and most compatible servers).
    Bearer,
    /// `api-key: <key>` (Azure OpenAI).
    ApiKeyHeader,
}

impl std::fmt::Debug for OpenAiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiProvider")
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .field("max_retries", &self.http.max_retries)
            .finish()
    }
}

impl OpenAiProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: Http::default(),
            api_key: api_key.into(),
            base_url: OPENAI_BASE_URL.to_string(),
            auth: Auth::Bearer,
            config_error: None,
        }
    }

    /// A resource or region becomes part of the host name, so it must be a plain DNS label;
    /// otherwise `"evil.com/"` would send the key to another host.
    fn with_host_label(mut self, what: &str, value: &str) -> Self {
        let valid = !value.is_empty()
            && value.len() <= 63
            && value.chars().all(|c| c.is_ascii_alphanumeric() || c == '-');
        if !valid {
            self.config_error = Some(format!(
                "OpenAiProvider: {what} `{value}` may only contain a-z, A-Z, 0-9 and '-'"
            ));
        }
        self
    }

    fn preset(api_key: impl Into<String>, base_url: &str) -> Self {
        Self::new(api_key).with_base_url(base_url)
    }

    /// Azure OpenAI v1 API: `https://{resource}.openai.azure.com/openai/v1`, key in the `api-key` header.
    /// Pass the deployment name as the model.
    pub fn azure(resource: &str, api_key: impl Into<String>) -> Self {
        let mut provider = Self::preset(
            api_key,
            &format!("https://{resource}.openai.azure.com/openai/v1"),
        );
        provider.auth = Auth::ApiKeyHeader;
        provider.with_host_label("resource", resource)
    }

    /// Amazon Bedrock, OpenAI-compatible endpoint with a Bedrock API key:
    /// `https://bedrock-runtime.{region}.amazonaws.com/openai/v1`. Use Bedrock model IDs.
    pub fn bedrock(region: &str, api_key: impl Into<String>) -> Self {
        Self::preset(
            api_key,
            &format!("https://bedrock-runtime.{region}.amazonaws.com/openai/v1"),
        )
        .with_host_label("region", region)
    }

    /// Groq: `https://api.groq.com/openai/v1`.
    pub fn groq(api_key: impl Into<String>) -> Self {
        Self::preset(api_key, "https://api.groq.com/openai/v1")
    }

    /// Mistral AI: `https://api.mistral.ai/v1`.
    pub fn mistral(api_key: impl Into<String>) -> Self {
        Self::preset(api_key, "https://api.mistral.ai/v1")
    }

    /// DeepSeek: `https://api.deepseek.com/v1`.
    pub fn deepseek(api_key: impl Into<String>) -> Self {
        Self::preset(api_key, "https://api.deepseek.com/v1")
    }

    /// xAI (Grok): `https://api.x.ai/v1`.
    pub fn xai(api_key: impl Into<String>) -> Self {
        Self::preset(api_key, "https://api.x.ai/v1")
    }

    /// OpenRouter: `https://openrouter.ai/api/v1`.
    pub fn openrouter(api_key: impl Into<String>) -> Self {
        Self::preset(api_key, "https://openrouter.ai/api/v1")
    }

    /// Together AI: `https://api.together.xyz/v1`.
    pub fn together(api_key: impl Into<String>) -> Self {
        Self::preset(api_key, "https://api.together.xyz/v1")
    }

    /// Fireworks AI: `https://api.fireworks.ai/inference/v1`.
    pub fn fireworks(api_key: impl Into<String>) -> Self {
        Self::preset(api_key, "https://api.fireworks.ai/inference/v1")
    }

    /// Perplexity: `https://api.perplexity.ai`.
    pub fn perplexity(api_key: impl Into<String>) -> Self {
        Self::preset(api_key, "https://api.perplexity.ai")
    }

    /// Ollama on this machine: `http://localhost:11434/v1`, no key.
    pub fn ollama() -> Self {
        Self::preset("", "http://localhost:11434/v1")
    }

    /// LM Studio on this machine: `http://localhost:1234/v1`, no key.
    pub fn lm_studio() -> Self {
        Self::preset("", "http://localhost:1234/v1")
    }

    /// Points the client at another server. The key may be empty if the server does not need one.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Uses your own `reqwest::Client`, for example to share a connection pool or set a proxy.
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.http.client = client;
        self
    }

    /// How many times a failed request is retried (default [`DEFAULT_MAX_RETRIES`](crate::DEFAULT_MAX_RETRIES); 0 disables).
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.http.max_retries = max_retries;
        self
    }

    fn request(&self, method: reqwest::Method, path: &str) -> Result<reqwest::RequestBuilder> {
        if let Some(error) = &self.config_error {
            return Err(LlmError::InvalidArgument(error.clone()));
        }
        let mut builder = self
            .http
            .client
            .request(method, format!("{}{path}", self.base_url));
        if !self.api_key.is_empty() {
            builder = match self.auth {
                Auth::Bearer => builder.header(
                    reqwest::header::AUTHORIZATION,
                    secret_header("OpenAiProvider", &format!("Bearer {}", self.api_key))?,
                ),
                Auth::ApiKeyHeader => {
                    builder.header("api-key", secret_header("OpenAiProvider", &self.api_key)?)
                }
            };
        }
        Ok(builder)
    }

    /// OpenAI itself and Azure take `max_completion_tokens`; other compatible servers
    /// (Ollama, vLLM, Mistral, …) reliably understand the older `max_tokens`.
    fn is_openai(&self) -> bool {
        self.base_url == OPENAI_BASE_URL || matches!(self.auth, Auth::ApiKeyHeader)
    }

    /// Mistral rejects unknown fields such as `stream_options` and reports usage on its own.
    fn stream_usage(&self) -> bool {
        !self.base_url.starts_with("https://api.mistral.ai")
    }

    pub(crate) fn build_body(req: &ChatRequest, openai: bool) -> Value {
        let mut messages = Vec::new();
        if let Some(system) = &req.system {
            messages.push(json!({ "role": "system", "content": system }));
        }
        for message in &req.messages {
            messages.extend(to_openai_messages(message));
        }

        let mut body = json!({ "model": req.model, "messages": messages });

        if !req.tools.is_empty() {
            let tools: Vec<Value> = req
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            body["tools"] = json!(tools);
        }
        if let Some(max_tokens) = req.max_tokens {
            let field = if openai {
                "max_completion_tokens"
            } else {
                "max_tokens"
            };
            body[field] = json!(max_tokens);
        }
        if let Some(temperature) = req.temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(format) = &req.response_format {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": { "name": format.name, "schema": format.schema, "strict": true },
            });
        }
        if let Some(effort) = req.reasoning {
            body["reasoning_effort"] = json!(effort.as_str());
        }
        body
    }
}

/// OpenAI-style `image_url` content part (also used by Cohere).
pub(crate) fn image_url_part(source: &ImageSource) -> Value {
    let url = match source {
        ImageSource::Url { url } => url.clone(),
        ImageSource::Base64 { media_type, data } => format!("data:{media_type};base64,{data}"),
    };
    json!({ "type": "image_url", "image_url": { "url": url } })
}

pub(crate) fn to_openai_messages(message: &Message) -> Vec<Value> {
    if message.role == Role::Tool {
        // One `tool` message per result.
        return message
            .parts
            .iter()
            .filter_map(|p| match p {
                Part::ToolResult {
                    call_id, content, ..
                } => Some(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                })),
                _ => None,
            })
            .collect();
    }

    let role = match message.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant | Role::Tool => "assistant",
    };
    let tool_calls: Vec<Value> = message
        .parts
        .iter()
        .filter_map(|p| match p {
            Part::ToolCall {
                id,
                name,
                arguments,
                ..
            } => Some(json!({
                "id": id,
                "type": "function",
                "function": { "name": name, "arguments": arguments_json(arguments) },
            })),
            _ => None,
        })
        .collect();

    let content = if message.has_images() {
        // Text and images as content parts, in order. Reasoning is not sent back.
        let parts: Vec<Value> = message
            .parts
            .iter()
            .filter_map(|p| match p {
                Part::Text { text } if !text.is_empty() => {
                    Some(json!({ "type": "text", "text": text }))
                }
                Part::Image { source } => Some(image_url_part(source)),
                _ => None,
            })
            .collect();
        json!(parts)
    } else {
        let text = message.text_content();
        if text.is_empty() && !tool_calls.is_empty() {
            Value::Null
        } else {
            json!(text)
        }
    };
    let mut msg = json!({ "role": role, "content": content });
    if !tool_calls.is_empty() {
        msg["tool_calls"] = json!(tool_calls);
    }
    vec![msg]
}

/// Tool-call arguments as the JSON string OpenAI-style APIs expect. Arguments that were kept
/// raw (invalid JSON) are sent back as they came, not encoded a second time.
pub(crate) fn arguments_json(arguments: &Value) -> String {
    match arguments {
        Value::String(raw) => raw.clone(),
        other => other.to_string(),
    }
}

/// Parses an OpenAI-style `tool_calls` array (also used by Cohere).
pub(crate) fn parse_tool_calls(calls: &Value) -> Vec<Part> {
    let Some(calls) = calls.as_array() else {
        return Vec::new();
    };
    calls
        .iter()
        .map(|call| Part::ToolCall {
            id: call["id"].as_str().unwrap_or_default().to_string(),
            name: call["function"]["name"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            arguments: parse_arguments(call["function"]["arguments"].as_str().unwrap_or_default()),
            signature: None,
        })
        .collect()
}

/// Arguments are a JSON string. An empty string means a tool without arguments; invalid JSON
/// (e.g. cut off by max tokens) is kept raw instead of failing the call.
pub(crate) fn parse_arguments(raw: &str) -> Value {
    if raw.trim().is_empty() {
        json!({})
    } else {
        serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
    }
}

fn parse_response(v: &Value) -> Result<ChatResponse> {
    let choice = v["choices"]
        .get(0)
        .ok_or_else(|| LlmError::InvalidResponse("missing field `choices[0]`".into()))?;
    let message = &choice["message"];

    let mut parts = Vec::new();
    // Some compatible servers return reasoning: DeepSeek `reasoning_content`, OpenRouter `reasoning`.
    let reasoning = message["reasoning_content"]
        .as_str()
        .or_else(|| message["reasoning"].as_str());
    if let Some(text) = reasoning.filter(|t| !t.is_empty()) {
        parts.push(Part::Reasoning {
            text: text.to_string(),
            raw: Value::Null,
        });
    }
    if let Some(text) = message["content"].as_str().filter(|t| !t.is_empty()) {
        parts.push(Part::text(text));
    }
    // A refusal arrives in its own `refusal` field while `content` is null.
    let refusal = message["refusal"].as_str().filter(|t| !t.is_empty());
    if let Some(text) = refusal {
        parts.push(Part::text(text));
    }
    parts.extend(parse_tool_calls(&message["tool_calls"]));

    let stop_reason = match choice["finish_reason"].as_str() {
        _ if refusal.is_some() => StopReason::Refusal,
        Some("stop") => StopReason::EndTurn,
        Some("tool_calls" | "function_call") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        Some("content_filter") => StopReason::Refusal,
        other => StopReason::Other(other.unwrap_or("unknown").to_string()),
    };

    Ok(ChatResponse {
        model: v["model"].as_str().unwrap_or_default().to_string(),
        message: Message::new(Role::Assistant, parts),
        stop_reason,
        usage: usage(&v["usage"], "prompt_tokens", "completion_tokens"),
    })
}

pub(crate) fn usage(u: &Value, input: &str, output: &str) -> Usage {
    let tokens = |field: &str| u32::try_from(u[field].as_u64().unwrap_or(0)).unwrap_or(u32::MAX);
    Usage {
        input_tokens: tokens(input),
        output_tokens: tokens(output),
    }
}

/// Builds the non-streaming response shape from `chat.completion.chunk` events.
#[derive(Default)]
struct OpenAiStream {
    model: String,
    content: String,
    reasoning: String,
    refusal: String,
    /// Tool calls by `index`: (id, name, arguments).
    calls: BTreeMap<u64, (String, String, String)>,
    finish_reason: Option<String>,
    usage: Value,
    done: bool,
}

impl Accumulator for OpenAiStream {
    fn on_event(&mut self, _event: &str, data: &str) -> Result<Vec<StreamEvent>> {
        if data.trim() == "[DONE]" {
            self.done = true;
            return Ok(Vec::new());
        }
        let chunk = parse_data(data)?;
        if chunk["error"].is_object() {
            return Err(LlmError::Api(Box::new(crate::ApiError::from_stream_error(
                &chunk["error"],
            ))));
        }
        if let Some(model) = chunk["model"].as_str() {
            self.model = model.to_string();
        }
        if chunk["usage"].is_object() {
            self.usage = chunk["usage"].clone();
        }
        let mut out = Vec::new();
        let Some(choice) = chunk["choices"].get(0) else {
            return Ok(out);
        };
        let delta = &choice["delta"];
        if let Some(text) = delta["reasoning_content"]
            .as_str()
            .or_else(|| delta["reasoning"].as_str())
        {
            if !text.is_empty() {
                self.reasoning.push_str(text);
                out.push(StreamEvent::ReasoningDelta(text.to_string()));
            }
        }
        if let Some(text) = delta["content"].as_str().filter(|t| !t.is_empty()) {
            self.content.push_str(text);
            out.push(StreamEvent::TextDelta(text.to_string()));
        }
        if let Some(text) = delta["refusal"].as_str().filter(|t| !t.is_empty()) {
            self.refusal.push_str(text);
            out.push(StreamEvent::TextDelta(text.to_string()));
        }
        for call in delta["tool_calls"].as_array().into_iter().flatten() {
            let index = call["index"].as_u64().unwrap_or(0);
            let entry = self.calls.entry(index).or_default();
            if let Some(id) = call["id"].as_str().filter(|s| !s.is_empty()) {
                entry.0 = id.to_string();
            }
            // The name comes once; some servers repeat it in every chunk, so take only the first.
            if let Some(name) = call["function"]["name"]
                .as_str()
                .filter(|s| !s.is_empty() && entry.1.is_empty())
            {
                entry.1 = name.to_string();
                out.push(StreamEvent::ToolCallStart {
                    id: entry.0.clone(),
                    name: entry.1.clone(),
                });
            }
            if let Some(args) = call["function"]["arguments"]
                .as_str()
                .filter(|s| !s.is_empty())
            {
                entry.2.push_str(args);
                out.push(StreamEvent::ToolCallDelta {
                    id: entry.0.clone(),
                    arguments: args.to_string(),
                });
            }
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            self.finish_reason = Some(reason.to_string());
            self.done = true;
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<ChatResponse> {
        if !self.done {
            return Err(crate::stream::ended_early());
        }
        let tool_calls: Vec<Value> = self
            .calls
            .values()
            .map(|(id, name, args)| json!({ "id": id, "function": { "name": name, "arguments": args } }))
            .collect();
        let mut message = json!({ "content": self.content, "tool_calls": tool_calls });
        if !self.reasoning.is_empty() {
            message["reasoning_content"] = json!(self.reasoning);
        }
        if !self.refusal.is_empty() {
            message["refusal"] = json!(self.refusal);
        }
        parse_response(&json!({
            "model": self.model,
            "choices": [{ "message": message, "finish_reason": self.finish_reason }],
            "usage": self.usage,
        }))
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        // OpenAI allows tool names up to 64 characters.
        crate::validate(
            "OpenAiProvider",
            &self.api_key,
            false,
            &self.base_url,
            64,
            req,
        )?;
        let builder = self
            .request(reqwest::Method::POST, "/chat/completions")?
            .json(&Self::build_body(req, self.is_openai()));
        parse_response(&self.http.json(builder).await?)
    }

    async fn chat_stream(&self, req: &ChatRequest) -> Result<ChatStream> {
        crate::validate(
            "OpenAiProvider",
            &self.api_key,
            false,
            &self.base_url,
            64,
            req,
        )?;
        let mut body = Self::build_body(req, self.is_openai());
        body["stream"] = json!(true);
        if self.stream_usage() {
            body["stream_options"] = json!({ "include_usage": true });
        }
        let builder = self
            .request(reqwest::Method::POST, "/chat/completions")?
            .json(&body);
        let response = self.http.send(builder).await?;
        Ok(stream::drive(response, Box::<OpenAiStream>::default()))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        crate::validate_endpoint("OpenAiProvider", &self.api_key, false, &self.base_url)?;
        let v = self
            .http
            .json(self.request(reqwest::Method::GET, "/models")?)
            .await?;
        Ok(v["data"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|m| m["id"].as_str())
            .map(|id| ModelInfo {
                id: id.to_string(),
                display_name: None,
            })
            .collect())
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiProvider {
    async fn embed(&self, model: &str, inputs: &[&str]) -> Result<Embeddings> {
        crate::validate_endpoint("OpenAiProvider", &self.api_key, false, &self.base_url)?;
        validate_embed_args(model, inputs)?;
        let builder = self
            .request(reqwest::Method::POST, "/embeddings")?
            .json(&json!({ "model": model, "input": inputs }));
        let v = self.http.json(builder).await?;
        let mut data: Vec<&Value> = v["data"].as_array().into_iter().flatten().collect();
        // Results carry their input index; order by it.
        data.sort_by_key(|d| d["index"].as_u64().unwrap_or(0));
        Ok(Embeddings {
            vectors: data.iter().map(|d| float_vec(&d["embedding"])).collect(),
            usage: usage(&v["usage"], "prompt_tokens", "completion_tokens"),
        })
    }
}

pub(crate) fn validate_embed_args(model: &str, inputs: &[&str]) -> Result<()> {
    if model.trim().is_empty() {
        return Err(LlmError::InvalidArgument("model must not be empty".into()));
    }
    if inputs.is_empty() {
        return Err(LlmError::InvalidArgument("inputs must not be empty".into()));
    }
    Ok(())
}

pub(crate) fn float_vec(v: &Value) -> Vec<f32> {
    v.as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_f64)
        .map(|x| x as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ReasoningEffort, Tool};

    #[test]
    fn builds_tool_request() {
        let req = ChatRequest::new("gpt-4o")
            .system("sys")
            .max_tokens(100)
            .tool(Tool::new("f", "d", json!({"type": "object"})))
            .message(Message::user("hi"));
        let body = OpenAiProvider::build_body(&req, true);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["content"], "hi");
        assert_eq!(body["tools"][0]["function"]["name"], "f");
        assert_eq!(body["max_completion_tokens"], 100);
    }

    #[test]
    fn builds_images_json_schema_and_reasoning() {
        let req = ChatRequest::new("gpt-4o")
            .json_schema("answer", json!({"type": "object"}))
            .reasoning(ReasoningEffort::High)
            .message(Message::user_with_image(
                "what is this?",
                Part::image_base64("image/png", "AAAA"),
            ));
        let body = OpenAiProvider::build_body(&req, true);
        let content = &body["messages"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,AAAA");
        assert_eq!(body["response_format"]["json_schema"]["name"], "answer");
        assert_eq!(body["reasoning_effort"], "high");
    }

    #[test]
    fn parses_tool_call_response() {
        let v = json!({
            "model": "gpt-4o",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "f", "arguments": "{\"x\":1}" }
                    }]
                }
            }],
            "usage": { "prompt_tokens": 5, "completion_tokens": 7 }
        });
        let resp = parse_response(&v).unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.tool_calls()[0].arguments["x"], 1);
        assert_eq!(resp.usage.output_tokens, 7);
    }

    #[test]
    fn tolerates_empty_and_truncated_arguments() {
        let v = json!({
            "choices": [{
                "finish_reason": "length",
                "message": {
                    "content": null,
                    "tool_calls": [
                        { "id": "a", "function": { "name": "f", "arguments": "" } },
                        { "id": "b", "function": { "name": "f", "arguments": "{\"x\":" } }
                    ]
                }
            }]
        });
        let resp = parse_response(&v).unwrap();
        let calls = resp.tool_calls();
        assert_eq!(calls[0].arguments, json!({}));
        assert_eq!(calls[1].arguments, json!("{\"x\":"));
        assert_eq!(resp.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn maps_refusal_field() {
        let v = json!({
            "choices": [{
                "finish_reason": "stop",
                "message": { "content": null, "refusal": "I can't help with that." }
            }]
        });
        let resp = parse_response(&v).unwrap();
        assert_eq!(resp.stop_reason, StopReason::Refusal);
        assert_eq!(resp.text(), "I can't help with that.");
    }

    #[test]
    fn stream_accumulates_text_and_tool_calls() {
        let mut acc = OpenAiStream::default();
        let events = [
            r#"{"model":"gpt-4o","choices":[{"delta":{"content":"Hel"}}]}"#,
            r#"{"choices":[{"delta":{"content":"lo"}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{\"x\""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":1}"}}]},"finish_reason":"tool_calls"}]}"#,
            r#"{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4}}"#,
            "[DONE]",
        ];
        let mut deltas = String::new();
        for e in events {
            for ev in acc.on_event("", e).unwrap() {
                if let StreamEvent::TextDelta(t) = ev {
                    deltas.push_str(&t);
                }
            }
        }
        let resp = acc.finish().unwrap();
        assert_eq!(deltas, "Hello");
        assert_eq!(resp.text(), "Hello");
        assert_eq!(resp.tool_calls()[0].arguments["x"], 1);
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.usage.output_tokens, 4);
    }

    #[test]
    fn debug_redacts_api_key() {
        let debug = format!("{:?}", OpenAiProvider::new("sk-secret"));
        assert!(!debug.contains("sk-secret"));
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[test]
    fn raw_string_arguments_are_not_double_encoded() {
        let msg = Message::new(
            Role::Assistant,
            vec![Part::ToolCall {
                id: "c".into(),
                name: "f".into(),
                arguments: json!("{\"x\":"),
                signature: None,
            }],
        );
        let sent = to_openai_messages(&msg);
        assert_eq!(sent[0]["tool_calls"][0]["function"]["arguments"], "{\"x\":");
    }

    #[test]
    fn compatible_servers_get_max_tokens_and_openai_gets_max_completion_tokens() {
        let req = ChatRequest::new("m")
            .max_tokens(9)
            .message(Message::user("x"));
        assert_eq!(
            OpenAiProvider::build_body(&req, true)["max_completion_tokens"],
            9
        );
        assert_eq!(OpenAiProvider::build_body(&req, false)["max_tokens"], 9);
        assert!(OpenAiProvider::new("k").is_openai());
        assert!(OpenAiProvider::azure("res", "k").is_openai());
        assert!(!OpenAiProvider::groq("k").is_openai());
        assert!(!OpenAiProvider::mistral("k").stream_usage());
    }

    #[test]
    fn repeated_tool_name_chunks_are_not_concatenated() {
        let mut acc = OpenAiStream::default();
        let chunk = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"f","arguments":"{}"}}]}}]}"#;
        let first = acc.on_event("", chunk).unwrap();
        let second = acc.on_event("", r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"f"}}]},"finish_reason":"tool_calls"}]}"#).unwrap();
        assert!(first
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolCallStart { .. })));
        assert!(!second
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolCallStart { .. })));
        assert_eq!(acc.finish().unwrap().tool_calls()[0].name, "f");
    }
}
