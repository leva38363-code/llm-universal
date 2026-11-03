use async_trait::async_trait;
use serde_json::{json, Value};

use crate::http::{secret_header, Http};
use crate::openai::{
    float_vec, parse_arguments, parse_tool_calls, usage, validate_embed_args, OpenAiProvider,
};
use crate::stream::{self, parse_data, Accumulator, ChatStream, StreamEvent};
use crate::{
    ChatRequest, ChatResponse, EmbeddingProvider, Embeddings, LlmError, LlmProvider, Message,
    ModelInfo, Part, Result, Role, StopReason,
};

/// Client for the Cohere Chat API v2 (`POST {base_url}/chat`).
///
/// Requests use the same message, image and tool shapes as OpenAI; responses differ.
#[derive(Clone)]
pub struct CohereProvider {
    http: Http,
    api_key: String,
    base_url: String,
}

impl std::fmt::Debug for CohereProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CohereProvider")
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .field("max_retries", &self.http.max_retries)
            .finish()
    }
}

impl CohereProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: Http::default(),
            api_key: api_key.into(),
            base_url: "https://api.cohere.com/v2".to_string(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Uses your own `reqwest::ClientBuilder` (proxy, pool, TLS settings). Redirects are always
    /// turned off, so the key is never sent to another host.
    pub fn with_client_builder(mut self, builder: reqwest::ClientBuilder) -> crate::Result<Self> {
        self.http.client = builder
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(crate::LlmError::from_reqwest)?;
        Ok(self)
    }

    /// How many times a failed request is retried (default [`DEFAULT_MAX_RETRIES`](crate::DEFAULT_MAX_RETRIES); 0 disables).
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.http.max_retries = max_retries;
        self
    }

    /// Also repeat requests the server may already have processed (5xx, 408, 409, timeouts after
    /// sending). Off by default: a repeat can generate and bill the answer twice.
    pub fn with_repeat_post(mut self, repeat: bool) -> Self {
        self.http.repeat_post = repeat;
        self
    }

    /// Limit for one non-streaming attempt (default 600 s).
    pub fn with_request_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http.request_timeout = timeout;
        self
    }

    /// Overall limit for a stream, including keep-alives (default 1 h).
    pub fn with_stream_deadline(mut self, deadline: std::time::Duration) -> Self {
        self.http.stream_deadline = deadline;
        self
    }

    fn build_body(req: &ChatRequest) -> Value {
        let mut body = OpenAiProvider::build_body(req, true);
        if let Some(obj) = body.as_object_mut() {
            // Cohere names the limit `max_tokens`, takes `json_object` + schema and its own thinking switch.
            if let Some(max_tokens) = obj.remove("max_completion_tokens") {
                obj.insert("max_tokens".into(), max_tokens);
            }
            obj.remove("response_format");
            obj.remove("reasoning_effort");
        }
        if let Some(format) = &req.response_format {
            body["response_format"] =
                json!({ "type": "json_object", "json_schema": format.schema });
        }
        if req.reasoning.is_some() {
            body["thinking"] = json!({ "type": "enabled" });
        }
        body
    }

    fn post(&self, path: &str) -> Result<reqwest::RequestBuilder> {
        Ok(self
            .http
            .client
            .post(format!("{}{path}", self.base_url))
            .header(
                reqwest::header::AUTHORIZATION,
                secret_header("CohereProvider", &format!("Bearer {}", self.api_key))?,
            ))
    }

    /// Model listing lives on the v1 API.
    fn models_url(&self) -> String {
        match self.base_url.strip_suffix("/v2") {
            Some(root) => format!("{root}/v1/models?page_size=1000"),
            None => format!("{}/models?page_size=1000", self.base_url),
        }
    }
}

fn parse_response(v: &Value, requested_model: &str) -> Result<ChatResponse> {
    let message = &v["message"];
    if !message.is_object() {
        return Err(LlmError::InvalidResponse("missing field `message`".into()));
    }

    let mut parts = Vec::new();
    for block in message["content"].as_array().into_iter().flatten() {
        match block["type"].as_str() {
            Some("text") => {
                if let Some(text) = block["text"].as_str().filter(|t| !t.is_empty()) {
                    parts.push(Part::text(text));
                }
            }
            Some("thinking") => parts.push(Part::Reasoning {
                text: block["thinking"].as_str().unwrap_or_default().to_string(),
                raw: Value::Null,
            }),
            _ => {}
        }
    }
    parts.extend(parse_tool_calls(&message["tool_calls"]));

    let stop_reason = match v["finish_reason"].as_str() {
        Some("COMPLETE" | "STOP_SEQUENCE") => StopReason::EndTurn,
        Some("TOOL_CALL") => StopReason::ToolUse,
        Some("MAX_TOKENS") => StopReason::MaxTokens,
        other => StopReason::Other(other.unwrap_or("unknown").to_string()),
    };

    Ok(ChatResponse {
        // The response does not echo the model.
        model: requested_model.to_string(),
        message: Message::new(Role::Assistant, parts),
        stop_reason,
        usage: usage(&v["usage"]["tokens"], "input_tokens", "output_tokens"),
    })
}

/// Rebuilds the response from `content-delta`, `tool-call-*` and `message-end` events.
struct CohereStream {
    model: String,
    text: String,
    thinking: String,
    /// (id, name, arguments)
    calls: Vec<(String, String, String)>,
    end: Value,
    done: bool,
}

impl Accumulator for CohereStream {
    fn on_event(&mut self, _event: &str, data: &str) -> Result<Vec<StreamEvent>> {
        let v = parse_data(data)?;
        let message = &v["delta"]["message"];
        let mut out = Vec::new();
        match v["type"].as_str() {
            Some("content-delta") => {
                if let Some(text) = message["content"]["text"]
                    .as_str()
                    .filter(|t| !t.is_empty())
                {
                    self.text.push_str(text);
                    out.push(StreamEvent::TextDelta(text.to_string()));
                }
                if let Some(text) = message["content"]["thinking"]
                    .as_str()
                    .filter(|t| !t.is_empty())
                {
                    self.thinking.push_str(text);
                    out.push(StreamEvent::ReasoningDelta(text.to_string()));
                }
            }
            Some("tool-call-start") => {
                let call = &message["tool_calls"];
                let id = call["id"].as_str().unwrap_or_default().to_string();
                let name = call["function"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let args = call["function"]["arguments"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                out.push(StreamEvent::ToolCallStart {
                    id: id.clone(),
                    name: name.clone(),
                });
                self.calls.push((id, name, args));
            }
            Some("tool-call-delta") => {
                let args = message["tool_calls"]["function"]["arguments"]
                    .as_str()
                    .unwrap_or_default();
                if let Some(call) = self.calls.last_mut() {
                    call.2.push_str(args);
                    out.push(StreamEvent::ToolCallDelta {
                        id: call.0.clone(),
                        arguments: args.to_string(),
                    });
                }
            }
            Some("message-end") => {
                self.end = v["delta"].clone();
                self.done = true;
            }
            _ => {}
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<ChatResponse> {
        if !self.done {
            return Err(crate::stream::ended_early());
        }
        let mut parts = Vec::new();
        if !self.thinking.is_empty() {
            parts.push(Part::Reasoning {
                text: std::mem::take(&mut self.thinking),
                raw: Value::Null,
            });
        }
        if !self.text.is_empty() {
            parts.push(Part::text(std::mem::take(&mut self.text)));
        }
        for (id, name, args) in self.calls.drain(..) {
            parts.push(Part::ToolCall {
                id,
                name,
                arguments: parse_arguments(&args),
                signature: None,
            });
        }
        let response = parse_response(
            &json!({ "message": {}, "finish_reason": self.end["finish_reason"], "usage": self.end["usage"] }),
            &self.model,
        )?;
        Ok(ChatResponse {
            message: Message::new(Role::Assistant, parts),
            ..response
        })
    }
}

#[async_trait]
impl LlmProvider for CohereProvider {
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        crate::validate(
            "CohereProvider",
            &self.api_key,
            true,
            &self.base_url,
            64,
            req,
        )?;
        let builder = self.post("/chat")?.json(&Self::build_body(req));
        parse_response(&self.http.json(builder).await?, &req.model)
    }

    async fn chat_stream(&self, req: &ChatRequest) -> Result<ChatStream> {
        crate::validate(
            "CohereProvider",
            &self.api_key,
            true,
            &self.base_url,
            64,
            req,
        )?;
        let mut body = Self::build_body(req);
        body["stream"] = json!(true);
        let response = self.http.send(self.post("/chat")?.json(&body)).await?;
        Ok(stream::drive(
            response,
            Box::new(CohereStream {
                model: req.model.clone(),
                text: String::new(),
                thinking: String::new(),
                calls: Vec::new(),
                end: Value::Null,
                done: false,
            }),
            self.http.stream_deadline,
        ))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        crate::validate_endpoint("CohereProvider", &self.api_key, true, &self.base_url)?;
        let builder = self.http.client.get(self.models_url()).header(
            reqwest::header::AUTHORIZATION,
            secret_header("CohereProvider", &format!("Bearer {}", self.api_key))?,
        );
        let v = self.http.json(builder).await?;
        Ok(v["models"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|m| m["name"].as_str())
            .map(|name| ModelInfo {
                id: name.to_string(),
                display_name: None,
            })
            .collect())
    }
}

#[async_trait]
impl EmbeddingProvider for CohereProvider {
    async fn embed(&self, model: &str, inputs: &[&str]) -> Result<Embeddings> {
        crate::validate_endpoint("CohereProvider", &self.api_key, true, &self.base_url)?;
        validate_embed_args(model, inputs)?;
        let builder = self.post("/embed")?.json(&json!({
            "model": model,
            "texts": inputs,
            "input_type": "search_document",
            "embedding_types": ["float"],
        }));
        let v = self.http.json(builder).await?;
        Ok(Embeddings {
            vectors: v["embeddings"]["float"]
                .as_array()
                .into_iter()
                .flatten()
                .map(float_vec)
                .collect(),
            usage: usage(&v["meta"]["billed_units"], "input_tokens", "output_tokens"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ReasoningEffort;

    #[test]
    fn builds_cohere_specific_fields() {
        let req = ChatRequest::new("command-a")
            .max_tokens(50)
            .json_schema("answer", json!({"type": "object"}))
            .reasoning(ReasoningEffort::High)
            .message(Message::user("hi"));
        let body = CohereProvider::build_body(&req);
        assert_eq!(body["max_tokens"], 50);
        assert!(body.get("max_completion_tokens").is_none());
        assert!(body.get("reasoning_effort").is_none());
        assert_eq!(body["response_format"]["type"], "json_object");
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["messages"][0]["content"], "hi");
    }

    #[test]
    fn parses_tool_call_response() {
        let v = json!({
            "finish_reason": "TOOL_CALL",
            "message": {
                "role": "assistant",
                "tool_plan": "I will look up the weather.",
                "tool_calls": [{
                    "id": "get_weather_abc",
                    "type": "function",
                    "function": { "name": "get_weather", "arguments": "{\"city\":\"Berlin\"}" }
                }]
            },
            "usage": { "tokens": { "input_tokens": 12, "output_tokens": 8 } }
        });
        let resp = parse_response(&v, "command-a").unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.tool_calls()[0].arguments["city"], "Berlin");
        assert_eq!(resp.usage.input_tokens, 12);
        assert_eq!(resp.model, "command-a");
    }

    #[test]
    fn parses_text_response() {
        let v = json!({
            "finish_reason": "COMPLETE",
            "message": { "role": "assistant", "content": [{ "type": "text", "text": "Hello" }] }
        });
        let resp = parse_response(&v, "command-a").unwrap();
        assert_eq!(resp.text(), "Hello");
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn stream_rebuilds_text_and_tool_calls() {
        let mut acc = CohereStream {
            model: "command-a".into(),
            text: String::new(),
            thinking: String::new(),
            calls: Vec::new(),
            end: Value::Null,
            done: false,
        };
        let events = [
            r#"{"type":"message-start","id":"x"}"#,
            r#"{"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"Hi"}}}}"#,
            r#"{"type":"tool-call-start","index":0,"delta":{"message":{"tool_calls":{"id":"c1","type":"function","function":{"name":"f","arguments":""}}}}}"#,
            r#"{"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":{"function":{"arguments":"{\"x\":1}"}}}}}"#,
            r#"{"type":"message-end","delta":{"finish_reason":"TOOL_CALL","usage":{"tokens":{"input_tokens":3,"output_tokens":5}}}}"#,
        ];
        for e in events {
            acc.on_event("", e).unwrap();
        }
        let resp = acc.finish().unwrap();
        assert_eq!(resp.text(), "Hi");
        assert_eq!(resp.tool_calls()[0].arguments["x"], 1);
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.usage.output_tokens, 5);
    }

    #[test]
    fn models_use_the_v1_api() {
        assert_eq!(
            CohereProvider::new("k").models_url(),
            "https://api.cohere.com/v1/models?page_size=1000"
        );
    }
}
