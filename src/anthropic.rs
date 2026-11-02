use std::collections::BTreeMap;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::http::{secret_header, Http};
use crate::openai::{parse_arguments, usage};
use crate::stream::{self, parse_data, Accumulator, ChatStream, StreamEvent};
use crate::{
    ApiError, ChatRequest, ChatResponse, ImageSource, LlmError, LlmProvider, Message, ModelInfo,
    Part, Result, Role, StopReason, DEFAULT_MAX_TOKENS,
};

/// Value of the `anthropic-version` header.
const API_VERSION: &str = "2023-06-01";
/// `anthropic_version` body field required by Vertex AI.
const VERTEX_API_VERSION: &str = "vertex-2023-10-16";

/// Client for the Anthropic Messages API (`POST {base_url}/v1/messages`), or for Claude on
/// Google Cloud Vertex AI via [`vertex`](Self::vertex).
#[derive(Clone)]
pub struct AnthropicProvider {
    http: Http,
    api_key: String,
    base_url: String,
    vertex: Option<Vertex>,
}

#[derive(Clone, Debug)]
pub(crate) struct Vertex {
    pub(crate) project: String,
    pub(crate) region: String,
}

impl Vertex {
    pub(crate) fn base_url(region: &str) -> String {
        if region == "global" {
            "https://aiplatform.googleapis.com/v1".to_string()
        } else {
            format!("https://{region}-aiplatform.googleapis.com/v1")
        }
    }

    /// `{base}/projects/{project}/locations/{region}/publishers/{publisher}/models/{model}:{method}`
    pub(crate) fn url(
        &self,
        provider: &str,
        base_url: &str,
        publisher: &str,
        model: &str,
        method: &str,
    ) -> Result<String> {
        let project = crate::checked_path_segment(provider, "project", &self.project, &[])?;
        let region = crate::checked_path_segment(provider, "region", &self.region, &[])?;
        let model = crate::checked_path_segment(provider, "model", model, &['@'])?;
        Ok(format!(
            "{base_url}/projects/{project}/locations/{region}/publishers/{publisher}/models/{model}:{method}"
        ))
    }
}

impl std::fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .field("vertex", &self.vertex)
            .field("max_retries", &self.http.max_retries)
            .finish()
    }
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: Http::default(),
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".to_string(),
            vertex: None,
        }
    }

    /// Claude on Google Cloud Vertex AI. `access_token` is an OAuth token
    /// (e.g. `gcloud auth print-access-token`); `region` is e.g. `global` or `us-east5`.
    pub fn vertex(
        project: impl Into<String>,
        region: impl Into<String>,
        access_token: impl Into<String>,
    ) -> Self {
        let region = region.into();
        Self {
            http: Http::default(),
            api_key: access_token.into(),
            base_url: Vertex::base_url(&region),
            vertex: Some(Vertex {
                project: project.into(),
                region,
            }),
        }
    }

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

    fn build_body(req: &ChatRequest) -> Value {
        // Anthropic has no `system` role in `messages`: system messages are merged into `system`.
        let mut system_parts: Vec<String> = req.system.iter().cloned().collect();
        let mut messages = Vec::new();
        for message in &req.messages {
            if message.role == Role::System {
                system_parts.push(message.text_content());
            } else {
                messages.push(to_anthropic_message(message));
            }
        }

        let mut body = json!({
            "model": req.model,
            "max_tokens": req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            "messages": messages,
        });
        if !system_parts.is_empty() {
            body["system"] = json!(system_parts.join("\n\n"));
        }
        if !req.tools.is_empty() {
            let tools: Vec<Value> = req
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.parameters,
                    })
                })
                .collect();
            body["tools"] = json!(tools);
        }
        if let Some(temperature) = req.temperature {
            body["temperature"] = json!(temperature);
        }
        let mut output_config = serde_json::Map::new();
        if let Some(format) = &req.response_format {
            output_config.insert(
                "format".into(),
                json!({ "type": "json_schema", "schema": format.schema }),
            );
        }
        if let Some(effort) = req.reasoning {
            // Adaptive thinking with readable summaries; depth is set by effort.
            body["thinking"] = json!({ "type": "adaptive", "display": "summarized" });
            output_config.insert("effort".into(), json!(effort.as_str()));
        }
        if !output_config.is_empty() {
            body["output_config"] = Value::Object(output_config);
        }
        body
    }

    fn request(&self, url: String) -> Result<reqwest::RequestBuilder> {
        let builder = self.http.client.post(url);
        Ok(if self.vertex.is_some() {
            builder.header(
                reqwest::header::AUTHORIZATION,
                secret_header("AnthropicProvider", &format!("Bearer {}", self.api_key))?,
            )
        } else {
            builder
                .header(
                    "x-api-key",
                    secret_header("AnthropicProvider", &self.api_key)?,
                )
                .header("anthropic-version", API_VERSION)
        })
    }

    /// URL and body for a chat call, streaming or not.
    fn chat_request(&self, req: &ChatRequest, stream: bool) -> Result<reqwest::RequestBuilder> {
        // Anthropic allows tool names up to 128 characters.
        crate::validate(
            "AnthropicProvider",
            &self.api_key,
            true,
            &self.base_url,
            128,
            req,
        )?;
        let mut body = Self::build_body(req);
        if stream {
            body["stream"] = json!(true);
        }
        let url = match &self.vertex {
            Some(vertex) => {
                // Vertex takes the model from the URL and the version from the body.
                if let Some(obj) = body.as_object_mut() {
                    obj.remove("model");
                    obj.insert("anthropic_version".into(), json!(VERTEX_API_VERSION));
                }
                let method = if stream {
                    "streamRawPredict"
                } else {
                    "rawPredict"
                };
                vertex.url(
                    "AnthropicProvider",
                    &self.base_url,
                    "anthropic",
                    &req.model,
                    method,
                )?
            }
            None => format!("{}/v1/messages", self.base_url),
        };
        Ok(self.request(url)?.json(&body))
    }
}

fn to_anthropic_message(message: &Message) -> Value {
    // Tool results are `tool_result` blocks inside a `user` turn.
    let role = if message.role == Role::Assistant {
        "assistant"
    } else {
        "user"
    };
    let content: Vec<Value> = message
        .parts
        .iter()
        .filter_map(to_anthropic_block)
        .collect();
    json!({ "role": role, "content": content })
}

fn to_anthropic_block(part: &Part) -> Option<Value> {
    Some(match part {
        // The API rejects empty text blocks.
        Part::Text { text } if text.is_empty() => return None,
        Part::Text { text } => json!({ "type": "text", "text": text }),
        Part::Image { source } => {
            let source = match source {
                ImageSource::Url { url } => json!({ "type": "url", "url": url }),
                ImageSource::Base64 { media_type, data } => {
                    json!({ "type": "base64", "media_type": media_type, "data": data })
                }
            };
            json!({ "type": "image", "source": source })
        }
        // Thinking blocks go back exactly as received (they are signed).
        Part::Reasoning { raw, .. } if raw.is_object() => raw.clone(),
        Part::Reasoning { .. } => return None,
        // Provider blocks this crate does not model (server tools, …) go back as received.
        Part::Other { raw } if raw.is_object() => raw.clone(),
        Part::Other { .. } => return None,
        Part::ToolCall {
            id,
            name,
            arguments,
            ..
        } => {
            // `input` must be an object; arguments kept raw (invalid JSON) cannot be sent as one.
            let input = if arguments.is_object() {
                arguments.clone()
            } else {
                json!({})
            };
            json!({ "type": "tool_use", "id": id, "name": name, "input": input })
        }
        Part::ToolResult {
            call_id,
            content,
            is_error,
        } => json!({
            "type": "tool_result",
            "tool_use_id": call_id,
            "content": content,
            "is_error": is_error,
        }),
    })
}

fn parse_response(v: &Value) -> Result<ChatResponse> {
    let blocks = v["content"]
        .as_array()
        .ok_or_else(|| LlmError::InvalidResponse("missing field `content`".into()))?;

    let mut parts = Vec::new();
    for block in blocks {
        match block["type"].as_str() {
            Some("text") => {
                if let Some(text) = block["text"].as_str().filter(|t| !t.is_empty()) {
                    parts.push(Part::text(text));
                }
            }
            Some("tool_use") => parts.push(Part::ToolCall {
                id: block["id"].as_str().unwrap_or_default().to_string(),
                name: block["name"].as_str().unwrap_or_default().to_string(),
                arguments: block["input"].clone(),
                signature: None,
            }),
            Some("thinking") => parts.push(Part::Reasoning {
                text: block["thinking"].as_str().unwrap_or_default().to_string(),
                raw: block.clone(),
            }),
            Some("redacted_thinking") => parts.push(Part::Reasoning {
                text: String::new(),
                raw: block.clone(),
            }),
            // Other block types (server tools, …) are kept so they can be sent back, e.g. to
            // continue a paused turn.
            Some(_) if block.is_object() => parts.push(Part::Other { raw: block.clone() }),
            _ => {}
        }
    }

    let stop_reason = match v["stop_reason"].as_str() {
        Some("end_turn" | "stop_sequence") => StopReason::EndTurn,
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens" | "model_context_window_exceeded") => StopReason::MaxTokens,
        Some("pause_turn") => StopReason::Paused,
        Some("refusal") => StopReason::Refusal,
        other => StopReason::Other(other.unwrap_or("unknown").to_string()),
    };

    Ok(ChatResponse {
        model: v["model"].as_str().unwrap_or_default().to_string(),
        message: Message::new(Role::Assistant, parts),
        stop_reason,
        usage: usage(&v["usage"], "input_tokens", "output_tokens"),
    })
}

/// Rebuilds the non-streaming message from `message_start` / `content_block_*` /
/// `message_delta` events.
#[derive(Default)]
struct AnthropicStream {
    message: Value,
    /// Content blocks by index; tool inputs are collected as JSON text first.
    blocks: BTreeMap<u64, Value>,
    tool_json: BTreeMap<u64, String>,
    done: bool,
}

impl Accumulator for AnthropicStream {
    fn on_event(&mut self, _event: &str, data: &str) -> Result<Vec<StreamEvent>> {
        let v = parse_data(data)?;
        let index = v["index"].as_u64().unwrap_or(0);
        let mut out = Vec::new();
        match v["type"].as_str() {
            Some("message_start") => {
                if !v["message"].is_object() {
                    return Err(LlmError::InvalidResponse(
                        "malformed message_start event".into(),
                    ));
                }
                self.message = v["message"].clone();
            }
            Some("content_block_start") => {
                let block = v["content_block"].clone();
                if !block.is_object() {
                    return Err(LlmError::InvalidResponse(
                        "malformed content_block_start event".into(),
                    ));
                }
                if block["type"] == "tool_use" {
                    out.push(StreamEvent::ToolCallStart {
                        id: block["id"].as_str().unwrap_or_default().to_string(),
                        name: block["name"].as_str().unwrap_or_default().to_string(),
                    });
                    self.tool_json.insert(index, String::new());
                }
                self.blocks.insert(index, block);
            }
            Some("content_block_delta") => {
                let delta = &v["delta"];
                let Some(block) = self.blocks.get_mut(&index) else {
                    return Ok(out);
                };
                match delta["type"].as_str() {
                    Some("text_delta") => {
                        let text = delta["text"].as_str().unwrap_or_default();
                        append(block, "text", text);
                        out.push(StreamEvent::TextDelta(text.to_string()));
                    }
                    Some("thinking_delta") => {
                        let text = delta["thinking"].as_str().unwrap_or_default();
                        append(block, "thinking", text);
                        out.push(StreamEvent::ReasoningDelta(text.to_string()));
                    }
                    Some("signature_delta") => {
                        append(
                            block,
                            "signature",
                            delta["signature"].as_str().unwrap_or_default(),
                        );
                    }
                    Some("input_json_delta") => {
                        let json = delta["partial_json"].as_str().unwrap_or_default();
                        self.tool_json.entry(index).or_default().push_str(json);
                        out.push(StreamEvent::ToolCallDelta {
                            id: block["id"].as_str().unwrap_or_default().to_string(),
                            arguments: json.to_string(),
                        });
                    }
                    _ => {}
                }
            }
            Some("content_block_stop") => {
                if let (Some(json), Some(block)) =
                    (self.tool_json.remove(&index), self.blocks.get_mut(&index))
                {
                    block["input"] = parse_arguments(&json);
                }
            }
            Some("message_delta") => {
                if !self.message.is_object() {
                    self.message = json!({});
                }
                if let Some(reason) = v["delta"]["stop_reason"].as_str() {
                    self.message["stop_reason"] = json!(reason);
                }
                if let Some(tokens) = v["usage"]["output_tokens"].as_u64() {
                    if !self.message["usage"].is_object() {
                        self.message["usage"] = json!({});
                    }
                    self.message["usage"]["output_tokens"] = json!(tokens);
                }
            }
            Some("message_stop") => self.done = true,
            Some("error") => {
                return Err(LlmError::Api(Box::new(ApiError::from_stream_error(
                    &v["error"],
                ))));
            }
            // `ping` and unknown events carry nothing to keep.
            _ => {}
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<ChatResponse> {
        if !self.done {
            return Err(crate::stream::ended_early());
        }
        let mut message = std::mem::take(&mut self.message);
        message["content"] = json!(std::mem::take(&mut self.blocks)
            .into_values()
            .collect::<Vec<_>>());
        parse_response(&message)
    }
}

/// Appends in place (no copy of the text so far). `block` is always an object here.
fn append(block: &mut Value, field: &str, text: &str) {
    match &mut block[field] {
        Value::String(current) => current.push_str(text),
        other => *other = json!(text),
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let builder = self.chat_request(req, false)?;
        parse_response(&self.http.json(builder).await?)
    }

    async fn chat_stream(&self, req: &ChatRequest) -> Result<ChatStream> {
        let builder = self.chat_request(req, true)?;
        let response = self.http.send(builder).await?;
        Ok(stream::drive(response, Box::<AnthropicStream>::default()))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        if self.vertex.is_some() {
            return Err(LlmError::InvalidArgument(
                "AnthropicProvider: list_models is not available on Vertex AI".into(),
            ));
        }
        crate::validate_endpoint("AnthropicProvider", &self.api_key, true, &self.base_url)?;
        let builder = self
            .http
            .client
            .get(format!("{}/v1/models?limit=1000", self.base_url))
            .header(
                "x-api-key",
                secret_header("AnthropicProvider", &self.api_key)?,
            )
            .header("anthropic-version", API_VERSION);
        let v = self.http.json(builder).await?;
        Ok(v["data"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|m| {
                Some(ModelInfo {
                    id: m["id"].as_str()?.to_string(),
                    display_name: m["display_name"].as_str().map(str::to_string),
                })
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ReasoningEffort, Tool};

    #[test]
    fn moves_system_and_maps_tool_results() {
        let req = ChatRequest::new("claude-opus-5-5")
            .system("sys")
            .tool(Tool::new("f", "d", json!({"type": "object"})))
            .message(Message::user("hi"))
            .message(Message::tool_result("toolu_1", "42", false));
        let body = AnthropicProvider::build_body(&req);
        assert_eq!(body["system"], "sys");
        assert_eq!(body["max_tokens"], DEFAULT_MAX_TOKENS);
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"][0]["type"], "tool_result");
        assert_eq!(body["messages"][1]["content"][0]["tool_use_id"], "toolu_1");
    }

    #[test]
    fn builds_images_schema_and_thinking() {
        let req = ChatRequest::new("claude-opus-5-5")
            .json_schema("answer", json!({"type": "object"}))
            .reasoning(ReasoningEffort::Low)
            .message(Message::user_with_image(
                "what?",
                Part::image_url("https://example.com/a.png"),
            ));
        let body = AnthropicProvider::build_body(&req);
        let image = &body["messages"][0]["content"][1];
        assert_eq!(image["type"], "image");
        assert_eq!(image["source"]["url"], "https://example.com/a.png");
        assert_eq!(body["output_config"]["format"]["type"], "json_schema");
        assert_eq!(body["output_config"]["effort"], "low");
        assert_eq!(body["thinking"]["type"], "adaptive");
    }

    #[test]
    fn thinking_blocks_round_trip_unchanged() {
        let v = json!({
            "model": "claude-opus-5-5",
            "content": [
                { "type": "thinking", "thinking": "plan", "signature": "sig123" },
                { "type": "tool_use", "id": "toolu_1", "name": "f", "input": { "x": 1 } }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 3, "output_tokens": 9 }
        });
        let resp = parse_response(&v).unwrap();
        assert_eq!(resp.reasoning(), "plan");
        let sent = to_anthropic_message(&resp.message);
        assert_eq!(sent["content"][0], v["content"][0]);
        assert_eq!(sent["content"][1]["type"], "tool_use");
    }

    #[test]
    fn stream_rebuilds_message() {
        let mut acc = AnthropicStream::default();
        let events = [
            r#"{"type":"message_start","message":{"model":"claude-opus-5-5","content":[],"usage":{"input_tokens":5,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"f","input":{}}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"x\":"}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"1}"}}"#,
            r#"{"type":"content_block_stop","index":1}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":12}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        for e in events {
            acc.on_event("", e).unwrap();
        }
        let resp = acc.finish().unwrap();
        assert_eq!(resp.text(), "Hi");
        assert_eq!(resp.tool_calls()[0].arguments["x"], 1);
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.usage.input_tokens, 5);
        assert_eq!(resp.usage.output_tokens, 12);
    }

    #[test]
    fn stream_error_event_maps_to_api_error() {
        let mut acc = AnthropicStream::default();
        let err = acc
            .on_event(
                "error",
                r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
            )
            .unwrap_err();
        assert_eq!(err.status(), Some(529));
        assert!(err.is_retryable());
    }

    #[test]
    fn vertex_url_and_body() {
        let p = AnthropicProvider::vertex("my-proj", "us-east5", "token");
        let req = ChatRequest::new("claude-opus-5-5").message(Message::user("hi"));
        let built = p.chat_request(&req, false).unwrap().build().unwrap();
        assert_eq!(
            built.url().as_str(),
            "https://us-east5-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-east5/publishers/anthropic/models/claude-opus-5-5:rawPredict"
        );
        let body: Value =
            serde_json::from_slice(built.body().unwrap().as_bytes().unwrap()).unwrap();
        assert!(body.get("model").is_none());
        assert_eq!(body["anthropic_version"], VERTEX_API_VERSION);
        assert!(built.headers()["authorization"].is_sensitive());

        let bad = AnthropicProvider::vertex("../x", "us-east5", "token");
        assert!(bad.chat_request(&req, false).is_err());
    }

    #[test]
    fn debug_redacts_api_key() {
        let debug = format!("{:?}", AnthropicProvider::new("sk-ant-secret"));
        assert!(!debug.contains("sk-ant-secret"));
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[test]
    fn server_tool_blocks_are_kept_for_a_paused_turn() {
        let v = json!({
            "content": [{ "type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {} }],
            "stop_reason": "pause_turn"
        });
        let resp = parse_response(&v).unwrap();
        assert_eq!(resp.stop_reason, StopReason::Paused);
        assert_eq!(
            to_anthropic_message(&resp.message)["content"][0],
            v["content"][0]
        );
    }

    #[test]
    fn raw_string_arguments_are_not_sent_as_input() {
        let msg = Message::new(
            Role::Assistant,
            vec![Part::ToolCall {
                id: "t".into(),
                name: "f".into(),
                arguments: json!("{\"x\":"),
                signature: None,
            }],
        );
        assert_eq!(to_anthropic_message(&msg)["content"][0]["input"], json!({}));
    }

    #[test]
    fn malformed_events_do_not_panic() {
        let mut acc = AnthropicStream::default();
        assert!(acc
            .on_event("", r#"{"type":"message_start","message":5}"#)
            .is_err());
        assert!(acc
            .on_event(
                "",
                r#"{"type":"content_block_start","index":0,"content_block":"x"}"#
            )
            .is_err());
        let mut acc = AnthropicStream::default();
        acc.on_event("", r#"{"type":"message_start","message":{"usage":1}}"#)
            .unwrap();
        acc.on_event(
            "",
            r#"{"type":"message_delta","delta":{},"usage":{"output_tokens":3}}"#,
        )
        .unwrap();
    }
}
