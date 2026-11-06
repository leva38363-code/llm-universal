use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::anthropic::Vertex;
use crate::http::{secret_header, Http};
use crate::openai::{float_vec, usage, validate_embed_args};
use crate::stream::{self, parse_data, Accumulator, ChatStream, StreamEvent};
use crate::{
    ApiError, ChatRequest, ChatResponse, EmbeddingProvider, Embeddings, ImageSource, LlmError,
    LlmProvider, Message, ModelInfo, Part, Result, Role, StopReason,
};

/// Prefix of tool-call ids this crate generates when Gemini returns a call without an id.
/// Such ids are never sent back to the API.
const LOCAL_ID_PREFIX: &str = "llm-universal-call-";

/// Client for the Google Gemini API
/// (`POST {base_url}/models/{model}:generateContent`, key in `x-goog-api-key`),
/// or for Gemini on Vertex AI via [`vertex`](Self::vertex).
#[derive(Clone)]
pub struct GeminiProvider {
    http: Http,
    api_key: String,
    base_url: String,
    vertex: Option<Vertex>,
}

impl std::fmt::Debug for GeminiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiProvider")
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .field("vertex", &self.vertex)
            .field("max_retries", &self.http.max_retries)
            .finish()
    }
}

impl GeminiProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: Http::default(),
            api_key: api_key.into(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            vertex: None,
        }
    }

    /// Gemini on Google Cloud Vertex AI. `access_token` is an OAuth token
    /// (e.g. `gcloud auth print-access-token`); `region` is e.g. `global` or `us-central1`.
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

    fn build_body(req: &ChatRequest) -> Result<Value> {
        // Gemini needs the function name in `functionResponse`; recover it from the call.
        let names: HashMap<&str, &str> = req
            .messages
            .iter()
            .flat_map(|m| &m.parts)
            .filter_map(|p| match p {
                Part::ToolCall { id, name, .. } => Some((id.as_str(), name.as_str())),
                _ => None,
            })
            .collect();

        let mut system_parts: Vec<Value> =
            req.system.iter().map(|s| json!({ "text": s })).collect();
        let mut contents = Vec::new();
        for message in &req.messages {
            if message.role == Role::System {
                system_parts.push(json!({ "text": message.text_content() }));
                continue;
            }
            let role = if message.role == Role::Assistant {
                "model"
            } else {
                "user"
            };
            let mut parts = Vec::new();
            for part in &message.parts {
                if let Some(p) = to_gemini_part(part, &names)? {
                    parts.push(p);
                }
            }
            contents.push(json!({ "role": role, "parts": parts }));
        }

        let mut body = json!({ "contents": contents });
        if !system_parts.is_empty() {
            body["systemInstruction"] = json!({ "parts": system_parts });
        }
        if !req.tools.is_empty() {
            let declarations: Vec<Value> = req
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    })
                })
                .collect();
            body["tools"] = json!([{ "functionDeclarations": declarations }]);
        }
        let mut config = serde_json::Map::new();
        if let Some(max_tokens) = req.max_tokens {
            config.insert("maxOutputTokens".into(), json!(max_tokens));
        }
        if let Some(temperature) = req.temperature {
            config.insert("temperature".into(), json!(temperature));
        }
        if let Some(format) = &req.response_format {
            config.insert("responseMimeType".into(), json!("application/json"));
            config.insert("responseJsonSchema".into(), format.schema.clone());
        }
        if let Some(effort) = req.reasoning {
            // Gemini 2.x sets depth with a token budget; Gemini 3 and later with a level.
            let mut thinking = json!({ "includeThoughts": true });
            if Self::model_name(&req.model).starts_with("gemini-2") {
                thinking["thinkingBudget"] = json!(match effort {
                    crate::ReasoningEffort::Low => 1024,
                    crate::ReasoningEffort::Medium => 8192,
                    crate::ReasoningEffort::High => 24576,
                });
            } else {
                thinking["thinkingLevel"] = json!(effort.as_str().to_uppercase());
            }
            config.insert("thinkingConfig".into(), thinking);
        }
        if !config.is_empty() {
            body["generationConfig"] = Value::Object(config);
        }
        Ok(body)
    }

    fn model_url(&self, model: &str, method: &str) -> Result<String> {
        match &self.vertex {
            Some(vertex) => vertex.url("GeminiProvider", &self.base_url, "google", model, method),
            None => {
                let model = crate::checked_path_segment(
                    "GeminiProvider",
                    "model",
                    Self::model_name(model),
                    &[],
                )?;
                Ok(format!("{}/models/{model}:{method}", self.base_url))
            }
        }
    }

    fn post(&self, url: String) -> Result<reqwest::RequestBuilder> {
        let builder = self.http.client.post(url);
        Ok(if self.vertex.is_some() {
            builder.header(
                reqwest::header::AUTHORIZATION,
                secret_header("GeminiProvider", &format!("Bearer {}", self.api_key))?,
            )
        } else {
            builder.header(
                "x-goog-api-key",
                secret_header("GeminiProvider", &self.api_key)?,
            )
        })
    }

    fn chat_request(&self, req: &ChatRequest, stream: bool) -> Result<reqwest::RequestBuilder> {
        crate::validate(
            "GeminiProvider",
            &self.api_key,
            true,
            &self.base_url,
            64,
            req,
        )?;
        let method = if stream {
            "streamGenerateContent?alt=sse"
        } else {
            "generateContent"
        };
        let url = self.model_url(&req.model, method)?;
        Ok(self.post(url)?.json(&Self::build_body(req)?))
    }

    fn model_name(model: &str) -> &str {
        model.strip_prefix("models/").unwrap_or(model)
    }
}

fn to_gemini_part(part: &Part, names: &HashMap<&str, &str>) -> Result<Option<Value>> {
    let sendable_id =
        |id: &str| (!id.is_empty() && !id.starts_with(LOCAL_ID_PREFIX)).then(|| id.to_string());
    Ok(Some(match part {
        Part::Text { text } if text.is_empty() => return Ok(None),
        Part::Text { text } => json!({ "text": text }),
        Part::Image { source } => match source {
            ImageSource::Base64 { media_type, data } => {
                json!({ "inlineData": { "mimeType": media_type, "data": data } })
            }
            ImageSource::Url { .. } => {
                return Err(LlmError::InvalidArgument(
                    "GeminiProvider: image URLs are not supported; use Part::image_base64".into(),
                ))
            }
        },
        // Thought parts go back exactly as received (they may carry signatures).
        Part::Reasoning { raw, .. } if raw.is_object() => raw.clone(),
        Part::Reasoning { .. } => return Ok(None),
        Part::Other { raw } if raw.is_object() => raw.clone(),
        Part::Other { .. } => return Ok(None),
        Part::ToolCall {
            id,
            name,
            arguments,
            signature,
        } => {
            let mut call = json!({ "name": name, "args": arguments });
            if let Some(id) = sendable_id(id) {
                call["id"] = json!(id);
            }
            let mut part = json!({ "functionCall": call });
            if let Some(signature) = signature {
                part["thoughtSignature"] = json!(signature);
            }
            part
        }
        Part::ToolResult {
            call_id,
            content,
            is_error,
        } => {
            // `response` must be an object: use the content if it is a JSON object,
            // otherwise wrap it the way Google's examples do.
            let response = match serde_json::from_str::<Value>(content) {
                Ok(v @ Value::Object(_)) if !is_error => v,
                _ if *is_error => json!({ "error": content }),
                _ => json!({ "result": content }),
            };
            let mut result = json!({
                "name": names.get(call_id.as_str()).copied().unwrap_or_default(),
                "response": response,
            });
            if let Some(id) = sendable_id(call_id) {
                result["id"] = json!(id);
            }
            json!({ "functionResponse": result })
        }
    }))
}

fn parse_response(v: &Value, requested_model: &str) -> Result<ChatResponse> {
    let mut usage = usage(
        &v["usageMetadata"],
        "promptTokenCount",
        "candidatesTokenCount",
    );
    // Thinking tokens are billed as output but counted separately.
    let thoughts = v["usageMetadata"]["thoughtsTokenCount"]
        .as_u64()
        .unwrap_or(0);
    usage.output_tokens = usage
        .output_tokens
        .saturating_add(u32::try_from(thoughts).unwrap_or(u32::MAX));
    let model = v["modelVersion"]
        .as_str()
        .unwrap_or(requested_model)
        .to_string();

    let Some(candidate) = v["candidates"].get(0) else {
        // The prompt itself was blocked: no candidates, only `promptFeedback.blockReason`.
        if let Some(reason) = v["promptFeedback"]["blockReason"].as_str() {
            return Ok(ChatResponse {
                model,
                message: Message::new(
                    Role::Assistant,
                    vec![Part::text(format!("prompt blocked: {reason}"))],
                ),
                stop_reason: StopReason::Refusal,
                usage,
            });
        }
        return Err(LlmError::InvalidResponse(
            "missing field `candidates[0]`".into(),
        ));
    };

    let mut parts = Vec::new();
    let mut calls = 0usize;
    for part in candidate["content"]["parts"]
        .as_array()
        .into_iter()
        .flatten()
    {
        if part["thought"].as_bool() == Some(true) {
            parts.push(Part::Reasoning {
                text: part["text"].as_str().unwrap_or_default().to_string(),
                raw: part.clone(),
            });
        } else if part["functionCall"].is_object() {
            let call = &part["functionCall"];
            let id = match call["id"].as_str() {
                Some(id) if !id.is_empty() => id.to_string(),
                _ => format!("{LOCAL_ID_PREFIX}{calls}"),
            };
            calls += 1;
            parts.push(Part::ToolCall {
                id,
                name: call["name"].as_str().unwrap_or_default().to_string(),
                arguments: if call["args"].is_null() {
                    json!({})
                } else {
                    call["args"].clone()
                },
                signature: part["thoughtSignature"].as_str().map(str::to_string),
            });
        } else if let Some(text) = part["text"].as_str().filter(|t| !t.is_empty()) {
            parts.push(Part::text(text));
        } else if is_meaningful_other(part) {
            // Parts this crate does not model (code execution, an empty text part carrying a
            // thought signature, …) are kept and sent back unchanged.
            parts.push(Part::Other { raw: part.clone() });
        }
    }

    let stop_reason = match candidate["finishReason"].as_str() {
        // Gemini reports STOP for function calls as well.
        Some("STOP") | None if calls > 0 => StopReason::ToolUse,
        Some("STOP") => StopReason::EndTurn,
        Some("MAX_TOKENS") => StopReason::MaxTokens,
        Some(
            "SAFETY"
            | "RECITATION"
            | "BLOCKLIST"
            | "PROHIBITED_CONTENT"
            | "SPII"
            | "IMAGE_SAFETY"
            | "IMAGE_PROHIBITED_CONTENT"
            | "IMAGE_RECITATION",
        ) => StopReason::Refusal,
        other => StopReason::Other(other.unwrap_or("unknown").to_string()),
    };

    Ok(ChatResponse {
        model,
        message: Message::new(Role::Assistant, parts),
        stop_reason,
        usage,
    })
}

/// An object part that is not plain empty text.
fn is_meaningful_other(part: &Value) -> bool {
    part.as_object()
        .is_some_and(|o| o.keys().any(|k| k != "text" && k != "thought"))
}

/// Each streamed chunk is a partial `GenerateContentResponse`; parts are concatenated and the
/// latest candidate fields, usage and model version are kept separately (the last chunk may
/// carry only usage).
struct GeminiStream {
    model: String,
    parts: Vec<Value>,
    candidate: Value,
    usage: Value,
    model_version: Value,
    prompt_feedback: Value,
    calls: usize,
    done: bool,
}

impl GeminiStream {
    fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            parts: Vec::new(),
            candidate: Value::Null,
            usage: Value::Null,
            model_version: Value::Null,
            prompt_feedback: Value::Null,
            calls: 0,
            done: false,
        }
    }

    fn push_part(&mut self, part: &Value) {
        // Merge consecutive plain text (or consecutive thought text) into one part.
        let mergeable = |p: &Value| {
            p["text"].is_string()
                && p.get("thoughtSignature").is_none()
                && p.get("functionCall").is_none()
        };
        if let Some(prev) = self.parts.last_mut() {
            if mergeable(prev) && mergeable(part) && prev["thought"] == part["thought"] {
                if let (Value::String(prev), Some(next)) =
                    (&mut prev["text"], part["text"].as_str())
                {
                    prev.push_str(next);
                }
                return;
            }
        }
        self.parts.push(part.clone());
    }
}

impl Accumulator for GeminiStream {
    fn on_event(&mut self, _event: &str, data: &str) -> Result<Vec<StreamEvent>> {
        let chunk = parse_data(data)?;
        if !chunk.is_object() {
            return Err(LlmError::InvalidResponse("malformed stream chunk".into()));
        }
        if chunk["error"].is_object() {
            return Err(LlmError::Api(Box::new(ApiError::from_stream_error(
                &chunk["error"],
            ))));
        }
        if chunk["usageMetadata"].is_object() {
            self.usage = chunk["usageMetadata"].clone();
        }
        if chunk["modelVersion"].is_string() {
            self.model_version = chunk["modelVersion"].clone();
        }
        if chunk["promptFeedback"]["blockReason"].is_string() {
            self.prompt_feedback = chunk["promptFeedback"].clone();
            self.done = true;
        }
        let candidate = &chunk["candidates"][0];
        if candidate.is_object() {
            self.candidate = candidate.clone();
            if candidate["finishReason"].is_string() {
                self.done = true;
            }
        }
        let mut out = Vec::new();
        for part in chunk["candidates"][0]["content"]["parts"]
            .as_array()
            .into_iter()
            .flatten()
        {
            if let Some(text) = part["text"].as_str().filter(|t| !t.is_empty()) {
                out.push(if part["thought"].as_bool() == Some(true) {
                    StreamEvent::ReasoningDelta(text.to_string())
                } else {
                    StreamEvent::TextDelta(text.to_string())
                });
            } else if part["functionCall"].is_object() {
                let call = &part["functionCall"];
                let name = call["name"].as_str().unwrap_or_default().to_string();
                // Same id the final response will use.
                let id = match call["id"].as_str() {
                    Some(id) if !id.is_empty() => id.to_string(),
                    _ => format!("{LOCAL_ID_PREFIX}{}", self.calls),
                };
                self.calls += 1;
                out.push(StreamEvent::ToolCallStart {
                    id: id.clone(),
                    name,
                });
                out.push(StreamEvent::ToolCallDelta {
                    id,
                    arguments: call["args"].to_string(),
                });
            }
            if part.is_object() {
                self.push_part(part);
            }
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<ChatResponse> {
        if !self.done {
            return Err(crate::stream::ended_early());
        }
        let mut response = json!({
            "usageMetadata": self.usage,
            "modelVersion": self.model_version,
            "promptFeedback": self.prompt_feedback,
        });
        if self.candidate.is_object() {
            let mut candidate = std::mem::take(&mut self.candidate);
            candidate["content"] =
                json!({ "role": "model", "parts": std::mem::take(&mut self.parts) });
            response["candidates"] = json!([candidate]);
        }
        parse_response(&response, &self.model)
    }
}

#[async_trait]
impl LlmProvider for GeminiProvider {
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let builder = self.chat_request(req, false)?;
        parse_response(
            &self.http.json(builder).await?,
            Self::model_name(&req.model),
        )
    }

    async fn chat_stream(&self, req: &ChatRequest) -> Result<ChatStream> {
        let builder = self.chat_request(req, true)?;
        let response = self.http.send(builder).await?;
        Ok(stream::drive(
            response,
            Box::new(GeminiStream::new(Self::model_name(&req.model))),
            self.http.stream_deadline,
            vec![self.api_key.clone()],
        ))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        if self.vertex.is_some() {
            return Err(LlmError::InvalidArgument(
                "GeminiProvider: list_models is not available on Vertex AI".into(),
            ));
        }
        crate::validate_endpoint("GeminiProvider", &self.api_key, true, &self.base_url)?;
        let builder = self
            .http
            .client
            .get(format!("{}/models?pageSize=1000", self.base_url))
            .header(
                "x-goog-api-key",
                secret_header("GeminiProvider", &self.api_key)?,
            );
        let v = self.http.json(builder).await?;
        Ok(v["models"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|m| {
                Some(ModelInfo {
                    id: Self::model_name(m["name"].as_str()?).to_string(),
                    display_name: m["displayName"].as_str().map(str::to_string),
                })
            })
            .collect())
    }
}

#[async_trait]
impl EmbeddingProvider for GeminiProvider {
    async fn embed(&self, model: &str, inputs: &[&str]) -> Result<Embeddings> {
        if self.vertex.is_some() {
            return Err(LlmError::InvalidArgument(
                "GeminiProvider: embeddings are not available on Vertex AI in this crate".into(),
            ));
        }
        crate::validate_endpoint("GeminiProvider", &self.api_key, true, &self.base_url)?;
        validate_embed_args(model, inputs)?;
        let model = Self::model_name(model);
        let requests: Vec<Value> = inputs
            .iter()
            .map(|text| json!({ "model": format!("models/{model}"), "content": { "parts": [{ "text": text }] } }))
            .collect();
        let builder = self
            .post(self.model_url(model, "batchEmbedContents")?)?
            .json(&json!({ "requests": requests }));
        let v = self.http.json(builder).await?;
        Ok(Embeddings {
            vectors: v["embeddings"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|e| float_vec(&e["values"]))
                .collect(),
            usage: crate::Usage::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ReasoningEffort, Tool};

    #[test]
    fn builds_request_with_tools_and_results() {
        let req = ChatRequest::new("gemini-2.5-flash")
            .system("sys")
            .max_tokens(100)
            .tool(Tool::new("get_weather", "d", json!({"type": "object"})))
            .message(Message::user("hi"))
            .message(Message::new(
                Role::Assistant,
                vec![Part::ToolCall {
                    id: format!("{LOCAL_ID_PREFIX}0"),
                    name: "get_weather".into(),
                    arguments: json!({"city": "Berlin"}),
                    signature: Some("sig".into()),
                }],
            ))
            .message(Message::tool_result(
                format!("{LOCAL_ID_PREFIX}0"),
                "18C",
                false,
            ));
        let body = GeminiProvider::build_body(&req).unwrap();
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "sys");
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 100);
        assert_eq!(
            body["tools"][0]["functionDeclarations"][0]["name"],
            "get_weather"
        );
        assert_eq!(body["contents"][1]["role"], "model");
        let call_part = &body["contents"][1]["parts"][0];
        assert_eq!(call_part["functionCall"]["args"]["city"], "Berlin");
        assert_eq!(call_part["thoughtSignature"], "sig");
        assert!(
            call_part["functionCall"].get("id").is_none(),
            "local ids are not sent"
        );
        let result = &body["contents"][2]["parts"][0]["functionResponse"];
        assert_eq!(result["name"], "get_weather");
        assert_eq!(result["response"]["result"], "18C");
    }

    #[test]
    fn builds_images_schema_and_thinking() {
        let req = ChatRequest::new("gemini-2.5-flash")
            .json_schema("answer", json!({"type": "object"}))
            .reasoning(ReasoningEffort::Medium)
            .message(Message::user_with_image(
                "what?",
                Part::image_base64("image/jpeg", "AAAA"),
            ));
        let body = GeminiProvider::build_body(&req).unwrap();
        assert_eq!(
            body["contents"][0]["parts"][1]["inlineData"]["mimeType"],
            "image/jpeg"
        );
        assert_eq!(
            body["generationConfig"]["responseMimeType"],
            "application/json"
        );
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["includeThoughts"],
            true
        );

        let url_image = ChatRequest::new("m").message(Message::user_with_image(
            "x",
            Part::image_url("https://e.com/a.png"),
        ));
        assert!(GeminiProvider::build_body(&url_image).is_err());
    }

    #[test]
    fn parses_function_call_with_signature_and_thoughts() {
        let v = json!({
            "candidates": [{
                "content": { "role": "model", "parts": [
                    { "text": "thinking...", "thought": true },
                    { "functionCall": { "name": "get_weather", "args": { "city": "Berlin" } }, "thoughtSignature": "abc" }
                ]},
                "finishReason": "STOP"
            }],
            "usageMetadata": { "promptTokenCount": 4, "candidatesTokenCount": 6 },
            "modelVersion": "gemini-2.5-flash"
        });
        let resp = parse_response(&v, "m").unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.text(), "");
        assert_eq!(resp.reasoning(), "thinking...");
        let calls = resp.tool_calls();
        assert_eq!(calls[0].name, "get_weather");
        assert!(calls[0].id.starts_with(LOCAL_ID_PREFIX));
        assert_eq!(resp.usage.output_tokens, 6);
        assert_eq!(resp.model, "gemini-2.5-flash");
        // The signature is replayed on the next turn.
        let replay = to_gemini_part(&resp.message.parts[1], &HashMap::new())
            .unwrap()
            .unwrap();
        assert_eq!(replay["thoughtSignature"], "abc");
    }

    #[test]
    fn blocked_prompt_is_a_refusal() {
        let v = json!({ "promptFeedback": { "blockReason": "SAFETY" } });
        let resp = parse_response(&v, "m").unwrap();
        assert_eq!(resp.stop_reason, StopReason::Refusal);
        assert_eq!(resp.text(), "prompt blocked: SAFETY");
    }

    #[test]
    fn stream_merges_text_chunks() {
        let mut acc = GeminiStream::new("gemini-2.5-flash");
        acc.on_event(
            "",
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]}}]}"#,
        )
        .unwrap();
        acc.on_event(
            "",
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3}}"#,
        )
        .unwrap();
        let resp = acc.finish().unwrap();
        assert_eq!(resp.text(), "Hello");
        assert_eq!(resp.message.parts.len(), 1);
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert_eq!(resp.usage.output_tokens, 3);
    }

    #[test]
    fn rejects_model_that_would_change_the_url() {
        let p = GeminiProvider::new("k");
        for model in ["../files", "gemini?key=x", "a/b", "gemini pro"] {
            assert!(p.model_url(model, "generateContent").is_err(), "{model}");
        }
        assert!(p
            .model_url("models/gemini-2.5-flash", "generateContent")
            .unwrap()
            .ends_with("/models/gemini-2.5-flash:generateContent"));
    }

    #[test]
    fn vertex_url() {
        let p = GeminiProvider::vertex("proj", "global", "token");
        assert_eq!(
            p.model_url("gemini-2.5-flash", "generateContent").unwrap(),
            "https://aiplatform.googleapis.com/v1/projects/proj/locations/global/publishers/google/models/gemini-2.5-flash:generateContent"
        );
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::ReasoningEffort;

    #[test]
    fn effort_maps_to_budget_or_level() {
        let req = |model: &str| {
            ChatRequest::new(model)
                .reasoning(ReasoningEffort::High)
                .message(Message::user("x"))
        };
        let v2 = GeminiProvider::build_body(&req("gemini-2.5-pro")).unwrap();
        assert_eq!(
            v2["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            24576
        );
        let v3 = GeminiProvider::build_body(&req("gemini-3-pro")).unwrap();
        assert_eq!(
            v3["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "HIGH"
        );
    }

    #[test]
    fn thought_tokens_count_as_output() {
        let v = json!({
            "candidates": [{ "content": { "parts": [{ "text": "hi" }] }, "finishReason": "STOP" }],
            "usageMetadata": { "promptTokenCount": 1, "candidatesTokenCount": 2, "thoughtsTokenCount": 30 }
        });
        assert_eq!(parse_response(&v, "m").unwrap().usage.output_tokens, 32);
    }

    #[test]
    fn unknown_parts_and_bare_signatures_are_replayed() {
        let v = json!({
            "candidates": [{ "content": { "parts": [
                { "text": "answer" },
                { "text": "", "thoughtSignature": "sig" },
                { "executableCode": { "code": "print(1)" } },
                { "text": "" }
            ] }, "finishReason": "STOP" }]
        });
        let resp = parse_response(&v, "m").unwrap();
        assert_eq!(resp.message.parts.len(), 3, "plain empty text is dropped");
        let replay: Vec<Value> = resp
            .message
            .parts
            .iter()
            .filter_map(|p| to_gemini_part(p, &HashMap::new()).unwrap())
            .collect();
        assert_eq!(replay[1]["thoughtSignature"], "sig");
        assert_eq!(replay[2]["executableCode"]["code"], "print(1)");
    }

    #[test]
    fn stream_keeps_text_when_last_chunk_has_only_usage() {
        let mut acc = GeminiStream::new("m");
        acc.on_event(
            "",
            r#"{"candidates":[{"content":{"parts":[{"text":"Hi"}]},"finishReason":"STOP"}]}"#,
        )
        .unwrap();
        acc.on_event(
            "",
            r#"{"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}"#,
        )
        .unwrap();
        let resp = acc.finish().unwrap();
        assert_eq!(resp.text(), "Hi");
        assert_eq!(resp.usage.output_tokens, 1);
    }

    #[test]
    fn stream_tool_call_ids_match_the_final_response() {
        let mut acc = GeminiStream::new("m");
        let events = acc
            .on_event(
                "",
                r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]},"finishReason":"STOP"}]}"#,
            )
            .unwrap();
        let StreamEvent::ToolCallStart { id, .. } = &events[0] else {
            panic!()
        };
        let resp = acc.finish().unwrap();
        assert_eq!(&resp.tool_calls()[0].id, id);
    }
}
