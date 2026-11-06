//! Integration tests against a local HTTP server: no network, no keys.

use llm_universal::{
    run_tools, AnthropicProvider, ApiErrorKind, ChatRequest, CohereProvider, GeminiProvider,
    LlmError, LlmProvider, Message, OpenAiProvider, StopReason, StreamEvent, StreamExt, Tool,
};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Starts a server that answers once with the given response
/// and returns the captured request (headers + body).
async fn one_shot(status: u16, body: &'static str) -> (String, tokio::task::JoinHandle<String>) {
    let (base, handle) = serve(vec![(status, "", body.to_string())]).await;
    let handle = tokio::spawn(async move { handle.await.unwrap().remove(0) });
    (base, handle)
}

/// Serves the responses in order, one connection each: (status, extra headers, body).
async fn serve(
    responses: Vec<(u16, &'static str, String)>,
) -> (String, tokio::task::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let mut captured = Vec::new();
        for (status, extra, body) in responses {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                let n = socket.read(&mut chunk).await.unwrap();
                buf.extend_from_slice(&chunk[..n]);
                if let Some(pos) = find_header_end(&buf) {
                    let headers = String::from_utf8_lossy(&buf[..pos]).to_string();
                    let len = content_length(&headers);
                    while buf.len() < pos + 4 + len {
                        let n = socket.read(&mut chunk).await.unwrap();
                        if n == 0 {
                            break;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                    }
                    break;
                }
                if n == 0 {
                    break;
                }
            }
            captured.push(String::from_utf8_lossy(&buf).to_string());
            let response = format!(
                "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nrequest-id: req_test_1\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        }
        captured
    });
    (format!("http://{addr}"), handle)
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn content_length(headers: &str) -> usize {
    headers
        .lines()
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.eq_ignore_ascii_case("content-length")
                .then(|| v.trim().parse().ok())?
        })
        .unwrap_or(0)
}

fn weather_tool() -> Tool {
    Tool::new("get_weather", "weather", json!({"type": "object"}))
}

#[tokio::test]
async fn anthropic_sends_auth_headers_and_parses_tool_use() {
    let body = r#"{"model":"claude-opus-5-5","content":[{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Berlin"}}],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":4}}"#;
    let (base, server) = one_shot(200, body).await;

    let provider = AnthropicProvider::new("test-key").with_base_url(base);
    let req = ChatRequest::new("claude-opus-5-5")
        .tool(weather_tool())
        .message(Message::user("weather?"));
    let resp = provider.chat(&req).await.unwrap();

    let raw = server.await.unwrap().to_lowercase();
    assert!(raw.starts_with("post /v1/messages "));
    assert!(raw.contains("x-api-key: test-key"));
    assert!(raw.contains("anthropic-version: 2023-06-01"));
    assert_eq!(resp.stop_reason, StopReason::ToolUse);
    assert_eq!(resp.tool_calls()[0].arguments["city"], "Berlin");
    assert_eq!(resp.usage.output_tokens, 4);
}

#[tokio::test]
async fn openai_sends_bearer_and_parses_text() {
    let body = r#"{"model":"gpt-4o","choices":[{"finish_reason":"stop","message":{"content":"hello"}}],"usage":{"prompt_tokens":2,"completion_tokens":1}}"#;
    let (base, server) = one_shot(200, body).await;

    let provider = OpenAiProvider::new("sk-test").with_base_url(format!("{base}/"));
    let req = ChatRequest::new("gpt-4o").message(Message::user("hi"));
    let resp = provider.chat(&req).await.unwrap();

    let raw = server.await.unwrap().to_lowercase();
    assert!(raw.starts_with("post /chat/completions "));
    assert!(raw.contains("authorization: bearer sk-test"));
    assert_eq!(resp.text(), "hello");
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
}

#[tokio::test]
async fn http_error_is_mapped_to_api_error() {
    let (base, server) = one_shot(429, r#"{"error":"rate limited"}"#).await;
    let provider = AnthropicProvider::new("k")
        .with_base_url(base)
        .with_max_retries(0);
    let err = provider
        .chat(&ChatRequest::new("m").message(Message::user("x")))
        .await
        .unwrap_err();
    server.await.unwrap();
    match err {
        LlmError::Api(ref e) => {
            assert_eq!(e.status, 429);
            assert_eq!(e.kind, ApiErrorKind::RateLimit);
            // `error` is a string, not an object: the message is the whole body, like JSON.stringify in the SDKs.
            assert_eq!(err.to_string(), r#"429 {"error":"rate limited"}"#);
            assert!(err.is_retryable());
        }
        ref other => panic!("expected Api error, got {other:?}"),
    }
}

#[tokio::test]
async fn anthropic_error_json_is_filtered_to_message() {
    let body =
        r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#;
    let (base, server) = one_shot(401, body).await;
    let provider = AnthropicProvider::new("bad").with_base_url(base);
    let err = provider
        .chat(&ChatRequest::new("m").message(Message::user("x")))
        .await
        .unwrap_err();
    server.await.unwrap();
    assert_eq!(err.to_string(), "401 invalid x-api-key");
    assert_eq!(err.request_id(), Some("req_test_1"));
    assert!(!err.is_retryable());
    let LlmError::Api(e) = err else {
        panic!("expected Api error")
    };
    assert_eq!(e.kind, ApiErrorKind::Authentication);
    assert_eq!(e.error_type.as_deref(), Some("authentication_error"));
}

#[tokio::test]
async fn openai_error_without_message_gets_hint() {
    let (base, server) = one_shot(404, "not found").await;
    let provider = OpenAiProvider::new("k").with_base_url(base);
    let err = provider
        .chat(&ChatRequest::new("no-such-model").message(Message::user("x")))
        .await
        .unwrap_err();
    server.await.unwrap();
    assert_eq!(err.to_string(), "404 not found");
    assert_eq!(err.status(), Some(404));
}

#[tokio::test]
async fn html_instead_of_json_is_reported() {
    let (base, server) = one_shot(200, "<html>login</html>").await;
    let provider = OpenAiProvider::new("k").with_base_url(base);
    let err = provider
        .chat(&ChatRequest::new("m").message(Message::user("x")))
        .await
        .unwrap_err();
    server.await.unwrap();
    match &err {
        LlmError::Deserialize { content, .. } => assert_eq!(content.as_str(), "<html>login</html>"),
        other => panic!("expected Deserialize, got {other:?}"),
    }
    // The body is kept in the error but never printed.
    assert!(!err.to_string().contains("login"), "{err}");
}

#[tokio::test]
async fn oversized_error_body_is_truncated() {
    let big: &'static str = Box::leak("x".repeat(200_000).into_boxed_str());
    let (base, server) = one_shot(500, big).await;
    let provider = OpenAiProvider::new("k")
        .with_base_url(base)
        .with_max_retries(0);
    let err = provider
        .chat(&ChatRequest::new("m").message(Message::user("x")))
        .await
        .unwrap_err();
    let _ = server.await;
    let LlmError::Api(e) = err else {
        panic!("expected Api error")
    };
    assert!(e.message.unwrap().len() <= 4096);
}

#[tokio::test]
async fn gemini_sends_key_header_and_model_in_path() {
    let body = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1},"modelVersion":"gemini-2.5-flash"}"#;
    let (base, server) = one_shot(200, body).await;
    let provider = GeminiProvider::new("g-key").with_base_url(base);
    let resp = provider
        .chat(&ChatRequest::new("gemini-2.5-flash").message(Message::user("x")))
        .await
        .unwrap();
    let raw = server.await.unwrap().to_lowercase();
    assert!(
        raw.starts_with("post /models/gemini-2.5-flash:generatecontent "),
        "{raw}"
    );
    assert!(raw.contains("x-goog-api-key: g-key"));
    assert!(!raw.contains("?key="), "key must not be in the URL");
    assert_eq!(resp.text(), "hi");
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
}

#[tokio::test]
async fn gemini_error_body_is_parsed() {
    let body =
        r#"{"error":{"code":400,"message":"API key not valid.","status":"INVALID_ARGUMENT"}}"#;
    let (base, server) = one_shot(400, body).await;
    let err = GeminiProvider::new("bad")
        .with_base_url(base)
        .chat(&ChatRequest::new("gemini-2.5-flash").message(Message::user("x")))
        .await
        .unwrap_err();
    server.await.unwrap();
    assert_eq!(err.to_string(), "400 API key not valid.");
    let LlmError::Api(e) = err else {
        panic!("expected Api error")
    };
    assert_eq!(e.error_type.as_deref(), Some("INVALID_ARGUMENT"));
}

#[tokio::test]
async fn cohere_sends_bearer_to_chat_endpoint() {
    let body = r#"{"finish_reason":"COMPLETE","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]},"usage":{"tokens":{"input_tokens":2,"output_tokens":1}}}"#;
    let (base, server) = one_shot(200, body).await;
    let resp = CohereProvider::new("co-key")
        .with_base_url(base)
        .chat(&ChatRequest::new("command-a").message(Message::user("x")))
        .await
        .unwrap();
    let raw = server.await.unwrap().to_lowercase();
    assert!(raw.starts_with("post /chat "), "{raw}");
    assert!(raw.contains("authorization: bearer co-key"));
    assert_eq!(resp.text(), "hi");
}

#[tokio::test]
async fn cohere_top_level_error_message_is_parsed() {
    let (base, server) = one_shot(401, r#"{"id":"x","message":"invalid api token"}"#).await;
    let err = CohereProvider::new("bad")
        .with_base_url(base)
        .chat(&ChatRequest::new("command-a").message(Message::user("x")))
        .await
        .unwrap_err();
    server.await.unwrap();
    assert_eq!(err.to_string(), "401 invalid api token");
}

#[test]
fn presets_point_at_the_documented_hosts() {
    let debug = |p: OpenAiProvider| format!("{p:?}");
    assert!(debug(OpenAiProvider::groq("k")).contains("https://api.groq.com/openai/v1"));
    assert!(debug(OpenAiProvider::mistral("k")).contains("https://api.mistral.ai/v1"));
    assert!(debug(OpenAiProvider::azure("myres", "k"))
        .contains("https://myres.openai.azure.com/openai/v1"));
    assert!(debug(OpenAiProvider::ollama()).contains("http://localhost:11434/v1"));
    assert!(!debug(OpenAiProvider::groq("secret-k")).contains("secret-k"));
}

#[tokio::test]
async fn azure_sends_api_key_header_not_bearer() {
    let body =
        r#"{"model":"gpt-4o","choices":[{"finish_reason":"stop","message":{"content":"hi"}}]}"#;
    let (base, server) = one_shot(200, body).await;
    OpenAiProvider::azure("res", "az-key")
        .with_base_url(base)
        .chat(&ChatRequest::new("my-deployment").message(Message::user("x")))
        .await
        .unwrap();
    let raw = server.await.unwrap().to_lowercase();
    assert!(raw.contains("api-key: az-key"), "{raw}");
    assert!(!raw.contains("authorization:"), "{raw}");
}

#[tokio::test]
async fn retries_429_honoring_retry_after_then_succeeds() {
    let ok =
        r#"{"model":"gpt-4o","choices":[{"finish_reason":"stop","message":{"content":"hi"}}]}"#;
    let (base, server) = serve(vec![
        (
            429,
            "retry-after-ms: 10\r\n",
            r#"{"error":{"message":"slow down"}}"#.into(),
        ),
        (503, "", "".into()),
        (200, "", ok.into()),
    ])
    .await;
    // 503 повторяется только по явному разрешению: сервер мог уже обработать запрос.
    let resp = OpenAiProvider::new("k")
        .with_base_url(base)
        .with_repeat_post(true)
        .chat(&ChatRequest::new("gpt-4o").message(Message::user("x")))
        .await
        .unwrap();
    assert_eq!(server.await.unwrap().len(), 3);
    assert_eq!(resp.text(), "hi");
}

/// По умолчанию 5xx не повторяется: один запрос, ошибка сразу.
#[tokio::test]
async fn server_error_is_not_repeated_by_default() {
    let (base, server) = serve(vec![
        (503, "", "".into()),
        (200, "", r#"{"model":"m","choices":[{"finish_reason":"stop","message":{"content":"x"}}]}"#.into()),
    ])
    .await;
    let r = OpenAiProvider::new("k")
        .with_base_url(base)
        .with_max_retries(3)
        .chat(&ChatRequest::new("m").message(Message::user("x")))
        .await;
    assert!(r.is_err());
    server.abort();
}

/// Таймаут запроса задаётся явно и срабатывает на молчащем сервере.
#[tokio::test]
async fn request_timeout_is_configurable() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let _h = tokio::spawn(async move {
        loop {
            if let Ok((s, _)) = listener.accept().await {
                tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                    drop(s);
                });
            }
        }
    });
    let started = std::time::Instant::now();
    let r = OpenAiProvider::new("k")
        .with_base_url(base)
        .with_max_retries(0)
        .with_request_timeout(std::time::Duration::from_millis(200))
        .chat(&ChatRequest::new("m").message(Message::user("x")))
        .await;
    assert!(matches!(r, Err(LlmError::Timeout(_))), "{r:?}");
    assert!(started.elapsed() < std::time::Duration::from_secs(5));
}

/// Дедлайн стрима: сервер шлёт только keep-alive каждые 50 мс, без событий.
#[tokio::test]
async fn stream_deadline_cuts_keepalive_only_stream() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut s, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 4096];
        let _ = s.read(&mut buf).await;
        let _ = s
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")
            .await;
        for _ in 0..200 {
            if s.write_all(b":\n\n").await.is_err() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    });
    let started = std::time::Instant::now();
    let stream = OpenAiProvider::new("k")
        .with_base_url(base)
        .with_stream_deadline(std::time::Duration::from_millis(300))
        .chat_stream(&ChatRequest::new("m").message(Message::user("x")))
        .await
        .unwrap();
    let mut stream = stream;
    let mut failed = false;
    while let Some(event) = stream.next().await {
        if event.is_err() {
            failed = true;
            break;
        }
    }
    assert!(failed, "стрим должен оборваться по дедлайну");
    assert!(started.elapsed() < std::time::Duration::from_secs(5));
}

#[tokio::test]
async fn does_not_retry_400_or_when_server_says_no() {
    let (base, server) = serve(vec![(400, "", r#"{"error":{"message":"bad"}}"#.into())]).await;
    let err = OpenAiProvider::new("k")
        .with_base_url(base)
        .chat(&ChatRequest::new("m").message(Message::user("x")))
        .await
        .unwrap_err();
    assert_eq!(server.await.unwrap().len(), 1);
    assert_eq!(err.status(), Some(400));

    let (base, server) = serve(vec![(500, "x-should-retry: false\r\n", "{}".into())]).await;
    let err = OpenAiProvider::new("k")
        .with_base_url(base)
        .chat(&ChatRequest::new("m").message(Message::user("x")))
        .await
        .unwrap_err();
    assert_eq!(server.await.unwrap().len(), 1);
    assert_eq!(err.status(), Some(500));
}

async fn collect(mut stream: llm_universal::ChatStream) -> (String, llm_universal::ChatResponse) {
    let mut text = String::new();
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            StreamEvent::TextDelta(t) => text.push_str(&t),
            StreamEvent::Done(resp) => return (text, resp),
            _ => {}
        }
    }
    panic!("stream ended without Done");
}

#[tokio::test]
async fn openai_stream_over_http() {
    let sse = "data: {\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n\
               data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n\
               data: [DONE]\n\n";
    let (base, server) = serve(vec![(200, "", sse.into())]).await;
    let stream = OpenAiProvider::new("k")
        .with_base_url(base)
        .chat_stream(&ChatRequest::new("gpt-4o").message(Message::user("x")))
        .await
        .unwrap();
    let (text, resp) = collect(stream).await;
    let raw = server.await.unwrap().remove(0);
    assert!(raw.contains("\"stream\":true"), "{raw}");
    assert_eq!(text, "Hello");
    assert_eq!(resp.text(), "Hello");
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
}

#[tokio::test]
async fn anthropic_stream_over_http() {
    let sse = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-opus-5-5\",\"content\":[],\"usage\":{\"input_tokens\":3}}}\n\n\
               event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
               event: ping\ndata: {\"type\":\"ping\"}\n\n\
               event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi there\"}}\n\n\
               event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
               event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n\
               event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let (base, server) = serve(vec![(200, "", sse.into())]).await;
    let stream = AnthropicProvider::new("k")
        .with_base_url(base)
        .chat_stream(&ChatRequest::new("claude-opus-5-5").message(Message::user("x")))
        .await
        .unwrap();
    let (text, resp) = collect(stream).await;
    server.await.unwrap();
    assert_eq!(text, "Hi there");
    assert_eq!(resp.usage.output_tokens, 2);
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
}

#[tokio::test]
async fn gemini_stream_uses_sse_endpoint() {
    let sse = "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hi\"}]},\"finishReason\":\"STOP\"}]}\n\n";
    let (base, server) = serve(vec![(200, "", sse.into())]).await;
    let stream = GeminiProvider::new("g")
        .with_base_url(base)
        .chat_stream(&ChatRequest::new("gemini-2.5-flash").message(Message::user("x")))
        .await
        .unwrap();
    let (text, _) = collect(stream).await;
    let raw = server.await.unwrap().remove(0).to_lowercase();
    assert!(
        raw.starts_with("post /models/gemini-2.5-flash:streamgeneratecontent?alt=sse "),
        "{raw}"
    );
    assert_eq!(text, "Hi");
}

#[tokio::test]
async fn run_tools_loops_until_answer() {
    let call = r#"{"model":"m","content":[{"type":"tool_use","id":"t1","name":"get_weather","input":{"city":"Berlin"}}],"stop_reason":"tool_use"}"#;
    let done = r#"{"model":"m","content":[{"type":"text","text":"18C"}],"stop_reason":"end_turn"}"#;
    let (base, server) = serve(vec![(200, "", call.into()), (200, "", done.into())]).await;
    let provider = AnthropicProvider::new("k").with_base_url(base);
    let mut req = ChatRequest::new("m")
        .tool(weather_tool())
        .message(Message::user("weather?"));
    let answer = run_tools(&provider, &mut req, 3, |call| async move {
        assert_eq!(call.arguments["city"], "Berlin");
        Ok("18C".to_string())
    })
    .await
    .unwrap();
    let requests = server.await.unwrap();
    assert_eq!(answer.text(), "18C");
    assert!(requests[1].contains("\"tool_result\""), "{}", requests[1]);
    assert_eq!(
        req.messages.len(),
        4,
        "user, tool call, tool result, answer"
    );
}

#[tokio::test]
async fn structured_output_parses_into_a_type() {
    #[derive(serde::Deserialize)]
    struct Answer {
        city: String,
        temp_c: i32,
    }
    let body = r#"{"model":"m","content":[{"type":"text","text":"{\"city\":\"Berlin\",\"temp_c\":18}"}],"stop_reason":"end_turn"}"#;
    let (base, server) = one_shot(200, body).await;
    let resp = AnthropicProvider::new("k")
        .with_base_url(base)
        .chat(
            &ChatRequest::new("m")
                .json_schema("answer", json!({"type": "object"}))
                .message(Message::user("x")),
        )
        .await
        .unwrap();
    let raw = server.await.unwrap();
    assert!(raw.contains("\"output_config\""), "{raw}");
    let answer: Answer = resp.json().unwrap();
    assert_eq!((answer.city.as_str(), answer.temp_c), ("Berlin", 18));
}

#[tokio::test]
async fn embeddings_and_models() {
    let (base, server) = one_shot(
        200,
        r#"{"data":[{"index":1,"embedding":[0.5]},{"index":0,"embedding":[0.25,0.75]}],"usage":{"prompt_tokens":4}}"#,
    )
    .await;
    let provider = OpenAiProvider::new("k").with_base_url(base);
    let e =
        llm_universal::EmbeddingProvider::embed(&provider, "text-embedding-3-small", &["a", "b"])
            .await
            .unwrap();
    assert!(server
        .await
        .unwrap()
        .to_lowercase()
        .starts_with("post /embeddings "));
    assert_eq!(e.vectors, vec![vec![0.25, 0.75], vec![0.5]]);
    assert_eq!(e.usage.input_tokens, 4);

    let (base, server) = one_shot(
        200,
        r#"{"data":[{"id":"claude-opus-5-5","display_name":"Claude Opus 5.5"}]}"#,
    )
    .await;
    let models = AnthropicProvider::new("k")
        .with_base_url(base)
        .list_models()
        .await
        .unwrap();
    assert!(server
        .await
        .unwrap()
        .to_lowercase()
        .starts_with("get /v1/models?limit=1000 "));
    assert_eq!(models[0].display_name.as_deref(), Some("Claude Opus 5.5"));
}

#[tokio::test]
async fn stream_cut_off_before_final_event_is_an_error() {
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n";
    let (base, server) = serve(vec![(200, "", sse.into())]).await;
    let mut stream = OpenAiProvider::new("k")
        .with_base_url(base)
        .chat_stream(&ChatRequest::new("m").message(Message::user("x")))
        .await
        .unwrap();
    let mut last = None;
    while let Some(event) = stream.next().await {
        last = Some(event);
    }
    server.await.unwrap();
    let err = last.unwrap().unwrap_err();
    assert!(
        err.to_string().contains("ended before the final event"),
        "{err}"
    );
}

#[tokio::test]
async fn malformed_stream_event_is_an_error_not_a_panic() {
    let sse = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":5}\n\n";
    let (base, server) = serve(vec![(200, "", sse.into())]).await;
    let mut stream = AnthropicProvider::new("k")
        .with_base_url(base)
        .chat_stream(&ChatRequest::new("m").message(Message::user("x")))
        .await
        .unwrap();
    let first = stream.next().await.unwrap();
    server.await.unwrap();
    assert!(
        matches!(first, Err(LlmError::InvalidResponse(_))),
        "{first:?}"
    );
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn run_tools_stops_on_truncated_tool_call() {
    let cut = r#"{"model":"m","content":[{"type":"tool_use","id":"t1","name":"get_weather","input":{}}],"stop_reason":"max_tokens"}"#;
    let (base, server) = serve(vec![(200, "", cut.into())]).await;
    let provider = AnthropicProvider::new("k").with_base_url(base);
    let mut req = ChatRequest::new("m")
        .tool(weather_tool())
        .message(Message::user("weather?"));
    let err = run_tools(&provider, &mut req, 3, |_| async {
        panic!("must not run a truncated call")
    })
    .await
    .unwrap_err();
    server.await.unwrap();
    assert!(err.to_string().contains("max_tokens"), "{err}");
    assert_eq!(
        req.messages.len(),
        1,
        "the cut-off assistant turn is not left in history"
    );
}

/// Регрессия пентеста: редирект на чужой хост не уносит ключ, если клиент собран через with_client_builder.
#[tokio::test]
async fn redirect_to_other_host_does_not_carry_the_key() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // Хост B: считает входящие соединения.
    let b = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let b_port = b.local_addr().unwrap().port();
    let hits = Arc::new(AtomicUsize::new(0));
    let hits2 = hits.clone();
    tokio::spawn(async move {
        while let Ok((mut s, _)) = b.accept().await {
            hits2.fetch_add(1, Ordering::SeqCst);
            let _ = s.write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}").await;
        }
    });
    // Хост A: отвечает 307 на B (другой хост: localhost против 127.0.0.1).
    let location = format!("http://localhost:{b_port}/v1/messages");
    let (base, _h) = serve(vec![(307, Box::leak(format!("location: {location}\r\n").into_boxed_str()), String::new())]).await;

    let p = AnthropicProvider::new("SECRET-ANT")
        .with_base_url(base)
        .with_max_retries(0)
        .with_client_builder(reqwest::Client::builder())
        .unwrap();
    let _ = p.chat(&ChatRequest::new("m").message(Message::user("hi"))).await;
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert_eq!(hits.load(Ordering::SeqCst), 0, "ключ ушёл на другой хост");
}

/// Регрессия пентеста: сервер возвращает ключ эхом в теле ошибки, а в текст ошибки он не попадает.
#[tokio::test]
async fn echoed_key_is_redacted_from_api_errors() {
    let body = r#"{"error":{"message":"bad header Authorization: Bearer sk-SUPERSECRET-123","type":"invalid_request_error"}}"#;
    let (base, _h) = one_shot(400, body).await;
    let p = OpenAiProvider::new("sk-SUPERSECRET-123").with_base_url(base).with_max_retries(0);
    let err = p.chat(&ChatRequest::new("m").message(Message::user("hi"))).await.unwrap_err();
    let shown = format!("{err} {err:?}");
    assert!(!shown.contains("sk-SUPERSECRET-123"), "{shown}");
    assert!(shown.contains("sk-****23"), "{shown}");
}

/// Регрессия пентеста: сервер присылает 100 вызовов, часть с необъявленным именем.
/// Обработчик видит не больше лимита за раунд и никогда не видит необъявленный инструмент.
#[tokio::test]
async fn hostile_tool_calls_are_capped_and_undeclared_names_rejected() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    let mut blocks = Vec::new();
    for i in 0..100 {
        let name = if i % 2 == 0 { "get_weather" } else { "rm_rf" };
        blocks.push(format!(r#"{{"type":"tool_use","id":"t{i}","name":"{name}","input":{{"city":"X"}}}}"#));
    }
    let call = format!(r#"{{"model":"m","content":[{}],"stop_reason":"tool_use"}}"#, blocks.join(","));
    let done = r#"{"model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}"#;
    let (base, _server) = serve(vec![(200, "", call), (200, "", done.into())]).await;
    let provider = AnthropicProvider::new("k").with_base_url(base);
    let mut req = ChatRequest::new("m").tool(weather_tool()).message(Message::user("go"));
    let seen = Arc::new(AtomicUsize::new(0));
    let seen2 = seen.clone();
    run_tools(&provider, &mut req, 3, move |call| {
        let seen = seen2.clone();
        async move {
            assert_eq!(call.name, "get_weather", "необъявленный инструмент дошёл до обработчика");
            seen.fetch_add(1, Ordering::SeqCst);
            Ok("ok".to_string())
        }
    })
    .await
    .unwrap();
    let n = seen.load(Ordering::SeqCst);
    assert!(n <= llm_universal::MAX_CALLS_PER_ROUND, "обработчик вызван {n} раз");
}
