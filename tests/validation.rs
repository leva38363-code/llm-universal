//! Client-side argument validation before the request is sent.

use llm_universal::{
    AnthropicProvider, ChatRequest, LlmError, LlmProvider, Message, OpenAiProvider, Tool,
};
use serde_json::json;

async fn invalid_args(provider: &dyn LlmProvider, req: ChatRequest) -> String {
    match provider.chat(&req).await.unwrap_err() {
        e @ LlmError::InvalidArgument(_) => e.to_string(),
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

#[tokio::test]
async fn empty_anthropic_key() {
    let msg = invalid_args(
        &AnthropicProvider::new(""),
        ChatRequest::new("m").message(Message::user("x")),
    )
    .await;
    assert_eq!(
        msg,
        "invalid args: AnthropicProvider: api_key must not be empty"
    );
}

#[tokio::test]
async fn missing_model_and_messages() {
    let p = OpenAiProvider::new("k");
    let msg = invalid_args(&p, ChatRequest::new("").message(Message::user("x"))).await;
    assert_eq!(msg, "invalid args: model must not be empty");
    let msg = invalid_args(&p, ChatRequest::new("m")).await;
    assert_eq!(msg, "invalid args: messages must not be empty");
}

#[tokio::test]
async fn bad_tool_and_base_url() {
    let req = ChatRequest::new("m")
        .message(Message::user("x"))
        .tool(Tool::new("bad name", "d", json!({"type": "object"})));
    let msg = invalid_args(&OpenAiProvider::new("k"), req).await;
    assert_eq!(
        msg,
        "invalid args: tool name `bad name` must match ^[a-zA-Z0-9_-]{1,64}$"
    );

    let p = OpenAiProvider::new("k").with_base_url("localhost:11434");
    let msg = invalid_args(&p, ChatRequest::new("m").message(Message::user("x"))).await;
    assert!(msg.contains("base_url must start with"), "{msg}");
}

fn hi() -> ChatRequest {
    ChatRequest::new("m").message(Message::user("x"))
}

#[tokio::test]
async fn refuses_key_over_plain_http_to_remote_host() {
    let p = OpenAiProvider::new("sk-x").with_base_url("http://api.example.com/v1");
    let msg = invalid_args(&p, hi()).await;
    assert!(msg.contains("plain http://"), "{msg}");
}

#[tokio::test]
async fn rejects_credentials_and_query_in_base_url() {
    for url in [
        "https://user:pass@example.com/v1",
        "https://example.com/v1?key=secret",
    ] {
        let msg = invalid_args(&OpenAiProvider::new("k").with_base_url(url), hi()).await;
        assert!(msg.contains("must not contain credentials"), "{msg}");
        assert!(!msg.contains("secret") && !msg.contains("pass"), "{msg}");
    }
}

#[tokio::test]
async fn rejects_duplicate_tools_and_empty_messages() {
    let tool = Tool::new("f", "d", json!({"type": "object"}));
    let req = hi().tool(tool.clone()).tool(tool);
    let msg = invalid_args(&OpenAiProvider::new("k"), req).await;
    assert_eq!(msg, "invalid args: duplicate tool name `f`");

    let req = ChatRequest::new("m").message(Message::user(""));
    let msg = invalid_args(&OpenAiProvider::new("k"), req).await;
    assert_eq!(msg, "invalid args: messages[0] has no content");
}

#[tokio::test]
async fn bad_key_characters_are_not_retryable() {
    let p = OpenAiProvider::new("sk-bad\nkey");
    let err = p.chat(&hi()).await.unwrap_err();
    assert!(matches!(err, LlmError::InvalidArgument(_)), "{err:?}");
    assert!(!err.is_retryable());
}

#[tokio::test]
async fn anthropic_allows_longer_tool_names() {
    // 100 characters: too long for OpenAI, valid for Anthropic. Fails later at the network, not in validation.
    let tool = Tool::new("a".repeat(100), "d", json!({"type": "object"}));
    let p = AnthropicProvider::new("k").with_base_url("http://127.0.0.1:9");
    let err = p.chat(&hi().tool(tool)).await.unwrap_err();
    assert!(!matches!(err, LlmError::InvalidArgument(_)), "{err:?}");
}

#[tokio::test]
async fn preset_host_labels_cannot_redirect_the_key() {
    for provider in [
        OpenAiProvider::azure("evil.com/", "k"),
        OpenAiProvider::azure("evil.com", "k"),
        OpenAiProvider::bedrock("evil.com/x", "k"),
    ] {
        let msg = invalid_args(&provider, hi()).await;
        assert!(msg.contains("may only contain"), "{msg}");
    }
}

#[tokio::test]
async fn vertex_rejects_dot_segments() {
    let p = llm_universal::GeminiProvider::vertex("..", "us-central1", "t");
    let msg = invalid_args(&p, hi()).await;
    assert!(msg.contains("project"), "{msg}");
}
