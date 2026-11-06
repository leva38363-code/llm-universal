//! Client-side argument validation before the request is sent.

use llm_universal::{
    AnthropicProvider, ChatRequest, LlmError, LlmProvider, Message, OpenAiProvider, Part, Tool,
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

/// Регрессия пентеста: ключ не должен уходить по http:// на хост, который только похож на локальный.
#[tokio::test]
async fn lookalike_local_hosts_do_not_get_the_key_over_http() {
    for url in [
        "http://127.attacker.example/v1",
        "http://127.0.0.1.nip.io/v1",
        "http://evil.com\\.localhost/v1",
        "http://evil.localhost.attacker.example/v1",
        "http://x.localhost/v1",
        "http://evil.com /v1",
    ] {
        let p = OpenAiProvider::new("sk-SECRET").with_base_url(url);
        let msg = invalid_args(&p, hi()).await;
        assert!(!msg.contains("sk-SECRET"), "{url}: {msg}");
    }
}

/// Настоящие локальные адреса по-прежнему разрешены: ошибка будет сетевой, а не отказ валидации.
#[tokio::test]
async fn real_loopback_hosts_still_allowed_over_http() {
    for url in ["http://localhost:9/v1", "http://127.0.0.1:9/v1", "http://[::1]:9/v1", "http://LOCALHOST:9/v1"] {
        let p = OpenAiProvider::new("sk-x").with_base_url(url).with_max_retries(0);
        let err = p.chat(&hi()).await.unwrap_err();
        assert!(!matches!(err, LlmError::InvalidArgument(_)), "{url}: {err}");
    }
}

#[tokio::test]
async fn image_urls_must_be_https_and_not_local() {
    for url in [
        "http://example.com/a.png",
        "https://127.0.0.1/a.png",
        "https://10.0.0.5/a.png",
        "https://169.254.169.254/latest/meta-data",
        "https://localhost/a.png",
        "https://printer.internal/a.png",
        "https://[::1]/a.png",
        "https://[fd00::1]/a.png",
        "https://example.com\\@evil/a.png",
    ] {
        let p = OpenAiProvider::new("sk-x");
        let req = ChatRequest::new("m").message(Message::user_with_image(
            "что на картинке",
            Part::image_url(url),
        ));
        let msg = invalid_args(&p, req).await;
        assert!(msg.contains("image url"), "{url}: {msg}");
    }
}

#[tokio::test]
async fn public_https_image_url_is_accepted_by_validation() {
    let p = OpenAiProvider::new("sk-x").with_base_url("http://127.0.0.1:9/v1").with_max_retries(0);
    let req = ChatRequest::new("m").message(Message::user_with_image(
        "что на картинке",
        Part::image_url("https://images.example.com/cat.png"),
    ));
    let err = p.chat(&req).await.unwrap_err();
    assert!(!matches!(err, LlmError::InvalidArgument(_)), "{err}");
}

#[tokio::test]
async fn too_many_messages_are_refused_before_sending() {
    let p = OpenAiProvider::new("sk-x");
    let mut req = ChatRequest::new("m");
    for _ in 0..1001 {
        req = req.message(Message::user("x"));
    }
    let msg = invalid_args(&p, req).await;
    assert!(msg.contains("too many messages"), "{msg}");
}

/// Регрессия: все локальные и приватные формы IPv6 (включая вложенный IPv4) отклоняются.
#[tokio::test]
async fn ipv6_local_and_embedded_ipv4_forms_are_refused() {
    for url in [
        "https://[::1]/a.png",
        "https://[::]/a.png",
        "https://[fe80::1]/a.png",
        "https://[fec0::1]/a.png",
        "https://[fd12:3456::1]/a.png",
        "https://[ff02::1]/a.png",
        "https://[::ffff:127.0.0.1]/a.png",
        "https://[::ffff:10.0.0.5]/a.png",
        "https://[::ffff:169.254.169.254]/a.png",
        "https://[64:ff9b::7f00:1]/a.png",
        "https://[64:ff9b::a9fe:a9fe]/a.png",
        "https://[::7f00:1]/a.png",
    ] {
        let p = OpenAiProvider::new("sk-x");
        let req = ChatRequest::new("m").message(Message::user_with_image(
            "что на картинке",
            Part::image_url(url),
        ));
        let msg = invalid_args(&p, req).await;
        assert!(msg.contains("image url"), "{url}: {msg}");
    }
}

/// Публичный IPv6 не должен отклоняться как локальный.
#[tokio::test]
async fn public_ipv6_image_url_is_accepted() {
    let p = OpenAiProvider::new("sk-x").with_base_url("http://127.0.0.1:9/v1").with_max_retries(0);
    let req = ChatRequest::new("m").message(Message::user_with_image(
        "что на картинке",
        Part::image_url("https://[2606:4700:4700::1111]/cat.png"),
    ));
    let err = p.chat(&req).await.unwrap_err();
    assert!(!matches!(err, LlmError::InvalidArgument(_)), "{err}");
}

#[tokio::test]
async fn wikipedia_and_mirrors_are_blocked_for_images() {
    for url in [
        "https://upload.wikimedia.org/wikipedia/commons/a/a9/Example.jpg",
        "https://en.wikipedia.org/wiki/File:Example.jpg",
        "https://ru.m.wikipedia.org/x.png",
        "https://commons.wikimedia.org/x.png",
        "https://www.wikiwand.com/x.png",
        "https://UPLOAD.WIKIMEDIA.ORG/x.png",
    ] {
        let p = OpenAiProvider::new("sk-x");
        let req = ChatRequest::new("m").message(Message::user_with_image("?", Part::image_url(url)));
        let msg = invalid_args(&p, req).await;
        assert!(msg.contains("blocked"), "{url}: {msg}");
    }
    // Похожий, но чужой домен не блокируется.
    let p = OpenAiProvider::new("sk-x").with_base_url("http://127.0.0.1:9/v1").with_max_retries(0);
    let req = ChatRequest::new("m").message(Message::user_with_image("?", Part::image_url("https://notwikipedia.org/x.png")));
    let err = p.chat(&req).await.unwrap_err();
    assert!(!matches!(err, LlmError::InvalidArgument(_)), "{err}");
}

#[tokio::test]
async fn image_url_bypass_forms_are_refused() {
    for url in [
        "https://2130706433/x.png",      // 127.0.0.1 as one integer
        "https://0x7f.1/x.png",          // 127.0.0.1 with hex octet
        "https://127.1/x.png",           // 127.0.0.1 short form
        "https://localhost./x.png",      // trailing dot
        "https://127.0.0.1./x.png",      // trailing dot on an IP
        "https://en.wikipedia.org./x.png", // trailing dot defeats the block list
        "https://100.64.0.1/x.png",      // carrier-grade NAT
        "https://user:pw@example.com/x.png", // credentials in the URL
    ] {
        let p = OpenAiProvider::new("sk-x");
        let req = ChatRequest::new("m").message(Message::user_with_image("?", Part::image_url(url)));
        let msg = invalid_args(&p, req).await;
        assert!(!msg.is_empty(), "{url} should be refused");
    }
}
