# llm-universal

[![crates.io](https://img.shields.io/crates/v/llm-universal.svg)](https://crates.io/crates/llm-universal) [![CI](https://github.com/leva38363-code/llm-universal/actions/workflows/ci.yml/badge.svg)](https://github.com/leva38363-code/llm-universal/actions/workflows/ci.yml) [![License: MIT-0](https://img.shields.io/badge/license-MIT--0-blue.svg)](LICENSE)

| Provider | Constructor | API |
|---|---|---|
| Anthropic (Claude) | `AnthropicProvider::new(key)` | Messages API |
| Claude on Vertex AI | `AnthropicProvider::vertex(project, region, access_token)` | `rawPredict` |
| OpenAI | `OpenAiProvider::new(key)` | Chat Completions |
| Google Gemini | `GeminiProvider::new(key)` | `generateContent` |
| Gemini on Vertex AI | `GeminiProvider::vertex(project, region, access_token)` | `generateContent` |
| Cohere | `CohereProvider::new(key)` | Chat API v2 |
| Azure OpenAI | `OpenAiProvider::azure(resource, key)` | Azure OpenAI v1 (`api-key` header) |
| Amazon Bedrock | `OpenAiProvider::bedrock(region, bedrock_api_key)` | Bedrock OpenAI-compatible endpoint |
| Groq | `OpenAiProvider::groq(key)` | OpenAI-compatible |
| Mistral | `OpenAiProvider::mistral(key)` | OpenAI-compatible |
| DeepSeek | `OpenAiProvider::deepseek(key)` | OpenAI-compatible |
| xAI (Grok) | `OpenAiProvider::xai(key)` | OpenAI-compatible |
| OpenRouter | `OpenAiProvider::openrouter(key)` | OpenAI-compatible |
| Together AI | `OpenAiProvider::together(key)` | OpenAI-compatible |
| Fireworks AI | `OpenAiProvider::fireworks(key)` | OpenAI-compatible |
| Perplexity | `OpenAiProvider::perplexity(key)` | OpenAI-compatible |
| Ollama (local) | `OpenAiProvider::ollama()` | OpenAI-compatible |
| LM Studio (local) | `OpenAiProvider::lm_studio()` | OpenAI-compatible |
| Anything else OpenAI-compatible (vLLM, LiteLLM, …) | `OpenAiProvider::new(key).with_base_url(url)` | OpenAI-compatible |

## Features

| Feature | How |
|---|---|
| Chat | `provider.chat(&req)` |
| Streaming | `provider.chat_stream(&req)` → `StreamEvent::TextDelta`, …, `StreamEvent::Done(response)` |
| Tool calling | `Tool::new(...)`, `response.tool_calls()`, or the whole loop with `run_tools(...)` |
| Images | `Message::user_with_image(text, Part::image_url(url))` / `Part::image_base64(type, data)` |
| Structured output | `.json_schema(name, schema)` then `response.json::<T>()` |
| Reasoning / thinking | `.reasoning(ReasoningEffort::High)`, read `response.reasoning()` |
| Embeddings | `EmbeddingProvider::embed(&provider, model, &inputs)` (OpenAI & compatible, Gemini, Cohere) |
| Model list | `provider.list_models()` |
| Retries | automatic, like the official SDKs; `.with_max_retries(n)` |
| Cost | `response.usage.cost(input_price_per_million, output_price_per_million)` |

Safe defaults:

- Connect timeout, idle read timeout, a 32 MiB response cap and a 4 MiB cap per stream event.
- No redirects, so the key never goes to another host.
- Keys are refused over plain `http://` to non-local hosts.
- Keys are redacted from `Debug` and marked sensitive in headers.
- URLs are stripped from connection errors.
- Model, project and region are checked before they go into a URL.

## Install

```toml
[dependencies]
llm-universal = "0.3"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

## Quick start

```rust,no_run
use llm_universal::{AnthropicProvider, ChatRequest, LlmProvider, Message, OpenAiProvider};

#[tokio::main]
async fn main() -> llm_universal::Result<()> {
    let provider: Box<dyn LlmProvider> = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(key) => Box::new(AnthropicProvider::new(key)),
        Err(_) => Box::new(OpenAiProvider::new(std::env::var("OPENAI_API_KEY").unwrap_or_default())),
    };

    let req = ChatRequest::new("claude-opus-5-5")
        .system("Be concise.")
        .max_tokens(512)
        .message(Message::user("Hello!"));

    let resp = provider.chat(&req).await?;
    println!("{}", resp.text());
    Ok(())
}
```

## Streaming

```rust,no_run
# use llm_universal::*;
# async fn run(provider: &dyn LlmProvider, req: &ChatRequest) -> Result<()> {
let mut stream = provider.chat_stream(req).await?;
while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::TextDelta(text) => print!("{text}"),
        StreamEvent::Done(response) => println!("\n{} tokens", response.usage.output_tokens),
        _ => {}
    }
}
# Ok(()) }
```

## Tool calling

`run_tools` sends the request, runs your handler for every tool call, sends the results back and
repeats until the model answers:

```rust,no_run
# use llm_universal::*;
# use serde_json::json;
# async fn run(provider: &dyn LlmProvider) -> Result<()> {
let weather = Tool::new(
    "get_weather",
    "Get the current weather in a city",
    json!({ "type": "object", "properties": { "city": { "type": "string" } }, "required": ["city"] }),
);
let mut req = ChatRequest::new("claude-opus-5-5")
    .tool(weather)
    .message(Message::user("What's the weather in Berlin?"));

let answer = run_tools(provider, &mut req, 5, |call| async move {
    match call.name.as_str() {
        "get_weather" => Ok(format!("18°C, cloudy in {}", call.arguments["city"])),
        other => Err(format!("unknown tool {other}")),
    }
})
.await?;
println!("{}", answer.text());
# Ok(()) }
```

The manual loop: check `response.tool_calls()`, push `response.message` to `req.messages`, then
push `Message::new(Role::Tool, results)` built with `Part::tool_result(id, content, is_error)`.

## Images

```rust,no_run
# use llm_universal::*;
# async fn run(provider: &dyn LlmProvider, png_base64: String) -> Result<()> {
let req = ChatRequest::new("claude-opus-5-5")
    .message(Message::user_with_image("What is in this picture?", Part::image_base64("image/png", png_base64)));
println!("{}", provider.chat(&req).await?.text());
# Ok(()) }
```

`Part::image_url(url)` works with OpenAI, Anthropic and Cohere; Gemini needs base64 data.

## Structured output

```rust,no_run
# use llm_universal::*;
# use serde_json::json;
# async fn run(provider: &dyn LlmProvider) -> Result<()> {
#[derive(serde::Deserialize)]
struct City { name: String, population: u64 }

let req = ChatRequest::new("gpt-4o")
    .json_schema("city", json!({
        "type": "object",
        "properties": { "name": { "type": "string" }, "population": { "type": "integer" } },
        "required": ["name", "population"],
        "additionalProperties": false
    }))
    .message(Message::user("Largest city in Germany?"));
let city: City = provider.chat(&req).await?.json()?;
println!("{} — {}", city.name, city.population);
# Ok(()) }
```

## Fallback

Opt-in: nothing changes unless you build a `FallbackProvider` yourself. It tries providers in order
and moves on only on errors worth retrying elsewhere (network, timeout, 408/409/429, 5xx).
Auth and bad-request errors are returned at once.

```rust,ignore
use llm_universal::{FallbackProvider, LlmProvider, OpenAiProvider, AnthropicProvider};

let chain = FallbackProvider::new(vec![
    Box::new(AnthropicProvider::new(anthropic_key)),
    Box::new(OpenAiProvider::new(openai_key)),
]).expect("at least one provider");
```

## Reliability

- Retries: connection failures and `429` are retried by default. A `5xx` or timeout is **not** retried by default, because the server may already have generated and billed the answer. Enable with `.with_repeat_post(true)` only when repeating the same request is safe.
- Time limits: `.with_request_timeout(..)` for one non-streaming attempt (default 10 min), `.with_stream_deadline(..)` for a whole stream (default 1 h).
- Fallback: `FallbackProvider` moves to the next provider on a dropped connection or `429`. Use `.with_server_errors(true)` to also move on `5xx` and timeouts.
- Images by URL must be `https://` and public: local and private addresses are refused.

## Errors

| `LlmError` variant | When | SDK equivalent |
|---|---|---|
| `Api(Box<ApiError>)` | non-2xx response (or an error event inside a stream) | `APIStatusError` and subclasses |
| `Connection` | DNS, TLS, dropped connection | `APIConnectionError` |
| `Timeout` | request timed out | `APIConnectionTimeoutError` |
| `Deserialize` | body is not JSON (`content` keeps the start of it; not printed) | `JSONDeserialize` (async-openai) |
| `InvalidResponse` | JSON is missing expected fields, or a size cap was hit | — |
| `InvalidArgument` | client-side validation failed | `InvalidArgument` (async-openai) |

`ApiError` carries `status`, `kind` (`ApiErrorKind::RateLimit`, …), `message`, `error_type`,
`code`, `param`, `request_id`, `retry_after` and the JSON `body`. The message format matches the SDKs:

```text
401 invalid x-api-key
429 Rate limit reached for requests
invalid args: model must not be empty
```

Retries follow the SDKs: 2 retries by default on connection errors, timeouts, 408, 409, 429 and
>= 500. `retry-after` / `retry-after-ms` (up to 60 s) and `x-should-retry` are honored; otherwise
the delay is 0.5 s doubling up to 8 s, with jitter. `err.is_retryable()` tells you whether a final
error is worth retrying later.

## Notes

- Anthropic's newer models accept only `temperature = 1.0`; leave `temperature` unset unless you need it.
- Reasoning blocks (`Part::Reasoning`) and Gemini's `thoughtSignature` are kept in the message and sent
  back unchanged on the next turn, as Anthropic and Gemini require. Keep `response.message` in the history.
- Gemini tool calls without an id get a local id (`llm-universal-call-N`) that is never sent back to the API.
- Vertex AI needs an OAuth access token (for example `gcloud auth print-access-token`); tokens expire after
  about an hour. Model lists and Gemini embeddings are not available on Vertex in this crate.
- Bedrock uses the OpenAI-compatible endpoint with a Bedrock API key; SigV4 signing is not implemented.

## License

[MIT-0](LICENSE) (MIT No Attribution): use, copy, modify and sell it for any purpose, commercial included, with no attribution required.
