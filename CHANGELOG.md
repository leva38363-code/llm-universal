## Unreleased

- `ChatRequest::cache_system()`: кэш системного промпта для Anthropic (`cache_control`), по умолчанию выключен.
- `Usage` складывается через `+=` (`AddAssign`) для подсчёта по сессии.
- `FallbackProvider`: опциональная цепочка провайдеров с переходом только на повторяемые ошибки.

## 0.3.1

- Релиз с подписью происхождения SLSA3 (release workflow).
- Версии 0.1.0 и 0.2.0 помечены как нерекомендуемые.

# Changelog

## 0.3.0

New:
- Streaming for every provider: `chat_stream` yields `StreamEvent`s and ends with the full response.
- Automatic retries like the official SDKs (connection errors, timeouts, 408/409/429/5xx;
  `retry-after`, `retry-after-ms`, `x-should-retry`; exponential backoff with jitter). `with_max_retries(n)`.
- Images in user messages: `Part::image_url`, `Part::image_base64`, `Message::user_with_image`.
- Structured output: `ChatRequest::json_schema` and `ChatResponse::json::<T>()`.
- Reasoning / thinking: `ChatRequest::reasoning`, `Part::Reasoning` (sent back unchanged), Gemini `thoughtSignature`.
- `run_tools`: the whole tool-calling loop in one call.
- Embeddings (`EmbeddingProvider`) for OpenAI and compatible servers, Gemini and Cohere.
- `list_models` for OpenAI, Anthropic, Gemini and Cohere.
- Vertex AI: `AnthropicProvider::vertex`, `GeminiProvider::vertex`. Amazon Bedrock: `OpenAiProvider::bedrock`.
- `Usage::cost`.

Fixes and hardening:
- `azure()` / `bedrock()` reject a resource or region that would change the host (key leak).
- Malformed stream events return an error instead of panicking.
- A stream that ends before the provider's final event is an error, not a silent partial answer.
- Unknown provider blocks (Anthropic server tools, Gemini code execution, bare thought signatures)
  are kept as `Part::Other` and sent back, so paused turns and Gemini 3 tool loops work.
- Tool arguments kept raw (invalid JSON) are no longer double-encoded or sent to Anthropic as a string.
- `run_tools` stops with an error when a tool call is cut off by `max_tokens`.
- Non-streaming calls have a 10-minute total limit per attempt; a broken 5xx body is retried.
- OpenAI-compatible servers get `max_tokens`; Mistral does not get `stream_options`.
- Gemini: effort maps to `thinkingBudget` (2.x) or `thinkingLevel` (3+); thought tokens count as output.
- Faster SSE parsing (linear), lone-CR line endings, `..` rejected in Vertex paths.

Breaking:
- `ChatResponse::tool_calls()` returns `Vec<ToolCall>` (`id`, `name`, `arguments`) instead of tuples.
- `Part` is `#[non_exhaustive]` and has new variants; `Part::ToolCall` has a `signature` field.
- `ApiError` has a `retry_after` field.
- The default client uses an idle read timeout (600 s) instead of a total timeout, so long streams are not cut off.

## 0.2.0

- New providers: `GeminiProvider` (Google Gemini) and `CohereProvider` (Cohere Chat v2).
- OpenAI-compatible presets: Azure OpenAI, Groq, Mistral, DeepSeek, xAI, OpenRouter, Together,
  Fireworks, Perplexity, Ollama, LM Studio.
- License changed to MIT-0 (MIT No Attribution): no attribution required.
- Security:
  - Response bodies are capped (32 MiB success, 64 KiB errors).
  - URLs are stripped from connection errors.
  - API keys are refused over plain `http://` to non-local hosts.
  - `base_url` with credentials or a query string is rejected.
  - Keys are validated and marked sensitive for every provider.
- Fixes:
  - Request-builder errors are `InvalidArgument` (not retryable) instead of `Connection`.
  - Per-provider tool-name length (Anthropic 128).
  - Duplicate tool names and messages without content are rejected.
- `LlmError::Deserialize` no longer prints the response body in its message.

## 0.1.0

- `AnthropicProvider` (Messages API) and `OpenAiProvider` (Chat Completions, OpenAI-compatible servers).
- Normalized messages, tool calls, tool results, stop reasons and usage.
- SDK-style errors: `LlmError`, `ApiError`, `ApiErrorKind`, request IDs, `is_retryable()`.
- Client-side validation, timeouts, no redirects, API keys redacted from `Debug`.
