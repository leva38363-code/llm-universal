use std::future::Future;

use crate::{
    ChatRequest, ChatResponse, LlmError, LlmProvider, Message, Part, Result, Role, StopReason,
    ToolCall,
};

/// Runs the tool loop until the model gives a final answer.
///
/// Each round sends `request`, calls `handler` for every tool call (in order) and sends the
/// results back. `handler` returns `Ok(content)` or `Err(message)`; errors are reported to the
/// model as failed tool results. Paused turns (`pause_turn`) are continued automatically and
/// count as a round. A tool call cut off by `max_tokens` is an error, not run.
///
/// The whole conversation, including the final answer, is left in `request.messages`, so the
/// next user turn can be appended to it. Fails after `max_rounds` rounds without a final answer.
///
/// ```no_run
/// # use llm_universal::*;
/// # use serde_json::json;
/// # async fn demo(provider: &dyn LlmProvider) -> Result<()> {
/// let mut req = ChatRequest::new("claude-opus-5-5")
///     .tool(Tool::new("get_weather", "Weather in a city", json!({"type": "object", "properties": {"city": {"type": "string"}}})))
///     .message(Message::user("Weather in Berlin?"));
/// let answer = run_tools(provider, &mut req, 5, |call| async move {
///     match call.name.as_str() {
///         "get_weather" => Ok(format!("18C in {}", call.arguments["city"])),
///         other => Err(format!("unknown tool {other}")),
///     }
/// })
/// .await?;
/// println!("{}", answer.text());
/// # Ok(()) }
/// ```
pub async fn run_tools<F, Fut>(
    provider: &dyn LlmProvider,
    request: &mut ChatRequest,
    max_rounds: usize,
    mut handler: F,
) -> Result<ChatResponse>
where
    F: FnMut(ToolCall) -> Fut,
    Fut: Future<Output = std::result::Result<String, String>>,
{
    for _ in 0..max_rounds {
        let response = provider.chat(request).await?;
        request.messages.push(response.message.clone());

        if response.stop_reason == StopReason::Paused {
            continue;
        }
        let calls = response.tool_calls();
        if calls.is_empty() {
            return Ok(response);
        }
        if response.stop_reason == StopReason::MaxTokens {
            // The tool calls were cut off mid-arguments; running them would use broken input,
            // and leaving them unanswered would break the next request.
            request.messages.pop();
            return Err(LlmError::InvalidResponse(
                "the model hit max_tokens while writing a tool call; raise max_tokens".into(),
            ));
        }

        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            let id = call.id.clone();
            let part = match handler(call).await {
                Ok(content) => Part::tool_result(id, content, false),
                Err(message) => Part::tool_result(id, message, true),
            };
            results.push(part);
        }
        request.messages.push(Message::new(Role::Tool, results));
    }
    Err(LlmError::InvalidResponse(format!(
        "model did not finish within {max_rounds} tool rounds"
    )))
}
