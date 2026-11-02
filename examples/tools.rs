//! The same tool loop runs against any provider.
//!
//! The provider is picked from environment variables:
//!   ANTHROPIC_API_KEY=...  -> Anthropic
//!   GEMINI_API_KEY=...     -> Google Gemini
//!   OPENAI_API_KEY=...     -> OpenAI (set OPENAI_BASE_URL for a compatible server)
//!   LLM_MODEL=...          -> optional, defaults below
//!
//! Run with: `cargo run --example tools`

use llm_universal::{
    run_tools, AnthropicProvider, ChatRequest, GeminiProvider, LlmProvider, Message,
    OpenAiProvider, Result, Tool,
};
use serde_json::json;

fn make_provider() -> (Box<dyn LlmProvider>, String) {
    let model = |default: &str| std::env::var("LLM_MODEL").unwrap_or_else(|_| default.into());
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        return (
            Box::new(AnthropicProvider::new(key)),
            model("claude-opus-5-5"),
        );
    }
    if let Ok(key) = std::env::var("GEMINI_API_KEY") {
        return (
            Box::new(GeminiProvider::new(key)),
            model("gemini-2.5-flash"),
        );
    }
    let mut provider = OpenAiProvider::new(std::env::var("OPENAI_API_KEY").unwrap_or_default());
    if let Ok(url) = std::env::var("OPENAI_BASE_URL") {
        provider = provider.with_base_url(url);
    }
    (Box::new(provider), model("gpt-4o"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let (provider, model) = make_provider();

    let weather = Tool::new(
        "get_weather",
        "Get the current weather in a city",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
            "additionalProperties": false
        }),
    );

    let mut req = ChatRequest::new(model)
        .system("Answer briefly.")
        .max_tokens(1024)
        .tool(weather)
        .message(Message::user("What's the weather in Berlin?"));

    // Calls the tools until the model answers; at most 5 rounds.
    let answer = run_tools(provider.as_ref(), &mut req, 5, |call| async move {
        match call.name.as_str() {
            "get_weather" => {
                let city = call.arguments["city"].as_str().unwrap_or("?");
                Ok(json!({ "city": city, "temp_c": 18, "sky": "cloudy" }).to_string())
            }
            other => Err(format!("unknown tool `{other}`")),
        }
    })
    .await?;

    println!("{}", answer.text());
    println!(
        "[{}] {} in / {} out tokens",
        answer.model, answer.usage.input_tokens, answer.usage.output_tokens
    );
    Ok(())
}
