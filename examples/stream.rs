//! Streams an answer token by token.
//!
//! `ANTHROPIC_API_KEY=... cargo run --example stream`
//! (or `OPENAI_API_KEY`, or nothing for a local Ollama).

use std::io::Write;

use llm_universal::{
    AnthropicProvider, ChatRequest, LlmProvider, Message, OpenAiProvider, Result, StreamEvent,
    StreamExt,
};

#[tokio::main]
async fn main() -> Result<()> {
    let (provider, model): (Box<dyn LlmProvider>, &str) =
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            (Box::new(AnthropicProvider::new(key)), "claude-opus-5-5")
        } else if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            (Box::new(OpenAiProvider::new(key)), "gpt-4o")
        } else {
            (Box::new(OpenAiProvider::ollama()), "llama3.2")
        };

    let req = ChatRequest::new(model)
        .max_tokens(1024)
        .message(Message::user("Write a haiku about Rust."));

    let mut stream = provider.chat_stream(&req).await?;
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta(text) => {
                print!("{text}");
                std::io::stdout().flush().ok();
            }
            StreamEvent::Done(response) => {
                println!(
                    "\n\n[{:?}, {} output tokens]",
                    response.stop_reason, response.usage.output_tokens
                );
            }
            _ => {}
        }
    }
    Ok(())
}
