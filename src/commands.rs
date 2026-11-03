//! Slash commands: named prompt templates. `add("tr", "Translate to English: {args}")`,
//! then `/tr привет` expands to `Translate to English: привет`.

use std::collections::BTreeMap;

use crate::{ChatRequest, LlmError, Message, Result};

/// Registry of `/name` commands. Names: 1–32 chars of `a-z`, `0-9`, `_`, `-`.
#[derive(Debug, Clone, Default)]
pub struct Commands {
    templates: BTreeMap<String, String>,
}

fn valid_name(name: &str) -> bool {
    (1..=32).contains(&name.len())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
}

impl Commands {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or replaces a command. `{args}` in the template is replaced by the text after the name.
    pub fn add(&mut self, name: &str, template: impl Into<String>) -> Result<()> {
        let name = name.trim_start_matches('/');
        if !valid_name(name) {
            return Err(LlmError::InvalidArgument(format!(
                "command name `{name}` must be 1-32 chars of a-z, 0-9, _ or -"
            )));
        }
        self.templates.insert(name.to_string(), template.into());
        Ok(())
    }

    /// Removes a command; returns whether it existed.
    pub fn remove(&mut self, name: &str) -> bool {
        self.templates.remove(name.trim_start_matches('/')).is_some()
    }

    /// Command names, sorted.
    pub fn list(&self) -> Vec<&str> {
        self.templates.keys().map(String::as_str).collect()
    }

    /// `/name args` -> the filled template. `None` if the input is not a known command.
    pub fn expand(&self, input: &str) -> Option<String> {
        let rest = input.trim_start().strip_prefix('/')?;
        let (name, args) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
        let template = self.templates.get(name)?;
        Some(template.replace("{args}", args.trim()))
    }

    /// Adds the expanded command (or the input as is, if it is not a command) as a user message.
    pub fn apply(&self, input: &str, request: ChatRequest) -> ChatRequest {
        let text = self.expand(input).unwrap_or_else(|| input.to_string());
        request.message(Message::user(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_expand_remove() {
        let mut c = Commands::new();
        c.add("/tr", "Translate to English: {args}").unwrap();
        c.add("sum", "Summarize:\n{args}").unwrap();
        assert_eq!(c.expand("/tr привет мир").as_deref(), Some("Translate to English: привет мир"));
        assert_eq!(c.expand("  /sum  text  ").as_deref(), Some("Summarize:\ntext"));
        assert_eq!(c.expand("/unknown x"), None);
        assert_eq!(c.expand("no slash"), None);
        assert_eq!(c.list(), vec!["sum", "tr"]);
        assert!(c.remove("tr"));
        assert!(!c.remove("tr"));
    }

    #[test]
    fn bad_names_are_rejected() {
        let mut c = Commands::new();
        for bad in ["", "Big", "a b", "x/y", "ы", &"a".repeat(33)] {
            assert!(c.add(bad, "t").is_err(), "{bad:?}");
        }
    }

    #[test]
    fn apply_adds_user_message() {
        let mut c = Commands::new();
        c.add("tr", "T: {args}").unwrap();
        let req = c.apply("/tr hi", ChatRequest::new("m"));
        assert_eq!(req.messages[0].text_content(), "T: hi");
        let req = c.apply("plain", ChatRequest::new("m"));
        assert_eq!(req.messages[0].text_content(), "plain");
    }
}
