//! Spending limits per user per UTC day. A budget is required: the default is a real limit,
//! and running without one needs an explicit `Budget::unlimited()`.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::{LlmError, Result, Usage};

/// Limits for one user for one UTC day.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Most input + output tokens per day.
    pub max_tokens: u64,
    /// Most money per day, in the same currency as the prices passed to `record`.
    pub max_cost: f64,
}

#[derive(Debug, Default, Clone, Copy)]
struct Spent {
    day: u64,
    tokens: u64,
    cost: f64,
}

/// Tracks spending per user id and refuses requests that would exceed the limits.
#[derive(Debug)]
pub struct Budget {
    limits: Option<Limits>,
    spent: Mutex<HashMap<String, Spent>>,
}

fn today() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() / 86_400)
        .unwrap_or(0)
}

impl Budget {
    /// Enforces the given per-user, per-day limits.
    pub fn new(limits: Limits) -> Self {
        Self { limits: Some(limits), spent: Mutex::new(HashMap::new()) }
    }

    /// No limits at all. Use only on purpose: every user can spend without bound.
    pub fn unlimited() -> Self {
        Self { limits: None, spent: Mutex::new(HashMap::new()) }
    }

    /// Fails before a request if the user has already reached a limit today.
    /// `estimated_tokens` is a rough size of the request; it is added to what is spent.
    pub fn check(&self, user: &str, estimated_tokens: u64) -> Result<()> {
        let Some(limits) = self.limits else { return Ok(()) };
        let spent = self.spent.lock().unwrap_or_else(|e| e.into_inner());
        let today_spent = spent.get(user).copied().filter(|s| s.day == today()).unwrap_or_default();
        if today_spent.tokens.saturating_add(estimated_tokens) > limits.max_tokens {
            return Err(LlmError::InvalidArgument(format!(
                "daily token limit reached for this user ({} of {})",
                today_spent.tokens, limits.max_tokens
            )));
        }
        if today_spent.cost >= limits.max_cost {
            return Err(LlmError::InvalidArgument("daily cost limit reached for this user".into()));
        }
        Ok(())
    }

    /// Records what a finished request cost. `input_per_million` / `output_per_million` are
    /// the model prices.
    pub fn record(&self, user: &str, usage: Usage, input_per_million: f64, output_per_million: f64) {
        let today = today();
        let mut spent = self.spent.lock().unwrap_or_else(|e| e.into_inner());
        let entry = spent.entry(user.to_string()).or_default();
        if entry.day != today {
            *entry = Spent { day: today, tokens: 0, cost: 0.0 };
        }
        entry.tokens = entry
            .tokens
            .saturating_add(u64::from(usage.input_tokens) + u64::from(usage.output_tokens));
        entry.cost += usage.cost(input_per_million, output_per_million);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lim() -> Limits {
        Limits { max_tokens: 1000, max_cost: 5.0 }
    }

    #[test]
    fn refuses_when_tokens_would_exceed() {
        let b = Budget::new(lim());
        assert!(b.check("u1", 900).is_ok());
        b.record("u1", Usage { input_tokens: 800, output_tokens: 0 }, 0.0, 0.0);
        assert!(b.check("u1", 300).is_err());
        assert!(b.check("u2", 300).is_ok(), "limits are per user");
    }

    #[test]
    fn refuses_when_cost_reached() {
        let b = Budget::new(lim());
        b.record("u1", Usage { input_tokens: 1_000_000, output_tokens: 0 }, 6.0, 0.0);
        assert!(b.check("u1", 1).is_err());
    }

    #[test]
    fn unlimited_only_when_asked_for() {
        let b = Budget::unlimited();
        b.record("u1", Usage { input_tokens: u32::MAX, output_tokens: 0 }, 1e6, 0.0);
        assert!(b.check("u1", u64::MAX / 2).is_ok());
    }
}
