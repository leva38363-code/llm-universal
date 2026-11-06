//! Spending limits per user per UTC day. A budget is required: the default is a real limit,
//! and running without one needs an explicit `Budget::unlimited()`.
//!
//! Use [`Budget::reserve`] before a request and [`Reservation::settle`] after it. The reservation
//! holds the estimated tokens while the request runs, so concurrent requests cannot all pass the
//! check together. Dropping a reservation without settling it releases the hold and records nothing.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::{LlmError, Result, Usage};

/// Limits for one user for one UTC day.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Most tokens per day (input, cache reads and writes, and output).
    pub max_tokens: u64,
    /// Most money per day, in the same currency as the prices passed to `record`.
    pub max_cost: f64,
}

#[derive(Debug, Default, Clone, Copy)]
struct Spent {
    day: u64,
    tokens: u64,
    cost: f64,
    /// Estimated tokens held by requests that are still running.
    reserved: u64,
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

/// Spending of `user` today, starting a fresh day when the stored one is old.
fn today_entry(spent: &mut HashMap<String, Spent>, user: &str, day: u64) -> Spent {
    let entry = spent.entry(user.to_string()).or_default();
    if entry.day != day {
        *entry = Spent {
            day,
            tokens: 0,
            cost: 0.0,
            reserved: entry.reserved,
        };
    }
    *entry
}

/// True when `cost` is at or over `max`, or when the two cannot be compared (NaN), so an
/// unusable cost refuses requests instead of letting them through.
fn cost_limit_reached(cost: f64, max: f64) -> bool {
    cost.partial_cmp(&max) != Some(std::cmp::Ordering::Less)
}

/// Drops users whose spending belongs to an earlier day and who have nothing in flight.
fn evict_old(spent: &mut HashMap<String, Spent>, day: u64) {
    spent.retain(|_, s| s.day == day || s.reserved > 0);
}

impl Budget {
    /// Enforces the given per-user, per-day limits.
    pub fn new(limits: Limits) -> Self {
        Self {
            limits: Some(limits),
            spent: Mutex::new(HashMap::new()),
        }
    }

    /// No limits at all. Use only on purpose: every user can spend without bound.
    pub fn unlimited() -> Self {
        Self {
            limits: None,
            spent: Mutex::new(HashMap::new()),
        }
    }

    /// Holds `estimated_tokens` for `user` until the returned [`Reservation`] is settled or
    /// dropped. Fails at once if the user has reached a limit today, counting requests that are
    /// still running.
    pub fn reserve(&self, user: &str, estimated_tokens: u64) -> Result<Reservation<'_>> {
        let Some(limits) = self.limits else {
            return Ok(Reservation {
                budget: self,
                user: user.to_string(),
                tokens: 0,
                open: false,
            });
        };
        let day = today();
        let mut spent = self.spent.lock().unwrap_or_else(|e| e.into_inner());
        evict_old(&mut spent, day);
        let current = today_entry(&mut spent, user, day);
        let in_use = current.tokens.saturating_add(current.reserved);
        if in_use.saturating_add(estimated_tokens) > limits.max_tokens {
            return Err(LlmError::BudgetExceeded(format!(
                "daily token limit reached for this user ({in_use} of {})",
                limits.max_tokens
            )));
        }
        if cost_limit_reached(current.cost, limits.max_cost) {
            return Err(LlmError::BudgetExceeded(
                "daily cost limit reached for this user".into(),
            ));
        }
        let entry = spent.entry(user.to_string()).or_default();
        entry.reserved = entry.reserved.saturating_add(estimated_tokens);
        Ok(Reservation {
            budget: self,
            user: user.to_string(),
            tokens: estimated_tokens,
            open: true,
        })
    }

    /// Fails before a request if the user has already reached a limit today. It does not hold
    /// anything, so concurrent requests can all pass; prefer [`Budget::reserve`].
    pub fn check(&self, user: &str, estimated_tokens: u64) -> Result<()> {
        let Some(limits) = self.limits else {
            return Ok(());
        };
        let day = today();
        let mut spent = self.spent.lock().unwrap_or_else(|e| e.into_inner());
        let current = today_entry(&mut spent, user, day);
        if current.tokens.saturating_add(estimated_tokens) > limits.max_tokens {
            return Err(LlmError::BudgetExceeded(format!(
                "daily token limit reached for this user ({} of {})",
                current.tokens, limits.max_tokens
            )));
        }
        if cost_limit_reached(current.cost, limits.max_cost) {
            return Err(LlmError::BudgetExceeded(
                "daily cost limit reached for this user".into(),
            ));
        }
        Ok(())
    }

    /// Records what a finished request cost. `input_per_million` / `output_per_million` are
    /// the model prices. Unusable prices (NaN, infinite, negative) block the user for the day
    /// rather than silently disabling the cost limit.
    pub fn record(
        &self,
        user: &str,
        usage: Usage,
        input_per_million: f64,
        output_per_million: f64,
    ) {
        let day = today();
        let mut spent = self.spent.lock().unwrap_or_else(|e| e.into_inner());
        evict_old(&mut spent, day);
        let _ = today_entry(&mut spent, user, day);
        let entry = spent.entry(user.to_string()).or_default();
        entry.tokens = entry.tokens.saturating_add(usage.total_tokens());
        let prices_ok = input_per_million.is_finite()
            && output_per_million.is_finite()
            && input_per_million >= 0.0
            && output_per_million >= 0.0;
        let cost = usage.cost(input_per_million, output_per_million);
        entry.cost = if prices_ok && cost.is_finite() {
            entry.cost + cost
        } else {
            f64::INFINITY
        };
    }
}

/// Tokens held for one request. Call [`Reservation::settle`] with the real usage when the request
/// finishes; dropping it releases the hold without recording anything.
#[derive(Debug)]
#[must_use = "a reservation holds tokens until it is settled or dropped"]
pub struct Reservation<'a> {
    budget: &'a Budget,
    user: String,
    tokens: u64,
    open: bool,
}

impl Reservation<'_> {
    /// Releases the hold and records the real usage and cost.
    pub fn settle(mut self, usage: Usage, input_per_million: f64, output_per_million: f64) {
        self.release();
        self.budget
            .record(&self.user, usage, input_per_million, output_per_million);
        self.open = false;
    }

    fn release(&mut self) {
        if !self.open {
            return;
        }
        let mut spent = self.budget.spent.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = spent.get_mut(&self.user) {
            entry.reserved = entry.reserved.saturating_sub(self.tokens);
        }
    }
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lim() -> Limits {
        Limits {
            max_tokens: 1000,
            max_cost: 5.0,
        }
    }

    fn tokens(n: u32) -> Usage {
        Usage {
            input_tokens: n,
            ..Default::default()
        }
    }

    #[test]
    fn refuses_when_tokens_would_exceed() {
        let b = Budget::new(lim());
        assert!(b.check("u1", 900).is_ok());
        b.record("u1", tokens(800), 0.0, 0.0);
        assert!(b.check("u1", 300).is_err());
        assert!(b.check("u2", 300).is_ok(), "limits are per user");
    }

    #[test]
    fn refuses_when_cost_reached() {
        let b = Budget::new(lim());
        b.record("u1", tokens(1_000_000), 6.0, 0.0);
        assert!(b.check("u1", 1).is_err());
    }

    #[test]
    fn refusal_is_its_own_error() {
        let b = Budget::new(lim());
        b.record("u1", tokens(1000), 0.0, 0.0);
        assert!(matches!(b.check("u1", 1), Err(LlmError::BudgetExceeded(_))));
    }

    #[test]
    fn concurrent_reservations_cannot_overspend() {
        let b = Budget::new(lim());
        let first = b.reserve("u1", 900).expect("first fits");
        assert!(
            b.reserve("u1", 900).is_err(),
            "second must see the first hold"
        );
        drop(first);
        assert!(
            b.reserve("u1", 900).is_ok(),
            "a dropped reservation releases its hold"
        );
    }

    #[test]
    fn settle_records_real_usage_and_releases_hold() {
        let b = Budget::new(lim());
        let r = b.reserve("u1", 900).expect("fits");
        r.settle(tokens(100), 0.0, 0.0);
        assert!(
            b.reserve("u1", 800).is_ok(),
            "only 100 real tokens are counted"
        );
        assert!(b.reserve("u1", 901).is_err());
    }

    #[test]
    fn unusable_price_blocks_instead_of_disabling_cost_limit() {
        let b = Budget::new(lim());
        b.record("u1", tokens(10), f64::NAN, 0.0);
        assert!(b.check("u1", 1).is_err());
        let c = Budget::new(lim());
        c.record("u1", tokens(10), -1.0, 0.0);
        assert!(c.check("u1", 1).is_err());
    }

    #[test]
    fn cache_tokens_count_against_the_token_limit() {
        let b = Budget::new(lim());
        b.record(
            "u1",
            Usage {
                input_tokens: 10,
                cache_read_tokens: 990,
                ..Default::default()
            },
            0.0,
            0.0,
        );
        assert!(b.check("u1", 1).is_err());
    }

    #[test]
    fn old_days_are_evicted() {
        let b = Budget::new(lim());
        {
            let mut spent = b.spent.lock().unwrap();
            spent.insert(
                "gone".into(),
                Spent {
                    day: 1,
                    tokens: 5,
                    cost: 0.0,
                    reserved: 0,
                },
            );
        }
        b.record("u1", tokens(1), 0.0, 0.0);
        assert!(!b.spent.lock().unwrap().contains_key("gone"));
    }

    #[test]
    fn unlimited_only_when_asked_for() {
        let b = Budget::unlimited();
        b.record("u1", tokens(u32::MAX), 1e6, 0.0);
        assert!(b.check("u1", u64::MAX / 2).is_ok());
        assert!(b.reserve("u1", u64::MAX / 2).is_ok());
    }
}
