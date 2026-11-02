//! Streaming: a small Server-Sent Events parser and a driver that turns provider events
//! into [`StreamEvent`]s, ending with the full [`ChatResponse`].

use std::collections::VecDeque;
use std::pin::Pin;

use futures_core::Stream;

use crate::http::MAX_RESPONSE_BYTES;
use crate::{ChatResponse, LlmError, Result};

/// One SSE event larger than this (without a blank line) is treated as an attack or a bug.
const MAX_EVENT_BYTES: usize = 4 * 1024 * 1024;

/// What a stream yields. The last item is always [`StreamEvent::Done`] (or an error).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum StreamEvent {
    /// More answer text.
    TextDelta(String),
    /// More reasoning text (when the provider shows it).
    ReasoningDelta(String),
    /// The model started a tool call.
    ToolCallStart { id: String, name: String },
    /// More JSON of a tool call's arguments.
    ToolCallDelta { id: String, arguments: String },
    /// The complete response, the same as a non-streaming call would return.
    Done(ChatResponse),
}

/// A stream of [`StreamEvent`]s. Read it with `StreamExt::next` (re-exported by this crate).
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>;

/// Per-provider state: turns SSE events into [`StreamEvent`]s and builds the final response.
pub(crate) trait Accumulator: Send + 'static {
    fn on_event(&mut self, event: &str, data: &str) -> Result<Vec<StreamEvent>>;
    fn finish(&mut self) -> Result<ChatResponse>;
}

/// Incremental SSE parser (`event:` / `data:` lines, events separated by a blank line).
/// Line endings may be `\n`, `\r\n` or a lone `\r`, as the SSE spec allows.
#[derive(Default)]
pub(crate) struct SseParser {
    buf: Vec<u8>,
    /// Where the search for the next blank line resumes, so bytes are scanned once.
    scanned: usize,
    /// The previous chunk ended with `\r`; a leading `\n` in the next one belongs to it.
    pending_cr: bool,
}

impl SseParser {
    /// Feeds bytes and returns every complete `(event, data)` pair.
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Result<Vec<(String, String)>> {
        for &b in bytes {
            match b {
                b'\r' => {
                    self.buf.push(b'\n');
                    self.pending_cr = true;
                }
                b'\n' if self.pending_cr => self.pending_cr = false,
                _ => {
                    self.buf.push(b);
                    self.pending_cr = false;
                }
            }
        }

        let mut events = Vec::new();
        let mut consumed = 0;
        let mut i = self.scanned.max(consumed + 1);
        while i < self.buf.len() {
            if self.buf[i] == b'\n' && self.buf[i - 1] == b'\n' {
                if let Some(event) = parse_block(&String::from_utf8_lossy(&self.buf[consumed..=i]))
                {
                    events.push(event);
                }
                consumed = i + 1;
                i = consumed + 1;
            } else {
                i += 1;
            }
        }
        // One drain per push, not per event.
        self.buf.drain(..consumed);
        self.scanned = self.buf.len().max(1);

        if self.buf.len() > MAX_EVENT_BYTES {
            return Err(LlmError::InvalidResponse(format!(
                "stream event exceeds {MAX_EVENT_BYTES} bytes"
            )));
        }
        Ok(events)
    }

    /// Whatever is left when the connection closes without a final blank line.
    pub(crate) fn flush(&mut self) -> Option<(String, String)> {
        self.scanned = 0;
        let rest = String::from_utf8_lossy(&std::mem::take(&mut self.buf)).into_owned();
        parse_block(&rest)
    }
}

/// The connection closed before the provider's final event: the answer may be cut off.
pub(crate) fn ended_early() -> LlmError {
    LlmError::InvalidResponse(
        "stream ended before the final event; the response may be incomplete".into(),
    )
}

fn parse_block(block: &str) -> Option<(String, String)> {
    let mut event = String::new();
    let mut data: Vec<&str> = Vec::new();
    for line in block.lines() {
        if let Some(value) = line.strip_prefix("event:") {
            event = value.trim().to_string();
        } else if let Some(value) = line.strip_prefix("data:") {
            data.push(value.strip_prefix(' ').unwrap_or(value));
        }
        // Lines starting with ':' are comments (keep-alives); other fields are ignored.
    }
    (!data.is_empty()).then(|| (event, data.join("\n")))
}

/// A stream may stay open for long answers, but not on keep-alives alone: without a single
/// data event for this long it is cut off (each keep-alive byte resets reqwest's idle timer).
pub(crate) const MAX_SILENCE: std::time::Duration = std::time::Duration::from_secs(300);

struct State {
    last_event: std::time::Instant,
    response: reqwest::Response,
    parser: SseParser,
    acc: Box<dyn Accumulator>,
    queue: VecDeque<Result<StreamEvent>>,
    read: usize,
    finished: bool,
}

impl State {
    fn handle(&mut self, events: Vec<(String, String)>) {
        for (event, data) in events {
            match self.acc.on_event(&event, &data) {
                Ok(out) => self.queue.extend(out.into_iter().map(Ok)),
                Err(e) => {
                    self.queue.push_back(Err(e));
                    self.finished = true;
                    return;
                }
            }
        }
    }

    fn fail(&mut self, error: LlmError) {
        self.queue.push_back(Err(error));
        self.finished = true;
    }
}

/// Turns an SSE response into a [`ChatStream`].
pub(crate) fn drive(response: reqwest::Response, acc: Box<dyn Accumulator>) -> ChatStream {
    let state = State {
        last_event: std::time::Instant::now(),
        response,
        parser: SseParser::default(),
        acc,
        queue: VecDeque::new(),
        read: 0,
        finished: false,
    };
    Box::pin(futures_util::stream::unfold(
        state,
        |mut state| async move {
            loop {
                if let Some(item) = state.queue.pop_front() {
                    return Some((item, state));
                }
                if state.finished {
                    return None;
                }
                match state.response.chunk().await {
                    Ok(Some(bytes)) => {
                        state.read += bytes.len();
                        if state.read > MAX_RESPONSE_BYTES {
                            state.fail(LlmError::InvalidResponse(format!(
                                "stream exceeds {MAX_RESPONSE_BYTES} bytes"
                            )));
                            continue;
                        }
                        match state.parser.push(&bytes) {
                            Ok(events) if events.is_empty() => {
                                if state.last_event.elapsed() > MAX_SILENCE {
                                    state.fail(LlmError::InvalidResponse(format!(
                                        "no stream events for {} s (keep-alives only)",
                                        MAX_SILENCE.as_secs()
                                    )));
                                }
                            }
                            Ok(events) => {
                                state.last_event = std::time::Instant::now();
                                state.handle(events);
                            }
                            Err(e) => state.fail(e),
                        }
                    }
                    Ok(None) => {
                        if let Some(last) = state.parser.flush() {
                            state.handle(vec![last]);
                        }
                        if !state.finished {
                            let done = state.acc.finish().map(StreamEvent::Done);
                            state.queue.push_back(done);
                            state.finished = true;
                        }
                    }
                    Err(e) => state.fail(LlmError::from_reqwest(e)),
                }
            }
        },
    ))
}

/// Parses a JSON SSE payload, or reports it.
pub(crate) fn parse_data(data: &str) -> Result<serde_json::Value> {
    serde_json::from_str(data).map_err(|source| LlmError::Deserialize {
        source,
        content: crate::error::truncate(data),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_events_split_across_chunks() {
        let mut p = SseParser::default();
        assert!(p.push(b"event: ping\ndata: {\"a\"").unwrap().is_empty());
        let events = p
            .push(b":1}\r\n\r\n: keep-alive\n\ndata: [DONE]\n\n")
            .unwrap();
        assert_eq!(
            events,
            vec![
                ("ping".to_string(), "{\"a\":1}".to_string()),
                (String::new(), "[DONE]".to_string())
            ]
        );
    }

    #[test]
    fn multi_line_data_is_joined() {
        let mut p = SseParser::default();
        let events = p.push(b"data: line1\ndata: line2\n\n").unwrap();
        assert_eq!(events[0].1, "line1\nline2");
    }

    #[test]
    fn lone_cr_and_split_crlf_line_endings() {
        let mut p = SseParser::default();
        assert!(p.push(b"data: a\r").unwrap().is_empty());
        let events = p.push(b"\n\r\ndata: b\r\r").unwrap();
        assert_eq!(
            events,
            vec![
                (String::new(), "a".to_string()),
                (String::new(), "b".to_string())
            ]
        );
    }

    #[test]
    fn many_events_in_one_chunk() {
        let mut p = SseParser::default();
        let chunk: Vec<u8> = (0..1000)
            .flat_map(|i| format!("data: {i}\n\n").into_bytes())
            .collect();
        let events = p.push(&chunk).unwrap();
        assert_eq!(events.len(), 1000);
        assert_eq!(events[999].1, "999");
    }

    #[test]
    fn oversized_event_is_rejected() {
        let mut p = SseParser::default();
        let big = vec![b'x'; MAX_EVENT_BYTES + 1];
        assert!(p.push(&big).is_err());
    }
}
