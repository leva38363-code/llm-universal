//! Opt-in fallback: tries providers in order and moves on only when the error is worth retrying
//! on another provider (network, timeout, 408/409/429, 5xx). Auth and bad-request errors are
//! returned at once, because the next provider cannot fix them.

use async_trait::async_trait;

use crate::{ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider, ModelInfo, Result};

/// A chain of providers. Nothing changes unless you build one explicitly.
pub struct FallbackProvider {
    providers: Vec<Box<dyn LlmProvider>>,
}

impl std::fmt::Debug for FallbackProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FallbackProvider")
            .field("providers", &self.providers.len())
            .finish()
    }
}

impl FallbackProvider {
    /// Returns `None` for an empty list: there is nothing to fall back to.
    pub fn new(providers: Vec<Box<dyn LlmProvider>>) -> Option<Self> {
        if providers.is_empty() {
            None
        } else {
            Some(Self { providers })
        }
    }
}

#[async_trait]
impl LlmProvider for FallbackProvider {
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let mut last = None;
        for provider in &self.providers {
            match provider.chat(request).await {
                Ok(response) => return Ok(response),
                Err(e) if e.is_retryable() => last = Some(e),
                Err(e) => return Err(e),
            }
        }
        Err(last.unwrap_or_else(|| LlmError::InvalidArgument("no providers".into())))
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let mut last = None;
        for provider in &self.providers {
            match provider.chat_stream(request).await {
                Ok(stream) => return Ok(stream),
                Err(e) if e.is_retryable() => last = Some(e),
                Err(e) => return Err(e),
            }
        }
        Err(last.unwrap_or_else(|| LlmError::InvalidArgument("no providers".into())))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let mut last = None;
        for provider in &self.providers {
            match provider.list_models().await {
                Ok(models) => return Ok(models),
                Err(e) if e.is_retryable() => last = Some(e),
                Err(e) => return Err(e),
            }
        }
        Err(last.unwrap_or_else(|| LlmError::InvalidArgument("no providers".into())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ApiError, ApiErrorKind, ChatResponse, Message, StopReason};

    struct Fake(Result<&'static str>);

    fn api(status: u16) -> LlmError {
        LlmError::Api(Box::new(ApiError {
            status,
            kind: ApiErrorKind::Other,
            message: Some("x".into()),
            error_type: None,
            code: None,
            param: None,
            request_id: None,
            body: None,
            retry_after: None,
        }))
    }

    #[async_trait]
    impl LlmProvider for Fake {
        async fn chat(&self, _: &ChatRequest) -> Result<ChatResponse> {
            match &self.0 {
                Ok(text) => Ok(ChatResponse {
                    model: "fake".into(),
                    message: Message::assistant(*text),
                    stop_reason: StopReason::EndTurn,
                    usage: Default::default(),
                }),
                Err(_) => Err(api(503)),
            }
        }
    }

    fn req() -> ChatRequest {
        ChatRequest::new("m").message(Message::user("hi"))
    }

    #[tokio::test]
    async fn empty_list_gives_none() {
        assert!(FallbackProvider::new(vec![]).is_none());
    }

    #[tokio::test]
    async fn falls_back_on_5xx() {
        let p = FallbackProvider::new(vec![Box::new(Fake(Err(api(503)))), Box::new(Fake(Ok("второй")))]).unwrap();
        assert_eq!(p.chat(&req()).await.unwrap().text(), "второй");
    }

    #[tokio::test]
    async fn stops_on_auth_error() {
        struct Auth;
        #[async_trait]
        impl LlmProvider for Auth {
            async fn chat(&self, _: &ChatRequest) -> Result<ChatResponse> {
                Err(LlmError::Api(Box::new(ApiError { status: 401, kind: ApiErrorKind::Authentication, message: Some("no".into()), error_type: None, code: None, param: None, request_id: None, body: None, retry_after: None })))
            }
        }
        let p = FallbackProvider::new(vec![Box::new(Auth), Box::new(Fake(Ok("не должен дойти")))]).unwrap();
        assert!(matches!(p.chat(&req()).await, Err(LlmError::Api(e)) if e.status == 401));
    }
}
