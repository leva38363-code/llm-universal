//! Opt-in fallback: tries providers in order and moves on only when the error is worth retrying
//! on another provider (network, timeout, 408/409/429, 5xx). Auth and bad-request errors are
//! returned at once, because the next provider cannot fix them.

use async_trait::async_trait;

use crate::{ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider, ModelInfo, Result};

/// A chain of providers. Nothing changes unless you build one explicitly.
pub struct FallbackProvider {
    providers: Vec<Box<dyn LlmProvider>>,
    server_errors: bool,
}

/// Moving on is safe when the failed provider most likely did not process the request:
/// the connection failed (`Connection`) or the server said "slow down" (429).
/// A 5xx or a timeout may mean the answer was generated and billed, so the next provider
/// would bill it again. Those go to the next provider only when `with_server_errors(true)`.
fn may_fall_back(error: &LlmError, server_errors: bool) -> bool {
    match error {
        LlmError::Connection(_) => true,
        LlmError::Api(e) if e.status == 429 => true,
        LlmError::Api(e) if server_errors && e.status >= 500 => true,
        LlmError::Timeout(_) => server_errors,
        _ => false,
    }
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
            Some(Self {
                providers,
                server_errors: false,
            })
        }
    }

    /// Also move to the next provider after 5xx and timeouts. Enable only when a second bill
    /// for the same request is acceptable.
    pub fn with_server_errors(mut self, enabled: bool) -> Self {
        self.server_errors = enabled;
        self
    }
}

#[async_trait]
impl LlmProvider for FallbackProvider {
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let mut last = None;
        for provider in &self.providers {
            match provider.chat(request).await {
                Ok(response) => return Ok(response),
                Err(e) if may_fall_back(&e, self.server_errors) => last = Some(e),
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
                Err(e) if may_fall_back(&e, self.server_errors) => last = Some(e),
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
                Err(e) if may_fall_back(&e, self.server_errors) => last = Some(e),
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

    /// Ошибка с заданным HTTP-статусом (Fake всегда отдаёт 503).
    struct Status(u16);

    #[async_trait]
    impl LlmProvider for Status {
        async fn chat(&self, _: &ChatRequest) -> Result<ChatResponse> {
            Err(api(self.0))
        }
    }

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
    async fn falls_back_on_5xx_only_when_enabled() {
        let p = FallbackProvider::new(vec![Box::new(Fake(Err(api(503)))), Box::new(Fake(Ok("второй")))])
            .unwrap()
            .with_server_errors(true);
        assert_eq!(p.chat(&req()).await.unwrap().text(), "второй");
    }

    #[tokio::test]
    async fn server_error_stops_chain_by_default() {
        // По умолчанию 5xx не переводит на второго: первый мог уже выставить счёт.
        let p = FallbackProvider::new(vec![Box::new(Fake(Err(api(503)))), Box::new(Fake(Ok("второй")))]).unwrap();
        assert!(matches!(p.chat(&req()).await, Err(LlmError::Api(e)) if e.status == 503));
    }

    #[tokio::test]
    async fn rate_limit_falls_back_by_default() {
        let p = FallbackProvider::new(vec![Box::new(Status(429)), Box::new(Fake(Ok("второй")))]).unwrap();
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
