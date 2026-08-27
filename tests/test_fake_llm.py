import pytest

from ke.llm.fake_llm import FakeLLM
from ke.llm.protocol import LlmClient
from ke.llm.types import LLMResponse, Message


def response(content: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def test_fake_llm_returns_queued_responses_in_order() -> None:
    fake = FakeLLM([response("first"), response("second")])
    messages = [Message(role="user", content="hello")]

    assert isinstance(fake, LlmClient)
    assert fake.complete(messages, []).message.content == "first"
    assert fake.complete(messages, []).message.content == "second"
    assert fake.remaining == 0
    assert len(fake.requests) == 2


def test_fake_llm_fails_immediately_when_queue_is_empty() -> None:
    fake = FakeLLM([])

    with pytest.raises(AssertionError, match="响应队列已耗尽"):
        fake.complete([], [])
