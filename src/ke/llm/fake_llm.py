from collections import deque
from collections.abc import Iterable, Sequence

from ke.llm.protocol import ToolSchema
from ke.llm.types import LLMResponse, Message


class FakeLLM:
    """A deterministic LLM test double backed by a response queue."""

    def __init__(self, responses: Iterable[LLMResponse]) -> None:
        self._responses = deque(responses)
        self.requests: list[tuple[list[Message], list[ToolSchema]]] = []

    @property
    def remaining(self) -> int:
        return len(self._responses)

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
    ) -> LLMResponse:
        self.requests.append((list(messages), list(tools)))
        if not self._responses:
            raise AssertionError("FakeLLM 响应队列已耗尽")
        return self._responses.popleft()
