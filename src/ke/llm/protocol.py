from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from ke.llm.types import LLMResponse, Message


ToolSchema = dict[str, Any]


@runtime_checkable
class LlmClient(Protocol):
    """Interface required by future agent code, independent of any SDK."""

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
    ) -> LLMResponse: ...
