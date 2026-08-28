import json
from dataclasses import dataclass, field, replace
from typing import Iterable

from ke.llm.types import Message
from ke.safety.output import truncate_output


DEFAULT_COMPACT_THRESHOLD_TOKENS = 24_000
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 8_000
DEFAULT_PRESERVED_TOOL_RESULTS = 2
MESSAGE_OVERHEAD_TOKENS = 8


def estimate_text_tokens(text: str | None) -> int:
    """Estimate tokens without depending on a provider-specific tokenizer."""

    if not text:
        return 0
    return (len(text) + 3) // 4


def _stable_arguments(arguments: object) -> str:
    return json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )


def estimate_message_tokens(message: Message) -> int:
    """Estimate one message, including tool names and serialized arguments."""

    estimated = MESSAGE_OVERHEAD_TOKENS
    estimated += estimate_text_tokens(message.content)
    estimated += estimate_text_tokens(message.name)
    for call in message.tool_calls:
        estimated += estimate_text_tokens(call.name)
        estimated += estimate_text_tokens(_stable_arguments(call.arguments))
        estimated += estimate_text_tokens(call.arguments_error)
    return estimated


def estimate_messages_tokens(messages: Iterable[Message]) -> int:
    """Estimate the total tokens for an iterable of internal messages."""

    return sum(estimate_message_tokens(message) for message in messages)


@dataclass
class AgentContext:
    """Own conversation messages and deterministic context compaction."""

    messages: list[Message] = field(default_factory=list)
    compact_threshold_tokens: int = DEFAULT_COMPACT_THRESHOLD_TOKENS
    max_tool_output_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS
    preserve_recent_tool_results: int = DEFAULT_PRESERVED_TOOL_RESULTS

    def __post_init__(self) -> None:
        if self.compact_threshold_tokens < 1:
            raise ValueError("compact_threshold_tokens 必须大于等于 1")
        if self.preserve_recent_tool_results < 0:
            raise ValueError("preserve_recent_tool_results 不能小于 0")
        truncate_output("", self.max_tool_output_chars)

        initial_messages = list(self.messages)
        self.messages = []
        for message in initial_messages:
            self.append(message)

    def append(self, message: Message) -> None:
        """Append one message while enforcing system order and tool limits."""

        if message.role == "system" and any(
            existing.role != "system" for existing in self.messages
        ):
            raise ValueError("system 消息必须位于上下文最前面")

        if message.role == "tool" and message.content is not None:
            message = replace(
                message,
                content=truncate_output(
                    message.content,
                    self.max_tool_output_chars,
                ),
            )
        self.messages.append(message)

    def estimate_tokens(self) -> int:
        return estimate_messages_tokens(self.messages)

    def should_compact(self) -> bool:
        return self.estimate_tokens() >= self.compact_threshold_tokens

    def compact(self) -> int:
        """Fold old tool observations while preserving recent tool results."""

        if not self.should_compact():
            return 0

        tool_indexes = [
            index
            for index, message in enumerate(self.messages)
            if message.role == "tool"
        ]
        if self.preserve_recent_tool_results:
            eligible_indexes = tool_indexes[: -self.preserve_recent_tool_results]
        else:
            eligible_indexes = tool_indexes

        collapsed_count = 0
        for index in eligible_indexes:
            message = self.messages[index]
            if message.collapsed:
                continue
            tool_name = message.name or "unknown"
            self.messages[index] = replace(
                message,
                content=f"[旧工具结果已折叠：{tool_name}]",
                collapsed=True,
            )
            collapsed_count += 1
            if not self.should_compact():
                break

        return collapsed_count
