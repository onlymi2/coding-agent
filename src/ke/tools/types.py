from dataclasses import dataclass


@dataclass(frozen=True)
class ToolResult:
    """A tool outcome that can later be returned to the model."""

    content: str
    is_error: bool = False

    @classmethod
    def success(cls, content: str) -> "ToolResult":
        return cls(content=content)

    @classmethod
    def error(cls, content: str) -> "ToolResult":
        return cls(content=content, is_error=True)
