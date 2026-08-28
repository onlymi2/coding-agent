import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ke.llm.types import ToolCall
from ke.safety.output import truncate_output
from ke.safety.sandbox import WorkspaceSandbox
from ke.tools.bash import bash
from ke.tools.fs import edit_file, list_dir, read_file, write_file
from ke.tools.search import grep
from ke.tools.types import ToolResult


ToolHandler = Callable[..., ToolResult]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": copy.deepcopy(self.parameters),
            },
        }


def _object_schema(
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _default_definitions() -> list[ToolDefinition]:
    path = {"type": "string", "minLength": 1, "description": "workspace 内相对路径"}
    positive_line = {"type": "integer", "minimum": 1}
    return [
        ToolDefinition(
            name="read_file",
            description=(
                "读取 workspace 内的 UTF-8 文本文件，可指定从 1 开始且包含端点的行范围。"
                "禁止读取 .env 等敏感环境文件。"
            ),
            parameters=_object_schema(
                {
                    "path": path,
                    "start_line": {
                        **positive_line,
                        "description": "可选的起始行号，从 1 开始",
                    },
                    "end_line": {
                        **positive_line,
                        "description": "可选的结束行号，包含该行",
                    },
                },
                ["path"],
            ),
            handler=read_file,
        ),
        ToolDefinition(
            name="write_file",
            description="完整写入 workspace 内的 UTF-8 文件，必要时自动创建父目录。",
            parameters=_object_schema(
                {
                    "path": path,
                    "content": {
                        "type": "string",
                        "description": "文件的完整新内容",
                    },
                },
                ["path", "content"],
            ),
            handler=write_file,
        ),
        ToolDefinition(
            name="edit_file",
            description=(
                "在 workspace 文件中把 old_text 精确替换为 new_text。"
                "old_text 必须恰好匹配一次；修改前应先 read_file。"
            ),
            parameters=_object_schema(
                {
                    "path": path,
                    "old_text": {
                        "type": "string",
                        "minLength": 1,
                        "description": "文件中必须唯一存在的原文本",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "替换后的新文本",
                    },
                },
                ["path", "old_text", "new_text"],
            ),
            handler=edit_file,
        ),
        ToolDefinition(
            name="list_dir",
            description="列出 workspace 内的目录内容，默认递归两层且不跟随符号链接。",
            parameters=_object_schema(
                {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": "目录路径，默认当前 workspace",
                    },
                    "max_depth": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "递归层数，默认 2",
                    },
                }
            ),
            handler=list_dir,
        ),
        ToolDefinition(
            name="grep",
            description=(
                "使用正则表达式搜索 workspace 内的 UTF-8 文本，返回路径、行号和匹配行。"
            ),
            parameters=_object_schema(
                {
                    "pattern": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Python 正则表达式",
                    },
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": "搜索路径，默认当前 workspace",
                    },
                },
                ["pattern"],
            ),
            handler=grep,
        ),
        ToolDefinition(
            name="bash",
            description=(
                "在 workspace 作为固定工作目录执行一条本地 shell 命令。"
                "返回退出码、stdout 和 stderr；超时与输出长度由 Runtime 固定控制。"
            ),
            parameters=_object_schema(
                {
                    "command": {
                        "type": "string",
                        "minLength": 1,
                        "description": "要执行的 shell 命令",
                    },
                },
                ["command"],
            ),
            handler=bash,
        ),
    ]


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    return True


def _validate_arguments(
    parameters: dict[str, Any],
    arguments: Any,
) -> str | None:
    if not isinstance(arguments, dict):
        return "工具参数必须是 JSON 对象"

    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    missing = [name for name in required if name not in arguments]
    if missing:
        return f"缺少必需参数：{', '.join(missing)}"

    if parameters.get("additionalProperties") is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            return f"包含未知参数：{', '.join(unknown)}"

    for name, value in arguments.items():
        rule = properties.get(name)
        if not rule:
            continue
        expected = rule.get("type")
        if expected and not _matches_type(value, expected):
            return f"参数 {name} 类型错误，应为 {expected}"
        if "minLength" in rule and len(value) < rule["minLength"]:
            return f"参数 {name} 长度不能小于 {rule['minLength']}"
        if "minimum" in rule and value < rule["minimum"]:
            return f"参数 {name} 不能小于 {rule['minimum']}"
        if "exclusiveMinimum" in rule and value <= rule["exclusiveMinimum"]:
            return f"参数 {name} 必须大于 {rule['exclusiveMinimum']}"
    return None


class ToolRegistry:
    """Own tool schemas, resolve tool names, validate arguments, and execute."""

    def __init__(
        self,
        sandbox: WorkspaceSandbox,
        max_output_chars: int = 8000,
    ) -> None:
        truncate_output("", max_output_chars)
        self.sandbox = sandbox
        self.max_output_chars = max_output_chars
        self._definitions: dict[str, ToolDefinition] = {}
        for definition in _default_definitions():
            self.register(definition)

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"工具已注册：{definition.name}")
        self._definitions[definition.name] = definition

    def schemas(self) -> list[dict[str, Any]]:
        return [definition.schema() for definition in self._definitions.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        if call.arguments_error:
            return ToolResult.error(f"工具参数解析失败：{call.arguments_error}")

        definition = self._definitions.get(call.name)
        if definition is None:
            return ToolResult.error(f"未知工具：{call.name}")

        validation_error = _validate_arguments(
            definition.parameters,
            call.arguments,
        )
        if validation_error:
            return ToolResult.error(f"工具参数校验失败：{validation_error}")

        try:
            result = definition.handler(self.sandbox, **call.arguments)
        except Exception as exc:
            result = ToolResult.error(
                f"工具执行异常：{type(exc).__name__}: {exc}"
            )
        if not isinstance(result, ToolResult):
            result = ToolResult.error("工具执行异常：handler 必须返回 ToolResult")

        return ToolResult(
            content=truncate_output(result.content, self.max_output_chars),
            is_error=result.is_error,
        )
