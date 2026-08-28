from pathlib import Path

from ke.llm.types import ToolCall
from ke.safety.sandbox import WorkspaceSandbox
from ke.tools.registry import ToolDefinition, ToolRegistry
from ke.tools.types import ToolResult


def test_registry_exposes_six_clear_tool_schemas(tmp_path: Path) -> None:
    schemas = ToolRegistry(WorkspaceSandbox(tmp_path)).schemas()

    assert [schema["function"]["name"] for schema in schemas] == [
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "grep",
        "bash",
    ]
    assert all(schema["type"] == "function" for schema in schemas)
    assert all(schema["function"]["description"] for schema in schemas)
    assert all(
        schema["function"]["parameters"]["type"] == "object"
        for schema in schemas
    )
    bash_schema = next(
        schema for schema in schemas if schema["function"]["name"] == "bash"
    )
    assert set(bash_schema["function"]["parameters"]["properties"]) == {
        "command"
    }


def test_registry_executes_registered_tool(tmp_path: Path) -> None:
    registry = ToolRegistry(WorkspaceSandbox(tmp_path))

    result = registry.execute(
        ToolCall(
            id="call_1",
            name="write_file",
            arguments={"path": "hello.txt", "content": "hello"},
        )
    )

    assert not result.is_error
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello"


def test_registry_returns_error_for_unknown_tool(tmp_path: Path) -> None:
    registry = ToolRegistry(WorkspaceSandbox(tmp_path))

    result = registry.execute(ToolCall("call_1", "unknown", {}))

    assert result.is_error
    assert "未知工具" in result.content


def test_registry_converts_handler_exception_to_error(tmp_path: Path) -> None:
    def explode(sandbox: WorkspaceSandbox) -> ToolResult:
        raise RuntimeError("boom")

    registry = ToolRegistry(WorkspaceSandbox(tmp_path))
    registry.register(
        ToolDefinition(
            name="explode",
            description="test handler",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=explode,
        )
    )

    result = registry.execute(ToolCall("call_1", "explode", {}))

    assert result.is_error
    assert "RuntimeError: boom" in result.content


def test_registry_rejects_invalid_arguments(tmp_path: Path) -> None:
    registry = ToolRegistry(WorkspaceSandbox(tmp_path))

    missing = registry.execute(
        ToolCall("call_1", "write_file", {"path": "hello.txt"})
    )
    wrong_type = registry.execute(
        ToolCall("call_2", "read_file", {"path": 123})
    )
    extra = registry.execute(
        ToolCall("call_3", "list_dir", {"unexpected": True})
    )
    runtime_control = registry.execute(
        ToolCall(
            "call_4",
            "bash",
            {"command": "echo test", "timeout": 99999999},
        )
    )

    assert missing.is_error and "缺少必需参数" in missing.content
    assert wrong_type.is_error and "类型错误" in wrong_type.content
    assert extra.is_error and "未知参数" in extra.content
    assert runtime_control.is_error and "未知参数" in runtime_control.content


def test_registry_does_not_run_handler_when_arguments_failed_to_parse(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(WorkspaceSandbox(tmp_path))
    call = ToolCall(
        id="call_1",
        name="write_file",
        arguments={},
        arguments_error="Expecting property name",
    )

    result = registry.execute(call)

    assert result.is_error
    assert "参数解析失败" in result.content
    assert not (tmp_path / "hello.txt").exists()


def test_registry_truncates_every_handler_result(tmp_path: Path) -> None:
    def verbose(sandbox: WorkspaceSandbox) -> ToolResult:
        return ToolResult.success("A" * 200 + "Z" * 200)

    registry = ToolRegistry(WorkspaceSandbox(tmp_path), max_output_chars=100)
    registry.register(
        ToolDefinition(
            name="verbose",
            description="test truncation",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=verbose,
        )
    )

    result = registry.execute(ToolCall("call_1", "verbose", {}))

    assert len(result.content) == 100
    assert "[truncated]" in result.content
    assert result.content.startswith("A")
    assert result.content.endswith("Z")
