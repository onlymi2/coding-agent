from pathlib import Path

from ke.safety.sandbox import SandboxError, WorkspaceSandbox
from ke.tools.types import ToolResult


def _failure(exc: Exception) -> ToolResult:
    return ToolResult.error(f"{type(exc).__name__}: {exc}")


def read_file(
    sandbox: WorkspaceSandbox,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> ToolResult:
    """Read a UTF-8 text file, optionally using inclusive 1-based lines."""

    try:
        target = sandbox.resolve_for_read(path)
        if not target.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")
        if start_line is not None and start_line < 1:
            raise ValueError("start_line 必须大于等于 1")
        if end_line is not None and end_line < 1:
            raise ValueError("end_line 必须大于等于 1")
        if start_line is not None and end_line is not None and start_line > end_line:
            raise ValueError("start_line 不能大于 end_line")

        content = target.read_text(encoding="utf-8")
        if start_line is None and end_line is None:
            return ToolResult.success(content)

        lines = content.splitlines(keepends=True)
        start_index = (start_line or 1) - 1
        end_index = end_line if end_line is not None else len(lines)
        return ToolResult.success("".join(lines[start_index:end_index]))
    except (OSError, UnicodeError, ValueError, SandboxError) as exc:
        return _failure(exc)


def write_file(
    sandbox: WorkspaceSandbox,
    path: str,
    content: str,
) -> ToolResult:
    """Write a complete UTF-8 file, creating parent directories as needed."""

    try:
        target = sandbox.resolve(path)
        if target.exists() and target.is_dir():
            raise IsADirectoryError(f"目标是目录：{path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult.success(
            f"已写入 {sandbox.display_path(target)}（{len(content)} 个字符）"
        )
    except (OSError, UnicodeError, ValueError, SandboxError) as exc:
        return _failure(exc)


def _preview(content: str, limit: int = 400) -> str:
    preview = content[:limit]
    if len(content) > limit:
        preview += "\n... [preview truncated]"
    return preview


def edit_file(
    sandbox: WorkspaceSandbox,
    path: str,
    old_text: str,
    new_text: str,
) -> ToolResult:
    """Replace old_text only when it occurs exactly once in the file."""

    try:
        if not old_text:
            raise ValueError("old_text 不能为空")
        target = sandbox.resolve_for_read(path)
        if not target.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")

        content = target.read_text(encoding="utf-8")
        matches = content.count(old_text)
        if matches == 0:
            return ToolResult.error(
                f"未找到 old_text，文件未修改。\n文件预览：\n{_preview(content)}"
            )
        if matches > 1:
            return ToolResult.error(
                f"old_text 匹配 {matches} 次，文件未修改；请提供更精确的文本。"
                f"\n文件预览：\n{_preview(content)}"
            )

        updated = content.replace(old_text, new_text, 1)
        target.write_text(updated, encoding="utf-8")
        return ToolResult.success(f"已更新 {sandbox.display_path(target)}")
    except (OSError, UnicodeError, ValueError, SandboxError) as exc:
        return _failure(exc)


def list_dir(
    sandbox: WorkspaceSandbox,
    path: str = ".",
    max_depth: int = 2,
) -> ToolResult:
    """List a directory recursively up to max_depth levels."""

    try:
        if max_depth < 1:
            raise ValueError("max_depth 必须大于等于 1")
        target = sandbox.resolve(path)
        if not target.is_dir():
            raise NotADirectoryError(f"目录不存在：{path}")

        entries: list[str] = []

        def visit(directory: Path, depth: int) -> None:
            for item in sorted(directory.iterdir(), key=lambda value: value.name.casefold()):
                display = item.relative_to(sandbox.workspace).as_posix()
                if item.is_symlink():
                    entries.append(display + "@")
                    continue
                if item.is_dir():
                    entries.append(display + "/")
                    if depth < max_depth:
                        visit(item, depth + 1)
                else:
                    entries.append(display)

        visit(target, 1)
        return ToolResult.success("\n".join(entries) if entries else "目录为空")
    except (OSError, ValueError, SandboxError) as exc:
        return _failure(exc)
