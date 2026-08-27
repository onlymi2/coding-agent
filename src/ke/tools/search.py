import os
import re
from pathlib import Path

from ke.safety.sandbox import SandboxError, WorkspaceSandbox
from ke.tools.types import ToolResult


def _candidate_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    files: list[Path] = []
    for root, directories, names in os.walk(target, followlinks=False):
        directories[:] = sorted(directories, key=str.casefold)
        for name in sorted(names, key=str.casefold):
            files.append(Path(root) / name)
    return files


def grep(
    sandbox: WorkspaceSandbox,
    pattern: str,
    path: str = ".",
) -> ToolResult:
    """Search UTF-8 text files with a regular expression."""

    try:
        expression = re.compile(pattern)
        target = sandbox.resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"路径不存在：{path}")

        matches: list[str] = []
        for candidate in _candidate_files(target):
            try:
                safe_candidate = sandbox.resolve_for_read(candidate)
            except SandboxError:
                continue
            if not safe_candidate.is_file():
                continue
            try:
                content = safe_candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            display = sandbox.display_path(safe_candidate)
            for line_number, line in enumerate(content.splitlines(), start=1):
                if expression.search(line):
                    matches.append(f"{display}:{line_number}:{line}")

        return ToolResult.success("\n".join(matches) if matches else "未找到匹配")
    except (OSError, ValueError, re.error, SandboxError) as exc:
        return ToolResult.error(f"{type(exc).__name__}: {exc}")
