import os
from pathlib import Path


class SandboxError(ValueError):
    """Raised when a path violates workspace safety rules."""


class WorkspaceSandbox:
    """Resolve and validate paths against one fixed workspace directory."""

    def __init__(self, workspace: str | Path) -> None:
        root = Path(workspace).resolve(strict=True)
        if not root.is_dir():
            raise SandboxError(f"workspace 不是目录：{root}")
        self.workspace = root

    def resolve(self, path: str | Path = ".") -> Path:
        raw_path = Path(path)
        candidate = raw_path if raw_path.is_absolute() else self.workspace / raw_path
        target = candidate.resolve(strict=False)

        root_text = os.path.normcase(str(self.workspace))
        target_text = os.path.normcase(str(target))
        try:
            common = os.path.commonpath([root_text, target_text])
        except ValueError as exc:
            raise SandboxError(f"路径不在 workspace 内：{path}") from exc
        if common != root_text:
            raise SandboxError(f"路径不在 workspace 内：{path}")
        return target

    def resolve_for_read(self, path: str | Path) -> Path:
        target = self.resolve(path)
        if self.is_sensitive_env_file(target):
            raise SandboxError(f"禁止读取敏感环境文件：{target.name}")
        return target

    @staticmethod
    def is_sensitive_env_file(path: str | Path) -> bool:
        name = Path(path).name.casefold()
        return name != ".env.example" and (
            name == ".env" or name.startswith(".env.")
        )

    def display_path(self, path: str | Path) -> str:
        return Path(path).relative_to(self.workspace).as_posix() or "."
