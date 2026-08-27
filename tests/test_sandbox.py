from pathlib import Path

import pytest

from ke.safety.sandbox import SandboxError, WorkspaceSandbox


def test_resolve_normal_relative_path(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path)

    assert sandbox.resolve("src/main.py") == (tmp_path / "src/main.py").resolve()


def test_reject_parent_path_escape(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path)

    with pytest.raises(SandboxError, match="不在 workspace 内"):
        sandbox.resolve("../outside.txt")


def test_reject_absolute_path_outside_workspace(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path)
    outside = tmp_path.parent / "outside.txt"

    with pytest.raises(SandboxError, match="不在 workspace 内"):
        sandbox.resolve(outside)


def test_windows_backslash_path_stays_inside_workspace(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path)

    assert sandbox.resolve(r"src\main.py") == (tmp_path / "src/main.py").resolve()


def test_reject_sensitive_env_files_but_allow_example(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path)

    for name in [".env", ".env.local", ".env.production"]:
        with pytest.raises(SandboxError, match="禁止读取"):
            sandbox.resolve_for_read(name)

    assert sandbox.resolve_for_read(".env.example") == (
        tmp_path / ".env.example"
    ).resolve()


def test_workspace_must_exist_and_be_a_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        WorkspaceSandbox(tmp_path / "missing")

    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(SandboxError, match="不是目录"):
        WorkspaceSandbox(file_path)
