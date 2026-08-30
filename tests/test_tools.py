from pathlib import Path

import pytest

from ke.safety.sandbox import WorkspaceSandbox
from ke.tools import edit_file, grep, list_dir, read_file, write_file


def test_write_then_read_file_and_line_range(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path)

    written = write_file(sandbox, "pkg/hello.py", "one\ntwo\nthree\n")
    complete = read_file(sandbox, "pkg/hello.py")
    partial = read_file(sandbox, "pkg/hello.py", start_line=2, end_line=3)

    assert not written.is_error
    assert complete.content == "one\ntwo\nthree\n"
    assert partial.content == "two\nthree\n"


def test_file_tools_reject_path_escape(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path)
    outside = tmp_path.parent / "outside-from-tool.txt"

    result = write_file(sandbox, "../outside-from-tool.txt", "blocked")

    assert result.is_error
    assert not outside.exists()


def test_read_file_rejects_env_but_allows_example(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path)
    (tmp_path / ".env").write_text("SECRET=value", encoding="utf-8")
    (tmp_path / ".env.example").write_text("SECRET=", encoding="utf-8")

    denied = read_file(sandbox, ".env")
    allowed = read_file(sandbox, ".env.example")

    assert denied.is_error
    assert "禁止读取" in denied.content
    assert allowed.content == "SECRET="


def test_edit_file_replaces_exactly_one_match(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path)
    target = tmp_path / "app.py"
    target.write_text("answer = 41\n", encoding="utf-8")

    result = edit_file(sandbox, "app.py", "41", "42")

    assert not result.is_error
    assert target.read_text(encoding="utf-8") == "answer = 42\n"


def test_edit_file_zero_matches_does_not_modify_file(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path)
    target = tmp_path / "app.py"
    original = "answer = 41\n"
    target.write_text(original, encoding="utf-8")

    result = edit_file(sandbox, "app.py", "missing", "replacement")

    assert result.is_error
    assert "未找到" in result.content
    assert target.read_text(encoding="utf-8") == original


def test_edit_file_multiple_matches_does_not_modify_file(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path)
    target = tmp_path / "app.py"
    original = "value\nvalue\n"
    target.write_text(original, encoding="utf-8")

    result = edit_file(sandbox, "app.py", "value", "changed")

    assert result.is_error
    assert "匹配 2 次" in result.content
    assert target.read_text(encoding="utf-8") == original


def test_list_dir_respects_depth_and_lists_nested_entries(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path)
    (tmp_path / "pkg/deep").mkdir(parents=True)
    (tmp_path / "pkg/top.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg/deep/hidden.py").write_text("", encoding="utf-8")

    depth_one = list_dir(sandbox, max_depth=1)
    depth_two = list_dir(sandbox, max_depth=2)

    assert "pkg/" in depth_one.content
    assert "pkg/top.py" not in depth_one.content
    assert "pkg/top.py" in depth_two.content
    assert "pkg/deep/" in depth_two.content
    assert "pkg/deep/hidden.py" not in depth_two.content


@pytest.mark.parametrize("error_type", [PermissionError, OSError])
def test_list_dir_skips_unreadable_child_and_keeps_other_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[OSError],
) -> None:
    sandbox = WorkspaceSandbox(tmp_path)
    readable = tmp_path / "readable"
    blocked = tmp_path / "blocked"
    readable.mkdir()
    blocked.mkdir()
    (readable / "visible.txt").write_text("ok", encoding="utf-8")
    (blocked / "hidden.txt").write_text("hidden", encoding="utf-8")
    original_iterdir = Path.iterdir

    def guarded_iterdir(directory: Path):
        if directory == blocked:
            raise error_type("denied")
        return original_iterdir(directory)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)

    result = list_dir(sandbox, max_depth=3)

    assert not result.is_error
    assert "blocked/" in result.content
    assert "blocked/hidden.txt" not in result.content
    assert "readable/" in result.content
    assert "readable/visible.txt" in result.content


def test_list_dir_root_read_error_still_returns_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = WorkspaceSandbox(tmp_path)
    original_iterdir = Path.iterdir

    def guarded_iterdir(directory: Path):
        if directory == tmp_path:
            raise PermissionError("root denied")
        return original_iterdir(directory)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)

    result = list_dir(sandbox)

    assert result.is_error
    assert "PermissionError" in result.content


def test_grep_finds_lines_and_skips_sensitive_env(tmp_path: Path) -> None:
    sandbox = WorkspaceSandbox(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text("alpha\nneedle here\n", encoding="utf-8")
    (tmp_path / "src/b.py").write_text("needle again\n", encoding="utf-8")
    (tmp_path / ".env").write_text("needle=secret\n", encoding="utf-8")

    result = grep(sandbox, "needle")

    assert not result.is_error
    assert "src/a.py:2:needle here" in result.content
    assert "src/b.py:1:needle again" in result.content
    assert ".env" not in result.content


def test_grep_rejects_invalid_regex(tmp_path: Path) -> None:
    result = grep(WorkspaceSandbox(tmp_path), "[")

    assert result.is_error
