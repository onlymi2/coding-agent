import os
import time
from pathlib import Path

from ke.safety.sandbox import WorkspaceSandbox
from ke.tools import bash


def test_bash_runs_python_command_successfully(tmp_path: Path) -> None:
    result = bash(WorkspaceSandbox(tmp_path), 'python -c "print(123)"')

    assert not result.is_error
    assert "exit_code: 0" in result.content
    assert "123" in result.content


def test_bash_uses_workspace_as_cwd(tmp_path: Path) -> None:
    result = bash(
        WorkspaceSandbox(tmp_path),
        'python -c "import os; print(os.getcwd())"',
    )

    assert not result.is_error
    expected_cwd = os.path.normcase(str(tmp_path.resolve()))
    assert expected_cwd in os.path.normcase(result.content)


def test_bash_nonzero_exit_is_error_and_preserves_code(tmp_path: Path) -> None:
    result = bash(
        WorkspaceSandbox(tmp_path),
        'python -c "import sys; print(\'failed\'); sys.exit(7)"',
    )

    assert result.is_error
    assert "exit_code: 7" in result.content
    assert "failed" in result.content


def test_bash_timeout_returns_error_without_crashing(tmp_path: Path) -> None:
    started_at = time.monotonic()
    result = bash(
        WorkspaceSandbox(tmp_path),
        'python -c "import time; print(\'started\', flush=True); time.sleep(5)"',
        timeout=0.2,
    )
    elapsed = time.monotonic() - started_at

    assert result.is_error
    assert elapsed < 2
    assert "超时" in result.content
    assert "exit_code: timeout" in result.content


def test_bash_captures_stdout_and_stderr(tmp_path: Path) -> None:
    result = bash(
        WorkspaceSandbox(tmp_path),
        (
            'python -c "import sys; print(\'from stdout\'); '
            "print('from stderr', file=sys.stderr)\""
        ),
    )

    assert not result.is_error
    assert "stdout:\nfrom stdout" in result.content
    assert "stderr:\nfrom stderr" in result.content


def test_bash_replaces_undecodable_output(tmp_path: Path) -> None:
    result = bash(
        WorkspaceSandbox(tmp_path),
        'python -c "import sys; sys.stdout.buffer.write(bytes([255]))"',
    )

    assert not result.is_error
    assert "�" in result.content


def test_bash_filters_agent_api_keys_but_preserves_normal_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KE_API_KEY", "ke-super-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-super-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-super-secret")
    monkeypatch.setenv("KE_TEST_VISIBLE", "hello")

    result = bash(
        WorkspaceSandbox(tmp_path),
        (
            'python -c "import os; '
            "print(os.getenv('KE_API_KEY')); "
            "print(os.getenv('OPENAI_API_KEY')); "
            "print(os.getenv('DEEPSEEK_API_KEY')); "
            "print(os.getenv('KE_TEST_VISIBLE'))\""
        ),
    )

    assert not result.is_error
    assert "ke-super-secret" not in result.content
    assert "openai-super-secret" not in result.content
    assert "deepseek-super-secret" not in result.content
    assert "hello" in result.content


def test_bash_truncates_long_output_and_keeps_head_and_tail(tmp_path: Path) -> None:
    result = bash(
        WorkspaceSandbox(tmp_path),
        'python -c "print(\'A\' * 1000); print(\'Z\' * 1000)"',
        max_output_chars=200,
    )

    assert not result.is_error
    assert len(result.content) == 200
    assert "A" * 20 in result.content
    assert "Z" * 20 in result.content
    assert "[truncated]" in result.content
