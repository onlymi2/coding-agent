import os
import signal
import subprocess

from ke.safety.output import truncate_output
from ke.safety.sandbox import WorkspaceSandbox
from ke.tools.types import ToolResult


DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_OUTPUT_CHARS = 8000


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _merge_output(partial: str | bytes | None, final: str) -> str:
    partial_text = _as_text(partial)
    if not partial_text or final.startswith(partial_text):
        return final
    return partial_text + final


def _format_result(exit_code: int | str, stdout: str, stderr: str) -> str:
    return (
        f"exit_code: {exit_code}\n"
        f"stdout:\n{stdout or '(empty)'}\n"
        f"stderr:\n{stderr or '(empty)'}"
    )


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    if os.name == "nt":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=0.5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    if process.poll() is None:
        process.kill()


def bash(
    sandbox: WorkspaceSandbox,
    command: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> ToolResult:
    """Run one shell command with a fixed workspace cwd and bounded output."""

    if not command.strip():
        return ToolResult.error("command 不能为空")
    if timeout <= 0:
        return ToolResult.error("timeout 必须大于 0")
    try:
        truncate_output("", max_output_chars)
    except ValueError as exc:
        return ToolResult.error(f"ValueError: {exc}")

    try:
        process = subprocess.Popen(
            command,
            cwd=sandbox.workspace,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=(
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if os.name == "nt"
                else 0
            ),
            start_new_session=os.name != "nt",
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_tree(process)
            final_stdout, final_stderr = process.communicate()
            stdout = _merge_output(exc.stdout, final_stdout)
            stderr = _merge_output(exc.stderr, final_stderr)
            output = (
                f"命令执行超时（timeout={timeout:g} 秒），进程已终止。\n"
                + _format_result("timeout", stdout, stderr)
            )
            output = truncate_output(output, max_output_chars)
            return ToolResult.error(output)

        output = _format_result(
            process.returncode,
            stdout,
            stderr,
        )
        output = truncate_output(output, max_output_chars)
        return ToolResult(
            content=output,
            is_error=process.returncode != 0,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return ToolResult.error(f"{type(exc).__name__}: {exc}")
