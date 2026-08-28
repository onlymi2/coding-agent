from pathlib import Path

import pytest

import ke.cli as cli_module
from ke.cli import build_parser, main
from ke.config import ConfigError, KeConfig


def test_help_is_available(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "usage: ke" in output
    assert "--version" in output
    assert "serve" in output
    assert "run" in output


def test_cli_without_arguments_prints_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 0
    assert "usage: ke" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["serve", "run"])
def test_subcommand_help_is_available(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([command, "--help"])

    assert exc_info.value.code == 0
    assert f"usage: ke {command}" in capsys.readouterr().out


def test_cli_has_no_api_key_option() -> None:
    assert "--api-key" not in build_parser().format_help()
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--api-key", "forbidden", "task"])
    assert exc_info.value.code == 2


def test_run_requires_task() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["run"])
    assert exc_info.value.code == 2


def fake_config(tmp_path: Path) -> KeConfig:
    return KeConfig(
        channel="test",
        base_url="https://unused.example/v1",
        model="fake-model",
        api_key="test-key",
        workspace=tmp_path,
    )


def test_cli_explicit_runtime_options_enter_config_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_load_config(*, overrides: dict[str, object]) -> KeConfig:
        captured.update(overrides)
        return fake_config(tmp_path)

    monkeypatch.setattr(cli_module, "load_config", fake_load_config)
    monkeypatch.setattr(cli_module, "run_task", lambda *args, **kwargs: 0)

    result = main(
        [
            "run",
            "--workspace",
            str(tmp_path),
            "--channel",
            "cli-channel",
            "--base-url",
            "https://cli.example/v1",
            "--model",
            "cli-model",
            "task",
        ]
    )

    assert result == 0
    assert captured == {
        "workspace": str(tmp_path),
        "channel": "cli-channel",
        "base_url": "https://cli.example/v1",
        "model": "cli-model",
    }


def test_cli_omits_unspecified_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    def fake_load_config(*, overrides: dict[str, object]) -> KeConfig:
        captured.append(overrides)
        return fake_config(tmp_path)

    monkeypatch.setattr(cli_module, "load_config", fake_load_config)
    monkeypatch.setattr(cli_module, "run_task", lambda *args, **kwargs: 0)

    assert main(["run", "task"]) == 0
    assert captured == [{}]


def test_run_yes_reaches_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda **kwargs: fake_config(tmp_path),
    )

    def fake_run(config: KeConfig, task: str, *, auto_approve: bool) -> int:
        captured.update(task=task, auto_approve=auto_approve)
        return 0

    monkeypatch.setattr(cli_module, "run_task", fake_run)

    assert main(["run", "--yes", "task"]) == 0
    assert captured == {"task": "task", "auto_approve": True}


@pytest.mark.parametrize("yes", [False, True])
def test_serve_passes_configured_host_port_and_auto_approve(
    yes: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    config = fake_config(tmp_path)
    monkeypatch.setattr(cli_module, "load_config", lambda **kwargs: config)

    def fake_serve(value: KeConfig, *, auto_approve: bool) -> None:
        captured.update(config=value, auto_approve=auto_approve)

    monkeypatch.setattr(cli_module, "serve_runtime", fake_serve)
    arguments = ["serve", "--host", "127.0.0.2", "--port", "9000"]
    if yes:
        arguments.append("--yes")

    assert main(arguments) == 0
    assert captured == {"config": config, "auto_approve": yes}


def test_serve_host_and_port_enter_config_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_load_config(*, overrides: dict[str, object]) -> KeConfig:
        captured.update(overrides)
        return fake_config(tmp_path)

    monkeypatch.setattr(cli_module, "load_config", fake_load_config)
    monkeypatch.setattr(cli_module, "serve_runtime", lambda *args, **kwargs: None)

    assert main(
        ["serve", "--host", "127.0.0.2", "--port", "9000"]
    ) == 0
    assert captured["host"] == "127.0.0.2"
    assert captured["port"] == "9000"


def test_config_error_is_friendly_and_does_not_print_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    key = "test-key-must-not-print"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KE_API_KEY", key)
    monkeypatch.setenv("KE_PORT", "invalid")

    assert main(["serve"]) == 2
    captured = capsys.readouterr()
    assert "配置错误" in captured.err
    assert key not in captured.out + captured.err


def test_cli_does_not_expose_unexpected_runtime_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    key = "test-key-hidden-in-error"
    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda **kwargs: fake_config(tmp_path),
    )
    monkeypatch.setattr(
        cli_module,
        "serve_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(key)),
    )

    assert main(["serve"]) == 1
    captured = capsys.readouterr()
    assert key not in captured.out + captured.err
