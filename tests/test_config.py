from pathlib import Path

import pytest

from ke.config import ConfigError, load_config


def write_yaml(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_yaml_loads_selected_channel(tmp_path: Path) -> None:
    yaml_path = write_yaml(
        tmp_path / "ke.yaml",
        """channel: openai
channels:
  deepseek:
    base_url: https://deepseek.example/v1
    model: deepseek-test
  openai:
    base_url: https://openai.example/v1
    model: openai-test
""",
    )

    config = load_config(
        yaml_path,
        environ={"OPENAI_API_KEY": "test-key"},
    )

    assert config.channel == "openai"
    assert config.base_url == "https://openai.example/v1"
    assert config.model == "openai-test"
    assert config.api_key == "test-key"


def test_environment_overrides_yaml(tmp_path: Path) -> None:
    yaml_path = write_yaml(
        tmp_path / "ke.yaml",
        """channel: deepseek
channels:
  deepseek:
    base_url: https://yaml.example/v1
    model: yaml-model
  openai:
    base_url: https://unused.example/v1
    model: unused-model
""",
    )

    config = load_config(
        yaml_path,
        environ={
            "KE_API_KEY": "test-key",
            "KE_CHANNEL": "openai",
            "KE_BASE_URL": "https://env.example/v1",
            "KE_MODEL": "env-model",
            "KE_MAX_TURNS": "12",
        },
    )

    assert config.channel == "openai"
    assert config.base_url == "https://env.example/v1"
    assert config.model == "env-model"
    assert config.max_turns == 12


def test_explicit_overrides_environment(tmp_path: Path) -> None:
    config = load_config(
        None,
        environ={
            "KE_API_KEY": "test-key",
            "KE_CHANNEL": "environment",
            "KE_BASE_URL": "https://env.example/v1",
            "KE_MODEL": "env-model",
            "KE_MAX_TURNS": "10",
        },
        overrides={
            "channel": "explicit",
            "base_url": "https://explicit.example/v1",
            "model": "explicit-model",
            "workspace": tmp_path,
            "max_turns": 5,
        },
    )

    assert config.channel == "explicit"
    assert config.base_url == "https://explicit.example/v1"
    assert config.model == "explicit-model"
    assert config.workspace == tmp_path.resolve()
    assert config.max_turns == 5


@pytest.mark.parametrize("field_name", ["api_key", "api-key"])
def test_yaml_rejects_api_key_fields(
    tmp_path: Path,
    field_name: str,
) -> None:
    yaml_path = write_yaml(
        tmp_path / "ke.yaml",
        f"channel: deepseek\n{field_name}: forbidden-value\n",
    )

    with pytest.raises(ConfigError, match="禁止包含") as captured:
        load_config(yaml_path, environ={"KE_API_KEY": "test-key"})

    assert "forbidden-value" not in str(captured.value)


@pytest.mark.parametrize(
    "field_name",
    ["secret", "token", "access_token", "client_secret", "refresh_token"],
)
def test_yaml_rejects_nested_secret_fields(
    tmp_path: Path,
    field_name: str,
) -> None:
    yaml_path = write_yaml(
        tmp_path / "ke.yaml",
        f"channels:\n  deepseek:\n    nested:\n      {field_name}: forbidden-value\n",
    )

    with pytest.raises(ConfigError, match="禁止包含") as captured:
        load_config(yaml_path, environ={"KE_API_KEY": "test-key"})

    assert "forbidden-value" not in str(captured.value)


def test_missing_api_key_has_clear_error() -> None:
    with pytest.raises(ConfigError, match="缺少 API Key"):
        load_config(None, environ={})


def test_ke_api_key_has_priority_over_channel_key() -> None:
    config = load_config(
        None,
        environ={
            "KE_API_KEY": "preferred-key",
            "DEEPSEEK_API_KEY": "channel-key",
            "OPENAI_API_KEY": "fallback-key",
        },
    )

    assert config.api_key == "preferred-key"


def test_deepseek_channel_does_not_fall_back_to_openai_key() -> None:
    with pytest.raises(ConfigError, match="缺少 API Key"):
        load_config(
            None,
            environ={
                "KE_CHANNEL": "deepseek",
                "OPENAI_API_KEY": "wrong-channel-key",
            },
        )


def test_process_environment_uses_fake_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "KE_API_KEY",
        "KE_BASE_URL",
        "KE_MODEL",
        "KE_CHANNEL",
        "KE_WORKSPACE",
        "KE_MAX_TURNS",
        "KE_MAX_TOOL_OUTPUT_CHARS",
        "KE_COMPACT_THRESHOLD_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KE_API_KEY", "test-key")
    monkeypatch.setenv("KE_MODEL", "test-model")

    config = load_config(None)

    assert config.api_key == "test-key"
    assert config.model == "test-model"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("KE_MAX_TURNS", "abc"),
        ("KE_MAX_TOOL_OUTPUT_CHARS", "0"),
        ("KE_COMPACT_THRESHOLD_TOKENS", "-1"),
    ],
)
def test_invalid_numeric_environment_value_is_rejected(
    name: str,
    value: str,
) -> None:
    with pytest.raises(ConfigError, match=name):
        load_config(
            None,
            environ={"KE_API_KEY": "test-key", name: value},
        )


def test_api_key_is_not_exposed_by_repr_or_errors(tmp_path: Path) -> None:
    key = "test-key-must-stay-private"
    config = load_config(None, environ={"KE_API_KEY": key})

    assert key not in repr(config)

    yaml_path = write_yaml(
        tmp_path / "ke.yaml",
        f"channels:\n  deepseek:\n    secret: {key}\n",
    )
    with pytest.raises(ConfigError) as captured:
        load_config(yaml_path, environ={"KE_API_KEY": key})
    assert key not in str(captured.value)


def test_api_key_cannot_be_supplied_as_explicit_override() -> None:
    with pytest.raises(ConfigError, match="不支持的显式配置项") as captured:
        load_config(
            None,
            environ={"KE_API_KEY": "environment-key"},
            overrides={"api_key": "override-key"},
        )

    assert "override-key" not in str(captured.value)


def test_yaml_rejects_camel_case_secret_fields_in_nested_structures(
    tmp_path: Path,
) -> None:
    access_token_yaml = write_yaml(
        tmp_path / "access-token.yaml",
        "channels:\n  deepseek:\n    auth:\n      accessToken: forbidden-value\n",
    )
    list_secret_yaml = write_yaml(
        tmp_path / "list-secret.yaml",
        "metadata:\n  entries:\n    - name: safe\n    - clientSecret: forbidden-value\n",
    )

    for yaml_path in (access_token_yaml, list_secret_yaml):
        with pytest.raises(ConfigError, match="禁止包含") as captured:
            load_config(yaml_path, environ={"KE_API_KEY": "test-key"})
        assert "forbidden-value" not in str(captured.value)
