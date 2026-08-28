import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ke.safety.output import TRUNCATION_MARKER


DEFAULT_CHANNEL = "deepseek"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_TURNS = 30
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 8_000
DEFAULT_COMPACT_THRESHOLD_TOKENS = 24_000

_SENSITIVE_YAML_KEYS = {
    "api_key",
    "apikey",
    "secret",
    "secret_key",
    "api_secret",
    "token",
    "access_token",
    "access_key",
}
_OVERRIDE_KEYS = {
    "channel",
    "base_url",
    "model",
    "workspace",
    "max_turns",
    "max_tool_output_chars",
    "compact_threshold_tokens",
}


class ConfigError(ValueError):
    """Raised when local configuration is missing, invalid, or unsafe."""


@dataclass(frozen=True)
class ChannelConfig:
    base_url: str
    model: str


@dataclass(frozen=True)
class KeConfig:
    channel: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    workspace: Path
    max_turns: int = DEFAULT_MAX_TURNS
    max_tool_output_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS
    compact_threshold_tokens: int = DEFAULT_COMPACT_THRESHOLD_TOKENS
    channels: dict[str, ChannelConfig] = field(default_factory=dict)


def _normalized_key(value: object) -> str:
    raw = str(value).strip()
    raw = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", raw)
    normalized = raw.casefold()
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def _is_sensitive_yaml_key(value: object) -> bool:
    normalized = _normalized_key(value)
    parts = set(normalized.split("_"))
    return normalized in _SENSITIVE_YAML_KEYS or bool(
        {"secret", "token"} & parts
    )


def _reject_yaml_secrets(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _is_sensitive_yaml_key(key):
                raise ConfigError("ke.yaml 禁止包含任何 API Key、secret 或 token 字段")
            _reject_yaml_secrets(child)
    elif isinstance(value, list):
        for child in value:
            _reject_yaml_secrets(child)


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    yaml_path = Path(path)
    if not yaml_path.exists():
        return {}
    try:
        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise ConfigError("ke.yaml 读取或解析失败") from None
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigError("ke.yaml 顶层必须是对象")
    _reject_yaml_secrets(loaded)
    return dict(loaded)


def _parse_channels(raw: object) -> dict[str, ChannelConfig]:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError("ke.yaml 的 channels 必须是对象")

    channels: dict[str, ChannelConfig] = {
        DEFAULT_CHANNEL: ChannelConfig(DEFAULT_BASE_URL, DEFAULT_MODEL)
    }
    for raw_name, raw_config in raw.items():
        name = str(raw_name).strip()
        if not name:
            raise ConfigError("ke.yaml 的渠道名称不能为空")
        if not isinstance(raw_config, Mapping):
            raise ConfigError(f"渠道 {name} 的配置必须是对象")
        base_url = raw_config.get("base_url")
        model = raw_config.get("model")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ConfigError(f"渠道 {name} 缺少有效 base_url")
        if not isinstance(model, str) or not model.strip():
            raise ConfigError(f"渠道 {name} 缺少有效 model")
        channels[name] = ChannelConfig(base_url.strip(), model.strip())
    return channels


def _pick(
    name: str,
    env_name: str,
    overrides: Mapping[str, object],
    environ: Mapping[str, str],
    yaml_value: object,
    default: object,
) -> object:
    override = overrides.get(name)
    if override is not None:
        return override
    env_value = environ.get(env_name)
    if env_value is not None and env_value != "":
        return env_value
    if yaml_value is not None:
        return yaml_value
    return default


def _required_text(name: str, value: object) -> str:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigError(f"{name} 必须是非空字符串")
    return str(value).strip()


def _positive_int(name: str, value: object, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} 必须是正整数")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} 必须是正整数") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ConfigError(f"{name} 必须是正整数")
    if parsed < minimum:
        raise ConfigError(f"{name} 必须是大于等于 {minimum} 的整数")
    return parsed


def _channel_key_name(channel: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", channel.upper()).strip("_")
    return f"{normalized}_API_KEY" if normalized else ""


def _read_api_key(channel: str, environ: Mapping[str, str]) -> str:
    names = ["KE_API_KEY", _channel_key_name(channel)]
    for name in names:
        if name and environ.get(name):
            return environ[name]
    raise ConfigError(
        "缺少 API Key；请通过 KE_API_KEY 或当前渠道专用环境变量提供"
    )


def load_config(
    yaml_path: str | Path | None = "ke.yaml",
    *,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> KeConfig:
    """Load config using overrides, environment, YAML, then built-in defaults."""

    environment = os.environ if environ is None else environ
    explicit = {} if overrides is None else dict(overrides)
    unknown_overrides = sorted(set(explicit) - _OVERRIDE_KEYS)
    if unknown_overrides:
        raise ConfigError(
            f"不支持的显式配置项：{', '.join(unknown_overrides)}"
        )

    yaml_data = _load_yaml(yaml_path)
    channels = _parse_channels(yaml_data.get("channels"))

    channel = _required_text(
        "channel",
        _pick(
            "channel",
            "KE_CHANNEL",
            explicit,
            environment,
            yaml_data.get("channel"),
            DEFAULT_CHANNEL,
        ),
    )
    selected = channels.get(channel)
    yaml_base_url = selected.base_url if selected else yaml_data.get("base_url")
    yaml_model = selected.model if selected else yaml_data.get("model")

    base_url = _required_text(
        "base_url",
        _pick(
            "base_url",
            "KE_BASE_URL",
            explicit,
            environment,
            yaml_base_url,
            DEFAULT_BASE_URL,
        ),
    )
    model = _required_text(
        "model",
        _pick(
            "model",
            "KE_MODEL",
            explicit,
            environment,
            yaml_model,
            DEFAULT_MODEL,
        ),
    )
    workspace = Path(
        _required_text(
            "workspace",
            _pick(
                "workspace",
                "KE_WORKSPACE",
                explicit,
                environment,
                yaml_data.get("workspace"),
                Path.cwd(),
            ),
        )
    ).expanduser().resolve()

    max_turns = _positive_int(
        "KE_MAX_TURNS",
        _pick(
            "max_turns",
            "KE_MAX_TURNS",
            explicit,
            environment,
            yaml_data.get("max_turns"),
            DEFAULT_MAX_TURNS,
        ),
    )
    max_tool_output_chars = _positive_int(
        "KE_MAX_TOOL_OUTPUT_CHARS",
        _pick(
            "max_tool_output_chars",
            "KE_MAX_TOOL_OUTPUT_CHARS",
            explicit,
            environment,
            yaml_data.get("max_tool_output_chars"),
            DEFAULT_MAX_TOOL_OUTPUT_CHARS,
        ),
        minimum=len(TRUNCATION_MARKER) + 2,
    )
    compact_threshold_tokens = _positive_int(
        "KE_COMPACT_THRESHOLD_TOKENS",
        _pick(
            "compact_threshold_tokens",
            "KE_COMPACT_THRESHOLD_TOKENS",
            explicit,
            environment,
            yaml_data.get("compact_threshold_tokens"),
            DEFAULT_COMPACT_THRESHOLD_TOKENS,
        ),
    )

    return KeConfig(
        channel=channel,
        base_url=base_url,
        model=model,
        api_key=_read_api_key(channel, environment),
        workspace=workspace,
        max_turns=max_turns,
        max_tool_output_chars=max_tool_output_chars,
        compact_threshold_tokens=compact_threshold_tokens,
        channels=channels,
    )
