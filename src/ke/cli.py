import argparse
import sys
from collections.abc import Sequence
from typing import Any

from ke import __version__
from ke.client.run import run_task
from ke.config import ConfigError, KeConfig, load_config
from ke.server.runtime import serve as serve_runtime


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default=None, help="Agent workspace 路径")
    parser.add_argument("--channel", default=None, help="模型渠道名称")
    parser.add_argument("--base-url", default=None, help="OpenAI 兼容 API 地址")
    parser.add_argument("--model", default=None, help="模型名称")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ke",
        description="ke：一个自研的本地编程智能体缰绳。",
        epilog="当前提供 ke serve 和无 TUI 的 ke run。",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser(
        "serve",
        help="启动本地 HTTP/SSE 服务",
        description="启动已有的六接口 Starlette 服务。",
    )
    _add_runtime_options(serve_parser)
    serve_parser.add_argument("--host", default=None, help="监听地址")
    serve_parser.add_argument("--port", default=None, help="监听端口")
    serve_parser.add_argument(
        "--yes",
        action="store_true",
        help="自动批准 write/edit/bash",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="通过内嵌 HTTP Server 执行一次任务",
        description="启动临时本地服务，通过 HTTP/SSE 完成一次 Agent 任务。",
    )
    _add_runtime_options(run_parser)
    run_parser.add_argument(
        "--yes",
        action="store_true",
        help="自动批准 write/edit/bash",
    )
    run_parser.add_argument("task", help="要交给 coding agent 的任务")
    return parser


def _config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("workspace", "channel", "base_url", "model", "host", "port"):
        if hasattr(args, name):
            value = getattr(args, name)
            if value is not None:
                result[name] = value
    return result


def _print_server_info(config: KeConfig) -> None:
    print(f"ke server: http://{config.host}:{config.port}")
    print(f"channel: {config.channel}")
    print(f"model: {config.model}")
    print(f"workspace: {config.workspace}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    try:
        config = load_config(overrides=_config_overrides(args))
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "serve":
            _print_server_info(config)
            serve_runtime(config, auto_approve=args.yes)
            return 0
        return run_task(
            config,
            args.task,
            auto_approve=args.yes,
        )
    except KeyboardInterrupt:
        print("已中止", file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            f"启动失败（{type(exc).__name__}）",
            file=sys.stderr,
        )
        return 1
