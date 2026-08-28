import argparse
from collections.abc import Sequence

from ke import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ke",
        description="ke：一个自研的本地编程智能体缰绳。",
        epilog="当前完成到阶段六，尚未接入真实模型、配置系统或 server。",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
