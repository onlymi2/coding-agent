import os
from pathlib import Path

from ke.cli import build_parser
from ke.config import load_config
from ke.llm.types import ToolCall
from ke.safety.sandbox import WorkspaceSandbox
from ke.tools.registry import ToolRegistry


DEMO_TASK = (
    "写一个命令行计算器 calculator.py：支持加减乘除，以及 "
    "tests/test_calculator.py，然后运行 pytest 直到通过。"
)


def test_documented_demo_command_is_accepted_by_argparse() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--yes",
            "--workspace",
            "examples/demo",
            DEMO_TASK,
        ]
    )

    assert args.command == "run"
    assert args.yes is True
    assert args.workspace == "examples/demo"
    assert args.task == DEMO_TASK


def test_demo_workspace_override_resolves_inside_repository() -> None:
    repository = Path(__file__).resolve().parents[1]
    config = load_config(
        None,
        environ={"KE_API_KEY": "test-key"},
        overrides={"workspace": repository / "examples/demo"},
    )

    assert config.workspace == (repository / "examples/demo").resolve()


def test_demo_file_tools_and_bash_stay_in_selected_workspace(
    tmp_path: Path,
) -> None:
    workspace = (tmp_path / "examples/demo").resolve()
    workspace.mkdir(parents=True)
    registry = ToolRegistry(WorkspaceSandbox(workspace))

    calculator = registry.execute(
        ToolCall(
            "write-calculator",
            "write_file",
            {"path": "calculator.py", "content": "print('demo')\n"},
        )
    )
    test_file = registry.execute(
        ToolCall(
            "write-test",
            "write_file",
            {
                "path": "tests/test_calculator.py",
                "content": "def test_demo():\n    assert True\n",
            },
        )
    )
    cwd = registry.execute(
        ToolCall(
            "cwd",
            "bash",
            {"command": 'python -c "import os; print(os.getcwd())"'},
        )
    )

    assert not calculator.is_error
    assert not test_file.is_error
    assert not cwd.is_error
    assert (workspace / "calculator.py").is_file()
    assert (workspace / "tests/test_calculator.py").is_file()
    assert not (tmp_path / "calculator.py").exists()
    assert os.path.normcase(str(workspace)) in os.path.normcase(cwd.content)
