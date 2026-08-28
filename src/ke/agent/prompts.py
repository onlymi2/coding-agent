from pathlib import Path

from ke.safety.sandbox import WorkspaceSandbox


PROJECT_INSTRUCTION_FILES = ("KE.md", "AGENTS.md")


def _read_project_instructions(sandbox: WorkspaceSandbox) -> tuple[str, str] | None:
    for name in PROJECT_INSTRUCTION_FILES:
        path = sandbox.resolve_for_read(name)
        if path.is_file():
            return name, path.read_text(encoding="utf-8")
    return None


def build_system_prompt(workspace: str | Path) -> str:
    """Build the local coding-agent prompt and optional project conventions."""

    sandbox = WorkspaceSandbox(workspace)
    prompt = f"""你是一个在本地项目中工作的 coding agent。

当前 workspace 根路径：{sandbox.workspace}

必须遵守以下规则：
1. 只能通过提供的本地工具读取、修改和验证项目内容。
2. 不要编造没有通过工具读取过的文件内容。
3. 修改文件前优先使用 read_file 查看当前内容。
4. 小范围修改优先使用 edit_file；只有创建文件或整体重写时才使用 write_file。
5. 使用 bash 运行相关测试或命令，验证修改确实有效。
6. 工具失败时阅读错误结果，修正参数或方案后继续处理，不要假装成功。
7. 完成任务并验证后停止调用工具，给出简洁的最终说明。
8. 所有操作都必须限制在当前 workspace 内。
"""

    project_instructions = _read_project_instructions(sandbox)
    if project_instructions is not None:
        name, content = project_instructions
        prompt += f"\n项目级约定（来自 {name}）：\n{content}"
    return prompt
