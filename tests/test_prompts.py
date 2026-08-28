from pathlib import Path

from ke.agent.prompts import build_system_prompt


def test_system_prompt_contains_runtime_rules_and_workspace(tmp_path: Path) -> None:
    prompt = build_system_prompt(tmp_path)

    assert str(tmp_path.resolve()) in prompt
    assert "本地项目" in prompt
    assert "不要编造" in prompt
    assert "read_file" in prompt
    assert "edit_file" in prompt
    assert "write_file" in prompt
    assert "bash" in prompt
    assert "工具失败" in prompt
    assert "停止调用工具" in prompt


def test_system_prompt_prefers_ke_md_and_never_reads_env(tmp_path: Path) -> None:
    (tmp_path / "KE.md").write_text("USE KE RULES", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("USE AGENT RULES", encoding="utf-8")
    (tmp_path / ".env").write_text("DO-NOT-READ", encoding="utf-8")

    prompt = build_system_prompt(tmp_path)

    assert "USE KE RULES" in prompt
    assert "USE AGENT RULES" not in prompt
    assert "DO-NOT-READ" not in prompt
