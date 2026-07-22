from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisExperience:
    thresholds: str = ""
    priority_rules: str = ""
    good_summaries: str = ""
    plan_schema: str = ""

    def as_prompt_context(self, *, include: tuple[str, ...] | None = None) -> str:
        selected = set(include or ("thresholds", "priority_rules", "good_summaries", "plan_schema"))
        sections = [
            ("thresholds", "业务阈值", self.thresholds),
            ("priority_rules", "优先级规则", self.priority_rules),
            ("good_summaries", "好结论范例", self.good_summaries),
            ("plan_schema", "Plan Schema", self.plan_schema),
        ]
        return "\n\n".join(
            f"【{title}】\n{content}"
            for key, title, content in sections
            if key in selected and content
        )


class ExperienceLoader:
    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root) if root is not None else Path(__file__).parent / "experience"

    def load(self) -> AnalysisExperience:
        return AnalysisExperience(
            thresholds=self._read("thresholds.json"),
            priority_rules=self._read("priority_rules.md"),
            good_summaries=self._read("good_summaries.md"),
            plan_schema=self._read("plan_schema.json"),
        )

    def _read(self, name: str) -> str:
        path = self._root / name
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()
