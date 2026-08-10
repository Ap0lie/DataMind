from __future__ import annotations

import json
from typing import Any

_ROUTING_REQUIRED_PHRASES = (
    "开始分析",
    "重新分析",
    "执行分析",
    "进一步分析",
    "帮我分析",
    "分析这个",
    "分析数据",
    "分析计划",
    "生成新报告",
    "重新生成",
    "修改报告",
    "编辑报告",
    "美化",
    "精简报告",
    "清洗",
    "上传",
    "导入",
    "删除",
    "恢复",
    "重命名",
    "发布",
    "授权",
    "取消",
    "重试",
    "任务状态",
    "运行状态",
    "处理进度",
    "有哪些字段",
    "字段类型",
    "列类型",
    "关系推荐",
    "保存关系",
    "语义模型",
)

_REPORT_READ_PHRASES = (
    "报告",
    "结论",
    "总结",
    "概括",
    "摘要",
    "发现",
    "建议",
    "风险",
    "趋势",
    "指标",
    "数据质量",
    "解读",
    "说明",
    "多少",
    "最高",
    "最低",
    "对比",
)

_NO_TOOL_CHAT_PHRASES = ("你好", "谢谢", "你是谁", "你能做什么", "帮助")


def should_skip_tool_router(
    *,
    question: str,
    execution_mode: str,
    scope_type: str,
    retrieved_reports: tuple[dict[str, Any], ...],
) -> bool:
    """Use one streaming model call when server-retrieved evidence is already sufficient."""

    if execution_mode != "ask":
        return False
    normalized = _normalize(question)
    if not normalized:
        return False
    if any(_normalize(phrase) in normalized for phrase in _ROUTING_REQUIRED_PHRASES):
        return False
    if any(_normalize(phrase) in normalized for phrase in _NO_TOOL_CHAT_PHRASES):
        return True
    if not retrieved_reports:
        return False
    if scope_type == "report":
        return True
    if any(_normalize(phrase) in normalized for phrase in _REPORT_READ_PHRASES):
        return True
    return any(_report_matches(question, report) for report in retrieved_reports)


def compact_message_text(value: str, *, max_chars: int = 4000) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    head = max(1, int(max_chars * 0.72))
    tail = max(1, max_chars - head - 20)
    return f"{text[:head]}\n...[truncated]...\n{text[-tail:]}"


def _report_matches(question: str, report: dict[str, Any]) -> bool:
    query = _normalize(question)
    searchable = _normalize(
        json.dumps(
            {
                "title": report.get("title"),
                "question": report.get("question"),
                "executive_summary": report.get("executive_summary"),
                "key_findings": report.get("key_findings"),
            },
            ensure_ascii=False,
            default=str,
        )
    )
    if len(query) >= 3 and query in searchable:
        return True
    query_pairs = _pairs(query)
    if not query_pairs:
        return False
    overlap = sum(1 for pair in query_pairs if pair in searchable)
    required = 1 if len(query) <= 6 else min(2, len(query_pairs))
    return overlap >= required


def _pairs(value: str) -> tuple[str, ...]:
    if len(value) < 2:
        return ()
    return tuple(
        pair
        for pair in {value[index : index + 2] for index in range(len(value) - 1)}
        if pair not in {"如何", "什么", "这个", "一下", "数据"}
    )


def _normalize(value: Any) -> str:
    return "".join(character.lower() for character in str(value or "") if character.isalnum())
