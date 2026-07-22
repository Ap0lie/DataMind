from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

from app.schemas.analysis import ChartResponse, TextAnalysisResultResponse

_TEXT_COLUMN_HINTS = (
    "comment",
    "content",
    "description",
    "feedback",
    "message",
    "review",
    "text",
    "评论",
    "内容",
    "反馈",
    "文本",
    "留言",
    "评价",
)
_GROUP_COLUMN_HINTS = (
    "category",
    "class",
    "label",
    "rating",
    "sentiment",
    "status",
    "type",
    "分类",
    "情绪",
    "标签",
    "类别",
    "状态",
)
_QUESTION_HINTS = (
    "comment",
    "feedback",
    "keyword",
    "negative",
    "positive",
    "review",
    "sentiment",
    "text",
    "主题",
    "关键词",
    "正面",
    "负面",
    "评论",
    "评价",
    "文本",
)
_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "but",
    "for",
    "from",
    "have",
    "into",
    "not",
    "that",
    "the",
    "this",
    "was",
    "were",
    "with",
    "you",
}


def run_text_analysis_toolbox(
    df: pd.DataFrame,
    *,
    question: str,
) -> tuple[TextAnalysisResultResponse, ...]:
    text_columns = _candidate_text_columns(df, question)
    if not text_columns:
        return ()

    results: list[TextAnalysisResultResponse] = []
    for text_column in text_columns[:2]:
        group_column = _candidate_group_column(df, text_column)
        results.append(_analyze_text_column(df, text_column=text_column, group_column=group_column))
    return tuple(results)


def _candidate_text_columns(df: pd.DataFrame, question: str) -> list[str]:
    if not _question_requests_text_analysis(question) and not _has_obvious_text_column(df):
        return []

    candidates: list[tuple[int, str]] = []
    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            continue
        text = series.astype(str)
        avg_length = float(text.str.len().mean())
        max_length = int(text.str.len().max())
        distinct_ratio = float(text.nunique(dropna=True) / max(len(text), 1))
        lowered_name = str(column).lower()
        score = 0
        if any(hint in lowered_name for hint in _TEXT_COLUMN_HINTS):
            score += 5
        if avg_length >= 20:
            score += 3
        if max_length >= 60:
            score += 3
        if distinct_ratio >= 0.5:
            score += 1
        if score >= 4:
            candidates.append((score, str(column)))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [column for _, column in candidates]


def _candidate_group_column(df: pd.DataFrame, text_column: str) -> str | None:
    candidates: list[tuple[int, str]] = []
    row_count = max(len(df), 1)
    for column in df.columns:
        if str(column) == text_column:
            continue
        series = df[column].dropna()
        if series.empty:
            continue
        distinct_count = int(series.astype(str).nunique(dropna=True))
        if distinct_count < 2 or distinct_count > min(30, max(2, row_count // 2)):
            continue
        lowered_name = str(column).lower()
        score = 1
        if any(hint in lowered_name for hint in _GROUP_COLUMN_HINTS):
            score += 5
        candidates.append((score, str(column)))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1] if candidates else None


def _analyze_text_column(
    df: pd.DataFrame,
    *,
    text_column: str,
    group_column: str | None,
) -> TextAnalysisResultResponse:
    text = df[text_column].fillna("").astype(str)
    non_empty = text[text.str.strip() != ""]
    lengths = non_empty.str.len()
    keywords = _top_keywords(non_empty.tolist(), limit=12)

    summary: dict[str, Any] = {
        "row_count": len(df),
        "non_empty_count": len(non_empty),
        "empty_count": int(len(df) - len(non_empty)),
        "avg_length": round(float(lengths.mean()), 2) if not lengths.empty else 0,
        "median_length": round(float(lengths.median()), 2) if not lengths.empty else 0,
        "max_length": int(lengths.max()) if not lengths.empty else 0,
        "top_keywords": [{"keyword": key, "count": int(count)} for key, count in keywords],
    }
    insights = [
        f"{text_column} 可分析文本 {summary['non_empty_count']} 条，平均长度 {summary['avg_length']} 字符。",
    ]
    charts = [
        ChartResponse(
            title=f"{text_column} 文本长度分布",
            chart_type="histogram",
            spec={"x": "label", "y": "value", "source_metric": "text_length"},
            data=tuple(_histogram_buckets([float(value) for value in lengths.tolist()])),
        )
    ]
    if keywords:
        charts.append(
            ChartResponse(
                title=f"{text_column} 高频关键词",
                chart_type="bar",
                spec={"x": "keyword", "y": "count"},
                data=tuple({"keyword": key, "count": int(count)} for key, count in keywords),
            )
        )

    if group_column and group_column in df.columns:
        grouped = _group_text_summary(df, text_column=text_column, group_column=group_column)
        if grouped:
            summary["group_column"] = group_column
            summary["groups"] = grouped
            top_group = grouped[0]
            insights.append(
                f"{group_column} 中 {top_group['group']} 样本最多，共 {top_group['count']} 条；"
                f"平均文本长度 {top_group['avg_length']}。"
            )
            if len(grouped) >= 2:
                longest_group = max(grouped, key=lambda item: float(item["avg_length"]))
                shortest_group = min(grouped, key=lambda item: float(item["avg_length"]))
                diff = round(float(longest_group["avg_length"]) - float(shortest_group["avg_length"]), 2)
                insights.append(
                    f"{group_column} 中 {longest_group['group']} 的平均文本长度最高，"
                    f"比 {shortest_group['group']} 高 {diff} 字符。"
                )
            charts.append(
                ChartResponse(
                    title=f"{group_column} 文本数量",
                    chart_type="bar",
                    spec={"x": "group", "y": "count"},
                    data=tuple({"group": item["group"], "count": item["count"]} for item in grouped),
                )
            )
            charts.append(
                ChartResponse(
                    title=f"{group_column} 平均文本长度",
                    chart_type="bar",
                    spec={"x": "group", "y": "avg_length"},
                    data=tuple(
                        {"group": item["group"], "avg_length": item["avg_length"]}
                        for item in grouped
                    ),
                )
            )

    return TextAnalysisResultResponse(
        task="text_profile_and_group_compare",
        text_column=text_column,
        group_column=group_column,
        summary=summary,
        insights=tuple(insights),
        charts=tuple(charts),
    )


def _group_text_summary(
    df: pd.DataFrame,
    *,
    text_column: str,
    group_column: str,
) -> list[dict[str, Any]]:
    working = df[[text_column, group_column]].copy()
    working[text_column] = working[text_column].fillna("").astype(str)
    working[group_column] = working[group_column].fillna("未标注").astype(str)
    working["text_length"] = working[text_column].str.len()
    rows: list[dict[str, Any]] = []
    for group, group_df in working.groupby(group_column, dropna=False):
        keywords = _top_keywords(group_df[text_column].tolist(), limit=6)
        rows.append(
            {
                "group": str(group),
                "count": len(group_df),
                "avg_length": round(float(group_df["text_length"].mean()), 2),
                "top_keywords": [
                    {"keyword": key, "count": int(count)} for key, count in keywords[:5]
                ],
            }
        )
    rows.sort(key=lambda item: (-int(item["count"]), str(item["group"])))
    return rows[:12]


def _top_keywords(texts: list[str], *, limit: int) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for text in texts[:1000]:
        counter.update(_tokens(text))
    return counter.most_common(limit)


def _tokens(text: str) -> list[str]:
    tokens = [
        english
        for english in re.findall(r"[A-Za-z][A-Za-z']{2,}", text.lower())
        if english not in _STOPWORDS
    ]
    for cjk_text in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        tokens.extend(cjk_text[index : index + 2] for index in range(len(cjk_text) - 1))
    return tokens


def _question_requests_text_analysis(question: str) -> bool:
    lowered = question.lower()
    return any(hint in lowered for hint in _QUESTION_HINTS)


def _has_obvious_text_column(df: pd.DataFrame) -> bool:
    return any(any(hint in str(column).lower() for hint in _TEXT_COLUMN_HINTS) for column in df.columns)


def _histogram_buckets(values: list[float], bucket_count: int = 10) -> list[dict[str, Any]]:
    clean_values = [float(value) for value in values if pd.notna(value)]
    if not clean_values:
        return []
    minimum = min(clean_values)
    maximum = max(clean_values)
    if minimum == maximum:
        return [{"label": _format_number(minimum), "value": len(clean_values)}]
    width = (maximum - minimum) / bucket_count
    buckets = [
        {
            "label": f"{_format_number(minimum + width * index)}-{_format_number(minimum + width * (index + 1))}",
            "value": 0,
        }
        for index in range(bucket_count)
    ]
    for value in clean_values:
        index = min(bucket_count - 1, int((value - minimum) / width))
        buckets[index]["value"] += 1
    return buckets


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"
