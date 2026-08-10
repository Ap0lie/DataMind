from __future__ import annotations

import json
import re
from dataclasses import dataclass
from math import isnan
from typing import Any
from uuid import UUID

import pandas as pd

from app.analysis.cleaning_sandbox import (
    GeneratedCleaningSandboxError,
    run_generated_cleaning_analysis,
)
from app.analysis.model_router import AnalysisModelRouter, MCPAnalysisModelRouter
from app.analysis.prompt_utils import (
    UNTRUSTED_INPUT_NOTICE,
    compact_prompt_columns,
    compact_prompt_records,
    untrusted_payload,
)
from app.core.settings import get_settings


@dataclass(frozen=True)
class DatasetCleaningResult:
    records: list[dict[str, Any]]
    provider: str
    model: str
    source: str
    result_markdown: str
    warnings: tuple[str, ...] = ()


class GeneratedCleaningSafetyError(ValueError):
    """Raised when generated cleaning code violates the local execution contract."""


class DataCleaningService:
    def __init__(self, model_router: AnalysisModelRouter | None = None) -> None:
        self._model_router = model_router or MCPAnalysisModelRouter()

    def clean(
        self,
        *,
        dataset_id: UUID,
        records: list[dict[str, Any]],
        requirement: str,
        use_llm: bool = True,
    ) -> DatasetCleaningResult:
        if not records:
            raise RuntimeError("Dataset has no raw records to clean.")

        raw_df = pd.DataFrame(records)
        fallback_df = _basic_clean_dataframe(raw_df)
        if not use_llm:
            return DatasetCleaningResult(
                records=_records(fallback_df),
                provider="rules",
                model="local-basic-cleaner",
                source="rules",
                result_markdown=(
                    "## DataMind 快速本地清洗\n\n"
                    "- 批量导入时使用本地规则完成快速清洗，未等待 LLM 生成代码。\n"
                    "- 已执行字段名去空格/去重、空白字符串转缺失、空行删除、重复行删除、"
                    "字符串去前后空格，以及高置信度数值/日期转换。\n"
                ),
            )
        settings = get_settings()
        failures: list[dict[str, str]] = []
        last_response = None
        for attempt in range(1, 4):
            try:
                response = self._model_router.complete(
                    messages=(
                        _cleaning_messages(dataset_id=dataset_id, df=raw_df, requirement=requirement)
                        if attempt == 1
                        else _cleaning_repair_messages(
                            dataset_id=dataset_id,
                            df=raw_df,
                            requirement=requirement,
                            failures=failures,
                        )
                    ),
                    provider=settings.cleaning_llm_provider,
                    model=None,
                    temperature=0.0,
                    max_tokens=1800,
                    metadata={
                        "agent": "cleaning",
                        "attempt": attempt,
                        "dataset_id": str(dataset_id),
                    },
                )
                last_response = response
                script = _extract_cleaning_script(response.content)
                cleaned_df = _basic_clean_dataframe(run_generated_cleaning_script(script, raw_df))
                _validate_cleaning_quality(
                    fallback_df=fallback_df,
                    cleaned_df=cleaned_df,
                    requirement=requirement,
                )
                return DatasetCleaningResult(
                    records=_records(cleaned_df),
                    provider=response.provider,
                    model=response.model,
                    source="model_router",
                    result_markdown=response.content,
                )
            except Exception as exc:
                failures.append(
                    {
                        "attempt": str(attempt),
                        "code": _truncate_text(
                            _extract_cleaning_script(last_response.content)
                            if last_response is not None and "def clean_dataset" in last_response.content
                            else (last_response.content if last_response is not None else ""),
                            5000,
                        ),
                        "error": _truncate_text(str(exc), 1200),
                    }
                )

        failure_summary = "; ".join(
            f"attempt {item['attempt']}: {item['error']}" for item in failures
        )
        return DatasetCleaningResult(
            records=_records(fallback_df),
            provider="rules",
            model="local-basic-cleaner",
            source="rules_fallback",
            result_markdown=(
                "## DataMind 基础清洗\n\n"
                "- LLM 清洗连续 3 次未通过执行或质量门禁，已使用本地规则完成基础清洗。\n"
                f"- 回退原因: {failure_summary}\n"
            ),
            warnings=tuple(item["error"] for item in failures),
        )


def run_generated_cleaning_script(script: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    try:
        return run_generated_cleaning_analysis(script, raw_df)
    except GeneratedCleaningSandboxError as exc:
        raise GeneratedCleaningSafetyError(str(exc)) from exc


def _cleaning_messages(
    *,
    dataset_id: UUID,
    df: pd.DataFrame,
    requirement: str,
) -> list[dict[str, str]]:
    selected_columns = list(df.columns[:60])
    sample_frame = df.loc[:, selected_columns].head(10)
    sample = sample_frame.where(pd.notna(sample_frame), None).to_dict(orient="records")
    profile = {
        "dataset_id": str(dataset_id),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": compact_prompt_columns(df.columns),
        "columns_truncated": max(int(df.shape[1]) - len(selected_columns), 0),
        "dtypes": {str(column): str(df[column].dtype) for column in selected_columns},
        "missing": {str(column): int(df[column].isna().sum()) for column in selected_columns},
        "sample_rows": compact_prompt_records(sample, max_rows=10, max_columns=60),
    }
    requirement_text = requirement.strip() or "执行通用分析前数据清洗。"
    return [
        {
            "role": "system",
            "content": (
                "你是 DataMind 的 DeepSeek 数据清洗工程师。返回中文 Markdown，"
                "先简述清洗策略，然后只给出一个 Python 代码块。代码块必须只定义 "
                "clean_dataset(df: pd.DataFrame) -> pd.DataFrame。不要 import。不要读取或写入文件。"
                "不要使用网络、系统调用、open、eval、exec、compile、query。运行环境已经提供 pd。"
                "函数必须复制输入 df，返回完整清洗后的 DataFrame。"
                "除非用户明确要求日期粒度，否则必须保留时间戳的时分秒、子秒和时区精度；"
                "只有原始值本身为纯日期时才规范为日期。"
                f" {UNTRUSTED_INPUT_NOTICE} "
                "只执行用户明确要求或由画像直接证明的清洗，不得凭样本猜测业务规则，不得虚构值。"
                "除非用户明确要求，不得删除业务列、过滤业务行或填充未知值。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"数据画像: {json.dumps(untrusted_payload(profile), ensure_ascii=False)}\n"
                f"用户清洗需求: {json.dumps(untrusted_payload(requirement_text), ensure_ascii=False)}\n\n"
                "请覆盖基础步骤: 字段名去空格并去重、空白字符串转缺失、明显空行删除、重复行删除、"
                "字符串值去前后空格、日期/数值字段只在转换成功率足够高时转换。"
            ),
        },
    ]


def _cleaning_repair_messages(
    *,
    dataset_id: UUID,
    df: pd.DataFrame,
    requirement: str,
    failures: list[dict[str, str]],
) -> list[dict[str, str]]:
    base = _cleaning_messages(dataset_id=dataset_id, df=df, requirement=requirement)
    repair_payload = {
        "failed_attempts": failures[-2:],
        "repair_requirement": (
            "Return a shorter complete clean_dataset(df) implementation. Fix every reported error. "
            "If a quality gate rejected row/column loss or expansion, preserve the original data shape "
            "and perform only conservative cleaning."
        ),
    }
    base[0]["content"] = (
        base[0]["content"]
        + " This is a repair attempt. Read all failure records and do not repeat the same error."
    )
    base[1]["content"] += f"\n失败记录: {json.dumps(repair_payload, ensure_ascii=False)}"
    return base


def _validate_cleaning_quality(
    *,
    fallback_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    requirement: str,
) -> None:
    if cleaned_df.empty or cleaned_df.shape[1] == 0:
        raise GeneratedCleaningSafetyError("Cleaning quality gate rejected an empty dataset.")
    baseline_rows = max(int(fallback_df.shape[0]), 1)
    baseline_columns = max(int(fallback_df.shape[1]), 1)
    lowered = requirement.lower()
    explicit_row_filter = any(
        token in lowered
        for token in ("过滤", "删除行", "保留", "排除", "filter", "drop rows", "keep only", "exclude")
    )
    explicit_column_drop = any(
        token in lowered
        for token in ("删除列", "移除列", "drop column", "remove column")
    )
    minimum_rows = 1 if explicit_row_filter else max(1, int(baseline_rows * 0.5))
    minimum_columns = 1 if explicit_column_drop else max(1, int(baseline_columns * 0.8))
    if int(cleaned_df.shape[0]) < minimum_rows:
        raise GeneratedCleaningSafetyError(
            f"Cleaning quality gate rejected excessive row loss: {cleaned_df.shape[0]}/{baseline_rows}."
        )
    if int(cleaned_df.shape[0]) > max(baseline_rows * 2, baseline_rows + 1000):
        raise GeneratedCleaningSafetyError("Cleaning quality gate rejected excessive row expansion.")
    if int(cleaned_df.shape[1]) < minimum_columns:
        raise GeneratedCleaningSafetyError(
            f"Cleaning quality gate rejected excessive column loss: {cleaned_df.shape[1]}/{baseline_columns}."
        )
    if int(cleaned_df.shape[1]) > baseline_columns + 10:
        raise GeneratedCleaningSafetyError("Cleaning quality gate rejected excessive new columns.")
    _validate_temporal_precision(
        fallback_df=fallback_df,
        cleaned_df=cleaned_df,
        requirement=requirement,
    )


def _validate_temporal_precision(
    *,
    fallback_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    requirement: str,
) -> None:
    if _allows_temporal_precision_loss(requirement):
        return
    for column in fallback_df.columns:
        if column not in cleaned_df.columns or not _looks_temporal_column(str(column)):
            continue
        baseline = pd.to_datetime(fallback_df[column], errors="coerce", format="mixed")
        current = pd.to_datetime(cleaned_df[column], errors="coerce", format="mixed")
        baseline_values = baseline.dropna()
        current_values = current.dropna()
        if len(baseline_values) < 5 or len(current_values) < 2:
            continue
        baseline_timed_rate = _explicit_time_component_rate(fallback_df[column])
        current_timed_rate = _explicit_time_component_rate(cleaned_df[column])
        baseline_timezone_rate, baseline_offsets = _timezone_profile(baseline_values)
        current_timezone_rate, current_offsets = _timezone_profile(current_values)
        same_temporal_count = len(baseline_values) == len(current_values)
        baseline_unique = int(baseline_values.nunique())
        current_unique = int(current_values.nunique())
        comparable_unique = min(baseline_unique, len(current_values))
        time_components_collapsed = (
            baseline_timed_rate > 0
            and (
                (same_temporal_count and current_timed_rate < baseline_timed_rate)
                or (
                    baseline_timed_rate >= 0.5
                    and current_timed_rate < baseline_timed_rate * 0.25
                )
            )
        )
        timestamp_cardinality_collapsed = (
            baseline_unique >= 10
            and comparable_unique >= 5
            and current_unique / comparable_unique < 0.5
        )
        timezone_awareness_collapsed = (
            baseline_timezone_rate > 0
            and (
                (same_temporal_count and current_timezone_rate < baseline_timezone_rate)
                or (
                    baseline_timezone_rate >= 0.5
                    and current_timezone_rate < baseline_timezone_rate * 0.5
                )
            )
        )
        timezone_offsets_changed = (
            baseline_timezone_rate >= 0.5
            and current_timezone_rate >= 0.5
            and same_temporal_count
            and baseline_offsets != current_offsets
        )
        if (
            time_components_collapsed
            or timestamp_cardinality_collapsed
            or timezone_awareness_collapsed
            or timezone_offsets_changed
        ):
            raise GeneratedCleaningSafetyError(
                "Cleaning quality gate rejected temporal precision loss "
                f"in column {column}: timed values {baseline_timed_rate:.0%} -> "
                f"{current_timed_rate:.0%}, unique timestamps "
                f"{baseline_unique} -> {current_unique}, timezone-aware values "
                f"{baseline_timezone_rate:.0%} -> {current_timezone_rate:.0%}, "
                f"UTC offsets {sorted(baseline_offsets)} -> {sorted(current_offsets)}."
            )


def _allows_temporal_precision_loss(requirement: str) -> bool:
    lowered = requirement.lower()
    if any(
        token in lowered
        for token in (
            "保留时间",
            "保留时分秒",
            "保留时区",
            "不要去掉时间",
            "不得去掉时间",
            "不删除时间",
            "preserve time",
            "keep time",
            "retain time",
            "preserve timezone",
            "keep timezone",
        )
    ):
        return False
    return any(
        token in lowered
        for token in (
            "仅保留日期",
            "只保留日期",
            "转为纯日期",
            "转换为日期",
            "日期粒度",
            "去掉时间",
            "移除时间",
            "删除时间",
            "截断到日期",
            "date only",
            "truncate to date",
            "day granularity",
            "drop time",
            "remove time",
        )
    )


def _time_component_rate(values: pd.Series) -> float:
    timed = sum(
        bool(
            value.hour
            or value.minute
            or value.second
            or value.microsecond
            or getattr(value, "nanosecond", 0)
        )
        for value in values
    )
    return timed / max(len(values), 1)


def _explicit_time_component_rate(values: pd.Series) -> float:
    present = values.dropna()
    if present.empty:
        return 0.0
    timed = present.astype("string").str.contains(
        r"(?:^|[T\s])\d{1,2}:\d{2}(?::\d{2})?",
        regex=True,
        na=False,
    )
    return float(timed.sum()) / len(present)


def _timezone_profile(values: pd.Series) -> tuple[float, set[int]]:
    if values.empty:
        return 0.0, set()
    aware_count = 0
    offsets: set[int] = set()
    for value in values:
        try:
            offset = value.utcoffset()
        except (AttributeError, TypeError, ValueError):
            offset = None
        if offset is None:
            continue
        aware_count += 1
        offsets.add(round(offset.total_seconds()))
    return aware_count / len(values), offsets


def _truncate_text(value: str, max_chars: int) -> str:
    return value if len(value) <= max_chars else f"{value[:max_chars]}... [truncated]"


def _extract_cleaning_script(markdown: str) -> str:
    matches = re.findall(
        r"```(?:python|py)?\s*(.*?)```",
        markdown,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if matches:
        return matches[-1].strip()
    if "def clean_dataset" in markdown:
        return markdown[markdown.find("def clean_dataset") :].strip()
    raise ValueError("DeepSeek did not return a clean_dataset Python code block.")


def _basic_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy(deep=True)
    cleaned = cleaned.dropna(axis=0, how="all").dropna(axis=1, how="all")
    cleaned.columns = _dedupe_columns([str(column).strip() or "column" for column in cleaned.columns])
    for column in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[column]) or pd.api.types.is_string_dtype(
            cleaned[column]
        ):
            cleaned[column] = cleaned[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            cleaned[column] = cleaned[column].replace(r"^\s*$", pd.NA, regex=True)
            numeric = pd.to_numeric(cleaned[column], errors="coerce")
            non_null_count = int(cleaned[column].notna().sum())
            numeric_count = int(numeric.notna().sum())
            if non_null_count and numeric_count / non_null_count >= 0.85:
                cleaned[column] = numeric
                continue
            lowered_name = str(column).lower()
            if any(token in lowered_name for token in ("date", "日期", "time", "时间")):
                parsed = pd.to_datetime(cleaned[column], errors="coerce", format="mixed")
                parsed_count = int(parsed.notna().sum())
                if non_null_count and parsed_count / non_null_count >= 0.75:
                    if _should_preserve_time(cleaned[column], parsed):
                        cleaned[column] = parsed.map(
                            lambda value: value.isoformat() if pd.notna(value) else pd.NA
                        )
                    else:
                        cleaned[column] = parsed.map(
                            lambda value: value.strftime("%Y-%m-%d")
                            if pd.notna(value)
                            else pd.NA
                        )
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned.where(pd.notna(cleaned), None)


def _dedupe_columns(values: list[str]) -> list[str]:
    columns: list[str] = []
    counts: dict[str, int] = {}
    for index, value in enumerate(values):
        base = value or f"column_{index + 1}"
        counts[base] = counts.get(base, 0) + 1
        columns.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return columns


def _looks_temporal_column(column: str) -> bool:
    lowered = column.lower()
    return any(token in lowered for token in ("date", "日期", "time", "时间"))


def _should_preserve_time(
    original: pd.Series,
    parsed: pd.Series,
) -> bool:
    if _explicit_time_component_rate(original) > 0:
        return True
    return _time_component_rate(parsed.dropna()) > 0


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [_jsonable(row) for row in df.to_dict(orient="records")]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, UUID):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, float) and isnan(value):
        return None
    return value
