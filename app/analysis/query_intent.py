from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.schemas.analysis import (
    AnalysisAggregationResponse,
    AnalysisFilterResponse,
    DatasetProfileResponse,
)

_NON_WORD_RE = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)
_IDENTIFIER_TOKENS = ("_id", " id", "编号", "编码", "code", "uuid", "key")
_NEGATED_CLAUSE_RE = re.compile(
    r"(?:不要|不得|请勿|禁止|无需|排除|忽略)[^,，;；。.!?！？\n]*"
    r"|(?:(?:do\s+not|don't|never)\s+[^,，;；。.!?！？\n]*"
    r"|without\s+[^,，;；。.!?！？\n]*|excluding?\s+[^,，;；。.!?！？\n]*)",
    re.IGNORECASE,
)
_NEGATED_GROUPING_RE = re.compile(
    r"(?:不要|不得|请勿|禁止)\s*(?:按|依据|根据|以)\s*"
    r"[^,，;；。.!?！？\n]*?(?:分组|汇总|聚合)"
    r"|(?:(?:do\s+not|don't|never)\s+(?:group|aggregate)\s+by"
    r"|without\s+(?:grouping|aggregating)\s+by)\s*"
    r"[^,，;；。.!?！？\n]*",
    re.IGNORECASE,
)

_DIMENSION_CONCEPT_ALIASES: tuple[
    tuple[tuple[str, ...], tuple[str, ...]], ...
] = (
    (
        ("segment", "customer_type", "user_group", "customer_group"),
        ("客户细分", "客户分群", "用户分群", "客群", "客户类型", "segment"),
    ),
    (
        ("region", "state", "province", "area"),
        ("地区", "区域", "省份", "州", "region", "state"),
    ),
    (
        ("city",),
        ("城市", "city"),
    ),
    (
        ("category", "product_type", "product_group"),
        ("商品类别", "产品类别", "品类", "类别", "category"),
    ),
    (
        ("status", "state"),
        ("状态", "status"),
    ),
)

_METRIC_CONCEPT_ALIASES: tuple[
    tuple[tuple[str, ...], tuple[str, ...]], ...
] = (
    (
        (
            "amount",
            "sales",
            "revenue",
            "total",
            "total_price",
            "gmv",
            "turnover",
        ),
        (
            "销售额",
            "成交额",
            "收入",
            "营收",
            "金额",
            "总额",
            "总计",
            "合计",
            "gmv",
            "sales",
            "revenue",
            "total",
        ),
    ),
    (
        ("payment", "paid_value", "pay_value"),
        ("支付金额", "付款金额", "支付", "付款", "payment"),
    ),
    (
        ("price", "item_price"),
        ("商品收入", "商品金额", "商品销售额", "价格", "price", "gmv"),
    ),
    (
        ("freight", "shipping"),
        ("运费", "配送费", "物流费", "freight", "shipping"),
    ),
    (
        ("profit", "margin"),
        ("利润", "毛利", "profit", "margin"),
    ),
)

SOURCE_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "price": ("price", "商品收入", "商品金额", "销售额", "gmv"),
    "freight_value": ("freightvalue", "freight", "运费"),
    "payment_value": ("paymentvalue", "支付总额", "支付金额", "付款总额"),
}

_VALUE_ALIASES: dict[str, tuple[str, ...]] = {
    "completed": ("已完成", "完成订单", "成交订单", "completed", "complete"),
    "complete": ("已完成", "完成订单", "成交订单", "completed", "complete"),
    "cancelled": ("已取消", "取消订单", "cancelled", "canceled"),
    "canceled": ("已取消", "取消订单", "cancelled", "canceled"),
    "delivered": ("已送达", "已交付", "delivered"),
    "paid": ("已支付", "支付成功", "paid"),
    "active": ("活跃", "启用", "active"),
    "inactive": ("不活跃", "停用", "inactive"),
}

_ENTITY_ALIASES: tuple[
    tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...
] = (
    (
        ("客户", "用户", "买家", "customer", "user", "buyer"),
        ("customer", "user", "buyer", "客户", "用户", "买家"),
        ("seller", "merchant", "卖家", "商家"),
    ),
    (
        ("卖家", "商家", "seller", "merchant"),
        ("seller", "merchant", "卖家", "商家"),
        ("customer", "user", "buyer", "客户", "用户", "买家"),
    ),
    (
        ("商品", "产品", "product", "item"),
        ("product", "item", "商品", "产品"),
        ("customer", "seller", "用户", "客户", "卖家"),
    ),
)


@dataclass(frozen=True)
class QueryIntent:
    required_dimensions: tuple[str, ...] = ()
    candidate_dimensions: tuple[str, ...] = ()
    required_metric: str | None = None
    candidate_metrics: tuple[str, ...] = ()
    aggregations: tuple[AnalysisAggregationResponse, ...] = ()
    filters: tuple[AnalysisFilterResponse, ...] = ()
    derived_metrics: tuple[str, ...] = ()

    @property
    def dimensions(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.required_dimensions, *self.candidate_dimensions)))

    @property
    def metric(self) -> str | None:
        return self.required_metric or next(iter(self.candidate_metrics), None)


def infer_query_intent(
    question: str,
    profile: DatasetProfileResponse,
) -> QueryIntent:
    semantic_question = strip_negated_clauses(question)
    columns = tuple(column.name for column in profile.columns)
    numeric = tuple(profile.numeric_columns)
    categorical = tuple(profile.categorical_columns)

    filters = _requested_filters(
        question=semantic_question,
        categorical_columns=categorical,
        sample_records=profile.sample_records,
    )
    filter_columns = {
        *(item.column for item in filters),
        *_explicit_filter_columns(semantic_question, categorical),
    }
    negated_dimension_columns = set(negated_grouping_columns(question, categorical))

    ranked_dimensions = tuple(
        (score, column)
        for score, column in sorted(
            (
                (
                    _column_mention_score(
                        semantic_question,
                        column,
                        aliases=_DIMENSION_CONCEPT_ALIASES,
                    ),
                    column,
                )
                for column in categorical
                if not _looks_like_identifier(column)
            ),
            reverse=True,
        )
        if score > 0
    )
    required_dimensions = _required_dimensions(
        semantic_question,
        ranked_dimensions,
        excluded_columns={*filter_columns, *negated_dimension_columns},
    )
    required_metric, candidate_metrics = _rank_metrics(semantic_question, numeric)
    mentioned_metrics = tuple(
        column
        for column in numeric
        if _column_mention_score(
            semantic_question,
            column,
            aliases=_METRIC_CONCEPT_ALIASES,
        )
        > 0
    )
    metric_dimension_columns = {
        column
        for column in categorical
        if _column_mention_score(
            semantic_question,
            column,
            aliases=_METRIC_CONCEPT_ALIASES,
        )
        > 0
    }
    excluded_dimension_columns = {
        *filter_columns,
        *negated_dimension_columns,
        *metric_dimension_columns,
        *((required_metric,) if required_metric else ()),
    }
    candidates = tuple(
        column
        for _, column in ranked_dimensions
        if column not in required_dimensions
        and column not in excluded_dimension_columns
        and _candidate_dimension_matches_question_entity(semantic_question, column)
    )
    aggregations = _requested_aggregations(
        question=semantic_question,
        metric=required_metric,
        metrics=(required_metric,) if required_metric else mentioned_metrics,
        columns=columns,
    )
    return QueryIntent(
        required_dimensions=required_dimensions,
        candidate_dimensions=candidates[:3],
        required_metric=required_metric,
        candidate_metrics=candidate_metrics,
        aggregations=aggregations,
        filters=filters,
        derived_metrics=("average_order_value",)
        if _wants_average_order_value(semantic_question)
        else (),
    )


def infer_source_aggregations(
    question: str,
    sources: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[AnalysisAggregationResponse, ...]:
    """Extract explicitly requested totals from their named source tables."""

    normalized_question = _normalize(question)
    aggregations: list[AnalysisAggregationResponse] = []
    for source, columns in sources:
        source_name = _normalize(
            source.casefold().removesuffix(".csv").removesuffix("_dataset")
        )
        if source_name and source_name not in normalized_question:
            continue
        for metric, aliases in SOURCE_METRIC_ALIASES.items():
            if metric not in columns or not any(
                _normalize(alias) in normalized_question for alias in aliases
            ):
                continue
            column = f"{source}__{metric}"
            aggregations.append(
                AnalysisAggregationResponse(
                    operation="sum",
                    column=column,
                    alias=f"total_{_safe_alias(column)}",
                )
            )
    return tuple(aggregations)


def _required_dimensions(
    question: str,
    ranked_dimensions: tuple[tuple[int, str], ...],
    *,
    excluded_columns: set[str],
) -> tuple[str, ...]:
    required = [
        column
        for _, column in ranked_dimensions
        if column not in excluded_columns
        and _explicit_column_name_mentioned(question, column)
    ]
    normalized_question = _normalize(question)
    question_has_entity = any(
        _normalize(token) in normalized_question
        for question_tokens, _, _ in _ENTITY_ALIASES
        for token in question_tokens
    )
    for column_tokens, aliases in _DIMENSION_CONCEPT_ALIASES:
        if not any(_normalize(alias) in normalized_question for alias in aliases):
            continue
        matches: list[tuple[int, int, str]] = []
        for score, column in ranked_dimensions:
            if column in excluded_columns:
                continue
            if not any(token in column.casefold() for token in column_tokens):
                continue
            alignment = _entity_alignment_score(question, column)
            if alignment < 0 or (
                question_has_entity
                and alignment == 0
                and _column_has_known_entity(column)
            ):
                continue
            leaf_length = len(_column_leaf(column)) if alignment > 0 else 0
            matches.append((score, -leaf_length, column))
        if not matches:
            continue
        matches.sort(reverse=True)
        best = matches[0]
        if len(matches) == 1 or best[:2] != matches[1][:2]:
            required.append(best[2])
    return tuple(dict.fromkeys(required))


def _explicit_column_name_mentioned(question: str, column: str) -> bool:
    normalized_question = _normalize(question)
    names = {_normalize(column), _normalize(_column_leaf(column))}
    return any(
        len(name) >= (2 if re.search(r"[\u3400-\u9fff]", name) else 3)
        and name in normalized_question
        for name in names
    )


def _column_leaf(column: str) -> str:
    return re.split(r"__|\.", column)[-1]


def _column_has_known_entity(column: str) -> bool:
    normalized_column = _normalize(column)
    return any(
        _normalize(token) in normalized_column
        for _, matching_tokens, conflicting_tokens in _ENTITY_ALIASES
        for token in (*matching_tokens, *conflicting_tokens)
    )


def _candidate_dimension_matches_question_entity(question: str, column: str) -> bool:
    normalized_question = _normalize(question)
    question_has_entity = any(
        _normalize(token) in normalized_question
        for question_tokens, _, _ in _ENTITY_ALIASES
        for token in question_tokens
    )
    if not question_has_entity or not _column_has_known_entity(column):
        return True
    return _entity_alignment_score(question, column) > 0


def _rank_metrics(
    question: str,
    numeric_columns: tuple[str, ...],
) -> tuple[str | None, tuple[str, ...]]:
    if not numeric_columns:
        return None, ()
    ranked = sorted(
        (
            (
                _column_mention_score(
                    question,
                    column,
                    aliases=_METRIC_CONCEPT_ALIASES,
                ),
                -index,
                column,
            )
            for index, column in enumerate(numeric_columns)
        ),
        reverse=True,
    )
    mentioned = tuple(item for item in ranked if item[0] > 0)
    required = _required_metric(question, mentioned)
    business_metrics = [
        column
        for column in numeric_columns
        if any(
            token in _normalize(column)
            for token in (
                "amount",
                "sales",
                "revenue",
                "profit",
                "price",
                "payment",
                "gmv",
                "收入",
                "销售",
                "利润",
                "金额",
            )
        )
    ]
    candidates = tuple(
        dict.fromkeys(
            (
                *(item[2] for item in mentioned),
                *business_metrics,
                *numeric_columns,
            )
        )
    )
    return required, tuple(item for item in candidates if item != required)[:5]


def _required_metric(
    question: str,
    ranked_metrics: tuple[tuple[int, int, str], ...],
) -> str | None:
    explicitly_named = [
        column
        for _, _, column in ranked_metrics
        if _explicit_column_name_mentioned(question, column)
    ]
    if len(explicitly_named) == 1:
        return explicitly_named[0]
    if len(explicitly_named) > 1:
        return None

    normalized_question = _normalize(question)
    if "gmv" in normalized_question:
        preferred_leaves = ("gmv", "price", "revenue", "sales", "amount", "total_price")
        for preferred in preferred_leaves:
            match = next(
                (
                    column
                    for _, _, column in ranked_metrics
                    if _column_leaf(column).casefold() == preferred
                ),
                None,
            )
            if match:
                return match
    semantic_matches: list[tuple[int, int, str]] = []
    for column_tokens, aliases in _METRIC_CONCEPT_ALIASES:
        if not any(_normalize(alias) in normalized_question for alias in aliases):
            continue
        semantic_matches.extend(
            item
            for item in ranked_metrics
            if any(token in item[2].casefold() for token in column_tokens)
        )
    ranked = sorted(set(semantic_matches), reverse=True)
    if not ranked:
        return None
    top_score = ranked[0][0]
    top_matches = {column for score, _, column in ranked if score == top_score}
    return next(iter(top_matches)) if len(top_matches) == 1 else None


def _requested_aggregations(
    *,
    question: str,
    metric: str | None,
    metrics: tuple[str, ...],
    columns: tuple[str, ...],
) -> tuple[AnalysisAggregationResponse, ...]:
    folded = question.casefold()
    requested: list[AnalysisAggregationResponse] = []
    sum_requested = any(
        token in folded
        for token in ("总计", "合计", "总额", "销售额", "成交额", "sum", "total", "gmv")
    )
    aggregate_metrics = tuple(dict.fromkeys((*metrics, *((metric,) if metric else ()))))
    if sum_requested:
        requested.extend(
            AnalysisAggregationResponse(
                operation="sum", column=column, alias=f"total_{_safe_alias(column)}"
            )
            for column in aggregate_metrics
        )

    average_order_value = _wants_average_order_value(question)
    count_requested = any(
        token in folded
        for token in ("数量", "订单数", "客户数", "用户数", "count", "number of")
    ) or average_order_value
    if count_requested:
        count_column = _count_entity_column(question, columns)
        requested.append(
            AnalysisAggregationResponse(
                operation="count_distinct" if count_column else "count",
                column=count_column,
                alias=_count_alias(count_column),
            )
        )

    if any(token in folded for token in ("准时率", "按时率", "on_time_rate", "on-time rate")):
        on_time_column = next(
            (
                column
                for column in columns
                if _column_leaf(column).casefold() in {"on_time", "on_time_rate"}
            ),
            None,
        )
        if on_time_column:
            requested.append(
                AnalysisAggregationResponse(
                    operation="avg",
                    column=on_time_column,
                    alias="on_time_rate",
                )
            )

    if metric and not average_order_value and any(
        token in folded
        for token in ("平均", "均值", "average", "avg", "mean")
    ):
        requested.append(
            AnalysisAggregationResponse(
                operation="avg",
                column=metric,
                alias=f"average_{_safe_alias(metric)}",
            )
        )

    if not requested and metric:
        requested.append(
            AnalysisAggregationResponse(
                operation="sum",
                column=metric,
                alias=f"total_{_safe_alias(metric)}",
            )
        )
    return tuple(_deduplicate_aggregations(requested))


def _wants_average_order_value(question: str) -> bool:
    folded = question.casefold()
    return any(token in folded for token in ("客单价", "aov", "average order value"))


def _requested_filters(
    *,
    question: str,
    categorical_columns: tuple[str, ...],
    sample_records: tuple[dict[str, Any], ...],
) -> tuple[AnalysisFilterResponse, ...]:
    folded = question.casefold()
    filters: list[AnalysisFilterResponse] = []
    inferred: list[tuple[str, str]] = []
    for column in categorical_columns:
        explicit_value = _explicit_filter_value(question, column)
        if explicit_value is not None:
            filters.append(
                AnalysisFilterResponse(
                    column=column,
                    operator="=",
                    value=explicit_value,
                )
            )
            continue
        values = {
            str(record[column]).strip()
            for record in sample_records
            if record.get(column) not in (None, "")
        }
        for value in sorted(values, key=len, reverse=True):
            aliases = _VALUE_ALIASES.get(value.casefold(), ())
            if _sample_value_mentioned(question, value) or any(
                alias.casefold() in folded for alias in aliases
            ):
                inferred.append((column, value))
                break
    inferred_by_value: dict[str, list[tuple[str, str]]] = {}
    for column, value in inferred:
        inferred_by_value.setdefault(value.casefold(), []).append((column, value))
    for candidates in inferred_by_value.values():
        selected = _resolve_inferred_filter_column(question, candidates)
        if selected is None:
            continue
        column, value = selected
        if _is_overall_slice_request(question, column, value):
            continue
        filters.append(
            AnalysisFilterResponse(column=column, operator="=", value=value)
        )
    return tuple(filters)


def _resolve_inferred_filter_column(
    question: str,
    candidates: list[tuple[str, str]],
) -> tuple[str, str] | None:
    if len(candidates) == 1:
        return candidates[0]
    explicitly_named = [
        item for item in candidates if _explicit_column_name_mentioned(question, item[0])
    ]
    if len(explicitly_named) == 1:
        return explicitly_named[0]
    aligned = sorted(
        (
            (_entity_alignment_score(question, column), column, value)
            for column, value in candidates
        ),
        reverse=True,
    )
    if aligned and aligned[0][0] > 0 and (
        len(aligned) == 1 or aligned[0][0] > aligned[1][0]
    ):
        _, column, value = aligned[0]
        return column, value
    return None


def _is_overall_slice_request(question: str, column: str, value: str) -> bool:
    folded = question.casefold()
    asks_for_overall = any(
        token in folded
        for token in ("总体", "全体", "全国", "全部", "overall", "grand total")
    )
    return bool(
        asks_for_overall
        and _explicit_column_name_mentioned(question, column)
        and _sample_value_mentioned(question, value)
    )


def _sample_value_mentioned(question: str, value: str) -> bool:
    """Match sampled categorical values without treating token fragments as filters."""

    folded_value = value.casefold().strip()
    if not folded_value:
        return False
    if re.search(r"[\u3400-\u9fff]", folded_value):
        return _normalize(folded_value) in _normalize(question)
    parts = tuple(part for part in re.split(r"[\s_.-]+", folded_value) if part)
    if not parts:
        return False
    pattern = r"(?<![a-z0-9])" + r"[\s_.-]+".join(
        re.escape(part) for part in parts
    ) + r"(?![a-z0-9])"
    return re.search(pattern, question.casefold()) is not None


def _explicit_filter_value(question: str, column: str) -> str | None:
    names = tuple(
        sorted(
            {column.casefold(), _column_leaf(column).casefold()},
            key=len,
            reverse=True,
        )
    )
    for name in names:
        if not name:
            continue
        matched = re.search(
            rf"(?<![\w]){re.escape(name)}\s*(?:==|=|:|：|为|是)\s*"
            r"['\"]?([^'\"\s,，;；。)）]+)",
            question,
            flags=re.IGNORECASE,
        )
        if matched:
            return matched.group(1).strip()
    return None


def _explicit_filter_columns(
    question: str,
    categorical_columns: tuple[str, ...],
) -> tuple[str, ...]:
    folded = question.casefold()
    matched: list[str] = []
    for column in categorical_columns:
        names = {column.casefold(), _column_leaf(column).casefold()}
        if any(
            re.search(
                rf"(?<![\w]){re.escape(name)}\s*(?:==|=|!=|<>|:|：|为|是)",
                folded,
            )
            for name in names
            if name
        ):
            matched.append(column)
    return tuple(matched)


def _negated_dimension_columns(
    question: str,
    categorical_columns: tuple[str, ...],
) -> set[str]:
    spans = tuple(match.span() for match in _NEGATED_GROUPING_RE.finditer(question))
    if not spans:
        return set()
    excluded: set[str] = set()
    for column in categorical_columns:
        for start, end in _column_reference_spans(question, column):
            if any(span_start <= start and end <= span_end for span_start, span_end in spans):
                excluded.add(column)
                break
    return excluded


def negated_grouping_columns(
    question: str,
    columns: tuple[str, ...],
) -> tuple[str, ...]:
    """Return columns explicitly forbidden as grouping dimensions by the user."""

    return tuple(
        column for column in columns if column in _negated_dimension_columns(question, columns)
    )


def _column_reference_spans(question: str, column: str) -> tuple[tuple[int, int], ...]:
    names = tuple(
        sorted(
            {column.casefold(), _column_leaf(column).casefold()},
            key=len,
            reverse=True,
        )
    )
    spans: list[tuple[int, int]] = []
    folded = question.casefold()
    for name in names:
        if not name:
            continue
        if re.search(r"[\u3400-\u9fff]", name):
            spans.extend(match.span() for match in re.finditer(re.escape(name), folded))
            continue
        parts = tuple(part for part in re.split(r"[\s_.-]+", name) if part)
        if not parts:
            continue
        pattern = r"(?<![a-z0-9])" + r"[\s_.-]+".join(
            re.escape(part) for part in parts
        ) + r"(?![a-z0-9])"
        spans.extend(match.span() for match in re.finditer(pattern, folded))
    return tuple(dict.fromkeys(spans))


def strip_negated_clauses(question: str) -> str:
    """Remove explicitly rejected clauses before intent-type keyword matching."""

    return _NEGATED_CLAUSE_RE.sub(" ", question)


def _column_mention_score(
    question: str,
    column: str,
    *,
    aliases: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
) -> int:
    normalized_question = _normalize(question)
    normalized_column = _normalize(column)
    if normalized_column and normalized_column in normalized_question:
        return 100 + len(normalized_column)
    score = 0
    folded_column = column.casefold()
    for column_tokens, concept_aliases in aliases:
        if not any(token in folded_column for token in column_tokens):
            continue
        matched = max(
            (
                len(_normalize(alias))
                for alias in concept_aliases
                if _normalize(alias) in normalized_question
            ),
            default=0,
        )
        if matched:
            score = max(score, 20 + matched)
    return score + _entity_alignment_score(question, column)


def _entity_alignment_score(question: str, column: str) -> int:
    normalized_question = _normalize(question)
    normalized_column = _normalize(column)
    for question_tokens, matching_tokens, conflicting_tokens in _ENTITY_ALIASES:
        if not any(_normalize(token) in normalized_question for token in question_tokens):
            continue
        if any(_normalize(token) in normalized_column for token in matching_tokens):
            return 60
        if any(_normalize(token) in normalized_column for token in conflicting_tokens):
            return -60
    return 0


def _count_entity_column(question: str, columns: tuple[str, ...]) -> str | None:
    folded = question.casefold()
    entity_tokens = (
        (("订单", "order"), ("order_id", "orderid", "订单编号", "订单id")),
        (("客户", "用户", "customer", "user"), ("customer_id", "userid", "客户编号", "用户id")),
        (("商品", "产品", "product"), ("product_id", "productid", "商品编号", "产品id")),
    )
    for question_tokens, column_tokens in entity_tokens:
        if not any(token in folded for token in question_tokens):
            continue
        match = next(
            (
                column
                for column in columns
                if any(token in _normalize(column) for token in map(_normalize, column_tokens))
            ),
            None,
        )
        if match:
            return match
    return next((column for column in columns if _looks_like_identifier(column)), None)


def _count_alias(column: str | None) -> str:
    if not column:
        return "row_count"
    base = _safe_alias(column)
    if base.endswith("_id"):
        base = base[:-3]
    return f"{base}_count"


def _deduplicate_aggregations(
    aggregations: list[AnalysisAggregationResponse],
) -> list[AnalysisAggregationResponse]:
    seen: set[tuple[str, str | None]] = set()
    result: list[AnalysisAggregationResponse] = []
    for item in aggregations:
        key = (item.operation, item.column)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _looks_like_identifier(column: str) -> bool:
    folded = column.casefold().replace("-", "_")
    return folded.endswith("_id") or any(token in folded for token in _IDENTIFIER_TOKENS)


def _safe_alias(value: str) -> str:
    alias = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return alias or "metric"


def _normalize(value: str) -> str:
    return _NON_WORD_RE.sub("", value.casefold()).replace("_", "")
