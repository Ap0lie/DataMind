from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from sqlglot import exp, parse

SQLResultScopeKind = Literal[
    "complete_query_result",
    "top_n_groups",
    "bottom_n_groups",
    "ranked_groups",
    "limited_groups",
    "limited_rows",
    "unknown",
]

SQLOrderDirection = Literal["ascending", "descending", "mixed"]


@dataclass(frozen=True)
class SQLResultScope:
    kind: SQLResultScopeKind
    grouped: bool
    ordered: bool
    order_direction: SQLOrderDirection | None
    limit: int | None
    returned_rows: int | None = None

    @property
    def is_partial(self) -> bool:
        return self.kind in {
            "top_n_groups",
            "bottom_n_groups",
            "ranked_groups",
            "limited_groups",
            "limited_rows",
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"is_partial": self.is_partial}


def analyze_sql_result_scope(
    sql: str | None,
    *,
    returned_rows: int | None = None,
) -> SQLResultScope:
    """Describe what the returned SQL rows represent."""

    try:
        statements = parse(str(sql or ""), read="duckdb")
    except Exception:
        statements = []
    query = next(
        (
            statement
            for statement in reversed(statements)
            if isinstance(statement, exp.Query)
        ),
        None,
    )
    if query is None:
        return SQLResultScope(
            kind="unknown",
            grouped=False,
            ordered=False,
            order_direction=None,
            limit=None,
            returned_rows=returned_rows,
        )

    grouped = query.args.get("group") is not None
    order = query.args.get("order")
    ordered = order is not None
    order_direction = _order_direction(order)
    limit = _literal_limit(query.args.get("limit"))
    if limit is not None and grouped and order_direction == "descending":
        kind: SQLResultScopeKind = "top_n_groups"
    elif limit is not None and grouped and order_direction == "ascending":
        kind = "bottom_n_groups"
    elif limit is not None and grouped and ordered:
        kind = "ranked_groups"
    elif limit is not None and grouped:
        kind = "limited_groups"
    elif limit is not None:
        kind = "limited_rows"
    else:
        kind = "complete_query_result"
    return SQLResultScope(
        kind=kind,
        grouped=grouped,
        ordered=ordered,
        order_direction=order_direction,
        limit=limit,
        returned_rows=returned_rows,
    )


def _literal_limit(value: Any) -> int | None:
    expression = value.expression if isinstance(value, exp.Limit) else None
    if not isinstance(expression, exp.Literal) or expression.is_string:
        return None
    try:
        parsed = int(expression.this)
    except (TypeError, ValueError):
        return None
    return max(0, parsed)


def _order_direction(value: Any) -> SQLOrderDirection | None:
    expressions = value.expressions if isinstance(value, exp.Order) else ()
    directions = {
        "descending" if bool(item.args.get("desc")) else "ascending"
        for item in expressions
        if isinstance(item, exp.Ordered)
    }
    if not directions:
        return None
    if len(directions) == 1:
        return next(iter(directions))
    return "mixed"
