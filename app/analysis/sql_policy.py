from __future__ import annotations

from sqlglot import exp, parse

_FORBIDDEN_STATEMENTS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Drop,
    exp.Command,
    exp.Copy,
    exp.Attach,
    exp.Pragma,
)
_FORBIDDEN_FUNCTIONS = {
    "glob",
    "httpfs",
    "postgres_scan",
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "sqlite_scan",
}


def validate_scoped_dataset_select(sql: str) -> str:
    """Validate one read-only query over the prepared ``dataset`` relation."""

    normalized = sql.strip().removesuffix(";").strip()
    if not normalized:
        raise ValueError("SQL is required.")
    statements = parse(normalized, read="duckdb")
    if len(statements) != 1:
        raise ValueError("Only one SQL statement is allowed.")
    if not isinstance(statements[0], (exp.Select, exp.Union)):
        raise ValueError("Only SELECT statements are allowed.")
    root = statements[0]
    if any(root.find(kind) is not None for kind in _FORBIDDEN_STATEMENTS):
        raise ValueError("SQL contains a forbidden statement.")
    if any(
        str(function.sql_name()).lower() in _FORBIDDEN_FUNCTIONS
        for function in root.find_all(exp.Func)
    ):
        raise ValueError("External table functions are forbidden.")

    tables = tuple(root.find_all(exp.Table))
    ctes = {str(cte.alias_or_name).casefold() for cte in root.find_all(exp.CTE)}
    invalid_tables = [
        table.sql()
        for table in tables
        if _qualified(table)
        or (
            str(table.name).casefold() != "dataset"
            and str(table.name).casefold() not in ctes
        )
    ]
    if invalid_tables:
        raise ValueError("SQL may only read the job-scoped dataset table.")

    dataset_scans = sum(
        1
        for table in tables
        if str(table.name).casefold() == "dataset" and not _qualified(table)
    )
    if dataset_scans != 1:
        raise ValueError(
            "SQL must scan the prepared dataset exactly once; direct dataset self-joins "
            "duplicate fact rows. Aggregate in the single scan or use a source-grain tool."
        )
    for join in root.find_all(exp.Join):
        kind = str(join.args.get("kind") or "").upper()
        if kind in {"CROSS", "NATURAL"} or (
            join.args.get("on") is None and not join.args.get("using")
        ):
            raise ValueError("CROSS, NATURAL, and comma joins are forbidden.")
    return normalized


def has_unsafe_dataset_self_join(sql: str) -> bool:
    """Return whether SQL physically scans the prepared dataframe more than once."""

    try:
        statements = parse(sql, read="duckdb")
    except Exception:
        return False
    return any(
        sum(
            1
            for table in statement.find_all(exp.Table)
            if str(table.name).casefold() == "dataset" and not _qualified(table)
        )
        > 1
        for statement in statements
    )


def _qualified(table: exp.Table) -> bool:
    return bool(table.args.get("db") or table.args.get("catalog"))
