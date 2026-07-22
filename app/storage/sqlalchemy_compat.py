from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=4)
def _engine(database_url: str) -> Any:
    from sqlalchemy import create_engine

    return create_engine(database_url, pool_pre_ping=True, future=True)


class RowAdapter:
    def __init__(self, row: Any) -> None:
        self._values = tuple(row)
        self._mapping = dict(row._mapping)

    def __getitem__(self, key: str | int) -> Any:
        return self._values[key] if isinstance(key, int) else self._mapping[key]

    def keys(self) -> Sequence[str]:
        return tuple(self._mapping.keys())


class ResultAdapter:
    def __init__(self, result: Any | None = None) -> None:
        self._result = result
        self.rowcount = int(getattr(result, "rowcount", 0) or 0) if result is not None else 0

    def fetchone(self) -> RowAdapter | None:
        if self._result is None:
            return None
        row = self._result.fetchone()
        return RowAdapter(row) if row is not None else None

    def fetchall(self) -> list[RowAdapter]:
        if self._result is None:
            return []
        return [RowAdapter(row) for row in self._result.fetchall()]

    def __iter__(self) -> Iterator[RowAdapter]:
        return iter(self.fetchall())


class SQLAlchemyConnectionAdapter:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._connection: Any | None = None
        self._transaction: Any | None = None

    def __enter__(self) -> SQLAlchemyConnectionAdapter:
        self._connection = _engine(self._database_url).connect()
        self._transaction = self._connection.begin()
        return self

    @property
    def dialect_name(self) -> str:
        return str(_engine(self._database_url).dialect.name)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._transaction is not None:
            self._transaction.rollback() if exc_type else self._transaction.commit()
        if self._connection is not None:
            self._connection.close()

    def execute(self, statement: str, parameters: Sequence[Any] | None = None) -> ResultAdapter:
        if self._connection is None:
            raise RuntimeError("Database connection is not open.")
        if statement.strip().upper() == "BEGIN IMMEDIATE":
            return ResultAdapter()
        sql, bindings = _bind_qmark(statement, parameters or ())
        from sqlalchemy import text

        return ResultAdapter(self._connection.execute(text(sql), bindings))

    def executemany(self, statement: str, parameters: Sequence[Sequence[Any]]) -> ResultAdapter:
        if self._connection is None:
            raise RuntimeError("Database connection is not open.")
        rows = list(parameters)
        if not rows:
            return ResultAdapter()
        sql, _ = _bind_qmark(statement, rows[0])
        from sqlalchemy import text

        return ResultAdapter(
            self._connection.execute(text(sql), [_bindings(row) for row in rows])
        )

    def executescript(self, script: str) -> None:
        for statement in _split_sql_script(script):
            self.execute(statement)

    def column_names(self, table: str) -> set[str]:
        from sqlalchemy import inspect

        target = self._connection if self._connection is not None else _engine(self._database_url)
        return {str(item["name"]) for item in inspect(target).get_columns(table)}


def _bind_qmark(statement: str, parameters: Sequence[Any]) -> tuple[str, dict[str, Any]]:
    output: list[str] = []
    index = 0
    quote: str | None = None
    for character in statement:
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
        if character == "?" and quote is None:
            output.append(f":p{index}")
            index += 1
        else:
            output.append(character)
    if index != len(parameters):
        raise ValueError(
            f"SQL parameter count mismatch: expected {index}, received {len(parameters)}."
        )
    return "".join(output), _bindings(parameters)


def _bindings(parameters: Sequence[Any]) -> dict[str, Any]:
    return {f"p{index}": value for index, value in enumerate(parameters)}


def _split_sql_script(script: str) -> tuple[str, ...]:
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for character in script:
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
        if character == ";" and quote is None:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
        else:
            current.append(character)
    trailing = "".join(current).strip()
    if trailing:
        statements.append(trailing)
    return tuple(statements)
