from __future__ import annotations


def resolve_column_reference(
    reference: str | None,
    available_columns: set[str] | tuple[str, ...],
) -> str | None:
    """Resolve a source-qualified field to the prepared dataframe column."""
    if not reference:
        return None
    value = reference.strip()
    columns = tuple(str(column) for column in available_columns)
    if value in columns:
        return value

    folded = value.casefold()
    exact_matches = [column for column in columns if column.casefold() == folded]
    if len(exact_matches) == 1:
        return exact_matches[0]

    leaf = value.rsplit("__", 1)[-1].casefold()
    direct_matches = [column for column in columns if column.casefold() == leaf]
    if len(direct_matches) == 1:
        return direct_matches[0]

    leaf_matches = [
        column
        for column in columns
        if column.rsplit("__", 1)[-1].casefold() == leaf
    ]
    return leaf_matches[0] if len(leaf_matches) == 1 else None
