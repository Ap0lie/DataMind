from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_alembic_revision_ids_fit_default_version_column() -> None:
    versions = Path("migrations/versions")
    revisions: dict[str, Path] = {}
    for path in versions.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        revision = _literal_assignment(tree, "revision")
        assert revision, f"{path} does not declare a literal revision id"
        assert len(revision) <= 32, (
            f"{path} revision id exceeds Alembic's default VARCHAR(32): {revision}"
        )
        assert revision not in revisions, f"duplicate revision id: {revision}"
        revisions[revision] = path


def _literal_assignment(tree: ast.Module, name: str) -> str | None:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    return None
