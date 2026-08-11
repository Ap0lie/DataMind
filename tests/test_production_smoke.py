from __future__ import annotations

import pytest

from scripts.production_smoke import _default_origin


@pytest.mark.unit
def test_default_origin_matches_first_compose_public_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "DATAMIND_PUBLIC_ORIGIN",
        "http://127.0.0.1:5173, https://example.test",
    )

    assert _default_origin() == "http://127.0.0.1:5173"


@pytest.mark.unit
def test_default_origin_falls_back_when_configuration_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATAMIND_PUBLIC_ORIGIN", " , ")

    assert _default_origin() == "https://localhost"
