from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from app.analysis.python_sandbox import _python_result_from_payload, _validate_generated_code
from app.schemas.analysis import PythonAnalysisResponse


class ScriptedPythonExecutor:
    def __init__(self, outcomes: Iterable[PythonAnalysisResponse | Exception] = ()) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[str, pd.DataFrame]] = []

    def __call__(self, code: str, dataframe: pd.DataFrame) -> PythonAnalysisResponse:
        self.calls.append((code, dataframe.copy(deep=True)))
        if self._outcomes:
            outcome = self._outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return _execute_validated_code_in_memory(code, dataframe)


def _execute_validated_code_in_memory(
    code: str,
    dataframe: pd.DataFrame,
) -> PythonAnalysisResponse:
    tree = ast.parse(code, mode="exec")
    _validate_generated_code(tree)
    namespace: dict[str, Any] = {"pd": pd, "np": np}
    exec(compile(tree, "<datamind-test-python>", "exec"), namespace, namespace)
    analyze = namespace.get("analyze")
    if not callable(analyze):
        raise RuntimeError("Generated code must define analyze(df).")
    payload = analyze(dataframe.copy(deep=True))
    if not isinstance(payload, dict):
        raise RuntimeError("analyze(df) must return a dict.")
    return _python_result_from_payload(payload)
