from __future__ import annotations

from typing import Protocol

import pandas as pd

from app.schemas.analysis import PythonAnalysisResponse


class PythonAnalysisExecutor(Protocol):
    def __call__(self, code: str, dataframe: pd.DataFrame) -> PythonAnalysisResponse: ...
