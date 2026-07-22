from __future__ import annotations

import pandas as pd

from app.analysis.python_sandbox import run_generated_python_analysis


def main() -> None:
    code = """
def analyze(df):
    return {
        "statistics": {"row_count": len(df)},
        "insights": ["runner-ok"],
        "charts": [],
    }
""".strip()
    result = run_generated_python_analysis(
        code,
        pd.DataFrame([{"value": 1}, {"value": 2}]),
    )
    if result.statistics.get("row_count") != 2:
        raise RuntimeError(f"Unexpected Python Runner result: {result}")
    print("python runner smoke passed")


if __name__ == "__main__":
    main()
