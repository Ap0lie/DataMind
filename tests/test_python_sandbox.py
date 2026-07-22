import pandas as pd
import pytest

from app.analysis.python_sandbox import (
    EXECUTION_TIMEOUT_SECONDS,
    GeneratedPythonSafetyError,
    PythonExecutionPolicy,
    run_generated_python_analysis,
)


def test_generated_python_analysis_runs_basic_dataframe_code() -> None:
    code = (
        "def analyze(df):\n"
        "    total = float(df['sales'].sum())\n"
        "    grouped = df.groupby('region', dropna=True)['sales'].sum().reset_index()\n"
        "    return {\n"
        "        'statistics': {'total_sales': total},\n"
        "        'insights': [f'Total sales are {total:.0f}.'],\n"
        "        'charts': [{\n"
        "            'title': 'Sales by region',\n"
        "            'chart_type': 'bar',\n"
        "            'spec': {'x': 'region', 'y': 'sales'},\n"
        "            'data': grouped.to_dict(orient='records'),\n"
        "        }],\n"
        "    }\n"
    )

    result = run_generated_python_analysis(
        code,
        pd.DataFrame(
            [
                {"region": "North", "sales": 100},
                {"region": "South", "sales": 180},
            ]
        ),
    )

    assert result.statistics["total_sales"] == 280.0
    assert result.insights == ("Total sales are 280.",)
    assert result.charts[0].chart_type == "bar"


def test_generated_python_analysis_allows_safe_imports_and_for_loops() -> None:
    code = (
        "import pandas as pd\n"
        "from collections import Counter\n"
        "def analyze(df):\n"
        "    counter = Counter()\n"
        "    for value in df['sentiment'].dropna().astype(str).tolist():\n"
        "        counter[value] += 1\n"
        "    rows = [{'sentiment': key, 'count': int(count)} for key, count in counter.items()]\n"
        "    total = int(pd.to_numeric(df['score'], errors='coerce').fillna(0).sum())\n"
        "    return {\n"
        "        'statistics': {'total_score': total, 'sentiment_count': dict(counter)},\n"
        "        'insights': [f'Found {len(counter)} sentiment groups.'],\n"
        "        'charts': [{\n"
        "            'title': 'Sentiment distribution',\n"
        "            'chart_type': 'bar',\n"
        "            'spec': {'x': 'sentiment', 'y': 'count'},\n"
        "            'data': rows,\n"
        "        }],\n"
        "    }\n"
    )

    result = run_generated_python_analysis(
        code,
        pd.DataFrame(
            [
                {"sentiment": "positive", "score": 5},
                {"sentiment": "negative", "score": 2},
                {"sentiment": "positive", "score": 4},
            ]
        ),
    )

    assert result.statistics["total_score"] == 11
    assert result.statistics["sentiment_count"] == {"positive": 2, "negative": 1}
    assert result.charts[0].data[0]["sentiment"] == "positive"


def test_generated_python_analysis_prebins_large_histogram_payload_before_stdout_limit() -> None:
    code = (
        "def analyze(df):\n"
        "    rows = df[['weight']].to_dict(orient='records')\n"
        "    return {\n"
        "        'statistics': {'nested': {'raw': rows}},\n"
        "        'insights': ['已生成重量分布。'],\n"
        "        'charts': [{\n"
        "            'title': '重量分布',\n"
        "            'chart_type': 'histogram',\n"
        "            'spec': {'x': 'weight'},\n"
        "            'data': rows,\n"
        "        }],\n"
        "    }\n"
    )
    dataframe = pd.DataFrame({"weight": list(range(5000))})

    result = run_generated_python_analysis(code, dataframe)

    assert len(result.charts[0].data) == 30
    assert set(result.charts[0].data[0]) == {"bin_start", "bin_end", "count"}
    assert sum(int(row["count"]) for row in result.charts[0].data) == 5000
    assert result.statistics["nested"]["raw"] == "list"


def test_generated_python_analysis_summarizes_large_box_plot_payload() -> None:
    code = (
        "def analyze(df):\n"
        "    rows = df[['category', 'amount']].to_dict(orient='records')\n"
        "    return {\n"
        "        'statistics': {},\n"
        "        'insights': ['已生成箱线图摘要。'],\n"
        "        'charts': [{\n"
        "            'title': '金额箱线图',\n"
        "            'chart_type': 'box_plot',\n"
        "            'spec': {'x': 'category', 'y': 'amount'},\n"
        "            'data': rows,\n"
        "        }],\n"
        "    }\n"
    )
    dataframe = pd.DataFrame(
        {
            "category": ["A"] * 700 + ["B"] * 700,
            "amount": list(range(700)) + list(range(100, 800)),
        }
    )

    result = run_generated_python_analysis(code, dataframe)

    rows = list(result.charts[0].data)
    assert len(rows) == 2
    assert set(rows[0]) == {"group", "min", "q1", "median", "q3", "max", "count"}
    assert {row["group"] for row in rows} == {"A", "B"}
    assert sum(int(row["count"]) for row in rows) == 1400


def test_generated_python_analysis_blocks_dangerous_imports() -> None:
    code = (
        "import os\n"
        "def analyze(df):\n"
        "    return {'statistics': {'cwd_length': len(os.getcwd())}, 'insights': [], 'charts': []}\n"
    )

    with pytest.raises(GeneratedPythonSafetyError, match="cannot import os"):
        run_generated_python_analysis(code, pd.DataFrame([{"sales": 100}]))


def test_generated_python_analysis_blocks_file_writes_but_keeps_while_loop_support() -> None:
    code = (
        "import pandas as pd\n"
        "def analyze(df):\n"
        "    i = 0\n"
        "    total = 0\n"
        "    while i < len(df):\n"
        "        total += int(df.iloc[i]['sales'])\n"
        "        i += 1\n"
        "    df.to_csv('result.csv', index=False)\n"
        "    return {'statistics': {'total': total}, 'insights': [], 'charts': []}\n"
    )

    with pytest.raises(GeneratedPythonSafetyError, match="cannot write files"):
        run_generated_python_analysis(
            code,
            pd.DataFrame([{"sales": 100}, {"sales": 180}]),
        )


def test_generated_python_analysis_blocks_open_file_access() -> None:
    code = (
        "def analyze(df):\n"
        "    with open('agent-output.txt', 'w') as handle:\n"
        "        handle.write(str(len(df)))\n"
        "    return {'statistics': {'rows': len(df)}, 'insights': [], 'charts': []}\n"
    )

    with pytest.raises(GeneratedPythonSafetyError, match="cannot call open"):
        run_generated_python_analysis(code, pd.DataFrame([{"sales": 100}]))

def test_generated_python_analysis_times_out() -> None:
    code = (
        "def analyze(df):\n"
        "    while True:\n"
        "        pass\n"
    )

    with pytest.raises(GeneratedPythonSafetyError, match="timed out"):
        run_generated_python_analysis(
            code,
            pd.DataFrame([{"sales": 100}]),
            policy=PythonExecutionPolicy(timeout_seconds=0.25),
        )


def test_python_execution_policy_keeps_production_timeout_default() -> None:
    assert PythonExecutionPolicy().timeout_seconds == EXECUTION_TIMEOUT_SECONDS == 8
