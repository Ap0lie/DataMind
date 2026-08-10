import pandas as pd

from app.analysis.text_analysis import run_text_analysis_toolbox


def test_text_analysis_toolbox_profiles_review_sentiment_data() -> None:
    result = run_text_analysis_toolbox(
        pd.DataFrame(
            [
                {
                    "review": "A warm movie with excellent acting and beautiful pacing.",
                    "sentiment": "positive",
                },
                {
                    "review": "Terrible pacing, weak story, and boring scenes.",
                    "sentiment": "negative",
                },
                {
                    "review": "Excellent story and excellent acting.",
                    "sentiment": "positive",
                },
            ]
        ),
        question="比较正面和负面评论的关键词与长度差异",
    )

    assert len(result) == 1
    analysis = result[0]
    assert analysis.text_column == "review"
    assert analysis.group_column == "sentiment"
    assert analysis.summary["non_empty_count"] == 3
    assert analysis.summary["groups"][0]["group"] == "positive"
    assert any(item["keyword"] == "excellent" for item in analysis.summary["top_keywords"])
    assert {chart.chart_type for chart in analysis.charts} >= {"histogram", "bar"}


def test_negated_review_table_does_not_trigger_text_analysis_for_identifier_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "customer_id": "0123456789abcdef0123456789abcdef",
                "payment_type": "credit_card",
                "payment_value": 120.0,
            },
            {
                "customer_id": "fedcba9876543210fedcba9876543210",
                "payment_type": "boleto",
                "payment_value": 80.0,
            },
        ]
    )

    result = run_text_analysis_toolbox(
        frame,
        question=(
            "统计 delivered 支付总额，不要使用 reviews，也不要按 payment_type 分组。"
        ),
    )

    assert result == ()


def test_explicit_text_analysis_respects_negated_grouping_column() -> None:
    frame = pd.DataFrame(
        [
            {"review": "great delivery and packaging", "payment_type": "card", "customer_state": "SP"},
            {"review": "late delivery and damaged box", "payment_type": "cash", "customer_state": "RJ"},
            {"review": "great product quality", "payment_type": "card", "customer_state": "SP"},
        ]
    )

    result = run_text_analysis_toolbox(
        frame,
        question="分析评论文本，但不要按 payment_type 分组。",
    )

    assert len(result) == 1
    assert result[0].group_column == "customer_state"
    assert all("payment_type" not in chart.title for chart in result[0].charts)
