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
