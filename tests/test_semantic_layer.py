from __future__ import annotations

import pytest

from app.semantic.download_model import missing_model_files
from app.semantic.dsl import SemanticDslError, compile_expression, quote_identifier
from app.semantic.embedding import MockEmbeddingProvider
from app.semantic.ranking import SemanticCandidateRanker
from app.semantic.service import SemanticLayerService, fit_pava, validate_semantic_sql
from app.storage.dataset_store import DatasetStoreRepository

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("weight_name", ["model.safetensors", "pytorch_model.bin"])
def test_semantic_model_verifier_accepts_supported_weight_formats(tmp_path, weight_name) -> None:
    for name in ("config.json", "tokenizer.json", weight_name):
        (tmp_path / name).write_text("test", encoding="utf-8")

    assert missing_model_files(tmp_path) == ()


def test_metric_dsl_compiles_safe_division_and_rejects_cycles() -> None:
    metrics = {
        "revenue": {"formula": {"op": "sum", "expr": {"op": "field", "entity": "orders", "field": "revenue"}}},
        "margin": {"formula": {"op": "divide", "left": {"op": "metric_ref", "metric_id": "profit"}, "right": {"op": "metric_ref", "metric_id": "revenue"}}},
        "profit": {"formula": {"op": "metric_ref", "metric_id": "margin"}},
    }
    compiled = compile_expression(
        {"op": "divide", "left": {"op": "literal", "value": 10}, "right": {"op": "literal", "value": 2}}
    )
    assert "NULLIF" in compiled.sql
    with pytest.raises(SemanticDslError, match="cycle"):
        compile_expression(metrics["margin"]["formula"], metric_definitions=metrics, stack=("margin",))


def test_semantic_model_draft_publish_plan_and_execute(tmp_path, monkeypatch) -> None:
    repository = DatasetStoreRepository(str(tmp_path / "datasets"), user_id="semantic-user")
    dataset = repository.create_dataset(name="orders", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"region": "North", "revenue": 10}, {"region": "South", "revenue": 20}],
    )
    repository.save_column_metadata(
        dataset_id=dataset.id,
        columns=[
            {"column_name": "region", "role": "dimension", "inferred_type": "text"},
            {"column_name": "revenue", "role": "metric", "inferred_type": "number"},
        ],
    )
    full_read = repository.read_analysis_records

    def fail_on_full_read(_dataset_id):
        raise AssertionError("semantic schema inspection must use a bounded preview")

    monkeypatch.setattr(repository, "read_analysis_records", fail_on_full_read)
    service = SemanticLayerService(repository)
    draft = service.create_draft(scope_type="dataset", scope_id=dataset.id, name="Sales")
    validation = service.validate(draft)
    assert validation["valid"], validation
    published = service.publish(draft["id"])
    assert published["status"] == "published"
    with pytest.raises(ValueError, match="immutable"):
        service.update_draft(published["id"], revision=published["revision"], name=None, definition=published["definition"])

    decision = service.create_planner_decision(dataset_id=dataset.id, dataset_group_id=None, question="按 region 分析 revenue")
    assert decision["semantic_source"] == "published"
    assert decision["semantic_plan"]["metric_ids"]
    monkeypatch.setattr(repository, "read_analysis_records", full_read)
    result = service.execute_semantic_plan(decision)
    assert len(result["rows"]) == 2
    assert "SUM" in result["sql"]


def test_semantic_sql_ast_rejects_external_and_unknown_join() -> None:
    relationship = frozenset((("orders", "customer_id"), ("customers", "customer_id")))
    ok, _ = validate_semantic_sql(
        "SELECT orders.customer_id, COUNT(*) FROM orders LEFT JOIN customers ON orders.customer_id = customers.customer_id GROUP BY orders.customer_id",
        allowed_tables={"orders", "customers"}, allowed_relationships={relationship},
    )
    assert ok
    external, _ = validate_semantic_sql("SELECT * FROM read_csv_auto('x.csv')", allowed_tables={"orders"}, allowed_relationships=set())
    assert not external
    invented, _ = validate_semantic_sql(
        "SELECT * FROM orders JOIN customers ON orders.region = customers.region",
        allowed_tables={"orders", "customers"}, allowed_relationships={relationship},
    )
    assert not invented


def test_semantic_execution_supports_delimited_list_relationship(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path / "datasets"), user_id="semantic-user")
    behavior = repository.create_dataset(name="behavior", source_type="txt", source_metadata={})
    products = repository.create_dataset(name="products", source_type="txt", source_metadata={})
    repository.append_raw_records(
        dataset_id=behavior.id,
        records=[
            {"candidate_wid_list": "P1_P2"},
            {"candidate_wid_list": "P2_P3"},
        ],
    )
    repository.append_raw_records(
        dataset_id=products.id,
        records=[
            {"wid": "P1", "price": 10},
            {"wid": "P2", "price": 20},
            {"wid": "P3", "price": 30},
        ],
    )
    repository.save_column_metadata(
        dataset_id=products.id,
        columns=[
            {"column_name": "wid", "role": "id", "inferred_type": "text"},
            {"column_name": "price", "role": "metric", "inferred_type": "number"},
        ],
    )
    group = repository.create_dataset_group(
        name="list relationship",
        dataset_ids=(behavior.id, products.id),
    )
    repository.update_dataset_group_relationships(
        group_id=group.id,
        relationships=(
            {
                "left_dataset_id": str(behavior.id),
                "right_dataset_id": str(products.id),
                "left_column": "candidate_wid_list",
                "right_column": "wid",
                "left_value_mode": "delimited",
                "right_value_mode": "scalar",
                "left_delimiter": "_",
                "join_type": "left",
                "relationship_type": "many_to_one",
            },
        ),
    )
    service = SemanticLayerService(repository)
    draft = service.create_draft(
        scope_type="dataset_group",
        scope_id=group.id,
        name="JDsearch semantic model",
    )
    published = service.publish(draft["id"])
    metric = next(item for item in published["definition"]["metrics"] if item["name"] == "price")

    result = service.execute_semantic_plan(
        {
            "semantic_model_id": str(published["id"]),
            "semantic_plan": {"metric_ids": [metric["id"]], "dimension_ids": []},
        }
    )

    assert "list_contains" in result["sql"]
    assert result["rows"] == [{metric["id"]: 80.0}]


def test_pava_returns_monotonic_calibration() -> None:
    fitted = fit_pava([(0.2, 1), (0.4, 0), (0.6, 1), (0.8, 1)])
    probabilities = [item[1] for item in fitted]
    assert probabilities == sorted(probabilities)


def test_chinese_dsl_v2_uses_stable_ids_and_quoted_source_columns(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path / "datasets"), user_id="中文用户")
    dataset = repository.create_dataset(name="订单明细（中文）", source_type="csv", source_metadata={})
    repository.append_raw_records(
        dataset_id=dataset.id,
        records=[{"客户\"地区": "华北", "销售额（元）": 10, "利润/金额": 3}],
    )
    repository.save_column_metadata(
        dataset_id=dataset.id,
        columns=[
            {"column_name": "客户\"地区", "role": "dimension", "inferred_type": "text"},
            {"column_name": "销售额（元）", "role": "metric", "inferred_type": "number"},
            {"column_name": "利润/金额", "role": "metric", "inferred_type": "number"},
        ],
    )
    service = SemanticLayerService(repository, embedding_provider=MockEmbeddingProvider())
    draft = service.create_draft(scope_type="dataset", scope_id=dataset.id, name="中文销售模型")
    definition = draft["definition"]
    assert definition["definition_schema_version"] == 2
    assert len({item["id"] for item in definition["metrics"]}) == 2
    assert all(item["formula"]["expr"].get("field_id") for item in definition["metrics"])
    chinese_validation = service.validate(draft)
    assert chinese_validation["valid"], chinese_validation
    service.publish(draft["id"])
    decision = service.create_planner_decision(dataset_id=dataset.id, dataset_group_id=None, question="按客户地区分析销售额")
    result = service.execute_semantic_plan(decision)
    assert '"客户""地区"' in result["sql"]
    assert '"销售额（元）"' in result["sql"]
    assert quote_identifier("利润/金额") == '"利润/金额"'


def test_chinese_candidate_benchmark_reaches_top_one_target() -> None:
    ranker = SemanticCandidateRanker(MockEmbeddingProvider())
    metrics = [
        {"id": "sales", "name": "销售额", "aliases": ["营收", "收入", "GMV"], "formula": {"op": "literal", "value": 1}},
        {"id": "profit", "name": "利润", "aliases": ["收益", "毛利"], "formula": {"op": "literal", "value": 1}},
    ]
    dimensions = [
        {"id": "region", "name": "地区", "aliases": ["区域", "大区", "地域"], "type": "dimension"},
        {"id": "product", "name": "产品", "aliases": ["商品", "品类"], "type": "dimension"},
    ]
    metric_questions = [f"按地区查看{term}{suffix}" for term in ("销售额", "营收", "收入", "GMV") for suffix in ("", "趋势", "合计", "表现", "同比")]
    dimension_questions = [f"按{term}分析销售额{suffix}" for term in ("地区", "区域", "大区", "地域") for suffix in ("", "趋势", "排名", "表现", "同比")]
    questions = (metric_questions * 3)[:50] + (dimension_questions * 3)[:50]
    correct = 0
    for index, question in enumerate(questions):
        ranked = ranker.rank(question, metrics if index < 50 else dimensions, expected_type="metric" if index < 50 else "dimension")
        correct += int(ranked[0].item["id"] == ("sales" if index < 50 else "region"))
    assert len(questions) == 100
    assert correct / len(questions) >= 0.9
