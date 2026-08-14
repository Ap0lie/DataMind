from __future__ import annotations

from copy import deepcopy

import pytest

from app.semantic.download_model import missing_model_files
from app.semantic.dsl import SemanticDslError, compile_expression, quote_identifier
from app.semantic.embedding import MockEmbeddingProvider
from app.semantic.ranking import SemanticCandidateRanker
from app.semantic.relationship_graph import plan_relationship_path
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
    confirmed = repository.set_planner_decision_confirmation(
        decision["id"],
        requires_confirmation=True,
        confirmed=True,
    )
    assert bool(confirmed["requires_confirmation"])
    assert bool(confirmed["confirmed"])
    monkeypatch.setattr(repository, "read_analysis_records", full_read)
    result = service.execute_semantic_plan(decision)
    assert len(result["rows"]) == 2
    assert "SUM" in result["sql"]


def test_olist_group_auto_draft_infers_roles_and_unique_semantic_names(tmp_path, monkeypatch) -> None:
    repository = DatasetStoreRepository(str(tmp_path / "datasets"), user_id="semantic-user")
    customer_id = "a" * 32
    order_id = "b" * 32
    product_id = "c" * 32
    seller_id = "d" * 32
    sources = {
        "olist_customers_dataset": {
            "customer_id": customer_id,
            "customer_unique_id": "e" * 32,
            "customer_state": "SP",
            "customer_zip_code_prefix": 1000,
        },
        "olist_orders_dataset": {
            "order_id": order_id,
            "customer_id": customer_id,
            "order_status": "delivered",
            "order_purchase_timestamp": "2018-01-01 10:30:00",
            "order_total": 100.0,
        },
        "olist_order_payments_dataset": {
            "order_id": order_id,
            "payment_sequential": 1,
            "payment_type": "credit_card",
            "payment_installments": 2,
            "payment_value": 100.5,
        },
        "olist_order_items_dataset": {
            "order_id": order_id,
            "order_item_id": 1,
            "product_id": product_id,
            "seller_id": seller_id,
            "shipping_limit_date": "2018-01-02 09:00:00",
            "price": 90.0,
            "freight_value": 10.5,
        },
        "olist_products_dataset": {
            "product_id": product_id,
            "product_category_name": "beleza_saude",
            "product_photos_qty": 5,
            "product_weight_g": 500,
        },
        "product_category_name_translation": {
            "product_category_name": "beleza_saude",
            "product_category_name_english": "health_beauty",
        },
        "olist_sellers_dataset": {
            "seller_id": seller_id,
            "seller_state": "SP",
            "seller_zip_code_prefix": 1000,
        },
        "olist_order_reviews_dataset": {
            "review_id": "f" * 32,
            "order_id": order_id,
            "review_score": 5,
            "review_comment_message": "excellent",
        },
    }
    datasets = {}
    for name, record in sources.items():
        dataset = repository.create_dataset(name=name, source_type="csv", source_metadata={})
        records = [record]
        if name == "olist_orders_dataset":
            records.append(
                record
                | {
                    "order_id": "2" * 32,
                    "order_purchase_timestamp": "2018-01-03 11:00:00",
                    "order_total": 200.0,
                }
            )
        elif name == "olist_order_payments_dataset":
            records.append(record | {"payment_sequential": 2, "payment_value": 50.0})
        elif name == "olist_order_items_dataset":
            records.append(
                record
                | {"order_item_id": 2, "price": 20.0, "freight_value": 5.0}
            )
        elif name == "olist_order_reviews_dataset":
            records.append(record | {"review_id": "1" * 32, "review_score": 4})
        repository.append_raw_records(dataset_id=dataset.id, records=records)
        datasets[name] = dataset

    group = repository.create_dataset_group(
        name="Brazilian ecommerce",
        dataset_ids=tuple(dataset.id for dataset in datasets.values()),
    )
    relationships = (
        ("olist_order_payments_dataset", "order_id", "olist_orders_dataset", "order_id", "one_to_one"),
        ("olist_orders_dataset", "customer_id", "olist_customers_dataset", "customer_id", "many_to_one"),
        ("olist_orders_dataset", "order_id", "olist_order_items_dataset", "order_id", "many_to_one"),
        ("olist_orders_dataset", "order_id", "olist_order_reviews_dataset", "order_id", "many_to_one"),
        ("olist_order_items_dataset", "product_id", "olist_products_dataset", "product_id", "many_to_one"),
        ("olist_order_items_dataset", "seller_id", "olist_sellers_dataset", "seller_id", "many_to_one"),
        ("olist_products_dataset", "product_category_name", "product_category_name_translation", "product_category_name", "many_to_one"),
    )
    repository.update_dataset_group_relationships(
        group_id=group.id,
        relationships=tuple(
            {
                "left_dataset_id": str(datasets[left_name].id),
                "right_dataset_id": str(datasets[right_name].id),
                "left_column": left_column,
                "right_column": right_column,
                "join_type": "left",
                "relationship_type": cardinality,
            }
            for left_name, left_column, right_name, right_column, cardinality in relationships
        ),
    )

    full_read = repository.read_analysis_records

    def fail_on_full_read(_dataset_id):
        raise AssertionError("semantic draft generation must not materialize full datasets")

    monkeypatch.setattr(repository, "read_analysis_records", fail_on_full_read)

    service = SemanticLayerService(repository, embedding_provider=MockEmbeddingProvider())
    draft = service.create_draft(
        scope_type="dataset_group",
        scope_id=group.id,
        name="Olist semantic model",
    )
    definition = draft["definition"]
    validation = service.validate(draft)
    generated_fields = [
        field
        for entity in definition["entities"]
        for field in entity["fields"]
    ]

    assert validation["valid"], validation
    assert not any("Duplicate semantic name" in error for error in validation["errors"])
    relationship_ids = [item["id"] for item in definition["relationships"]]
    assert len(relationship_ids) == len(set(relationship_ids)) == 7
    cardinalities = {
        (
            next(
                entity["name"]
                for entity in definition["entities"]
                if entity["id"] == relationship["left_entity_id"]
            ),
            next(
                entity["name"]
                for entity in definition["entities"]
                if entity["id"] == relationship["right_entity_id"]
            ),
        ): relationship["cardinality"]
        for relationship in definition["relationships"]
    }
    assert cardinalities[("olist_order_payments_dataset", "olist_orders_dataset")] == "many_to_one"
    assert cardinalities[("olist_orders_dataset", "olist_order_items_dataset")] == "one_to_many"
    assert cardinalities[("olist_orders_dataset", "olist_order_reviews_dataset")] == "one_to_many"
    assert len(definition["metrics"]) >= 1
    metric_field_ids = {
        str(metric["formula"]["expr"]["field_id"])
        for metric in definition["metrics"]
    }
    metric_sources = {
        str(field["source_name"])
        for entity in definition["entities"]
        for field in entity["fields"]
        if str(field["field_id"]) in metric_field_ids
    }
    assert {"payment_value", "price", "freight_value"} <= metric_sources
    assert not any(source.endswith("_id") for source in metric_sources)
    assert all(
        field["role"] == "id"
        for field in generated_fields
        if str(field["source_name"]).endswith("_id")
    )
    purchase_timestamp = next(
        field
        for field in generated_fields
        if field["source_name"] == "order_purchase_timestamp"
    )
    assert purchase_timestamp["role"] == "date"
    assert purchase_timestamp["type"] == "datetime"
    non_additive_numeric_fields = {
        "payment_sequential",
        "payment_installments",
        "product_photos_qty",
        "product_weight_g",
        "review_score",
    }
    assert all(
        field["role"] == "dimension"
        for field in generated_fields
        if field["source_name"] in non_additive_numeric_fields
    )
    products_entity = next(
        entity
        for entity in definition["entities"]
        if entity["name"] == "olist_products_dataset"
    )
    assert products_entity["entity_type"] == "dimension"
    assert products_entity["primary_key"] is None
    product_category_names = [
        item["name"]
        for item in definition["dimensions"]
        if str(item["name"]).endswith(".product_category_name")
    ]
    assert len(product_category_names) == 2
    assert len(set(product_category_names)) == 2

    legacy_definition = deepcopy(definition)
    source_name_by_field_id = {
        str(field["field_id"]): str(field["source_name"])
        for entity in legacy_definition["entities"]
        for field in entity["fields"]
    }
    for dimension in legacy_definition["dimensions"]:
        if source_name_by_field_id.get(str(dimension["field_id"])) == "product_category_name":
            dimension["name"] = "product_category_name"
    legacy_identifier_dimensions = []
    for entity in legacy_definition["entities"]:
        for field in entity["fields"]:
            source_name = str(field["source_name"])
            if source_name not in {"customer_id", "order_id"}:
                continue
            legacy_identifier_dimensions.append(
                {
                    "id": f"legacy_dimension_{len(legacy_identifier_dimensions)}",
                    "name": source_name,
                    "aliases": [],
                    "entity_id": entity["id"],
                    "field_id": field["field_id"],
                    "type": "categorical",
                    "time_grains": [],
                }
            )
    legacy_definition["dimensions"].extend(legacy_identifier_dimensions)
    legacy_draft = service.update_draft(
        draft["id"],
        revision=draft["revision"],
        name="Olist legacy v1",
        definition=legacy_definition,
    )
    copied = service.create_draft(
        scope_type="dataset_group",
        scope_id=group.id,
        name="Olist semantic model v2",
        source_model_id=legacy_draft["id"],
    )
    copied_validation = service.validate(copied)
    copied_dimension_names = [
        str(item["name"])
        for item in copied["definition"]["dimensions"]
    ]

    assert copied_validation["valid"], copied_validation
    assert copied["definition"]["unresolved_bindings"] == []
    assert "customer_id" not in copied_dimension_names
    assert "order_id" not in copied_dimension_names
    assert len(
        [name for name in copied_dimension_names if name.endswith(".product_category_name")]
    ) == 2

    copied_definition = copied["definition"]
    copied_entities = {
        str(entity["id"]): entity for entity in copied_definition["entities"]
    }
    copied_field_sources = {
        (str(entity["id"]), str(field["field_id"])): str(field["source_name"])
        for entity in copied_definition["entities"]
        for field in entity["fields"]
    }

    def metric_source(metric):
        expression = metric["formula"]["expr"]
        return copied_field_sources[(str(expression["entity_id"]), str(expression["field_id"]))]

    def dimension_source(dimension):
        return copied_field_sources[(str(dimension["entity_id"]), str(dimension["field_id"]))]

    payment_metric = next(
        metric
        for metric in copied_definition["metrics"]
        if metric_source(metric) == "payment_value"
    )
    order_metric = next(
        metric
        for metric in copied_definition["metrics"]
        if metric_source(metric) == "order_total"
    )
    customer_state_dimension = next(
        dimension
        for dimension in copied_definition["dimensions"]
        if dimension_source(dimension) == "customer_state"
    )
    payment_type_dimension = next(
        dimension
        for dimension in copied_definition["dimensions"]
        if dimension_source(dimension) == "payment_type"
    )
    order_status_dimension = next(
        dimension
        for dimension in copied_definition["dimensions"]
        if dimension_source(dimension) == "order_status"
    )
    payment_entity_id = next(
        entity_id
        for entity_id, entity in copied_entities.items()
        if entity["name"] == "olist_order_payments_dataset"
    )
    reviews_entity_id = next(
        entity_id
        for entity_id, entity in copied_entities.items()
        if entity["name"] == "olist_order_reviews_dataset"
    )
    assert copied_entities[payment_entity_id]["entity_type"] == "fact"

    payment_plan = plan_relationship_path(
        copied_definition,
        metric_ids=(str(payment_metric["id"]),),
        dimension_ids=(str(customer_state_dimension["id"]),),
    )
    assert payment_plan["safe"], payment_plan
    assert payment_plan["metric_entity_ids"] == (payment_entity_id,)
    assert reviews_entity_id not in {
        str(step["from_entity_id"])
        for step in payment_plan["steps"]
    } | {
        str(step["to_entity_id"])
        for step in payment_plan["steps"]
    }

    reverse_plan = plan_relationship_path(
        copied_definition,
        metric_ids=(str(order_metric["id"]),),
        dimension_ids=(str(payment_type_dimension["id"]),),
    )
    assert not reverse_plan["safe"]
    assert any("One-to-many" in warning for warning in reverse_plan["warnings"])

    monkeypatch.setattr(repository, "read_analysis_records", full_read)
    published = service.publish(copied["id"])
    result = service.execute_semantic_plan(
        {
            "semantic_model_id": str(published["id"]),
            "semantic_plan": {
                "metric_ids": [payment_metric["id"]],
                "dimension_ids": [customer_state_dimension["id"]],
            },
        }
    )
    assert result["rows"] == [
        {
            customer_state_dimension["id"]: "SP",
            payment_metric["id"]: 150.5,
        }
    ]
    assert "review" not in result["sql"].lower()

    scoped_decision = service.create_planner_decision(
        dataset_id=datasets["olist_order_payments_dataset"].id,
        dataset_group_id=group.id,
        question=(
            "仅使用 customers、orders、order_payments 三张表，过滤 "
            "order_status=delivered，按 customer_state 统计 payment_value 总额；"
            "不要按 order_status 或 payment_type 分组。"
        ),
    )
    semantic_plan = scoped_decision["semantic_plan"]
    assert semantic_plan["dimension_ids"] == [customer_state_dimension["id"]]
    assert payment_type_dimension["id"] not in semantic_plan["dimension_ids"]
    assert semantic_plan["filters"] == [
        {
            "dimension_id": order_status_dimension["id"],
            "operator": "=",
            "value": "delivered",
        }
    ]

    scoped_result = service.execute_semantic_plan(scoped_decision)
    assert '"order_status" = \'delivered\'' in scoped_result["sql"]
    assert "GROUP BY" in scoped_result["sql"]
    assert '"customer_state"' in scoped_result["sql"]
    assert "payment_type" not in scoped_result["sql"]
    assert scoped_result["rows"] == [
        {
            customer_state_dimension["id"]: "SP",
            payment_metric["id"]: 150.5,
        }
    ]


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


def test_semantic_metric_does_not_expand_through_unproven_delimited_relationship(tmp_path) -> None:
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

    relationship = published["definition"]["relationships"][0]
    assert relationship["cardinality"] == "unknown"
    assert relationship["enabled"] is False
    assert "JOIN" not in result["sql"]
    assert result["rows"] == [{metric["id"]: 60.0}]


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
