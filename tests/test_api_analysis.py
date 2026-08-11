from __future__ import annotations

import asyncio
import io

import pandas as pd
import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from app.analysis.dataset_groups import suggest_dataset_group_relationships
from app.api.v1 import analysis as analysis_api
from app.core.settings import get_settings
from app.main import create_app
from app.mcp.tool_schemas import ModelRouterResponse
from app.schemas.analysis import AnalysisRunRequest, DatasetJoinConfig
from app.storage.assistant_memory_repository import AssistantMemoryRepository
from app.storage.dataset_store import DatasetStoreRepository


def test_create_job_persists_question_scoped_dataset_ids(tmp_path, monkeypatch) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    datasets = {
        name: repository.create_dataset(
            name=f"olist_{name}_dataset.csv",
            source_type="csv",
            source_metadata={},
        )
        for name in (
            "order_items",
            "orders",
            "customers",
            "order_payments",
            "products",
            "product_translation",
            "sellers",
            "order_reviews",
            "geolocation",
        )
    }
    rows = {
        "order_items": [
            {"order_id": "O1", "product_id": "P1", "seller_id": "S1"}
        ],
        "orders": [
            {"order_id": "O1", "customer_id": "C1", "order_status": "delivered"}
        ],
        "customers": [
            {
                "customer_id": "C1",
                "customer_state": "SP",
                "customer_zip_code_prefix": "01000",
            }
        ],
        "order_payments": [
            {"order_id": "O1", "payment_type": "credit_card", "payment_value": 10.0}
        ],
        "products": [{"product_id": "P1", "product_category_name": "books"}],
        "product_translation": [
            {"product_category_name": "books", "product_category_name_english": "books"}
        ],
        "sellers": [{"seller_id": "S1", "seller_state": "SP"}],
        "order_reviews": [{"order_id": "O1", "review_comment_message": "ok"}],
        "geolocation": [
            {
                "geolocation_zip_code_prefix": "01000",
                "geolocation_state": "SP",
            }
        ],
    }
    for name, records in rows.items():
        repository.append_raw_records(dataset_id=datasets[name].id, records=records)
    relationship_plan = (
        DatasetJoinConfig(
            left_dataset_id=datasets["order_items"].id,
            right_dataset_id=datasets["orders"].id,
            left_column="order_id",
            right_column="order_id",
        ),
        DatasetJoinConfig(
            left_dataset_id=datasets["orders"].id,
            right_dataset_id=datasets["customers"].id,
            left_column="customer_id",
            right_column="customer_id",
        ),
        DatasetJoinConfig(
            left_dataset_id=datasets["orders"].id,
            right_dataset_id=datasets["order_payments"].id,
            left_column="order_id",
            right_column="order_id",
        ),
        DatasetJoinConfig(
            left_dataset_id=datasets["order_items"].id,
            right_dataset_id=datasets["products"].id,
            left_column="product_id",
            right_column="product_id",
        ),
        DatasetJoinConfig(
            left_dataset_id=datasets["products"].id,
            right_dataset_id=datasets["product_translation"].id,
            left_column="product_category_name",
            right_column="product_category_name",
        ),
        DatasetJoinConfig(
            left_dataset_id=datasets["order_items"].id,
            right_dataset_id=datasets["sellers"].id,
            left_column="seller_id",
            right_column="seller_id",
        ),
        DatasetJoinConfig(
            left_dataset_id=datasets["orders"].id,
            right_dataset_id=datasets["order_reviews"].id,
            left_column="order_id",
            right_column="order_id",
        ),
        DatasetJoinConfig(
            left_dataset_id=datasets["customers"].id,
            right_dataset_id=datasets["geolocation"].id,
            left_column="customer_zip_code_prefix",
            right_column="geolocation_zip_code_prefix",
        ),
    )
    question = (
        "仅使用 customers、orders、order_payments 三张表，过滤 "
        "order_status=delivered，按 customer_state 统计 payment_value 总额，"
        "并给出总体支付总额和 SP 州支付总额。不要使用 order_items、reviews、"
        "products、sellers 或 geolocation，也不要按 order_status 或 "
        "payment_type 分组。"
    )
    monkeypatch.setattr(analysis_api, "_repository", lambda _user_id: repository)
    monkeypatch.setattr(analysis_api, "start_analysis_job", lambda **_kwargs: None)

    response = analysis_api.create_analysis_job(
        AnalysisRunRequest(
            dataset_id=datasets["order_items"].id,
            additional_dataset_ids=tuple(
                dataset.id
                for name, dataset in datasets.items()
                if name != "order_items"
            ),
            join_plan=relationship_plan,
            relationship_plan=relationship_plan,
            question=question,
            agent_mode="legacy",
        ),
        user_id="default",
    )

    stored = repository.get_analysis_job(response.job_id)
    assert stored.dataset_id == datasets["order_payments"].id
    assert set(stored.additional_dataset_ids) == {
        datasets["orders"].id,
        datasets["customers"].id,
    }
    assert len(stored.join_plan) == 2
    assert len(stored.relationship_plan) == 2
    assert {stored.dataset_id, *stored.additional_dataset_ids} == {
        datasets["order_payments"].id,
        datasets["orders"].id,
        datasets["customers"].id,
    }

    with pytest.raises(HTTPException) as exc_info:
        analysis_api.create_analysis_job(
            AnalysisRunRequest(
                dataset_id=datasets["order_payments"].id,
                question=question,
                agent_mode="legacy",
            ),
            user_id="default",
        )

    assert exc_info.value.status_code == 400
    assert "not submitted" in str(exc_info.value.detail)
    assert len(repository.list_analysis_jobs()) == 1


@pytest.mark.asyncio
async def test_file_import_endpoint_uses_backend_tabular_parser(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path / "store"))
    get_settings.cache_clear()
    app = create_app()
    workbook = io.BytesIO()
    valid_sheet = pd.DataFrame(
        [
            [None, None, None],
            ["期末考试未完成名单", None, None],
            ["姓名", "班级", "未完成"],
            ["张三", "6班", 1],
            ["李四", "6班", 0],
        ]
    )
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame().to_excel(writer, sheet_name="空表", index=False)
        valid_sheet.to_excel(writer, sheet_name="名单", index=False, header=False)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        import_response = await client.post(
            "/api/v1/store/files/import",
            data={"dataset_name": "complex.xlsx"},
            files={
                "file": (
                    "complex.xlsx",
                    workbook.getvalue(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        dataset_id = import_response.json()["dataset"]["dataset_id"]
        preview_response = await client.get(
            f"/api/v1/store/datasets/{dataset_id}/preview?source=raw"
        )

    get_settings.cache_clear()
    assert import_response.status_code == 200
    payload = import_response.json()
    assert payload["inserted"] == 2
    assert payload["dataset"]["source_type"] == "xlsx"
    assert payload["dataset"]["source_metadata"]["parser"] == "backend_tabular_import"
    assert payload["preview_records"] == [
        {"姓名": "张三", "班级": "6班", "未完成": 1},
        {"姓名": "李四", "班级": "6班", "未完成": 0},
    ]
    assert preview_response.json()["records"][1]["姓名"] == "李四"


@pytest.mark.asyncio
async def test_file_import_endpoint_accepts_json_and_txt(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path / "store"))
    get_settings.cache_clear()
    app = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        json_response = await client.post(
            "/api/v1/store/files/import",
            data={"dataset_name": "reviews.json"},
            files={"file": ("reviews.json", b'[{"review":"great","sentiment":"positive"}]', "application/json")},
        )
        txt_response = await client.post(
            "/api/v1/store/files/import",
            data={"dataset_name": "comments.txt"},
            files={"file": ("comments.txt", "第一条评论\n第二条评论\n".encode(), "text/plain")},
        )

    get_settings.cache_clear()
    assert json_response.status_code == 200
    assert json_response.json()["dataset"]["source_type"] == "json"
    assert json_response.json()["preview_records"][0]["review"] == "great"
    assert txt_response.status_code == 200
    assert txt_response.json()["dataset"]["source_type"] == "txt"
    assert txt_response.json()["preview_records"][1]["text"] == "第二条评论"


@pytest.mark.asyncio
async def test_xlsx_sheet_preview_and_selected_sheet_import(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path / "store"))
    get_settings.cache_clear()
    app = create_app()
    workbook = io.BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame([["region", "sales"], ["North", 100]]).to_excel(
            writer,
            sheet_name="sales",
            index=False,
            header=False,
        )
        pd.DataFrame([["name", "score"], ["Alice", 95], ["Bob", 88]]).to_excel(
            writer,
            sheet_name="scores",
            index=False,
            header=False,
        )
    file_bytes = workbook.getvalue()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        sheets_response = await client.post(
            "/api/v1/store/files/xlsx-sheets",
            files={
                "file": (
                    "multi.xlsx",
                    file_bytes,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        import_response = await client.post(
            "/api/v1/store/files/import",
            data={"dataset_name": "multi.xlsx", "sheet_name": "scores"},
            files={
                "file": (
                    "multi.xlsx",
                    file_bytes,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

    get_settings.cache_clear()
    assert sheets_response.status_code == 200
    assert [sheet["sheet_name"] for sheet in sheets_response.json()["sheets"]] == ["scores", "sales"]
    assert import_response.status_code == 200
    assert import_response.json()["inserted"] == 2
    assert import_response.json()["preview_records"][0]["name"] == "Alice"
    assert import_response.json()["dataset"]["source_metadata"]["sheet_name"] == "scores"


@pytest.mark.asyncio
async def test_analysis_endpoint_runs_dataset_workflow(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path))
    monkeypatch.setenv("DATAMIND_LLM_API_KEY", "")
    monkeypatch.setenv("DATAMIND_LLM_MODEL", "deepseek-chat")
    get_settings.cache_clear()
    app = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        create_response = await client.post(
            "/api/v1/store/datasets",
            json={"name": "sales.csv", "source_type": "csv", "source_metadata": {}},
        )
        dataset_id = create_response.json()["dataset_id"]
        await client.post(
            f"/api/v1/store/datasets/{dataset_id}/raw-records",
            json={
                "records": [
                    {"region": "North", "sales": 100, "profit": 20},
                    {"region": "South", "sales": 180, "profit": 45},
                ]
            },
        )

        profile_response = await client.get(f"/api/v1/store/datasets/{dataset_id}/profile")
        analysis_response = await client.post(
            "/api/v1/analysis/run",
            json={"dataset_id": dataset_id, "question": "Which region has the highest sales?"},
        )
        reports_response = await client.get("/api/v1/store/reports")
        report_summaries_response = await client.get(
            "/api/v1/store/reports?include_content=false"
        )
        dataset_reports_response = await client.get(f"/api/v1/store/datasets/{dataset_id}/reports")
        report_id = reports_response.json()["reports"][0]["id"]
        delete_report_response = await client.delete(f"/api/v1/store/reports/{report_id}")
        reports_after_delete_response = await client.get("/api/v1/store/reports")

    get_settings.cache_clear()
    assert profile_response.status_code == 200
    assert profile_response.json()["row_count"] == 2
    assert analysis_response.status_code == 200
    payload = analysis_response.json()
    assert payload["plan"]["route"] == "sql"
    assert payload["planner_metadata"]["confidence"] > 0
    assert "选择" in payload["planner_metadata"]["route_reason"]
    assert payload["workflow_trace"]
    assert payload["sql_result"]["rows"][0]["category"] == "South"
    assert payload["html_report"]
    assert "<!doctype html>" in payload["html_report"]
    assert "DataMind 分析报告" in payload["report_markdown"]
    assert reports_response.status_code == 200
    assert reports_response.json()["reports"][0]["dataset_id"] == dataset_id
    assert reports_response.json()["reports"][0]["metadata"]["workflow"] == "langgraph_analysis"
    assert "html_report" in reports_response.json()["reports"][0]["metadata"]
    summary = report_summaries_response.json()["reports"][0]
    assert report_summaries_response.status_code == 200
    assert summary["markdown"] == ""
    assert summary["metadata"]["route"] == "sql"
    assert summary["metadata"]["sql_source"] != "none"
    assert summary["metadata"]["python_source"] != "none"
    assert "html_report" not in summary["metadata"]
    assert dataset_reports_response.status_code == 200
    assert "DataMind 分析报告" in dataset_reports_response.json()["reports"][0]["markdown"]
    assert delete_report_response.status_code == 200
    assert delete_report_response.json() == {"report_id": report_id, "deleted": True}
    assert reports_after_delete_response.status_code == 200
    assert reports_after_delete_response.json()["reports"] == []


@pytest.mark.asyncio
async def test_multi_dataset_join_suggestions_run_and_job_retry(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path))
    monkeypatch.setenv("DATAMIND_LLM_API_KEY", "")
    get_settings.cache_clear()
    app = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        customers_response = await client.post(
            "/api/v1/store/datasets",
            json={"name": "customers.csv", "source_type": "csv", "source_metadata": {}},
        )
        orders_response = await client.post(
            "/api/v1/store/datasets",
            json={"name": "orders.csv", "source_type": "csv", "source_metadata": {}},
        )
        unmatched_orders_response = await client.post(
            "/api/v1/store/datasets",
            json={"name": "unmatched_orders.csv", "source_type": "csv", "source_metadata": {}},
        )
        customers_id = customers_response.json()["dataset_id"]
        orders_id = orders_response.json()["dataset_id"]
        unmatched_orders_id = unmatched_orders_response.json()["dataset_id"]
        await client.post(
            f"/api/v1/store/datasets/{customers_id}/raw-records",
            json={
                "records": [
                    {"customer_id": "C1", "segment": "Enterprise"},
                    {"customer_id": "C2", "segment": "SMB"},
                ]
            },
        )
        await client.post(
            f"/api/v1/store/datasets/{orders_id}/raw-records",
            json={
                "records": [
                    {"customer_id": "C1", "amount": 300},
                    {"customer_id": "C2", "amount": 120},
                ]
            },
        )
        await client.post(
            f"/api/v1/store/datasets/{unmatched_orders_id}/raw-records",
            json={"records": [{"customer_id": "C9", "amount": 999}]},
        )
        await client.post(
            f"/api/v1/store/datasets/{customers_id}/columns",
            json={
                "columns": [
                    {
                        "column_name": "customer_id",
                        "inferred_type": "text",
                        "role": "id",
                        "description": "Customer id",
                    },
                    {
                        "column_name": "segment",
                        "inferred_type": "text",
                        "role": "dimension",
                        "description": "Customer segment",
                    },
                ]
            },
        )
        await client.post(
            f"/api/v1/store/datasets/{orders_id}/columns",
            json={
                "columns": [
                    {
                        "column_name": "customer_id",
                        "inferred_type": "text",
                        "role": "id",
                        "description": "Customer id",
                    },
                    {
                        "column_name": "amount",
                        "inferred_type": "number",
                        "role": "metric",
                        "description": "Order amount",
                    },
                ]
            },
        )

        suggestions_response = await client.post(
            "/api/v1/analysis/join-suggestions",
            json={"dataset_id": customers_id, "additional_dataset_ids": [orders_id]},
        )
        top_suggestion = suggestions_response.json()["suggestions"][0]
        join_plan = [
            {
                "left_dataset_id": customers_id,
                "right_dataset_id": orders_id,
                "left_column": "customer_id",
                "right_column": "customer_id",
                "join_type": "left",
            }
        ]
        run_response = await client.post(
            "/api/v1/analysis/run",
            json={
                "dataset_id": customers_id,
                "additional_dataset_ids": [orders_id],
                "join_plan": join_plan,
                "question": "Which segment has the highest amount?",
            },
        )
        empty_join_response = await client.post(
            "/api/v1/analysis/run",
            json={
                "dataset_id": customers_id,
                "additional_dataset_ids": [unmatched_orders_id],
                "join_plan": [
                    {
                        "left_dataset_id": customers_id,
                        "right_dataset_id": unmatched_orders_id,
                        "left_column": "customer_id",
                        "right_column": "customer_id",
                        "join_type": "inner",
                    }
                ],
                "question": "Analyze unmatched joined records.",
            },
        )
        report_response = await client.get(f"/api/v1/store/datasets/{customers_id}/reports")
        create_job_response = await client.post(
            "/api/v1/analysis/jobs",
            json={
                "dataset_id": customers_id,
                "additional_dataset_ids": [orders_id],
                "join_plan": join_plan,
                "question": "Analyze joined customer order amount.",
            },
        )
        job_id = create_job_response.json()["job_id"]
        final_job = None
        for _ in range(30):
            job_response = await client.get(f"/api/v1/analysis/jobs/{job_id}")
            payload = job_response.json()
            if payload["status"] in {"completed", "failed", "canceled", "interrupted"}:
                final_job = payload
                break
            await asyncio.sleep(0.1)

        result_response = await client.get(f"/api/v1/analysis/jobs/{job_id}/result")
        retry_response = await client.post(f"/api/v1/analysis/jobs/{job_id}/retry")
        retry_job_id = retry_response.json()["job_id"]
        retry_final_job = None
        for _ in range(30):
            retry_job_response = await client.get(f"/api/v1/analysis/jobs/{retry_job_id}")
            retry_payload = retry_job_response.json()
            if retry_payload["status"] in {"completed", "failed", "canceled", "interrupted"}:
                retry_final_job = retry_payload
                break
            await asyncio.sleep(0.1)

    get_settings.cache_clear()
    assert suggestions_response.status_code == 200
    assert top_suggestion["left_column"] == "customer_id"
    assert top_suggestion["right_column"] == "customer_id"
    assert top_suggestion["score"] > 0.5
    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["multi_dataset_context"]["join_summary"]["mode"] == "joined"
    assert run_payload["multi_dataset_context"]["join_summary"]["joined_row_count"] == 2
    assert "orders_csv__amount" in run_payload["multi_dataset_context"]["column_source_map"]
    assert run_payload["planner_metadata"]["multi_dataset_summary"]["additional_datasets"] == ["orders.csv"]
    assert any("join_prepare" == node["node"] for node in run_payload["workflow_trace"])
    assert empty_join_response.status_code == 200
    empty_payload = empty_join_response.json()
    assert empty_payload["profile"]["row_count"] == 2
    assert any(
        "join 后结果为空" in issue["issue"]
        for issue in empty_payload["multi_dataset_context"]["validation_issues"]
    )
    assert report_response.status_code == 200
    assert report_response.json()["reports"][0]["metadata"]["join_summary"]["mode"] == "joined"
    assert create_job_response.status_code == 202
    assert create_job_response.json()["additional_dataset_ids"] == [orders_id]
    assert final_job is not None
    assert final_job["status"] == "completed"
    assert result_response.status_code == 200
    assert result_response.json()["multi_dataset_context"]["join_plan"][0]["right_dataset_id"] == orders_id
    assert retry_response.status_code == 202
    assert retry_response.json()["additional_dataset_ids"] == [orders_id]
    assert retry_final_job is not None
    assert retry_final_job["status"] == "completed"


@pytest.mark.asyncio
async def test_dataset_group_relationship_suggestions_and_analysis(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path))
    get_settings.cache_clear()
    app = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        customers_response = await client.post(
            "/api/v1/store/datasets",
            json={"name": "customers_dataset.csv", "source_type": "csv", "source_metadata": {}},
        )
        orders_response = await client.post(
            "/api/v1/store/datasets",
            json={"name": "orders_dataset.csv", "source_type": "csv", "source_metadata": {}},
        )
        customers_id = customers_response.json()["dataset_id"]
        orders_id = orders_response.json()["dataset_id"]
        await client.post(
            f"/api/v1/store/datasets/{customers_id}/raw-records",
            json={"records": [{"customer_id": "C1", "segment": "A"}, {"customer_id": "C2", "segment": "B"}]},
        )
        await client.post(
            f"/api/v1/store/datasets/{orders_id}/raw-records",
            json={"records": [{"order_id": "O1", "customer_id": "C1", "amount": 10}, {"order_id": "O2", "customer_id": "C2", "amount": 20}]},
        )
        group_response = await client.post(
            "/api/v1/store/dataset-groups",
            json={
                "name": "Brazilian E-commerce sample",
                "dataset_ids": [orders_id, customers_id],
                "description": "batch upload",
            },
        )
        group_id = group_response.json()["group_id"]
        suggestions_response = await client.post(
            f"/api/v1/store/dataset-groups/{group_id}/relationship-suggestions",
            json={},
        )
        candidate = suggestions_response.json()["candidates"][0]
        relationship = {
            "left_dataset_id": candidate["left_dataset_id"],
            "right_dataset_id": candidate["right_dataset_id"],
            "left_column": candidate["left_column"],
            "right_column": candidate["right_column"],
            "join_type": "left",
            "enabled": True,
            "confidence": candidate["confidence"],
            "source": candidate["source"],
            "reason": candidate["reason"],
            "relationship_type": candidate["relationship_type"],
            "risk_note": candidate["risk_note"],
        }
        update_response = await client.patch(
            f"/api/v1/store/dataset-groups/{group_id}/relationships",
            json={"relationships": [relationship]},
        )
        run_response = await client.post(
            "/api/v1/analysis/run",
            json={
                "dataset_id": relationship["left_dataset_id"],
                "dataset_group_id": group_id,
                "relationship_plan": [
                    {
                        "left_dataset_id": relationship["left_dataset_id"],
                        "right_dataset_id": relationship["right_dataset_id"],
                        "left_column": relationship["left_column"],
                        "right_column": relationship["right_column"],
                        "join_type": "left",
                    }
                ],
                "question": "Analyze customer order amount by segment.",
            },
        )
        delete_group_response = await client.delete(
            f"/api/v1/store/dataset-groups/{group_id}?delete_datasets=true",
        )
        list_after_delete_response = await client.get("/api/v1/store/datasets")

    get_settings.cache_clear()
    assert group_response.status_code == 200
    assert suggestions_response.status_code == 200
    suggestions_payload = suggestions_response.json()
    assert suggestions_payload["compact_context"]["contains_full_records"] is False
    assert len(suggestions_payload["compact_context"]["tables"][0]["preview_records"]) <= 5
    assert candidate["left_column"] == "customer_id"
    assert candidate["right_column"] == "customer_id"
    assert candidate["confidence"] > 0.5
    assert update_response.status_code == 200
    assert update_response.json()["relationships"][0]["left_column"] == "customer_id"
    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["dataset_group_id"] == group_id
    assert run_payload["multi_dataset_context"]["join_summary"]["mode"] == "joined"
    assert delete_group_response.status_code == 200
    assert set(delete_group_response.json()["deleted_dataset_ids"]) == {orders_id, customers_id}
    assert list_after_delete_response.json()["datasets"] == []


@pytest.mark.asyncio
async def test_dataset_group_relationships_auto_configure_and_persist(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path))
    monkeypatch.setenv("DATAMIND_PLANNER_LLM_PROVIDER", "mock")
    get_settings.cache_clear()
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        orders_response = await client.post(
            "/api/v1/store/datasets",
            json={"name": "orders.csv", "source_type": "csv", "source_metadata": {}},
        )
        customers_response = await client.post(
            "/api/v1/store/datasets",
            json={"name": "customers.csv", "source_type": "csv", "source_metadata": {}},
        )
        orders_id = orders_response.json()["dataset_id"]
        customers_id = customers_response.json()["dataset_id"]
        await client.post(
            f"/api/v1/store/datasets/{orders_id}/raw-records",
            json={"records": [{"order_id": "O1", "customer_id": "C1"}, {"order_id": "O2", "customer_id": "C2"}]},
        )
        await client.post(
            f"/api/v1/store/datasets/{customers_id}/raw-records",
            json={"records": [{"customer_id": "C1", "segment": "A"}, {"customer_id": "C2", "segment": "B"}]},
        )
        group_response = await client.post(
            "/api/v1/store/dataset-groups",
            json={"name": "Auto relationship package", "dataset_ids": [orders_id, customers_id]},
        )
        group_id = group_response.json()["group_id"]

        auto_response = await client.post(
            f"/api/v1/store/dataset-groups/{group_id}/relationships/auto-configure",
            json={},
        )
        persisted_response = await client.get(f"/api/v1/store/dataset-groups/{group_id}")

    get_settings.cache_clear()
    assert auto_response.status_code == 200
    payload = auto_response.json()
    assert len(payload["saved_relationships"]) == 1
    assert payload["saved_relationships"][0]["left_column"] == "customer_id"
    assert payload["saved_relationships"][0]["right_column"] == "customer_id"
    assert payload["unresolved_dataset_ids"] == []
    assert persisted_response.json()["relationships"] == payload["saved_relationships"]


def test_dataset_group_llm_suggestions_reject_hallucinated_columns(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path))
    get_settings.cache_clear()
    repository = DatasetStoreRepository(str(tmp_path), user_id="default")
    left = repository.create_dataset(name="buyers.csv", source_type="csv", source_metadata={})
    right = repository.create_dataset(name="transactions.csv", source_type="csv", source_metadata={})
    repository.append_raw_records(dataset_id=left.id, records=[{"buyer_key": "B1", "segment": "A"}])
    repository.append_raw_records(dataset_id=right.id, records=[{"client_ref": "B1", "amount": 12}])
    group = repository.create_dataset_group(name="odd names", dataset_ids=(left.id, right.id))

    class FakeRouter:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        def complete(self, **kwargs: object) -> ModelRouterResponse:
            self.messages = list(kwargs.get("messages") or [])
            return ModelRouterResponse(
                provider="mock",
                model="mock",
                content=(
                    '{"relationships":[{"left_dataset_id":"'
                    f'{left.id}","right_dataset_id":"{right.id}'
                    '","left_column":"missing_left","right_column":"missing_right",'
                    '"relationship_type":"many_to_one","confidence":0.99,'
                    '"reason":"looks semantic"}]}'
                ),
            )

    router = FakeRouter()
    suggestions = suggest_dataset_group_relationships(repository, group_id=group.id, router=router)

    get_settings.cache_clear()
    assert suggestions.llm_used is False
    assert all(candidate.left_column != "missing_left" for candidate in suggestions.candidates)
    assert suggestions.compact_context["contains_full_records"] is False
    assert "buyer_key" in str(router.messages)


@pytest.mark.asyncio
async def test_dataset_store_is_scoped_by_logged_in_user(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path))
    get_settings.cache_clear()
    app = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        alice_login = await client.post(
            "/api/v1/auth/login",
            json={"username": "alice@example.com", "password": "secret"},
        )
        bob_login = await client.post(
            "/api/v1/auth/login",
            json={"username": "bob@example.com", "password": "secret"},
        )
        alice_headers = {"X-DataMind-User": alice_login.json()["user_id"]}
        bob_headers = {"X-DataMind-User": bob_login.json()["user_id"]}

        create_response = await client.post(
            "/api/v1/store/datasets",
            headers=alice_headers,
            json={"name": "alice.csv", "source_type": "csv", "source_metadata": {}},
        )
        dataset_id = create_response.json()["dataset_id"]
        bob_dataset_response = await client.post(
            "/api/v1/store/datasets",
            headers=bob_headers,
            json={"name": "bob.csv", "source_type": "csv", "source_metadata": {}},
        )
        bob_dataset_id = bob_dataset_response.json()["dataset_id"]

        alice_list = await client.get("/api/v1/store/datasets", headers=alice_headers)
        bob_list = await client.get("/api/v1/store/datasets", headers=bob_headers)
        bob_get = await client.get(f"/api/v1/store/datasets/{dataset_id}", headers=bob_headers)
        cross_user_join = await client.post(
            "/api/v1/analysis/join-suggestions",
            headers=alice_headers,
            json={"dataset_id": dataset_id, "additional_dataset_ids": [bob_dataset_id]},
        )

    get_settings.cache_clear()
    assert alice_login.status_code == 200
    assert bob_login.status_code == 200
    assert create_response.status_code == 200
    assert create_response.json()["user_id"] == alice_headers["X-DataMind-User"]
    assert len(alice_list.json()["datasets"]) == 1
    assert len(bob_list.json()["datasets"]) == 1
    assert bob_get.status_code == 404
    assert cross_user_join.status_code == 404


@pytest.mark.asyncio
async def test_dataset_cleaning_versions_columns_and_report_detail_api(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path))
    monkeypatch.setenv("DATAMIND_LLM_API_KEY", "")
    get_settings.cache_clear()
    app = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        create_response = await client.post(
            "/api/v1/store/datasets",
            json={"name": "customers.csv", "source_type": "csv", "source_metadata": {}},
        )
        dataset_id = create_response.json()["dataset_id"]
        await client.post(
            f"/api/v1/store/datasets/{dataset_id}/raw-records",
            json={
                "records": [
                    {"customer_id": "001", "sales": "100", "region": "North"},
                    {"customer_id": "002", "sales": "180", "region": "South"},
                ]
            },
        )
        clean_response = await client.post(
            f"/api/v1/store/datasets/{dataset_id}/cleaning-runs",
            json={"requirement": "clean blanks"},
        )
        fast_clean_dataset_response = await client.post(
            "/api/v1/store/datasets",
            json={"name": "fast.csv", "source_type": "csv", "source_metadata": {}},
        )
        fast_dataset_id = fast_clean_dataset_response.json()["dataset_id"]
        await client.post(
            f"/api/v1/store/datasets/{fast_dataset_id}/raw-records",
            json={"records": [{" name ": " Alice "}, {" name ": " Alice "}]},
        )
        fast_clean_response = await client.post(
            f"/api/v1/store/datasets/{fast_dataset_id}/cleaning-runs",
            json={"requirement": "fast batch clean", "use_llm": False},
        )
        run_id = clean_response.json()["run_id"]
        runs_response = await client.get(f"/api/v1/store/datasets/{dataset_id}/cleaning-runs")
        detail_response = await client.get(
            f"/api/v1/store/datasets/{dataset_id}/cleaning-runs/{run_id}"
        )
        activate_response = await client.post(
            f"/api/v1/store/datasets/{dataset_id}/cleaning-runs/{run_id}/activate"
        )

        columns_response = await client.post(
            f"/api/v1/store/datasets/{dataset_id}/columns",
            json={
                "columns": [
                    {
                        "column_name": "customer_id",
                        "inferred_type": "text",
                        "override_type": "text",
                        "role": "id",
                        "description": "Customer identifier",
                    },
                    {
                        "column_name": "sales",
                        "inferred_type": "number",
                        "override_type": "number",
                        "role": "metric",
                        "description": "Sales amount",
                    },
                ]
            },
        )
        patch_column_response = await client.patch(
            f"/api/v1/store/datasets/{dataset_id}/columns/region",
            json={"inferred_type": "text", "role": "dimension", "description": "Sales region"},
        )
        profile_response = await client.get(f"/api/v1/store/datasets/{dataset_id}/profile")
        rules_preview_response = await client.post(
            f"/api/v1/store/datasets/{dataset_id}/cleaning-rules/preview",
            json={
                "rules": [
                    {"rule_type": "trim_text", "column": "region", "enabled": True},
                    {"rule_type": "rename_column", "column": "sales", "new_name": "revenue", "enabled": True},
                ]
            },
        )
        rules_apply_response = await client.post(
            f"/api/v1/store/datasets/{dataset_id}/cleaning-rules/apply",
            json={
                "rules": [
                    {"rule_type": "trim_text", "column": "region", "enabled": True},
                    {"rule_type": "rename_column", "column": "sales", "new_name": "revenue", "enabled": True},
                ]
            },
        )
        runs_after_rules_response = await client.get(f"/api/v1/store/datasets/{dataset_id}/cleaning-runs")

        save_report_response = await client.post(
            f"/api/v1/store/datasets/{dataset_id}/reports",
            json={"title": "Initial report", "markdown": "# Report\nRevenue", "metadata": {"question": "Revenue?"}},
        )
        report_id = save_report_response.json()["id"]
        report_detail_response = await client.get(f"/api/v1/store/reports/{report_id}")
        rename_response = await client.patch(
            f"/api/v1/store/reports/{report_id}",
            json={"title": "Renamed report"},
        )
        search_response = await client.get("/api/v1/store/reports?query=renamed&limit=5")
        versions_response = await client.get(f"/api/v1/store/datasets/{dataset_id}/report-versions")

    get_settings.cache_clear()
    assert clean_response.status_code == 200
    assert clean_response.json()["version"] == 1
    assert fast_clean_response.status_code == 200
    assert fast_clean_response.json()["provider"] == "rules"
    assert fast_clean_response.json()["source"] == "rules"
    assert fast_clean_response.json()["cleaned_row_count"] == 1
    assert runs_response.status_code == 200
    assert runs_response.json()["runs"][0]["is_active"] is True
    assert detail_response.status_code == 200
    assert detail_response.json()["diff_summary"]["raw_row_count"] == 2
    assert activate_response.status_code == 200
    assert activate_response.json()["is_active"] is True
    assert columns_response.status_code == 200
    assert patch_column_response.status_code == 200
    profile_payload = profile_response.json()
    assert "sales" in profile_payload["numeric_columns"]
    assert "customer_id" in profile_payload["categorical_columns"]
    assert rules_preview_response.status_code == 200
    assert rules_preview_response.json()["diff_summary"]["changed_cells"] >= 2
    assert rules_apply_response.status_code == 200
    assert rules_apply_response.json()["version"] == 2
    assert runs_after_rules_response.json()["runs"][0]["version"] == 2
    assert report_detail_response.status_code == 200
    assert report_detail_response.json()["title"] == "Initial report"
    assert rename_response.status_code == 200
    assert rename_response.json()["title"] == "Renamed report"
    assert search_response.status_code == 200
    assert search_response.json()["reports"][0]["id"] == report_id
    assert versions_response.status_code == 200
    assert versions_response.json()["versions"][0]["report_id"] == report_id


@pytest.mark.asyncio
async def test_analysis_job_lifecycle_and_user_scope(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path))
    monkeypatch.setenv("DATAMIND_LLM_API_KEY", "")
    get_settings.cache_clear()
    app = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        alice_headers = {"X-DataMind-User": "alice"}
        bob_headers = {"X-DataMind-User": "bob"}
        create_response = await client.post(
            "/api/v1/store/datasets",
            headers=alice_headers,
            json={"name": "sales.csv", "source_type": "csv", "source_metadata": {}},
        )
        dataset_id = create_response.json()["dataset_id"]
        await client.post(
            f"/api/v1/store/datasets/{dataset_id}/raw-records",
            headers=alice_headers,
            json={
                "records": [
                    {"region": "North", "sales": 100, "profit": 20},
                    {"region": "South", "sales": 180, "profit": 45},
                ]
            },
        )

        create_job_response = await client.post(
            "/api/v1/analysis/jobs",
            headers=alice_headers,
            json={"dataset_id": dataset_id, "question": "Which region has the highest sales?"},
        )
        job_id = create_job_response.json()["job_id"]
        bob_get_response = await client.get(
            f"/api/v1/analysis/jobs/{job_id}",
            headers=bob_headers,
        )

        final_job = None
        for _ in range(30):
            job_response = await client.get(
                f"/api/v1/analysis/jobs/{job_id}",
                headers=alice_headers,
            )
            payload = job_response.json()
            if payload["status"] in {"completed", "failed", "canceled", "interrupted"}:
                final_job = payload
                break
            await asyncio.sleep(0.1)

        result_response = await client.get(
            f"/api/v1/analysis/jobs/{job_id}/result",
            headers=alice_headers,
        )
        list_response = await client.get(
            f"/api/v1/analysis/jobs?dataset_id={dataset_id}",
            headers=alice_headers,
        )
        retry_response = await client.post(
            f"/api/v1/analysis/jobs/{job_id}/retry",
            headers=alice_headers,
        )
        retry_job_id = retry_response.json()["job_id"]
        retry_final_job = None
        for _ in range(30):
            retry_job_response = await client.get(
                f"/api/v1/analysis/jobs/{retry_job_id}",
                headers=alice_headers,
            )
            retry_payload = retry_job_response.json()
            if retry_payload["status"] in {"completed", "failed", "canceled", "interrupted"}:
                retry_final_job = retry_payload
                break
            await asyncio.sleep(0.1)

        experiences = AssistantMemoryRepository(
            str(tmp_path), user_id="alice"
        ).list(memory_kind="episodic", status="active")

    get_settings.cache_clear()
    assert create_job_response.status_code == 202
    assert create_job_response.json()["status"] == "queued"
    assert bob_get_response.status_code == 404
    assert final_job is not None
    assert final_job["status"] == "completed"
    assert final_job["progress"] == 100
    assert final_job["report_id"]
    assert result_response.status_code == 200
    assert result_response.json()["dataset_id"] == dataset_id
    assert result_response.json()["report_id"] == final_job["report_id"]
    assert list_response.status_code == 200
    assert any(job["job_id"] == job_id for job in list_response.json()["jobs"])
    assert retry_response.status_code == 202
    assert retry_response.json()["retry_of"] == job_id
    assert retry_final_job is not None
    assert retry_final_job["status"] == "completed"
    assert len(experiences) == 1
    assert experiences[0]["memory_type"] == "analysis_experience"


def test_queued_analysis_job_can_be_canceled(tmp_path) -> None:
    repository = DatasetStoreRepository(str(tmp_path), user_id="alice")
    dataset = repository.create_dataset(
        name="sales.csv",
        source_type="csv",
        source_metadata={},
    )
    job = repository.create_analysis_job(
        dataset_id=dataset.id,
        question="Analyze sales.",
    )

    canceled = repository.request_analysis_job_cancel(job.id)

    assert canceled.status == "canceled"
    assert canceled.cancel_requested is True
    assert canceled.completed_at is not None


@pytest.mark.asyncio
async def test_analysis_job_cancel_endpoint_handles_queued_job(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATAMIND_DATASET_STORE_PATH", str(tmp_path))
    get_settings.cache_clear()
    app = create_app()
    repository = DatasetStoreRepository(str(tmp_path), user_id="alice")
    dataset = repository.create_dataset(
        name="sales.csv",
        source_type="csv",
        source_metadata={},
    )
    job = repository.create_analysis_job(
        dataset_id=dataset.id,
        question="Analyze sales.",
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"/api/v1/analysis/jobs/{job.id}/cancel",
            headers={"X-DataMind-User": "alice"},
        )

    get_settings.cache_clear()
    assert response.status_code == 200
    assert response.json()["status"] == "canceled"
    assert response.json()["cancel_requested"] is True
