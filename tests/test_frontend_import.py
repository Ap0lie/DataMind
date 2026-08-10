from __future__ import annotations

import io

import pandas as pd

from app.services.tabular_import import (
    record_batches_from_file_path,
    records_from_file_bytes,
    xlsx_sheet_previews_from_path,
)


def test_xlsx_import_selects_non_empty_sheet_and_detects_header_offset() -> None:
    workbook = io.BytesIO()
    valid_sheet = pd.DataFrame(
        [
            [None, None, None],
            ["考试统计", None, None],
            ["姓名", "班级", "未完成"],
            ["张三", "6班", 1],
            ["李四", "6班", 0],
        ]
    )
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame().to_excel(writer, sheet_name="空表", index=False)
        valid_sheet.to_excel(writer, sheet_name="名单", index=False, header=False)

    result = records_from_file_bytes(
        file_bytes=workbook.getvalue(),
        source_type="xlsx",
    )

    assert result["ok"]
    assert result["data"] == [
        {"姓名": "张三", "班级": "6班", "未完成": 1},
        {"姓名": "李四", "班级": "6班", "未完成": 0},
    ]


def test_xlsx_import_returns_clear_error_for_empty_workbook() -> None:
    workbook = io.BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame().to_excel(writer, sheet_name="空表", index=False)

    result = records_from_file_bytes(
        file_bytes=workbook.getvalue(),
        source_type="xlsx",
    )

    assert not result["ok"]
    assert "未从 XLSX 中解析到有效表格数据" in result["error"]


def test_json_import_accepts_records_array() -> None:
    result = records_from_file_bytes(
        file_bytes=b'[{"name":"Alice","score":95},{"name":"Bob","score":88}]',
        source_type="json",
    )

    assert result["ok"]
    assert result["data"] == [
        {"name": "Alice", "score": 95},
        {"name": "Bob", "score": 88},
    ]


def test_json_import_accepts_nested_records_array() -> None:
    result = records_from_file_bytes(
        file_bytes=b'{"records":[{"review":"great","sentiment":"positive"}]}',
        source_type="json",
    )

    assert result["ok"]
    assert result["data"] == [{"review": "great", "sentiment": "positive"}]


def test_txt_import_parses_delimited_text() -> None:
    result = records_from_file_bytes(
        file_bytes=b"name\tscore\nAlice\t95\nBob\t88\n",
        source_type="txt",
    )

    assert result["ok"]
    assert result["data"] == [
        {"name": "Alice", "score": "95"},
        {"name": "Bob", "score": "88"},
    ]


def test_txt_import_parses_long_list_valued_tsv_with_control_separators() -> None:
    candidate_ids = "_".join(str(index) for index in range(250))
    content = (
        "query\tcandidate_wid_list\tcandidate_label_list\thistory_qry_list\t"
        "history_wid_list\thistory_type_list\thistory_time_list\n"
        f"12\x1832\t{candidate_ids}\t1.0_0.0\t-1\t889_256\tORD_CLICK\t0_12_4\n"
        f"54\x1856\t{candidate_ids}\t0.0_1.0\t-1\t345_789\tCART_ORD\t0_8_2\n"
    )

    result = records_from_file_bytes(file_bytes=content.encode(), source_type="txt")

    assert result["ok"]
    assert len(result["data"]) == 2
    assert tuple(result["data"][0]) == (
        "query",
        "candidate_wid_list",
        "candidate_label_list",
        "history_qry_list",
        "history_wid_list",
        "history_type_list",
        "history_time_list",
    )
    assert result["data"][0]["candidate_wid_list"] == candidate_ids


def test_txt_import_falls_back_to_line_records() -> None:
    result = records_from_file_bytes(
        file_bytes="第一条评论\n第二条评论\n".encode(),
        source_type="txt",
    )

    assert result["ok"]
    assert result["data"] == [
        {"line_number": 1, "text": "第一条评论"},
        {"line_number": 2, "text": "第二条评论"},
    ]


def test_path_import_streams_csv_and_nested_json_in_bounded_batches(tmp_path) -> None:
    csv_path = tmp_path / "large.csv"
    csv_path.write_text(
        "id,value\n" + "".join(f"{index},{index * 2}\n" for index in range(2505)),
        encoding="utf-8",
    )
    csv_stream = record_batches_from_file_path(
        csv_path,
        source_type="csv",
        batch_size=128,
    )
    csv_batches = list(csv_stream.batches)

    json_path = tmp_path / "nested.json"
    json_path.write_text(
        '{"records":['
        + ",".join(f'{{"id":{index},"value":"v{index}"}}' for index in range(205))
        + "]}",
        encoding="utf-8",
    )
    json_stream = record_batches_from_file_path(
        json_path,
        source_type="json",
        batch_size=64,
    )
    json_batches = list(json_stream.batches)

    assert sum(len(batch) for batch in csv_batches) == 2505
    assert max(map(len, csv_batches)) == 128
    assert csv_batches[-1][-1] == {"id": "2504", "value": "5008"}
    assert sum(len(batch) for batch in json_batches) == 205
    assert max(map(len, json_batches)) == 64
    assert json_batches[0][0] == {"id": 0, "value": "v0"}


def test_path_xlsx_preview_scans_sheets_without_loading_all_at_once(tmp_path) -> None:
    workbook_path = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        pd.DataFrame([["region", "sales"], ["North", 100]]).to_excel(
            writer, sheet_name="sales", index=False, header=False
        )
        pd.DataFrame([["name", "score"], ["Alice", 95], ["Bob", 88]]).to_excel(
            writer, sheet_name="scores", index=False, header=False
        )

    result = xlsx_sheet_previews_from_path(workbook_path)

    assert result["ok"]
    assert [item["sheet_name"] for item in result["sheets"]] == ["scores", "sales"]
    assert result["sheets"][0]["selected"] is True
