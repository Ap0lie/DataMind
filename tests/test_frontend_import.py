from __future__ import annotations

import io

import pandas as pd

from app.services.tabular_import import records_from_file_bytes


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
