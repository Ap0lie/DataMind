from __future__ import annotations

import csv
import io
import json
from typing import Any

import pandas as pd


def records_from_file_bytes(
    *,
    file_bytes: bytes,
    source_type: str,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    if source_type == "csv":
        return {"ok": True, "data": _csv_records_from_bytes(file_bytes)}
    if source_type == "json":
        return _json_records_from_bytes(file_bytes)
    if source_type == "txt":
        return {"ok": True, "data": _txt_records_from_bytes(file_bytes)}
    if source_type != "xlsx":
        return {"ok": False, "error": f"不支持的数据源类型: {source_type}"}

    dataframe_result = xlsx_dataframe_from_bytes(file_bytes, sheet_name=sheet_name)
    if not dataframe_result["ok"]:
        return dataframe_result
    return {"ok": True, "data": dataframe_to_json_records(dataframe_result["data"])}


def xlsx_dataframe_from_bytes(
    file_bytes: bytes,
    *,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    selected_sheet_name = sheet_name
    try:
        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None)
    except Exception as exc:
        return {"ok": False, "error": f"解析 XLSX 原始行数据失败: {exc}"}

    candidates = []
    for current_sheet_name, raw_df in sheets.items():
        normalized = _normalize_excel_sheet(raw_df)
        if normalized is None:
            continue
        score = int(normalized.shape[0] * max(normalized.shape[1], 1))
        candidates.append((score, str(current_sheet_name), normalized))

    if not candidates:
        return {
            "ok": False,
            "error": "未从 XLSX 中解析到有效表格数据，请检查 sheet、表头或空行。",
        }

    if selected_sheet_name:
        for _, candidate_name, normalized in candidates:
            if candidate_name == selected_sheet_name:
                return {"ok": True, "data": normalized}
        return {"ok": False, "error": f"未找到可导入的 sheet: {selected_sheet_name}"}

    candidates.sort(key=lambda item: item[0], reverse=True)
    return {"ok": True, "data": candidates[0][2]}


def xlsx_sheet_previews_from_bytes(
    file_bytes: bytes,
    *,
    preview_limit: int = 8,
) -> dict[str, Any]:
    try:
        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None)
    except Exception as exc:
        return {"ok": False, "error": f"解析 XLSX sheet 失败: {exc}"}

    previews: list[dict[str, Any]] = []
    for sheet_name, raw_df in sheets.items():
        normalized = _normalize_excel_sheet(raw_df)
        if normalized is None:
            continue
        score = int(normalized.shape[0] * max(normalized.shape[1], 1))
        previews.append(
            {
                "sheet_name": str(sheet_name),
                "row_count": int(normalized.shape[0]),
                "column_count": int(normalized.shape[1]),
                "score": score,
                "preview_records": dataframe_to_json_records(normalized.head(preview_limit)),
            }
        )
    if not previews:
        return {"ok": False, "error": "未从 XLSX 中解析到有效 sheet。"}
    best_score = max(int(item["score"]) for item in previews)
    selected = False
    for item in previews:
        item["selected"] = not selected and int(item["score"]) == best_score
        selected = selected or bool(item["selected"])
    previews.sort(key=lambda item: int(item["score"]), reverse=True)
    return {"ok": True, "sheets": previews}


def dataframe_to_json_records(dataframe: Any) -> list[dict[str, Any]]:
    return json.loads(dataframe.to_json(orient="records", force_ascii=False, date_format="iso"))


def _csv_records_from_bytes(file_bytes: bytes) -> list[dict[str, str]]:
    text = _decode_text(file_bytes)
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader]


def _json_records_from_bytes(file_bytes: bytes) -> dict[str, Any]:
    text = _decode_text(file_bytes)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"解析 JSON 失败: {exc}"}

    records = _records_from_json_payload(payload)
    if not records:
        return {"ok": False, "error": "JSON 中没有可导入的记录数组或对象。"}
    return {"ok": True, "data": records}


def _records_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [_json_row(item, index) for index, item in enumerate(payload, start=1)]
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                records = [_json_row(item, index) for index, item in enumerate(value, start=1)]
                if records:
                    return records
        return [_json_row(payload, 1)]
    return [{"value": payload}]


def _json_row(item: Any, index: int) -> dict[str, Any]:
    if isinstance(item, dict):
        return {str(key): value for key, value in item.items()}
    return {"row_number": index, "value": item}


def _txt_records_from_bytes(file_bytes: bytes) -> list[dict[str, Any]]:
    text = _decode_text(file_bytes)
    tabular = _delimited_text_records(text)
    if tabular:
        return tabular

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [{"line_number": index, "text": line} for index, line in enumerate(lines, start=1)]


def _delimited_text_records(text: str) -> list[dict[str, str]]:
    delimiter = _detect_text_delimiter(text)
    if delimiter is None:
        return []
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    non_empty_rows = [[cell.strip() for cell in row] for row in rows if any(cell.strip() for cell in row)]
    if len(non_empty_rows) < 2 or max(len(row) for row in non_empty_rows) < 2:
        return []

    header = _dedupe_columns([cell or f"column_{index + 1}" for index, cell in enumerate(non_empty_rows[0])])
    records: list[dict[str, str]] = []
    for row in non_empty_rows[1:]:
        padded = row + [""] * max(len(header) - len(row), 0)
        records.append({header[index]: padded[index] for index in range(len(header))})
    return records


def _detect_text_delimiter(text: str) -> str | None:
    """Detect a table delimiter without relying only on csv.Sniffer.

    Sniffer can reject otherwise valid TSV files when the first data row contains
    very long list-valued cells or ASCII control separators (for example the
    public JDsearch samples).  A delimited header is stronger evidence and also
    avoids reading an arbitrary prefix that ends halfway through a large cell.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    header = lines[0]
    candidates = ("\t", ",", ";", "|")
    header_ranked = sorted(candidates, key=header.count, reverse=True)
    for delimiter in header_ranked:
        column_count = header.count(delimiter) + 1
        if column_count < 2:
            continue
        matching_rows = sum(
            1
            for line in lines[1:11]
            if line.count(delimiter) + 1 == column_count
        )
        if matching_rows:
            return delimiter

    sample = "\n".join(lines[:10])[:32_768]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        return None


def _decode_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def _normalize_excel_sheet(raw_df: Any) -> Any | None:
    if raw_df is None or raw_df.empty:
        return None
    df = raw_df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        return None

    header_index = _detect_excel_header_row(df)
    header_values = [str(value).strip() for value in df.iloc[header_index].tolist()]
    data = df.iloc[header_index + 1 :].copy()
    data = data.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if data.empty:
        return None

    columns = _dedupe_columns(header_values[: len(data.columns)])
    if len(columns) < len(data.columns):
        columns.extend(f"column_{index + 1}" for index in range(len(columns), len(data.columns)))
    data.columns = columns[: len(data.columns)]
    data = data.loc[:, [column for column in data.columns if str(column).strip()]]
    data = data.where(pd.notna(data), None)
    return data if not data.empty else None


def _detect_excel_header_row(df: Any) -> int:
    best_index = 0
    best_score = -1
    max_scan = min(len(df), 20)
    for index in range(max_scan):
        row = df.iloc[index]
        values = [value for value in row.tolist() if not _is_blank_cell(value)]
        if not values:
            continue
        string_like = sum(1 for value in values if isinstance(value, str))
        unique_values = len({str(value).strip() for value in values})
        following_rows = max(len(df) - index - 1, 0)
        score = unique_values * 3 + string_like * 2 + min(following_rows, 5)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _dedupe_columns(values: list[str]) -> list[str]:
    columns: list[str] = []
    counts: dict[str, int] = {}
    for index, value in enumerate(values):
        base = value.strip() or f"column_{index + 1}"
        counts[base] = counts.get(base, 0) + 1
        columns.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return columns


def _is_blank_cell(value: Any) -> bool:
    return bool(pd.isna(value)) or str(value).strip() == ""
