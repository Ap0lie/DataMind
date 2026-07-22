import { Database, Loader2 } from "lucide-react";
import type { AssistantImportBatch } from "./types";

type Props = {
  batch: AssistantImportBatch | null;
  importing: boolean;
  sheetSelections: Record<string, string>;
  onSheetChange: (attachmentId: string, sheet: string) => void;
  onPreview: () => void;
  onCommit: (allowPartial: boolean) => void;
};

export function AssistantImportPreview({ batch, importing, sheetSelections, onSheetChange, onPreview, onCommit }: Props) {
  const missingSheet = Boolean(batch?.preview.files?.some((item) => item.valid && item.requires_sheet_selection && !sheetSelections[item.attachment_id]));
  const stage = batch?.status === "completed" ? 3 : batch ? 2 : 1;
  return (
    <section className="assistant-import-panel">
      <div><Database size={17} /><span><b>{stage === 3 ? "数据包已创建并授权" : "导入数据并授权 Kimi 管理"}</b><small>{batch ? `${batch.preview.valid_count ?? 0} 个文件可导入，${batch.preview.invalid_count ?? 0} 个需处理` : "先预览字段、行数和 Sheet，不会立即创建数据集。"}</small></span></div>
      <ol className="assistant-import-steps" aria-label="数据导入进度"><li className={stage >= 1 ? "active" : ""}>上传</li><li className={stage >= 2 ? "active" : ""}>预览 / Sheet</li><li className={stage >= 3 ? "active" : ""}>导入并授权</li></ol>
      {batch?.preview.files?.map((item) => (
        <details key={item.attachment_id} open={item.requires_sheet_selection}>
          <summary><span className={item.valid ? "is-valid" : "is-invalid"}>{item.valid ? "可导入" : "失败"}</span><b>{item.file_name}</b><small>{item.valid ? `${item.row_count} 行 · ${item.column_count} 列${item.selected_sheet ? ` · ${item.selected_sheet}` : item.requires_sheet_selection ? " · 需选择 Sheet" : ""}` : item.error}</small></summary>
          {item.columns.length > 0 && <p>{item.columns.slice(0, 12).join("、")}{item.columns.length > 12 ? "…" : ""}</p>}
          {item.requires_sheet_selection && <label className="assistant-sheet-select"><span>导入 Sheet</span><select value={sheetSelections[item.attachment_id] ?? ""} onChange={(event) => onSheetChange(item.attachment_id, event.target.value)}><option value="">请选择</option>{item.sheets?.map((sheet) => <option key={sheet.sheet_name} value={sheet.sheet_name}>{sheet.sheet_name} · {sheet.row_count} 行 × {sheet.column_count} 列</option>)}</select></label>}
        </details>
      ))}
      {batch?.status !== "completed" && <div className="assistant-import-actions"><button type="button" disabled={importing || missingSheet} onClick={() => batch ? onCommit(false) : onPreview()}>{importing ? <Loader2 size={15} className="animate-spin" /> : <Database size={15} />}{batch ? "确认导入并授权" : "预览数据包"}</button>{Boolean(batch?.preview.invalid_count) && <button type="button" className="secondary" disabled={importing || missingSheet} onClick={() => onCommit(true)}>仅导入成功文件</button>}</div>}
    </section>
  );
}
