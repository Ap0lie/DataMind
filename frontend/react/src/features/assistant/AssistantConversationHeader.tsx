import { Bot, ChevronDown, Layers3, Menu, PanelRightOpen, ShieldCheck } from "lucide-react";
import type { AssistantExecutionMode, AssistantScopeAsset } from "./types";

type Props = {
  executionMode: AssistantExecutionMode;
  scopeValue: string;
  datasets: AssistantScopeAsset[];
  datasetGroups: AssistantScopeAsset[];
  reports: AssistantScopeAsset[];
  active: boolean;
  workbenchCount: number;
  onScopeChange: (value: string) => void;
  onOpenHistory: () => void;
  onOpenWorkbench: () => void;
};

export function AssistantConversationHeader({ executionMode, scopeValue, datasets, datasetGroups, reports, active, workbenchCount, onScopeChange, onOpenHistory, onOpenWorkbench }: Props) {
  return (
    <header className="assistant-thread-toolbar">
      <button type="button" className="assistant-mobile-menu" title="打开消息记录" onClick={onOpenHistory}><Menu size={18} /></button>
      <div className="assistant-model">
        <span><Bot size={17} /></span>
        <div><h2>数据分析助手</h2><small>Kimi · {executionMode === "execute" ? "执行任务：当前范围自动获得完整管理权限" : "问答：仅读取与计划预览"}</small></div>
      </div>
      <div className="assistant-toolbar-controls">
        <div className="assistant-context-control">
          <span><Layers3 size={16} /></span>
          <label className="assistant-scope">
            <b>当前范围 · 选择具体资产会自动授权</b>
            <select aria-label="数据范围" value={scopeValue} onChange={(event) => onScopeChange(event.target.value)} disabled={active}>
              <option value="auto">自动检索全部资产</option>
              <optgroup label="数据包">{datasetGroups.map((item) => <option key={item.id} value={`dataset_group:${item.id}`}>{item.name}</option>)}</optgroup>
              <optgroup label="数据集">{datasets.map((item) => <option key={item.id} value={`dataset:${item.id}`}>{item.name}</option>)}</optgroup>
              <optgroup label="报告">{reports.map((item) => <option key={item.id} value={`report:${item.id}`}>{item.name}</option>)}</optgroup>
            </select>
            <ChevronDown size={15} />
          </label>
        </div>
        <button type="button" className="assistant-manage-button" aria-label={`打开 Kimi 工作台，${workbenchCount} 项待关注`} title="Kimi 权限与操作" onClick={onOpenWorkbench}>
          <ShieldCheck size={16} /><span>工作台</span>{workbenchCount > 0 && <b>{workbenchCount}</b>}<PanelRightOpen size={14} />
        </button>
      </div>
    </header>
  );
}
