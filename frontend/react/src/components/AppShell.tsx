import type { ReactNode } from "react";
import {
  ArrowRight,
  Database,
  FileText,
  Home,
  Loader2,
  LogOut,
  Play,
  Sparkles,
} from "lucide-react";

import type { AuthUser } from "../api-client";
import type { ActiveTask, Page } from "../domain-types";

export function Sidebar({
  page,
  user,
  assistantActiveRuns,
  onPageChange,
  onLogout,
}: {
  page: Page;
  user: AuthUser;
  assistantActiveRuns: number;
  onPageChange: (page: Page) => void;
  onLogout: () => void;
}) {
  const items: { page: Page; icon: ReactNode }[] = [
    { page: "首页", icon: <Home size={16} /> },
    { page: "数据集", icon: <Database size={16} /> },
    { page: "分析任务", icon: <Play size={16} /> },
    { page: "报告", icon: <FileText size={16} /> },
    { page: "Kimi", icon: <Sparkles size={16} /> },
  ];
  return (
    <aside className="fixed inset-x-0 bottom-0 z-20 flex h-[76px] w-full items-center border-t border-line bg-white/95 px-2 py-2 shadow-[0_-12px_32px_rgba(15,23,42,0.08)] backdrop-blur md:inset-y-0 md:left-0 md:right-auto md:h-auto md:w-[236px] md:flex-col md:items-stretch md:border-r md:border-t-0 md:px-8 md:py-8 md:shadow-[18px_0_44px_rgba(15,23,42,0.05)]">
      <div className="mb-14 hidden items-center gap-3 md:flex">
        <div className="grid h-10 w-10 place-items-center rounded-2xl bg-slate-950 text-xs font-black text-white shadow-[0_14px_28px_rgba(15,23,42,0.2)]">
          DM
        </div>
        <strong className="text-lg tracking-tight">DataMind</strong>
      </div>
      <nav className="flex min-w-0 flex-1 items-center justify-around gap-1 md:block md:space-y-3">
        {items.map((item) => (
          <button
            key={item.page}
            onClick={() => onPageChange(item.page)}
            className={`flex h-14 min-w-0 flex-1 flex-col items-center justify-center gap-1 rounded-lg px-1 text-center text-[10px] font-black transition md:h-12 md:w-full md:flex-row md:justify-start md:gap-3 md:rounded-xl md:px-5 md:text-left md:text-sm ${
              page === item.page
                ? "bg-acid text-black shadow-[0_14px_24px_rgba(200,251,79,0.34)]"
                : "bg-transparent text-slate-700 hover:bg-slate-100 hover:text-slate-950"
            }`}
          >
            {item.icon}
            <span className="relative">
              {item.page}
              {item.page === "Kimi" && assistantActiveRuns > 0 && (
                <i className="absolute -right-5 -top-2 grid min-h-4 min-w-4 place-items-center rounded-full bg-emerald-600 px-1 text-[9px] font-black not-italic text-white">
                  {assistantActiveRuns}
                </i>
              )}
            </span>
          </button>
        ))}
        <button
          type="button"
          aria-label="Log Out"
          onClick={onLogout}
          className="flex h-14 min-w-0 flex-1 flex-col items-center justify-center gap-1 rounded-lg px-1 text-[10px] font-black text-slate-600 transition hover:bg-slate-100 hover:text-slate-950 md:hidden"
        >
          <LogOut size={16} />
          退出
        </button>
      </nav>
      <div className="mt-auto hidden w-full min-w-0 space-y-4 overflow-hidden text-center md:block">
        <div className="mx-auto grid h-10 w-10 place-items-center rounded-full border border-line bg-slate-100 text-sm font-black">
          {user.display_name.slice(0, 1).toUpperCase()}
        </div>
        <div
          className="w-full min-w-0 truncate px-1 text-sm font-black"
          data-testid="sidebar-account-name"
          title={user.display_name}
        >
          {user.display_name}
        </div>
        <button
          type="button"
          onClick={onLogout}
          className="mx-auto flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-bold text-slate-600 transition hover:bg-slate-100 hover:text-slate-950"
        >
          <LogOut size={15} /> Log Out
        </button>
      </div>
    </aside>
  );
}

export function Topbar({ user }: { user: AuthUser }) {
  return (
    <header className="mb-10 flex items-start justify-between">
      <h1 className="text-3xl font-black tracking-tight">DataMind</h1>
      <div className="flex items-center gap-3">
        <span className="grid h-9 w-9 place-items-center rounded-full border border-[#eadfd4] bg-[#f4eadf] text-xs font-black shadow-[0_10px_20px_rgba(15,23,42,0.08)]">
          {user.display_name.slice(0, 1).toUpperCase()}
        </span>
        <div>
          <b className="block text-sm">{user.display_name}</b>
          <small className="text-xs text-slate-500">DataMind User</small>
        </div>
      </div>
    </header>
  );
}

export function FloatingTaskProgress({
  task,
  activeCount,
  onOpen,
}: {
  task: ActiveTask;
  activeCount: number;
  onOpen: () => void;
}) {
  const progress = Math.max(0, Math.min(task.progress, 100));
  const Icon = task.kind === "cleaning" ? Database : task.kind === "assistant" ? Sparkles : Loader2;
  const motionClass = task.kind === "analysis" ? "animate-spin" : "animate-pulse";
  const accessibleLabel = task.kind === "analysis"
    ? `查看运行中的分析：${task.title}，${progress}%`
    : task.kind === "cleaning"
      ? `查看后台清洗进度，${progress}%`
      : `查看后台助手进度，${progress}%`;
  return (
    <button
      type="button"
      className="floating-task-progress"
      onClick={onOpen}
      aria-label={accessibleLabel}
      title={`前往${task.page}`}
    >
      <span className="floating-task-icon">
        <Icon className={motionClass} size={18} />
      </span>
      <span className="floating-task-copy">
        <span>
          {activeCount > 1 ? `${activeCount} 个任务运行中` : task.stage}
          <b>{progress}%</b>
        </span>
        <strong>{task.title}</strong>
      </span>
      <ArrowRight className="floating-task-arrow" size={17} />
      <span className="floating-task-track" aria-hidden="true">
        <i style={{ width: `${Math.max(progress, 3)}%` }} />
      </span>
    </button>
  );
}
