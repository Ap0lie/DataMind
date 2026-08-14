import type React from "react";

export function Metric({
  label,
  value,
  caption,
  icon,
}: {
  label: string;
  value: number | string;
  caption: string;
  icon?: React.ReactNode;
}) {
  return (
    <div className="metric-card">
      <div className="metric-card-heading">
        <span>{label}</span>
        {icon && <i>{icon}</i>}
      </div>
      <strong className="mt-4 block text-3xl font-black tracking-tight text-slate-950">{value}</strong>
      <small className="mt-2 block text-sm text-slate-500">{caption}</small>
    </div>
  );
}

export function Alert({
  children,
  tone = "info",
}: {
  children: React.ReactNode;
  tone?: "info" | "error";
}) {
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      aria-live={tone === "error" ? "assertive" : "polite"}
      className={`mt-4 rounded-xl border px-4 py-3 text-sm ${
        tone === "error"
          ? "border-rose-200 bg-rose-50 text-rose-950"
          : "border-sky-200 bg-sky-50 text-slate-900"
      }`}
    >
      {children}
    </div>
  );
}

export function LoadingLine() {
  return (
    <div className="rounded-xl border border-line bg-white px-4 py-3 text-sm font-bold text-slate-500 shadow-[0_8px_24px_rgba(15,23,42,0.04)]">
      正在从数据库加载...
    </div>
  );
}
