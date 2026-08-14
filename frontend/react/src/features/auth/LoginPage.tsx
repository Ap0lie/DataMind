import { useState, type FormEvent } from "react";
import { Loader2 } from "lucide-react";

import { Alert } from "../../components/primitives";
import { loginErrorMessage } from "../../formatters";

type FocusTarget = "idle" | "username" | "password";

export function LoginPage({
  notice,
  onLogin,
}: {
  notice?: string | null;
  onLogin: (username: string, password: string) => Promise<void>;
}) {
  const [username, setUsername] = useState("default");
  const [password, setPassword] = useState("");
  const [focus, setFocus] = useState<FocusTarget>("idle");
  const [pointer, setPointer] = useState({ x: 0, y: 0 });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!username.trim()) {
      setError("请输入用户名或邮箱。");
      return;
    }
    if (!password) {
      setError("请输入密码。");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await onLogin(username, password);
    } catch (submitError) {
      setError(loginErrorMessage(submitError));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main
      className="login-shell"
      onMouseMove={(event) =>
        setPointer({
          x: event.clientX / Math.max(window.innerWidth, 1) - 0.5,
          y: event.clientY / Math.max(window.innerHeight, 1) - 0.5,
        })
      }
    >
      <section className="login-panel">
        <div className="character-stage" aria-hidden="true">
          <LoginCharacter tone="purple" focus={focus} pointer={pointer} />
          <LoginCharacter tone="dark" focus={focus} pointer={pointer} delay />
          <LoginCharacter tone="orange" focus={focus} pointer={pointer} half />
          <LoginCharacter tone="yellow" focus={focus} pointer={pointer} small />
        </div>
        <form className="login-form" onSubmit={submit}>
          <div>
            <p className="label mb-3">DATA ANALYSIS AGENT</p>
            <h1 className="text-4xl font-black tracking-normal text-slate-950">
              Welcome to DataMind
            </h1>
            <p className="mt-3 max-w-md text-sm font-semibold leading-6 text-slate-500">
              登录后你的数据集、清洗结果、分析报告会和其他用户隔离保存。
            </p>
          </div>
          <label className="label mt-8" htmlFor="datamind-username">
            用户名 / 邮箱
          </label>
          <input
            id="datamind-username"
            value={username}
            onChange={(event) => {
              setUsername(event.target.value);
              setError(null);
            }}
            onFocus={() => setFocus("username")}
            onBlur={() => setFocus("idle")}
            className="input"
            placeholder="例如 default 或 nora@datamind.local"
            autoComplete="username"
            aria-invalid={!!error}
            aria-describedby={error ? "login-error" : undefined}
          />
          <label className="label mt-4" htmlFor="datamind-password">
            密码
          </label>
          <input
            id="datamind-password"
            value={password}
            onChange={(event) => {
              setPassword(event.target.value);
              setError(null);
            }}
            onFocus={() => setFocus("password")}
            onBlur={() => setFocus("idle")}
            className="input"
            placeholder="首次登录会创建本地用户"
            type="password"
            autoComplete="current-password"
            aria-invalid={!!error}
            aria-describedby={error ? "login-error" : undefined}
          />
          {(error || notice) && (
            <div id="login-error">
              <Alert tone={error ? "error" : "info"}>{error ?? notice}</Alert>
            </div>
          )}
          <button type="submit" disabled={busy} className="login-button mt-6">
            <span>{busy ? "登录中" : "Log in"}</span>
            {busy ? (
              <Loader2 className="animate-spin" size={18} />
            ) : (
              <span aria-hidden="true">→</span>
            )}
          </button>
        </form>
      </section>
    </main>
  );
}

function LoginCharacter({
  tone,
  focus,
  pointer,
  delay = false,
  half = false,
  small = false,
}: {
  tone: "purple" | "dark" | "orange" | "yellow";
  focus: FocusTarget;
  pointer: { x: number; y: number };
  delay?: boolean;
  half?: boolean;
  small?: boolean;
}) {
  const eyeX = focus === "password" ? -7 : focus === "username" ? 6 : pointer.x * 10;
  const eyeY = focus === "password" ? -2 : pointer.y * 5;
  return (
    <div
      className={`login-character ${tone} ${half ? "half" : ""} ${small ? "small" : ""} ${delay ? "delay" : ""} ${focus}`}
      style={{ transform: `rotate(${pointer.x * 7}deg) translateY(${pointer.y * 8}px)` }}
    >
      <span className="eye left" style={{ transform: `translate(${eyeX}px, ${eyeY}px)` }} />
      <span className="eye right" style={{ transform: `translate(${eyeX}px, ${eyeY}px)` }} />
      <span className="mouth" />
    </div>
  );
}
