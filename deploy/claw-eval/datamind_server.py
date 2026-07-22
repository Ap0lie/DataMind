from __future__ import annotations

import argparse
import ctypes
import os
import threading
import time

import uvicorn


def _validate_evaluation_configuration() -> None:
    expected = os.environ.get("DATAMIND_EVAL_EXPECTED_PROVIDER", "").strip().lower()
    if not expected:
        return
    from app.core.settings import get_settings

    settings = get_settings()
    provider_fields = {
        "llm_provider": settings.llm_provider,
        "default_llm_provider": settings.default_llm_provider,
        "planner_llm_provider": settings.planner_llm_provider,
        "sql_llm_provider": settings.sql_llm_provider,
        "python_llm_provider": settings.python_llm_provider,
        "reflection_llm_provider": settings.reflection_llm_provider,
        "report_llm_provider": settings.report_llm_provider,
        "review_llm_provider": settings.review_llm_provider,
        "multimodal_llm_provider": settings.multimodal_llm_provider,
        "agent_loop_provider": settings.agent_loop_provider,
    }
    mismatched = {
        name: value
        for name, value in provider_fields.items()
        if str(value or "").strip().lower() != expected
    }
    if mismatched:
        raise RuntimeError(
            f"DataMind evaluation provider mismatch: expected {expected}, got {mismatched}"
        )
    if settings.llm_allow_provider_fallback:
        raise RuntimeError("DataMind evaluation requires provider fallback to be disabled.")
    expected_model = "kimi-k2.6" if expected == "kimi" else "mock-model"
    if settings.agent_loop_model != expected_model:
        raise RuntimeError(
            "DataMind evaluation agent model mismatch: "
            f"expected {expected_model}, got {settings.agent_loop_model}"
        )


def _parent_alive(parent_pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(parent_pid, 0)
        except OSError:
            return False
        return True

    synchronize = 0x00100000
    wait_timeout = 0x00000102
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(synchronize, False, parent_pid)
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
    finally:
        kernel32.CloseHandle(handle)


def _watch_parent(parent_pid: int) -> None:
    while _parent_alive(parent_pid):
        time.sleep(1)
    os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DataMind with a parent-process guard")
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    _validate_evaluation_configuration()
    threading.Thread(
        target=_watch_parent,
        args=(args.parent_pid,),
        name="datamind-claw-parent-guard",
        daemon=True,
    ).start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
