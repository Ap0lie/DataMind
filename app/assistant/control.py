from __future__ import annotations

from uuid import UUID

from app.storage.assistant_repository import AssistantRepository


class AssistantRunCanceled(RuntimeError):
    pass


class AssistantRunPaused(RuntimeError):
    pass


def ensure_run_continuable(repository: AssistantRepository, run_id: UUID) -> None:
    run = repository.get_run(run_id)
    if run.cancel_requested or run.status in {"cancel_requested", "canceled"}:
        raise AssistantRunCanceled("Assistant run was canceled.")
    if run.status in {"pause_requested", "paused"}:
        raise AssistantRunPaused("Assistant run was paused.")
