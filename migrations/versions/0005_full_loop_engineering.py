"""Add autonomous cleaning jobs and report loop metadata.

Revision ID: 0005_full_loop_engineering
Revises: 0004_agent_loop
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_full_loop_engineering"
down_revision = "0004_agent_loop"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cleaning_jobs",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("dataset_id", sa.Text(), nullable=False),
        sa.Column("requirement", sa.Text(), nullable=False, server_default=""),
        sa.Column("cleaning_strategy", sa.Text(), nullable=False, server_default="auto"),
        sa.Column("selected_strategy", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("current_stage", sa.Text(), nullable=False, server_default="queued"),
        sa.Column("events", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("loop_summary", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("terminal_reason", sa.Text()),
        sa.Column("result", sa.Text()),
        sa.Column("error", sa.Text()),
        sa.Column("cleaning_run_id", sa.Text()),
        sa.Column("retry_of", sa.Text()),
        sa.Column("cancel_requested", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("broker_task_id", sa.Text()),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("lease_owner", sa.Text()),
        sa.Column("lease_expires_at", sa.Text()),
        sa.Column("heartbeat_at", sa.Text()),
        sa.Column("checkpoint_thread_id", sa.Text()),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("started_at", sa.Text()),
        sa.Column("completed_at", sa.Text()),
    )
    op.create_table(
        "cleaning_job_events",
        sa.Column("job_id", sa.Text(), primary_key=True),
        sa.Column("sequence", sa.Integer(), primary_key=True),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("event_type", sa.Text()),
        sa.Column("iteration", sa.Integer()),
        sa.Column("strategy", sa.Text()),
        sa.Column("payload", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.Text(), nullable=False),
    )
    op.create_index("idx_cleaning_jobs_user", "cleaning_jobs", ["user_id", "created_at"])
    op.create_index("idx_cleaning_jobs_dataset", "cleaning_jobs", ["dataset_id"])
    op.create_index("idx_cleaning_job_events_job", "cleaning_job_events", ["job_id"])
    op.add_column("cleaning_runs", sa.Column("job_id", sa.Text()))
    op.create_index("uq_cleaning_runs_job_id", "cleaning_runs", ["job_id"], unique=True)
    op.add_column("analysis_jobs", sa.Column("report_strategy", sa.Text()))
    op.add_column("analysis_jobs", sa.Column("report_revision_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("analysis_jobs", sa.Column("report_terminal_reason", sa.Text()))


def downgrade() -> None:
    op.drop_column("analysis_jobs", "report_terminal_reason")
    op.drop_column("analysis_jobs", "report_revision_count")
    op.drop_column("analysis_jobs", "report_strategy")
    op.drop_index("uq_cleaning_runs_job_id", table_name="cleaning_runs")
    op.drop_column("cleaning_runs", "job_id")
    op.drop_index("idx_cleaning_job_events_job", table_name="cleaning_job_events")
    op.drop_index("idx_cleaning_jobs_dataset", table_name="cleaning_jobs")
    op.drop_index("idx_cleaning_jobs_user", table_name="cleaning_jobs")
    op.drop_table("cleaning_job_events")
    op.drop_table("cleaning_jobs")
