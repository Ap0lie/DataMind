"""Add bounded agent loop job and event metadata.

Revision ID: 0004_agent_loop
Revises: 0003_semantic_embedding_cache
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_agent_loop"
down_revision = "0003_semantic_embedding_cache"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "analysis_jobs",
        sa.Column("agent_mode", sa.Text(), nullable=False, server_default="legacy"),
    )
    op.add_column(
        "analysis_jobs",
        sa.Column("loop_summary", sa.Text(), nullable=False, server_default="{}"),
    )
    op.add_column("analysis_jobs", sa.Column("loop_terminal_reason", sa.Text()))
    op.add_column("analysis_job_events", sa.Column("event_type", sa.Text()))
    op.add_column("analysis_job_events", sa.Column("iteration", sa.Integer()))
    op.add_column("analysis_job_events", sa.Column("tool_name", sa.Text()))
    op.add_column("analysis_job_events", sa.Column("repair_of_sequence", sa.Integer()))
    op.add_column(
        "analysis_job_events",
        sa.Column("payload", sa.Text(), nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("analysis_job_events", "payload")
    op.drop_column("analysis_job_events", "repair_of_sequence")
    op.drop_column("analysis_job_events", "tool_name")
    op.drop_column("analysis_job_events", "iteration")
    op.drop_column("analysis_job_events", "event_type")
    op.drop_column("analysis_jobs", "loop_terminal_reason")
    op.drop_column("analysis_jobs", "loop_summary")
    op.drop_column("analysis_jobs", "agent_mode")
