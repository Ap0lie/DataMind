"""Add trustworthy, versioned Assistant memory.

Revision ID: 0012_trustworthy_memory
Revises: 0011_assistant_memory
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0012_trustworthy_memory"
down_revision = "0011_assistant_memory"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    conversation_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("assistant_conversations")
    }
    if "summary_payload" not in conversation_columns:
        op.add_column(
            "assistant_conversations",
            sa.Column("summary_payload", sa.Text(), nullable=False, server_default="{}"),
        )

    memory_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("assistant_memories")
    }
    additions = (
        sa.Column("memory_kind", sa.Text(), nullable=False, server_default="semantic"),
        sa.Column("subject_key", sa.Text(), nullable=False, server_default=""),
        sa.Column("structured_value", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("supersedes_id", sa.Text()),
        sa.Column("superseded_by_id", sa.Text()),
        sa.Column("valid_from", sa.Text()),
        sa.Column("valid_to", sa.Text()),
        sa.Column("application_policy", sa.Text(), nullable=False, server_default="relevant"),
        sa.Column("source_kind", sa.Text(), nullable=False, server_default="user_message"),
        sa.Column("source_job_id", sa.Text()),
    )
    for column in additions:
        if column.name not in memory_columns:
            op.add_column("assistant_memories", column)
    op.execute("UPDATE assistant_memories SET subject_key=normalized_key WHERE subject_key='' OR subject_key IS NULL")
    op.execute("UPDATE assistant_memories SET valid_from=created_at WHERE valid_from IS NULL")
    op.execute(
        "UPDATE assistant_memories SET application_policy='always' "
        "WHERE explicit IS TRUE AND memory_type='workflow_preference' "
        "AND normalized_key IN ('language','detail','visual_style','report_style')"
    )

    op.create_table(
        "assistant_memory_settings",
        sa.Column("user_id", sa.Text(), primary_key=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("updated_at", sa.Text(), nullable=False),
    )
    op.create_table(
        "assistant_memory_usage",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("memory_id", sa.Text(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("lexical_score", sa.Float(), nullable=False),
        sa.Column("embedding_score", sa.Float(), nullable=False),
        sa.Column("scope_score", sa.Float(), nullable=False),
        sa.Column("recency_score", sa.Float(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("scope_type", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.UniqueConstraint("run_id", "memory_id", name="uq_assistant_memory_usage"),
    )
    op.create_index("idx_assistant_memory_usage_run", "assistant_memory_usage", ["user_id", "run_id", "created_at"])
    op.create_table(
        "assistant_memory_maintenance_jobs",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("conversation_id", sa.Text(), nullable=False),
        sa.Column("user_message_id", sa.Text(), nullable=False),
        sa.Column("assistant_message_id", sa.Text(), nullable=False),
        sa.Column("analysis_job_id", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("broker_task_id", sa.Text()),
        sa.Column("lease_owner", sa.Text()),
        sa.Column("lease_expires_at", sa.Text()),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("completed_at", sa.Text()),
        sa.UniqueConstraint("user_message_id", name="uq_assistant_memory_maintenance_message"),
    )
    op.create_index(
        "idx_assistant_memory_maintenance_status",
        "assistant_memory_maintenance_jobs",
        ["status", "lease_expires_at", "updated_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_assistant_memory_maintenance_status", table_name="assistant_memory_maintenance_jobs")
    op.drop_table("assistant_memory_maintenance_jobs")
    op.drop_index("idx_assistant_memory_usage_run", table_name="assistant_memory_usage")
    op.drop_table("assistant_memory_usage")
    op.drop_table("assistant_memory_settings")
    for name in (
        "source_job_id",
        "source_kind",
        "application_policy",
        "valid_to",
        "valid_from",
        "superseded_by_id",
        "supersedes_id",
        "version",
        "structured_value",
        "subject_key",
        "memory_kind",
    ):
        op.drop_column("assistant_memories", name)
    op.drop_column("assistant_conversations", "summary_payload")
