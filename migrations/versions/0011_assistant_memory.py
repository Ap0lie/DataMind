"""Add persistent Assistant conversation summaries and long-term memory.

Revision ID: 0011_assistant_memory
Revises: 0010_data_reliability_graph
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0011_assistant_memory"
down_revision = "0010_data_reliability_graph"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    conversation_columns = {
        column["name"] for column in inspector.get_columns("assistant_conversations")
    }
    for name, column in (
        ("summary_through_message_id", sa.Column("summary_through_message_id", sa.Text())),
        (
            "summary_version",
            sa.Column("summary_version", sa.Integer(), nullable=False, server_default="0"),
        ),
        ("summary_updated_at", sa.Column("summary_updated_at", sa.Text())),
    ):
        if name not in conversation_columns:
            op.add_column("assistant_conversations", column)

    inspector = sa.inspect(bind)
    if not inspector.has_table("assistant_memories"):
        op.create_table(
            "assistant_memories",
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column("user_id", sa.Text(), nullable=False),
            sa.Column("scope_type", sa.Text(), nullable=False),
            sa.Column("scope_id", sa.Text()),
            sa.Column("scope_key", sa.Text(), nullable=False),
            sa.Column("memory_type", sa.Text(), nullable=False),
            sa.Column("normalized_key", sa.Text(), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("source_conversation_id", sa.Text()),
            sa.Column("source_message_id", sa.Text()),
            sa.Column("source_message_ids", sa.Text(), nullable=False, server_default="[]"),
            sa.Column("explicit", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("pinned", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("last_used_at", sa.Text()),
            sa.Column("recycle_from_status", sa.Text()),
            sa.Column("deleted_at", sa.Text()),
            sa.Column("purge_after", sa.Text()),
            sa.Column("created_at", sa.Text(), nullable=False),
            sa.Column("updated_at", sa.Text(), nullable=False),
            sa.UniqueConstraint(
                "user_id",
                "scope_type",
                "scope_key",
                "memory_type",
                "normalized_key",
                name="uq_assistant_memory_fingerprint",
            ),
        )
    inspector = sa.inspect(bind)
    memory_indexes = {
        item["name"] for item in inspector.get_indexes("assistant_memories")
    }
    if "idx_assistant_memories_user_status" not in memory_indexes:
        op.create_index(
            "idx_assistant_memories_user_status",
            "assistant_memories",
            ["user_id", "status", "pinned", "updated_at"],
        )
    if "idx_assistant_memories_scope" not in memory_indexes:
        op.create_index(
            "idx_assistant_memories_scope",
            "assistant_memories",
            ["user_id", "scope_type", "scope_key", "status"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("assistant_memories"):
        indexes = {item["name"] for item in inspector.get_indexes("assistant_memories")}
        for name in ("idx_assistant_memories_scope", "idx_assistant_memories_user_status"):
            if name in indexes:
                op.drop_index(name, table_name="assistant_memories")
        op.drop_table("assistant_memories")
    conversation_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("assistant_conversations")
    }
    for name in ("summary_updated_at", "summary_version", "summary_through_message_id"):
        if name in conversation_columns:
            op.drop_column("assistant_conversations", name)
