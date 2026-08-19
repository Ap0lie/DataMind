"""Converge Memory usage observability and production indexes.

Revision ID: 0015_memory_convergence
Revises: 0014_conversation_idempotency
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0015_memory_convergence"
down_revision = "0014_conversation_idempotency"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    usage_columns = {
        column["name"]
        for column in sa.inspect(bind).get_columns("assistant_memory_usage")
    }
    additions = (
        sa.Column("agent", sa.Text(), nullable=False, server_default="unknown"),
        sa.Column(
            "validation_result",
            sa.Text(),
            nullable=False,
            server_default="not_validated",
        ),
        sa.Column("validated_at", sa.Text()),
    )
    for column in additions:
        if column.name not in usage_columns:
            op.add_column("assistant_memory_usage", column)

    _create_index(
        "assistant_memories",
        "idx_assistant_memories_namespace",
        ["user_id", "scope_type", "scope_key", "memory_kind", "status", "updated_at"],
    )
    _create_index(
        "assistant_memories",
        "idx_assistant_memories_source_message",
        ["user_id", "source_message_id", "created_at"],
    )
    _create_index(
        "assistant_memory_usage",
        "idx_assistant_memory_usage_agent",
        ["user_id", "agent", "run_id", "retrieval_rank"],
    )


def downgrade() -> None:
    _drop_index("assistant_memory_usage", "idx_assistant_memory_usage_agent")
    _drop_index("assistant_memories", "idx_assistant_memories_source_message")
    _drop_index("assistant_memories", "idx_assistant_memories_namespace")
    columns = {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("assistant_memory_usage")
    }
    for name in ("validated_at", "validation_result", "agent"):
        if name in columns:
            op.drop_column("assistant_memory_usage", name)


def _create_index(table: str, name: str, columns: list[str]) -> None:
    indexes = {
        item["name"] for item in sa.inspect(op.get_bind()).get_indexes(table)
    }
    if name not in indexes:
        op.create_index(name, table, columns)


def _drop_index(table: str, name: str) -> None:
    indexes = {
        item["name"] for item in sa.inspect(op.get_bind()).get_indexes(table)
    }
    if name in indexes:
        op.drop_index(name, table_name=table)
