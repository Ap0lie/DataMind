"""Add idempotent Kimi conversation creation.

Revision ID: 0014_conversation_idempotency
Revises: 0013_memory_effectiveness
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0014_conversation_idempotency"
down_revision = "0013_memory_effectiveness"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {
        column["name"] for column in inspector.get_columns("assistant_conversations")
    }
    if "idempotency_key" not in columns:
        op.add_column(
            "assistant_conversations", sa.Column("idempotency_key", sa.Text())
        )
    if "request_fingerprint" not in columns:
        op.add_column(
            "assistant_conversations", sa.Column("request_fingerprint", sa.Text())
        )

    indexes = {
        item["name"]
        for item in sa.inspect(bind).get_indexes("assistant_conversations")
    }
    if "uq_assistant_conversation_idempotency" not in indexes:
        op.create_index(
            "uq_assistant_conversation_idempotency",
            "assistant_conversations",
            ["user_id", "idempotency_key"],
            unique=True,
        )


def downgrade() -> None:
    bind = op.get_bind()
    indexes = {
        item["name"]
        for item in sa.inspect(bind).get_indexes("assistant_conversations")
    }
    if "uq_assistant_conversation_idempotency" in indexes:
        op.drop_index(
            "uq_assistant_conversation_idempotency",
            table_name="assistant_conversations",
        )
    columns = {
        column["name"]
        for column in sa.inspect(bind).get_columns("assistant_conversations")
    }
    for name in ("request_fingerprint", "idempotency_key"):
        if name in columns:
            op.drop_column("assistant_conversations", name)
