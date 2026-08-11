"""Add Memory v3 effectiveness feedback and utility tracking.

Revision ID: 0013_memory_effectiveness
Revises: 0012_trustworthy_memory
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0013_memory_effectiveness"
down_revision = "0012_trustworthy_memory"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    memory_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("assistant_memories")
    }
    memory_additions = (
        sa.Column("entity_key", sa.Text(), nullable=False, server_default=""),
        sa.Column("predicate", sa.Text(), nullable=False, server_default="value"),
        sa.Column("typed_value", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("unit", sa.Text()),
        sa.Column("utility_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("helpful_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("irrelevant_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("wrong_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("correction_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("validated_reuse_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("feedback_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_validated_at", sa.Text()),
        sa.Column("dormant_reason", sa.Text()),
    )
    for column in memory_additions:
        if column.name not in memory_columns:
            op.add_column("assistant_memories", column)
    op.execute(
        "UPDATE assistant_memories SET entity_key=subject_key "
        "WHERE entity_key='' OR entity_key IS NULL"
    )
    op.execute(
        "UPDATE assistant_memories SET typed_value=structured_value "
        "WHERE typed_value='{}' OR typed_value IS NULL"
    )

    usage_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("assistant_memory_usage")
    }
    usage_additions = (
        sa.Column("assistant_message_id", sa.Text()),
        sa.Column("retrieval_rank", sa.Integer()),
        sa.Column("final_selected", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("relevance_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("utility_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("final_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("suppression_reason", sa.Text()),
        sa.Column("outcome_recorded", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    for column in usage_additions:
        if column.name not in usage_columns:
            op.add_column("assistant_memory_usage", column)
    op.execute(
        "UPDATE assistant_memory_usage SET relevance_score=score,final_score=score "
        "WHERE final_score=0"
    )

    op.create_table(
        "assistant_memory_feedback",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("usage_id", sa.Text(), nullable=False),
        sa.Column("memory_id", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("feedback", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text()),
        sa.Column("source", sa.Text(), nullable=False, server_default="user"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.UniqueConstraint("user_id", "usage_id", name="uq_assistant_memory_feedback_usage"),
    )
    op.create_index(
        "idx_assistant_memory_feedback_memory",
        "assistant_memory_feedback",
        ["user_id", "memory_id", "updated_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_assistant_memory_feedback_memory",
        table_name="assistant_memory_feedback",
    )
    op.drop_table("assistant_memory_feedback")
    for name in (
        "outcome_recorded",
        "suppression_reason",
        "final_score",
        "utility_score",
        "relevance_score",
        "final_selected",
        "retrieval_rank",
        "assistant_message_id",
    ):
        op.drop_column("assistant_memory_usage", name)
    for name in (
        "dormant_reason",
        "last_validated_at",
        "feedback_count",
        "validated_reuse_count",
        "correction_count",
        "wrong_count",
        "irrelevant_count",
        "helpful_count",
        "utility_score",
        "unit",
        "typed_value",
        "predicate",
        "entity_key",
    ):
        op.drop_column("assistant_memories", name)
