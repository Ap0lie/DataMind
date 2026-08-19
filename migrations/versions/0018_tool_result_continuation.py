"""Add bounded tool-result continuation projections.

Revision ID: 0018_tool_continuation
Revises: 0017_tool_map_reduce
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0018_tool_continuation"
down_revision = "0017_tool_map_reduce"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tool_result_projections",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "artifact_id",
            sa.Text(),
            sa.ForeignKey("tool_result_artifacts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("query_hash", sa.Text(), nullable=False),
        sa.Column("projection", sa.Text(), nullable=False),
        sa.Column("selected_paths", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("context_size_bytes", sa.Integer(), nullable=False),
        sa.Column("scanned_bytes", sa.BigInteger(), nullable=False),
        sa.Column("truncated", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "user_id",
            "run_id",
            "artifact_id",
            "query_hash",
            name="uq_tool_result_projection_query",
        ),
    )
    op.create_index(
        "idx_tool_result_projections_user_run",
        "tool_result_projections",
        ["user_id", "run_id"],
    )
    op.create_index(
        "idx_tool_result_projections_artifact",
        "tool_result_projections",
        ["artifact_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_tool_result_projections_artifact",
        table_name="tool_result_projections",
    )
    op.drop_index(
        "idx_tool_result_projections_user_run",
        table_name="tool_result_projections",
    )
    op.drop_table("tool_result_projections")
