"""Add verified Map-Reduce chunk summaries.

Revision ID: 0017_tool_map_reduce
Revises: 0016_tool_results
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0017_tool_map_reduce"
down_revision = "0016_tool_results"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tool_result_summary_chunks", sa.Column("summary", sa.Text()))
    op.add_column("tool_result_summary_chunks", sa.Column("provider", sa.Text()))
    op.add_column("tool_result_summary_chunks", sa.Column("model", sa.Text()))
    op.add_column("tool_result_summary_chunks", sa.Column("input_tokens", sa.Integer()))
    op.add_column("tool_result_summary_chunks", sa.Column("output_tokens", sa.Integer()))
    op.add_column(
        "tool_result_summary_chunks",
        sa.Column("verified", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "tool_result_summary_chunks",
        sa.Column(
            "verification_issues",
            sa.Text(),
            nullable=False,
            server_default="[]",
        ),
    )


def downgrade() -> None:
    op.drop_column("tool_result_summary_chunks", "verification_issues")
    op.drop_column("tool_result_summary_chunks", "verified")
    op.drop_column("tool_result_summary_chunks", "output_tokens")
    op.drop_column("tool_result_summary_chunks", "input_tokens")
    op.drop_column("tool_result_summary_chunks", "model")
    op.drop_column("tool_result_summary_chunks", "provider")
    op.drop_column("tool_result_summary_chunks", "summary")
