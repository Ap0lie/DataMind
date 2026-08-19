"""Add short-term tool result artifacts and distillation metadata.

Revision ID: 0016_tool_results
Revises: 0015_memory_convergence
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0016_tool_results"
down_revision = "0015_memory_convergence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tool_result_artifacts",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("tool_name", sa.Text(), nullable=False),
        sa.Column("action_hash", sa.Text(), nullable=False),
        sa.Column("payload_sha256", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("result_kind", sa.Text(), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("compressed_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("metadata", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("expires_at", sa.Text()),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "user_id",
            "run_id",
            "action_hash",
            "payload_sha256",
            name="uq_tool_result_artifact_action_payload",
        ),
    )
    op.create_index(
        "idx_tool_result_artifacts_user_run",
        "tool_result_artifacts",
        ["user_id", "run_id"],
    )
    op.create_index(
        "idx_tool_result_artifacts_sha",
        "tool_result_artifacts",
        ["payload_sha256"],
    )
    op.create_index(
        "idx_tool_result_artifacts_expires",
        "tool_result_artifacts",
        ["expires_at"],
    )
    op.create_table(
        "tool_result_summaries",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "artifact_id",
            sa.Text(),
            sa.ForeignKey("tool_result_artifacts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("summary_version", sa.Integer(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text()),
        sa.Column("model", sa.Text()),
        sa.Column("input_tokens", sa.Integer()),
        sa.Column("output_tokens", sa.Integer()),
        sa.Column("verified", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("verification_issues", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "artifact_id",
            "summary_version",
            name="uq_tool_result_summary_version",
        ),
    )
    op.create_index(
        "idx_tool_result_summaries_artifact_version",
        "tool_result_summaries",
        ["artifact_id", "summary_version"],
    )
    op.create_table(
        "tool_result_summary_chunks",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "artifact_id",
            sa.Text(),
            sa.ForeignKey("tool_result_artifacts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("section", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "summary_id",
            sa.Text(),
            sa.ForeignKey("tool_result_summaries.id", ondelete="SET NULL"),
        ),
        sa.Column("idempotency_key", sa.Text(), nullable=False, unique=True),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "artifact_id",
            "chunk_index",
            name="uq_tool_result_summary_chunk",
        ),
    )


def downgrade() -> None:
    op.drop_table("tool_result_summary_chunks")
    op.drop_index(
        "idx_tool_result_summaries_artifact_version",
        table_name="tool_result_summaries",
    )
    op.drop_table("tool_result_summaries")
    op.drop_index("idx_tool_result_artifacts_expires", table_name="tool_result_artifacts")
    op.drop_index("idx_tool_result_artifacts_sha", table_name="tool_result_artifacts")
    op.drop_index("idx_tool_result_artifacts_user_run", table_name="tool_result_artifacts")
    op.drop_table("tool_result_artifacts")
