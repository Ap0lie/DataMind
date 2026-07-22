"""Add Kimi capability grants, audited actions, imports, and recycle metadata.

Revision ID: 0007_kimi_capabilities
Revises: 0006_ai_assistant
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_kimi_capabilities"
down_revision = "0006_ai_assistant"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "assistant_permission_grants",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("asset_type", sa.Text(), nullable=False),
        sa.Column("asset_id", sa.Text(), nullable=False),
        sa.Column("capabilities", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("status", sa.Text(), nullable=False, server_default="active"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("revoked_at", sa.Text()),
    )
    op.create_table(
        "assistant_action_log",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text()),
        sa.Column("conversation_id", sa.Text()),
        sa.Column("grant_id", sa.Text()),
        sa.Column("tool_name", sa.Text(), nullable=False),
        sa.Column("arguments_hash", sa.Text(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("asset_type", sa.Text()),
        sa.Column("asset_id", sa.Text()),
        sa.Column("before_state", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("after_state", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("result", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("reversible", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("undone_at", sa.Text()),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("completed_at", sa.Text()),
    )
    op.create_table(
        "assistant_import_batches",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("conversation_id", sa.Text(), nullable=False),
        sa.Column("attachment_ids", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("preview", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("dataset_ids", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("dataset_group_id", sa.Text()),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("completed_at", sa.Text()),
    )
    op.create_index("uq_assistant_grant_active", "assistant_permission_grants", ["user_id", "asset_type", "asset_id"], unique=True)
    op.create_index("uq_assistant_action_idempotency", "assistant_action_log", ["idempotency_key"], unique=True)
    op.create_index("idx_assistant_actions_user", "assistant_action_log", ["user_id", "created_at"])
    op.create_index("idx_assistant_imports_conversation", "assistant_import_batches", ["conversation_id", "created_at"])

    with op.batch_alter_table("assistant_runs") as batch:
        batch.add_column(sa.Column("execution_mode", sa.Text(), nullable=False, server_default="ask"))
        batch.add_column(sa.Column("execution_plan", sa.Text(), nullable=False, server_default="{}"))
        batch.add_column(sa.Column("current_action_id", sa.Text()))
        batch.add_column(sa.Column("required_permission", sa.Text()))
    with op.batch_alter_table("assistant_attachments") as batch:
        batch.add_column(sa.Column("attachment_kind", sa.Text(), nullable=False, server_default="image"))
        batch.add_column(sa.Column("import_status", sa.Text()))
        batch.add_column(sa.Column("dataset_id", sa.Text()))
        batch.add_column(sa.Column("import_batch_id", sa.Text()))
    for table in ("datasets", "dataset_groups", "reports", "semantic_models"):
        with op.batch_alter_table(table) as batch:
            batch.add_column(sa.Column("deleted_at", sa.Text()))
            batch.add_column(sa.Column("purge_after", sa.Text()))
            batch.add_column(sa.Column("deleted_by_batch_id", sa.Text()))


def downgrade() -> None:
    for table in ("semantic_models", "reports", "dataset_groups", "datasets"):
        with op.batch_alter_table(table) as batch:
            batch.drop_column("deleted_by_batch_id")
            batch.drop_column("purge_after")
            batch.drop_column("deleted_at")
    with op.batch_alter_table("assistant_attachments") as batch:
        batch.drop_column("import_batch_id")
        batch.drop_column("dataset_id")
        batch.drop_column("import_status")
        batch.drop_column("attachment_kind")
    with op.batch_alter_table("assistant_runs") as batch:
        batch.drop_column("required_permission")
        batch.drop_column("current_action_id")
        batch.drop_column("execution_plan")
        batch.drop_column("execution_mode")
    op.drop_index("idx_assistant_imports_conversation", table_name="assistant_import_batches")
    op.drop_index("idx_assistant_actions_user", table_name="assistant_action_log")
    op.drop_index("uq_assistant_action_idempotency", table_name="assistant_action_log")
    op.drop_index("uq_assistant_grant_active", table_name="assistant_permission_grants")
    op.drop_table("assistant_import_batches")
    op.drop_table("assistant_action_log")
    op.drop_table("assistant_permission_grants")
