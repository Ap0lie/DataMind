"""Add persistent Kimi assistant conversations and runs.

Revision ID: 0006_ai_assistant
Revises: 0005_full_loop_engineering
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_ai_assistant"
down_revision = "0005_full_loop_engineering"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("assistant_conversations", sa.Column("id", sa.Text(), primary_key=True), sa.Column("user_id", sa.Text(), nullable=False), sa.Column("title", sa.Text(), nullable=False), sa.Column("scope_type", sa.Text(), nullable=False), sa.Column("scope_id", sa.Text()), sa.Column("summary", sa.Text(), nullable=False, server_default=""), sa.Column("deleted_at", sa.Text()), sa.Column("created_at", sa.Text(), nullable=False), sa.Column("updated_at", sa.Text(), nullable=False), sa.Column("last_message_at", sa.Text()))
    op.create_table("assistant_messages", sa.Column("id", sa.Text(), primary_key=True), sa.Column("conversation_id", sa.Text(), nullable=False), sa.Column("user_id", sa.Text(), nullable=False), sa.Column("role", sa.Text(), nullable=False), sa.Column("content", sa.Text(), nullable=False), sa.Column("status", sa.Text(), nullable=False), sa.Column("provider", sa.Text()), sa.Column("model", sa.Text()), sa.Column("token_usage", sa.Text(), nullable=False, server_default="{}"), sa.Column("citations", sa.Text(), nullable=False, server_default="[]"), sa.Column("metadata", sa.Text(), nullable=False, server_default="{}"), sa.Column("created_at", sa.Text(), nullable=False))
    op.create_table("assistant_runs", sa.Column("id", sa.Text(), primary_key=True), sa.Column("conversation_id", sa.Text(), nullable=False), sa.Column("user_id", sa.Text(), nullable=False), sa.Column("user_message_id", sa.Text(), nullable=False), sa.Column("assistant_message_id", sa.Text(), nullable=False), sa.Column("status", sa.Text(), nullable=False), sa.Column("current_stage", sa.Text(), nullable=False), sa.Column("analysis_job_id", sa.Text()), sa.Column("pending_confirmation", sa.Text(), nullable=False, server_default="{}"), sa.Column("error", sa.Text()), sa.Column("cancel_requested", sa.Integer(), nullable=False, server_default="0"), sa.Column("broker_task_id", sa.Text()), sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"), sa.Column("lease_owner", sa.Text()), sa.Column("lease_expires_at", sa.Text()), sa.Column("heartbeat_at", sa.Text()), sa.Column("checkpoint_thread_id", sa.Text()), sa.Column("created_at", sa.Text(), nullable=False), sa.Column("updated_at", sa.Text(), nullable=False), sa.Column("completed_at", sa.Text()))
    op.create_table("assistant_run_events", sa.Column("run_id", sa.Text(), primary_key=True), sa.Column("sequence", sa.Integer(), primary_key=True), sa.Column("event_type", sa.Text(), nullable=False), sa.Column("status", sa.Text(), nullable=False), sa.Column("message", sa.Text(), nullable=False), sa.Column("tool_name", sa.Text()), sa.Column("payload", sa.Text(), nullable=False, server_default="{}"), sa.Column("created_at", sa.Text(), nullable=False))
    op.create_table("assistant_attachments", sa.Column("id", sa.Text(), primary_key=True), sa.Column("conversation_id", sa.Text(), nullable=False), sa.Column("message_id", sa.Text()), sa.Column("user_id", sa.Text(), nullable=False), sa.Column("file_name", sa.Text(), nullable=False), sa.Column("media_type", sa.Text(), nullable=False), sa.Column("size_bytes", sa.Integer(), nullable=False), sa.Column("sha256", sa.Text(), nullable=False), sa.Column("width", sa.Integer(), nullable=False), sa.Column("height", sa.Integer(), nullable=False), sa.Column("storage_path", sa.Text(), nullable=False), sa.Column("created_at", sa.Text(), nullable=False))
    op.create_index("idx_assistant_conversations_user", "assistant_conversations", ["user_id", "last_message_at"])
    op.create_index("idx_assistant_messages_conversation", "assistant_messages", ["conversation_id", "created_at"])
    op.create_index("idx_assistant_runs_conversation", "assistant_runs", ["conversation_id", "created_at"])
    op.create_index("idx_assistant_events_run", "assistant_run_events", ["run_id", "sequence"])
    op.create_index("idx_assistant_attachments_message", "assistant_attachments", ["message_id"])


def downgrade() -> None:
    for name, table in (("idx_assistant_attachments_message", "assistant_attachments"), ("idx_assistant_events_run", "assistant_run_events"), ("idx_assistant_runs_conversation", "assistant_runs"), ("idx_assistant_messages_conversation", "assistant_messages"), ("idx_assistant_conversations_user", "assistant_conversations")):
        op.drop_index(name, table_name=table)
    op.drop_table("assistant_attachments")
    op.drop_table("assistant_run_events")
    op.drop_table("assistant_runs")
    op.drop_table("assistant_messages")
    op.drop_table("assistant_conversations")
