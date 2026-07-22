"""Create DataMind production schema.

Revision ID: 0001_production_foundation
Revises: None
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_production_foundation"
down_revision = None
branch_labels = None
depends_on = None


def _create_table(name: str, *columns: sa.Column) -> None:
    op.create_table(name, *columns, if_not_exists=True)


def upgrade() -> None:
    _create_table(
        "users",
        sa.Column("user_id", sa.Text(), primary_key=True),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("salt", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("last_login_at", sa.Text(), nullable=False),
    )
    _create_table(
        "user_sessions",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False, unique=True),
        sa.Column("csrf_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("last_seen_at", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.Text(), nullable=False),
        sa.Column("absolute_expires_at", sa.Text(), nullable=False),
        sa.Column("revoked_at", sa.Text()),
    )
    _create_table(
        "datasets",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False, server_default="default"),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("source_metadata", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
    )
    _create_table(
        "records",
        sa.Column("dataset_id", sa.Text(), primary_key=True),
        sa.Column("source", sa.Text(), primary_key=True),
        sa.Column("row_number", sa.Integer(), primary_key=True),
        sa.Column("record", sa.Text(), nullable=False),
    )
    _create_table(
        "artifacts",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("dataset_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("artifact_type", sa.Text(), nullable=False),
        sa.Column("file_name", sa.Text()),
        sa.Column("content", sa.Text(), nullable=False),
    )
    _create_table(
        "cleaning_runs",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("dataset_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False, server_default="default"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("is_active", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("result_markdown", sa.Text(), nullable=False),
        sa.Column("cleaned_dataset", sa.Text(), nullable=False),
        sa.Column("raw_summary", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("previous_summary", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("current_summary", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("diff_summary", sa.Text(), nullable=False, server_default="{}"),
    )
    _create_table(
        "dataset_columns",
        sa.Column("dataset_id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), primary_key=True),
        sa.Column("column_name", sa.Text(), primary_key=True),
        sa.Column("inferred_type", sa.Text(), nullable=False, server_default="text"),
        sa.Column("override_type", sa.Text()),
        sa.Column("role", sa.Text(), nullable=False, server_default="dimension"),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
    )
    _create_table(
        "dataset_groups",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False, server_default="default"),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("dataset_ids", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("relationships", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("metadata", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
    )
    _create_table(
        "charts",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("dataset_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("chart_type", sa.Text(), nullable=False),
        sa.Column("chart_spec", sa.Text(), nullable=False),
        sa.Column("chart_data", sa.Text(), nullable=False),
    )
    _create_table(
        "reports",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("dataset_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False, server_default="default"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text()),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("markdown", sa.Text(), nullable=False),
        sa.Column("metadata", sa.Text(), nullable=False),
        sa.Column("job_id", sa.Text(), unique=True),
    )
    _create_table(
        "analysis_jobs",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False, server_default="default"),
        sa.Column("dataset_id", sa.Text(), nullable=False),
        sa.Column("dataset_group_id", sa.Text()),
        sa.Column("additional_dataset_ids", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("join_plan", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("relationship_plan", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("multimodal_inputs", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("current_stage", sa.Text(), nullable=False, server_default="queued"),
        sa.Column("events", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("result", sa.Text()),
        sa.Column("error", sa.Text()),
        sa.Column("report_id", sa.Text()),
        sa.Column("retry_of", sa.Text()),
        sa.Column("cancel_requested", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("broker_task_id", sa.Text()),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("lease_owner", sa.Text()),
        sa.Column("lease_expires_at", sa.Text()),
        sa.Column("heartbeat_at", sa.Text()),
        sa.Column("checkpoint_thread_id", sa.Text()),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("started_at", sa.Text()),
        sa.Column("completed_at", sa.Text()),
    )
    _create_table(
        "analysis_job_events",
        sa.Column("job_id", sa.Text(), primary_key=True),
        sa.Column("sequence", sa.Integer(), primary_key=True),
        sa.Column("node", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("duration_ms", sa.Float()),
        sa.Column("provider", sa.Text()),
        sa.Column("model", sa.Text()),
        sa.Column("token_usage", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("error_code", sa.Text()),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
    )
    for table, column in (
        ("user_sessions", "token_hash"),
        ("artifacts", "dataset_id"),
        ("cleaning_runs", "dataset_id"),
        ("charts", "dataset_id"),
        ("reports", "dataset_id"),
        ("analysis_jobs", "dataset_id"),
        ("analysis_job_events", "job_id"),
        ("dataset_groups", "user_id"),
    ):
        op.create_index(f"idx_{table}_{column}", table, [column], if_not_exists=True)


def downgrade() -> None:
    for table in (
        "analysis_job_events",
        "analysis_jobs",
        "reports",
        "charts",
        "dataset_groups",
        "dataset_columns",
        "cleaning_runs",
        "artifacts",
        "records",
        "datasets",
        "user_sessions",
        "users",
    ):
        op.drop_table(table)
