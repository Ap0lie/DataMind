"""Add versioned semantic models and planner calibration records.

Revision ID: 0002_semantic_layer
Revises: 0001_production_foundation
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_semantic_layer"
down_revision = "0001_production_foundation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "semantic_models",
        sa.Column("id", sa.Text(), primary_key=True), sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("scope_type", sa.Text(), nullable=False), sa.Column("scope_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False), sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.Text(), nullable=False), sa.Column("source", sa.Text(), nullable=False),
        sa.Column("parent_model_id", sa.Text()), sa.Column("definition", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("schema_fingerprint", sa.Text(), nullable=False, server_default=""),
        sa.Column("validation", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.Text(), nullable=False), sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("published_at", sa.Text()),
    )
    op.create_index("idx_semantic_models_scope", "semantic_models", ["user_id", "scope_type", "scope_id"])
    op.create_table(
        "planner_decisions",
        sa.Column("id", sa.Text(), primary_key=True), sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("dataset_id", sa.Text(), nullable=False), sa.Column("dataset_group_id", sa.Text()),
        sa.Column("question", sa.Text(), nullable=False), sa.Column("semantic_model_id", sa.Text()),
        sa.Column("semantic_model_version", sa.Integer()), sa.Column("semantic_source", sa.Text(), nullable=False),
        sa.Column("semantic_plan", sa.Text(), nullable=False), sa.Column("component_scores", sa.Text(), nullable=False),
        sa.Column("raw_confidence", sa.Float(), nullable=False), sa.Column("calibrated_confidence", sa.Float(), nullable=False),
        sa.Column("confidence_level", sa.Text(), nullable=False),
        sa.Column("requires_confirmation", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("confirmed", sa.Integer(), nullable=False, server_default="0"), sa.Column("created_at", sa.Text(), nullable=False),
    )
    op.create_index("idx_planner_decisions_user", "planner_decisions", ["user_id", "created_at"])
    op.create_table(
        "planner_feedback",
        sa.Column("id", sa.Text(), primary_key=True), sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("decision_id", sa.Text(), nullable=False), sa.Column("action", sa.Text(), nullable=False),
        sa.Column("corrected_plan", sa.Text(), nullable=False, server_default="{}"), sa.Column("created_at", sa.Text(), nullable=False),
    )
    op.create_table(
        "planner_calibrators",
        sa.Column("id", sa.Text(), primary_key=True), sa.Column("user_id", sa.Text()),
        sa.Column("version", sa.Integer(), nullable=False), sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("breakpoints", sa.Text(), nullable=False), sa.Column("metrics", sa.Text(), nullable=False),
        sa.Column("active", sa.Integer(), nullable=False, server_default="0"), sa.Column("created_at", sa.Text(), nullable=False),
    )
    op.add_column("analysis_jobs", sa.Column("planner_decision_id", sa.Text()))
    op.add_column("analysis_jobs", sa.Column("semantic_model_id", sa.Text()))
    op.add_column("analysis_jobs", sa.Column("semantic_model_version", sa.Integer()))


def downgrade() -> None:
    op.drop_column("analysis_jobs", "semantic_model_version")
    op.drop_column("analysis_jobs", "semantic_model_id")
    op.drop_column("analysis_jobs", "planner_decision_id")
    op.drop_table("planner_calibrators")
    op.drop_table("planner_feedback")
    op.drop_index("idx_planner_decisions_user", table_name="planner_decisions")
    op.drop_table("planner_decisions")
    op.drop_index("idx_semantic_models_scope", table_name="semantic_models")
    op.drop_table("semantic_models")
