"""Add data drift snapshots and reliability events.

Revision ID: 0010_data_reliability_graph
Revises: 0009_p1_security_reliability
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0010_data_reliability_graph"
down_revision = "0009_p1_security_reliability"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("data_snapshots"):
        op.create_table(
            "data_snapshots",
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column("user_id", sa.Text(), nullable=False),
            sa.Column("dataset_id", sa.Text(), nullable=False),
            sa.Column("source", sa.Text(), nullable=False),
            sa.Column("row_count", sa.Integer(), nullable=False),
            sa.Column("sample_size", sa.Integer(), nullable=False),
            sa.Column("fingerprint", sa.Text(), nullable=False),
            sa.Column("profile", sa.Text(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.Text(), nullable=False),
        )
    if not inspector.has_table("data_drift_events"):
        op.create_table(
            "data_drift_events",
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column("user_id", sa.Text(), nullable=False),
            sa.Column("dataset_id", sa.Text(), nullable=False),
            sa.Column("baseline_snapshot_id", sa.Text(), nullable=False),
            sa.Column("current_snapshot_id", sa.Text(), nullable=False),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("changes", sa.Text(), nullable=False, server_default="[]"),
            sa.Column("affected_assets", sa.Text(), nullable=False, server_default="[]"),
            sa.Column(
                "recommended_actions",
                sa.Text(),
                nullable=False,
                server_default="[]",
            ),
            sa.Column("created_at", sa.Text(), nullable=False),
            sa.Column("acknowledged_at", sa.Text()),
        )

    inspector = sa.inspect(bind)
    snapshot_indexes = {
        item["name"] for item in inspector.get_indexes("data_snapshots")
    }
    if "idx_data_snapshots_dataset" not in snapshot_indexes:
        op.create_index(
            "idx_data_snapshots_dataset",
            "data_snapshots",
            ["user_id", "dataset_id", "created_at"],
        )
    event_indexes = {
        item["name"] for item in inspector.get_indexes("data_drift_events")
    }
    if "idx_data_drift_events_dataset" not in event_indexes:
        op.create_index(
            "idx_data_drift_events_dataset",
            "data_drift_events",
            ["user_id", "dataset_id", "created_at"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("data_drift_events"):
        event_indexes = {
            item["name"] for item in inspector.get_indexes("data_drift_events")
        }
        if "idx_data_drift_events_dataset" in event_indexes:
            op.drop_index(
                "idx_data_drift_events_dataset",
                table_name="data_drift_events",
            )
        op.drop_table("data_drift_events")
    if inspector.has_table("data_snapshots"):
        snapshot_indexes = {
            item["name"] for item in inspector.get_indexes("data_snapshots")
        }
        if "idx_data_snapshots_dataset" in snapshot_indexes:
            op.drop_index(
                "idx_data_snapshots_dataset",
                table_name="data_snapshots",
            )
        op.drop_table("data_snapshots")
