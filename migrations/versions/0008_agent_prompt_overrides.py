"""Persist scoped user prompt overrides for cleaning and analysis jobs.

Revision ID: 0008_agent_prompt_overrides
Revises: 0007_kimi_capabilities
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008_agent_prompt_overrides"
down_revision = "0007_kimi_capabilities"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("cleaning_jobs") as batch:
        batch.add_column(
            sa.Column(
                "prompt_overrides",
                sa.Text(),
                nullable=False,
                server_default="{}",
            )
        )
    with op.batch_alter_table("analysis_jobs") as batch:
        batch.add_column(
            sa.Column(
                "prompt_overrides",
                sa.Text(),
                nullable=False,
                server_default="{}",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("analysis_jobs") as batch:
        batch.drop_column("prompt_overrides")
    with op.batch_alter_table("cleaning_jobs") as batch:
        batch.drop_column("prompt_overrides")
