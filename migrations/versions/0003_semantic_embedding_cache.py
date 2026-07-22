"""Add user-scoped semantic embedding cache.

Revision ID: 0003_semantic_embedding_cache
Revises: 0002_semantic_layer
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_semantic_embedding_cache"
down_revision = "0002_semantic_layer"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "semantic_embedding_cache",
        sa.Column("user_id", sa.Text(), primary_key=True),
        sa.Column("model_revision", sa.Text(), primary_key=True),
        sa.Column("text_hash", sa.Text(), primary_key=True),
        sa.Column("vector", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("semantic_embedding_cache")
