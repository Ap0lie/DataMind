"""Harden identity and Assistant task sequencing.

Revision ID: 0009_p1_security_reliability
Revises: 0008_agent_prompt_overrides
"""

from __future__ import annotations

import unicodedata

import sqlalchemy as sa
from alembic import op

revision = "0009_p1_security_reliability"
down_revision = "0008_agent_prompt_overrides"
branch_labels = None
depends_on = None


def _login_name(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().lower()


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("login_name_normalized", sa.Text(), nullable=True))

    connection = op.get_bind()
    rows = connection.execute(
        sa.text("SELECT user_id,display_name FROM users ORDER BY user_id")
    ).mappings()
    used: set[str] = set()
    for row in rows:
        candidate = _login_name(str(row["display_name"])) or str(row["user_id"])
        if candidate in used:
            candidate = f"{candidate}#{row['user_id']}"
        connection.execute(
            sa.text(
                """
                UPDATE users
                SET login_name_normalized=:login_name
                WHERE user_id=:user_id
                """
            ),
            {"login_name": candidate, "user_id": str(row["user_id"])},
        )
        used.add(candidate)

    with op.batch_alter_table("users") as batch:
        batch.alter_column(
            "login_name_normalized",
            existing_type=sa.Text(),
            nullable=False,
        )
        batch.create_unique_constraint(
            "uq_users_login_name_normalized",
            ["login_name_normalized"],
        )

    with op.batch_alter_table("assistant_runs") as batch:
        batch.add_column(
            sa.Column(
                "next_event_sequence",
                sa.Integer(),
                nullable=False,
                server_default="1",
            )
        )
    connection.execute(
        sa.text(
            """
            UPDATE assistant_runs
            SET next_event_sequence = COALESCE(
                (
                    SELECT MAX(event.sequence) + 1
                    FROM assistant_run_events event
                    WHERE event.run_id = assistant_runs.id
                ),
                1
            )
            """
        )
    )


def downgrade() -> None:
    with op.batch_alter_table("assistant_runs") as batch:
        batch.drop_column("next_event_sequence")
    with op.batch_alter_table("users") as batch:
        batch.drop_constraint("uq_users_login_name_normalized", type_="unique")
        batch.drop_column("login_name_normalized")
