from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from sqlalchemy import MetaData, create_engine, inspect

TABLES = (
    "users",
    "datasets",
    "records",
    "artifacts",
    "cleaning_runs",
    "cleaning_jobs",
    "cleaning_job_events",
    "dataset_columns",
    "dataset_groups",
    "charts",
    "reports",
    "analysis_jobs",
    "analysis_job_events",
    "semantic_models",
    "semantic_embedding_cache",
    "data_snapshots",
    "data_drift_events",
    "planner_decisions",
    "planner_feedback",
    "planner_calibrators",
    "assistant_conversations",
    "assistant_messages",
    "assistant_runs",
    "assistant_run_events",
    "assistant_attachments",
    "assistant_permission_grants",
    "assistant_action_log",
    "assistant_import_batches",
    "assistant_memories",
    "assistant_memory_settings",
    "assistant_memory_usage",
    "assistant_memory_maintenance_jobs",
    "user_sessions",
)


def migrate(source: Path, target_url: str) -> dict[str, int]:
    if not source.exists():
        raise FileNotFoundError(source)
    source_connection = sqlite3.connect(source)
    source_connection.row_factory = sqlite3.Row
    target_engine = create_engine(target_url, future=True)
    metadata = MetaData()
    metadata.reflect(bind=target_engine)
    target_tables = set(inspect(target_engine).get_table_names())
    counts: dict[str, int] = {}
    with target_engine.begin() as target:
        for table_name in TABLES:
            source_exists = source_connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table_name,),
            ).fetchone()
            if source_exists is None or table_name not in target_tables:
                continue
            table = metadata.tables[table_name]
            target_columns = {column.name for column in table.columns}
            rows = source_connection.execute(f'SELECT * FROM "{table_name}"').fetchall()
            payloads = [
                {key: row[key] for key in row.keys() if key in target_columns}
                for row in rows
            ]
            if payloads:
                statement = table.insert()
                if target_engine.dialect.name == "postgresql":
                    from sqlalchemy.dialects.postgresql import insert

                    statement = insert(table).on_conflict_do_nothing()
                target.execute(statement, payloads)
            counts[table_name] = len(payloads)
    source_connection.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate DataMind SQLite data to PostgreSQL.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    counts = migrate(args.source, args.target)
    for table, count in counts.items():
        print(f"{table}: {count}")


if __name__ == "__main__":
    main()
