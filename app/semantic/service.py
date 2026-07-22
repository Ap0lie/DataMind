from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from app.semantic.dsl import SemanticDslError, compile_expression
from app.semantic.embedding import (
    PersistentEmbeddingProvider,
    SemanticEmbeddingProvider,
    get_semantic_embedding_provider,
)
from app.semantic.ranking import SemanticCandidateRanker
from app.storage.dataset_store import DatasetStoreRepository

CONFIDENCE_WEIGHTS = {
    "intent": 0.10,
    "metric": 0.30,
    "dimension": 0.15,
    "time": 0.10,
    "join": 0.20,
    "data_quality": 0.10,
    "route": 0.15,
}
DEFAULT_CALIBRATION = ((0.0, 0.05), (0.4, 0.35), (0.55, 0.58), (0.7, 0.76), (0.85, 0.9), (1.0, 0.97))


class SemanticLayerService:
    def __init__(self, repository: DatasetStoreRepository, embedding_provider: SemanticEmbeddingProvider | None = None) -> None:
        self.repository = repository
        base_provider = embedding_provider or get_semantic_embedding_provider()
        self.embedding_provider = PersistentEmbeddingProvider(base_provider, repository)

    def create_draft(
        self,
        *,
        scope_type: str,
        scope_id: UUID,
        name: str | None = None,
        source_model_id: UUID | None = None,
    ) -> dict[str, Any]:
        source = (
            self.repository.get_semantic_model(source_model_id)
            if source_model_id
            else None
        )
        definition = self._auto_definition(scope_type, scope_id)
        if source is not None:
            definition["unresolved_bindings"] = self._transfer_semantic_labels(source["definition"], definition)
            source_name = source["name"]
        else:
            source_name = "Semantic model"
        existing = self.repository.list_semantic_models(scope_type=scope_type, scope_id=scope_id)
        version = max((int(item["version"]) for item in existing), default=0) + 1
        return self.repository.save_semantic_model(
            {
                "id": uuid4(), "scope_type": scope_type, "scope_id": scope_id,
                "name": name or source_name, "version": version, "revision": 1,
                "status": "draft", "source": "copy" if source_model_id else "auto",
                "parent_model_id": source_model_id, "definition": definition,
                "schema_fingerprint": self.schema_fingerprint(scope_type, scope_id),
            }
        )

    def update_draft(self, model_id: UUID, *, revision: int, name: str | None, definition: dict[str, Any]) -> dict[str, Any]:
        current = self.repository.get_semantic_model(model_id)
        if current["status"] != "draft":
            raise ValueError("Published semantic models are immutable; create a new draft version.")
        if int(current["revision"]) != revision:
            raise ValueError("Semantic model revision conflict.")
        current.update({"name": name or current["name"], "definition": definition, "revision": revision + 1})
        return self.repository.save_semantic_model(current)

    def validate(self, model: dict[str, Any]) -> dict[str, Any]:
        definition = model.get("definition") or {}
        errors: list[str] = []
        entities = {str(item.get("id")): item for item in definition.get("entities") or [] if isinstance(item, dict)}
        metrics = {str(item.get("id")): item for item in definition.get("metrics") or [] if isinstance(item, dict)}
        dimensions = [item for item in definition.get("dimensions") or [] if isinstance(item, dict)]
        aliases: set[str] = set()
        actual_fields: dict[str, set[str]] = {}
        for entity_id, entity in entities.items():
            try:
                dataset_id = UUID(str(entity.get("dataset_id")))
                records = self.repository.preview_analysis_records(dataset_id, limit=20)
                actual_fields[entity_id] = {str(key) for record in records for key in record}
                actual_fields[str(entity.get("sql_alias") or entity_id)] = actual_fields[entity_id]
            except Exception:
                errors.append(f"Entity {entity_id} references an unavailable dataset.")
        for item in (*dimensions, *metrics.values()):
            name = _normalized(str(item.get("name") or item.get("id") or ""))
            item_aliases = [name, *(_normalized(str(alias)) for alias in item.get("aliases") or [])]
            for alias in filter(None, item_aliases):
                if alias in aliases:
                    errors.append(f"Duplicate semantic name or alias: {alias}")
                aliases.add(alias)
        for dimension in dimensions:
            entity, field = _dimension_source_binding(definition, dimension)
            if entity not in entities or field not in actual_fields.get(entity, set()):
                errors.append(f"Dimension {dimension.get('id')} references unknown field {entity}.{field}.")
        for metric_id, metric in metrics.items():
            try:
                compiled = compile_expression(
                    metric.get("formula") or {},
                    metric_definitions=metrics,
                    stack=(metric_id,),
                    field_resolver=_definition_field_resolver(definition),
                )
                for reference in compiled.fields:
                    entity, field = reference.split(".", 1)
                    if entity not in actual_fields or field not in actual_fields.get(entity, set()):
                        errors.append(f"Metric {metric_id} references unknown field {reference}.")
            except SemanticDslError as exc:
                errors.append(f"Metric {metric_id}: {exc}")
        relationships = [item for item in definition.get("relationships") or [] if isinstance(item, dict) and item.get("enabled", True)]
        for relationship in relationships:
            left = str(relationship.get("left_entity_id") or relationship.get("left_entity") or "")
            right = str(relationship.get("right_entity_id") or relationship.get("right_entity") or "")
            if left not in entities or right not in entities:
                errors.append("Relationship references an unknown entity.")
            if relationship.get("cardinality") in {"one_to_many", "many_to_many"} and not relationship.get("deduplication_strategy"):
                errors.append(f"Relationship {left}->{right} requires an explicit aggregation grain or deduplication strategy.")
        current_fingerprint = self.schema_fingerprint(model["scope_type"], model["scope_id"])
        if model.get("schema_fingerprint") and model["schema_fingerprint"] != current_fingerprint:
            errors.append("Dataset schema changed after this semantic model was created.")
        if definition.get("unresolved_bindings"):
            errors.append("Copied semantic model contains unresolved field bindings.")
        return {"valid": not errors, "errors": tuple(dict.fromkeys(errors)), "warnings": (), "schema_fingerprint": current_fingerprint}

    def publish(self, model_id: UUID) -> dict[str, Any]:
        model = self.repository.get_semantic_model(model_id)
        validation = self.validate(model)
        if not validation["valid"]:
            raise ValueError("Semantic model validation failed: " + "; ".join(validation["errors"]))
        for item in self.repository.list_semantic_models(scope_type=model["scope_type"], scope_id=model["scope_id"]):
            if item["status"] == "published":
                item["status"] = "archived"
                self.repository.save_semantic_model(item)
        model.update({"status": "published", "validation": validation, "schema_fingerprint": validation["schema_fingerprint"], "published_at": datetime.now(UTC).isoformat()})
        return self.repository.save_semantic_model(model)

    def active_model(self, *, dataset_id: UUID, dataset_group_id: UUID | None) -> dict[str, Any] | None:
        scope_type = "dataset_group" if dataset_group_id else "dataset"
        scope_id = dataset_group_id or dataset_id
        models = self.repository.list_semantic_models(scope_type=scope_type, scope_id=scope_id)
        return next((item for item in models if item["status"] == "published"), None)

    def create_planner_decision(self, *, dataset_id: UUID, dataset_group_id: UUID | None, question: str) -> dict[str, Any]:
        model = self.active_model(dataset_id=dataset_id, dataset_group_id=dataset_group_id)
        if model is None:
            scores = {"intent": 0.72, "metric": 0.58, "dimension": 0.6, "time": None, "join": None, "data_quality": 0.7, "route": 0.65}
            raw = min(_weighted_confidence(scores), 0.69)
            plan = {"route": "hybrid", "metric_ids": [], "dimension_ids": [], "time_dimension_id": None, "filters": [], "ambiguities": ["No published semantic model; legacy planner will resolve fields."]}
            semantic_source = "legacy"
        else:
            plan, scores = self._resolve(question, model)
            raw = _weighted_confidence(scores)
            semantic_source = "published"
        stored_calibrator = self.repository.active_planner_calibrator()
        breakpoints = tuple(tuple(float(value) for value in item) for item in stored_calibrator["breakpoints"]) if stored_calibrator else DEFAULT_CALIBRATION
        calibrated = min(_calibrate(raw, breakpoints), 0.69) if semantic_source == "legacy" else _calibrate(raw, breakpoints)
        level = "high" if calibrated >= 0.8 else "medium" if calibrated >= 0.55 else "low"
        ambiguities = tuple(str(item) for item in plan.get("ambiguities") or ())
        if any("ambiguous" in item.lower() for item in ambiguities):
            level = "medium" if level == "high" else level
            calibrated = min(calibrated, 0.79)
        return self.repository.save_planner_decision(
            {"dataset_id": dataset_id, "dataset_group_id": dataset_group_id, "question": question,
             "semantic_model_id": model["id"] if model else None, "semantic_model_version": model["version"] if model else None,
             "semantic_source": semantic_source, "semantic_plan": plan, "component_scores": scores,
             "raw_confidence": raw, "calibrated_confidence": calibrated,
             "confidence_level": level, "requires_confirmation": level == "low"}
        )

    def rebuild_user_calibrator(self, *, minimum_samples: int = 30) -> dict[str, Any] | None:
        samples = list(self.repository.planner_training_samples())
        if len(samples) < minimum_samples:
            return None
        breakpoints = fit_pava(samples)
        predictions = [_calibrate(score, breakpoints) for score, _ in samples]
        brier = sum((prediction - label) ** 2 for prediction, (_, label) in zip(predictions, samples, strict=True)) / len(samples)
        ece = sum(abs(prediction - label) for prediction, (_, label) in zip(predictions, samples, strict=True)) / len(samples)
        return self.repository.save_planner_calibrator(breakpoints=breakpoints, sample_count=len(samples), metrics={"brier_score": brier, "ece": ece})

    def execute_semantic_plan(self, decision: dict[str, Any]) -> dict[str, Any]:
        model_id = decision.get("semantic_model_id")
        if not model_id:
            raise ValueError("Semantic execution requires a published semantic model.")
        model = self.repository.get_semantic_model(UUID(str(model_id)))
        if model["status"] not in {"published", "archived", "stale"}:
            raise ValueError("Semantic execution requires an immutable published model version.")
        definition = model["definition"]
        plan = decision.get("semantic_plan") or {}
        entities = {str(item["id"]): item for item in definition.get("entities") or []}
        metrics = {str(item["id"]): item for item in definition.get("metrics") or []}
        dimensions = {str(item["id"]): item for item in definition.get("dimensions") or []}
        metric_ids = [str(item) for item in plan.get("metric_ids") or [] if str(item) in metrics]
        dimension_ids = [str(item) for item in plan.get("dimension_ids") or [] if str(item) in dimensions]
        if not metric_ids:
            raise ValueError("Semantic plan did not resolve a published metric.")
        from app.semantic.dsl import quote_identifier

        resolver = _definition_field_resolver(definition)
        select_parts: list[str] = []
        group_parts: list[str] = []
        for dimension_id in dimension_ids:
            dimension = dimensions[dimension_id]
            entity_alias, source_name = resolver(
                str(dimension.get("entity_id") or dimension.get("entity")),
                str(dimension.get("field_id") or dimension.get("field")),
            )
            reference = f"{quote_identifier(entity_alias)}.{quote_identifier(source_name)}"
            select_parts.append(f'{reference} AS "{dimension_id}"')
            group_parts.append(reference)
        for metric_id in metric_ids:
            compiled = compile_expression(
                metrics[metric_id]["formula"],
                metric_definitions=metrics,
                stack=(metric_id,),
                field_resolver=resolver,
            )
            select_parts.append(f'{compiled.sql} AS "{metric_id}"')
        root = next(iter(entities))
        root_alias = str(entities[root].get("sql_alias") or root)
        sql = "SELECT " + ", ".join(select_parts) + f" FROM {quote_identifier(root_alias)}"
        relationships = [item for item in definition.get("relationships") or [] if item.get("enabled", True)]
        connected = {root}
        remaining = list(relationships)
        while remaining:
            progress = False
            for relationship in tuple(remaining):
                left = str(relationship.get("left_entity_id") or relationship.get("left_entity"))
                right = str(relationship.get("right_entity_id") or relationship.get("right_entity"))
                left_alias = str(entities[left].get("sql_alias") or left)
                right_alias = str(entities[right].get("sql_alias") or right)
                _, left_field = resolver(left, str(relationship.get("left_field_id") or relationship.get("left_field")))
                _, right_field = resolver(right, str(relationship.get("right_field_id") or relationship.get("right_field")))
                join_condition = _semantic_relationship_condition(
                    relationship,
                    left_alias=left_alias,
                    left_field=left_field,
                    right_alias=right_alias,
                    right_field=right_field,
                )
                if left in connected and right not in connected:
                    sql += f" {str(relationship.get('join_type') or 'left').upper()} JOIN {quote_identifier(right_alias)} ON {join_condition}"
                    connected.add(right)
                    remaining.remove(relationship)
                    progress = True
                elif right in connected and left not in connected:
                    sql += f" {str(relationship.get('join_type') or 'left').upper()} JOIN {quote_identifier(left_alias)} ON {join_condition}"
                    connected.add(left)
                    remaining.remove(relationship)
                    progress = True
            if not progress:
                break
        if group_parts:
            sql += " GROUP BY " + ", ".join(group_parts)
        sql += " LIMIT 1000"
        allowed_relationships = set()
        for item in relationships:
            left = str(item.get("left_entity_id") or item.get("left_entity"))
            right = str(item.get("right_entity_id") or item.get("right_entity"))
            left_alias, left_field = resolver(left, str(item.get("left_field_id") or item.get("left_field")))
            right_alias, right_field = resolver(right, str(item.get("right_field_id") or item.get("right_field")))
            allowed_relationships.add(frozenset(((left_alias.lower(), left_field), (right_alias.lower(), right_field))))
        valid, message = validate_semantic_sql(
            sql,
            allowed_tables={str(entity.get("sql_alias") or entity_id) for entity_id, entity in entities.items()},
            allowed_relationships=allowed_relationships,
        )
        if not valid:
            raise ValueError(message)
        import duckdb
        import pandas as pd
        connection = duckdb.connect(":memory:")
        try:
            for entity_id, entity in entities.items():
                records = self.repository.read_analysis_records(UUID(str(entity["dataset_id"])))
                connection.register(str(entity.get("sql_alias") or entity_id), pd.DataFrame(records))
            frame = connection.execute(sql).fetchdf()
            rows = frame.where(frame.notna(), None).to_dict(orient="records")
        finally:
            connection.close()
        return {"sql": sql, "rows": rows, "explanation": f"Deterministic semantic query using model {model['name']} v{model['version']}."}

    def schema_fingerprint(self, scope_type: str, scope_id: UUID) -> str:
        dataset_ids = self.repository.get_dataset_group(scope_id).dataset_ids if scope_type == "dataset_group" else (scope_id,)
        payload = []
        for dataset_id in dataset_ids:
            dataset = self.repository.get_dataset(dataset_id)
            records = self.repository.preview_analysis_records(dataset_id, limit=50)
            columns = sorted({str(key) for row in records for key in row})
            metadata = self.repository.list_column_metadata(dataset_id)
            payload.append({"id": str(dataset_id), "name": dataset.name, "columns": columns, "metadata": metadata})
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()

    def _auto_definition(self, scope_type: str, scope_id: UUID) -> dict[str, Any]:
        group = self.repository.get_dataset_group(scope_id) if scope_type == "dataset_group" else None
        dataset_ids = group.dataset_ids if group else (scope_id,)
        entities, dimensions, metrics = [], [], []
        entity_by_dataset: dict[str, str] = {}
        for index, dataset_id in enumerate(dataset_ids):
            dataset = self.repository.get_dataset(dataset_id)
            entity_id = _stable_semantic_id(scope_id, dataset_id, dataset.name, "entity")
            alias = f"t_{dataset_id.hex[:12]}"
            entity_by_dataset[str(dataset_id)] = entity_id
            records = self.repository.preview_analysis_records(dataset_id, limit=50)
            fields = sorted({str(key) for row in records for key in row})
            metadata = {item["column_name"]: item for item in self.repository.list_column_metadata(dataset_id)}
            field_definitions = []
            for field in fields:
                item = metadata.get(field, {})
                role = str(item.get("role") or "dimension")
                field_definitions.append({"field_id": _stable_semantic_id(scope_id, dataset_id, field, role), "source_name": field, "type": str(item.get("override_type") or item.get("inferred_type") or "text"), "role": role, "description": str(item.get("description") or "")})
            entities.append({"id": entity_id, "entity_id": entity_id, "name": dataset.name, "sql_alias": alias, "dataset_id": str(dataset_id), "entity_type": "fact" if index == 0 else "dimension", "primary_key": None, "grain": "one row per source record", "fields": field_definitions})
            for field in fields:
                item = metadata.get(field, {})
                role = str(item.get("role") or "dimension")
                field_id = _stable_semantic_id(scope_id, dataset_id, field, role)
                if role in {"dimension", "date", "text"}:
                    dimensions.append({"id": _stable_semantic_id(scope_id, dataset_id, field, "dimension"), "name": field, "aliases": [], "entity_id": entity_id, "field_id": field_id, "type": "time" if role == "date" else "categorical", "time_grains": ["day", "week", "month", "quarter", "year"] if role == "date" else []})
                elif role == "metric":
                    metrics.append({"id": _stable_semantic_id(scope_id, dataset_id, field, "metric"), "name": field, "aliases": [], "description": str(item.get("description") or ""), "unit": "", "format": "number", "direction": "neutral", "formula": {"op": "sum", "expr": {"op": "field", "entity_id": entity_id, "field_id": field_id}}, "default_time_dimension": None, "allowed_dimensions": []})
        relationships = []
        for relationship in (group.relationships if group else ()):
            left = entity_by_dataset.get(str(relationship.get("left_dataset_id")))
            right = entity_by_dataset.get(str(relationship.get("right_dataset_id")))
            if left and right:
                left_entity = next(item for item in entities if item["id"] == left)
                right_entity = next(item for item in entities if item["id"] == right)
                left_field_id = next((item["field_id"] for item in left_entity["fields"] if item["source_name"] == relationship.get("left_column")), "")
                right_field_id = next((item["field_id"] for item in right_entity["fields"] if item["source_name"] == relationship.get("right_column")), "")
                relationships.append({"id": _stable_semantic_id(scope_id, UUID(str(relationship.get("left_dataset_id"))), f"{relationship.get('left_column')}:{relationship.get('right_column')}", "relationship"), "left_entity_id": left, "right_entity_id": right, "left_field_id": left_field_id, "right_field_id": right_field_id, "join_type": relationship.get("join_type", "left"), "left_value_mode": relationship.get("left_value_mode", "scalar"), "right_value_mode": relationship.get("right_value_mode", "scalar"), "left_delimiter": relationship.get("left_delimiter"), "right_delimiter": relationship.get("right_delimiter"), "cardinality": relationship.get("relationship_type", "unknown"), "enabled": relationship.get("enabled", True), "deduplication_strategy": None, "risk_note": relationship.get("risk_note", "")})
        return {"definition_schema_version": 2, "entities": entities, "relationships": relationships, "dimensions": dimensions, "metrics": metrics, "unresolved_bindings": []}

    def _transfer_semantic_labels(self, source: dict[str, Any], target: dict[str, Any]) -> list[dict[str, Any]]:
        unresolved: list[dict[str, Any]] = []
        ranker = SemanticCandidateRanker(self.embedding_provider)
        for collection, expected_type in (("metrics", "metric"), ("dimensions", "dimension")):
            target_items = [item for item in target.get(collection) or [] if isinstance(item, dict)]
            for source_item in source.get(collection) or []:
                if not isinstance(source_item, dict) or not target_items:
                    continue
                query = " ".join([str(source_item.get("name") or ""), *(str(alias) for alias in source_item.get("aliases") or [])])
                ranked = ranker.rank(query, target_items, expected_type=expected_type)
                top = ranked[0] if ranked else None
                gap = top.final_score - ranked[1].final_score if top and len(ranked) > 1 else 1.0
                if top and top.final_score >= 0.85 and gap >= 0.08:
                    top.item["name"] = source_item.get("name") or top.item.get("name")
                    top.item["aliases"] = list(source_item.get("aliases") or [])
                    if source_item.get("description"):
                        top.item["description"] = source_item["description"]
                    top.item["binding_source"] = "auto_rebound"
                    top.item["binding_score"] = round(top.final_score, 4)
                else:
                    unresolved.append({"source_id": source_item.get("id"), "source_name": source_item.get("name"), "semantic_type": expected_type, "candidates": [candidate.evidence() for candidate in ranked[:3]]})
        return unresolved

    def _resolve(self, question: str, model: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float | None]]:
        definition = model["definition"]
        ranker = SemanticCandidateRanker(self.embedding_provider)
        metrics = ranker.rank(question, definition.get("metrics") or [], expected_type="metric")
        dimensions = ranker.rank(question, definition.get("dimensions") or [], expected_type="dimension")
        selected_metric = metrics[0].item if metrics and metrics[0].final_score >= 0.25 else None
        selected_dimension = dimensions[0].item if dimensions and dimensions[0].final_score >= 0.25 else None
        ambiguity = []
        if len(metrics) > 1 and metrics[0].final_score - metrics[1].final_score < 0.08:
            ambiguity.append("Ambiguous metric: " + ", ".join(str(item.item.get("name")) for item in metrics[:2]))
        evidence = [metrics[0].evidence() if metrics else {"reason": "No metric matched"}, dimensions[0].evidence() if dimensions else {"reason": "No dimension matched"}]
        plan = {"route": "hybrid" if selected_metric else "python", "metric_ids": [selected_metric["id"]] if selected_metric else [], "dimension_ids": [selected_dimension["id"]] if selected_dimension else [], "time_dimension_id": selected_dimension["id"] if selected_dimension and selected_dimension.get("type") == "time" else None, "filters": [], "join_path": definition.get("relationships") or [], "ambiguities": ambiguity, "evidence": evidence, "embedding_model_revision": self.embedding_provider.model_revision}
        relationship_count = len(definition.get("relationships") or [])
        scores = {"intent": 0.86 if selected_metric else 0.65, "metric": metrics[0].final_score if selected_metric else 0.15, "dimension": dimensions[0].final_score if selected_dimension else None, "time": 0.9 if plan["time_dimension_id"] else None, "join": 0.9 if relationship_count else None, "data_quality": 0.82, "route": 0.86 if selected_metric else 0.45}
        return plan, scores


def validate_semantic_sql(sql: str, *, allowed_tables: set[str], allowed_relationships: set[frozenset[tuple[str, str]]]) -> tuple[bool, str]:
    try:
        from sqlglot import exp, parse
    except ImportError:
        return False, "sqlglot is required for semantic multi-table SQL validation."
    try:
        statements = parse(sql, read="duckdb")
    except Exception as exc:
        return False, f"SQL parse failed: {exc}"
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return False, "Only one SELECT/CTE query is allowed."
    root = statements[0]
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Command, exp.Copy)
    if any(root.find(kind) is not None for kind in forbidden):
        return False, "SQL contains a forbidden statement."
    ctes = {str(cte.alias_or_name).lower() for cte in root.find_all(exp.CTE)}
    tables = {str(table.name).lower() for table in root.find_all(exp.Table) if str(table.name).lower() not in ctes}
    if not tables.issubset({item.lower() for item in allowed_tables}):
        return False, "SQL references a table outside the published semantic model."
    forbidden_functions = {"read_csv", "read_csv_auto", "read_parquet", "sqlite_scan", "postgres_scan", "httpfs", "glob"}
    if any(str(func.sql_name()).lower() in forbidden_functions for func in root.find_all(exp.Func)):
        return False, "External or table-reading functions are forbidden."
    for join in root.find_all(exp.Join):
        kind = str(join.args.get("kind") or "").upper()
        if kind in {"CROSS", "NATURAL"} or join.args.get("on") is None:
            return False, "Only explicit INNER/LEFT joins are allowed."
        pairs = set()
        for equality in join.args["on"].find_all(exp.EQ):
            if isinstance(equality.left, exp.Column) and isinstance(equality.right, exp.Column):
                pairs.add(frozenset(((str(equality.left.table).lower(), str(equality.left.name)), (str(equality.right.table).lower(), str(equality.right.name)))))
        if pairs and not pairs.issubset(allowed_relationships):
            return False, "Join condition is not declared by the published semantic model."
    return True, ""


def _semantic_relationship_condition(
    relationship: dict[str, Any],
    *,
    left_alias: str,
    left_field: str,
    right_alias: str,
    right_field: str,
) -> str:
    from app.semantic.dsl import quote_identifier

    left_reference = f"{quote_identifier(left_alias)}.{quote_identifier(left_field)}"
    right_reference = f"{quote_identifier(right_alias)}.{quote_identifier(right_field)}"
    left_mode = str(relationship.get("left_value_mode") or "scalar")
    right_mode = str(relationship.get("right_value_mode") or "scalar")
    if left_mode == "delimited" and right_mode == "scalar":
        delimiter = _sql_string_literal(str(relationship.get("left_delimiter") or "_"))
        return (
            f"list_contains(string_split(CAST({left_reference} AS VARCHAR), {delimiter}), "
            f"CAST({right_reference} AS VARCHAR))"
        )
    if right_mode == "delimited" and left_mode == "scalar":
        delimiter = _sql_string_literal(str(relationship.get("right_delimiter") or "_"))
        return (
            f"list_contains(string_split(CAST({right_reference} AS VARCHAR), {delimiter}), "
            f"CAST({left_reference} AS VARCHAR))"
        )
    return f"{left_reference} = {right_reference}"


def _sql_string_literal(value: str) -> str:
    if not value or len(value) > 4 or any(ord(character) < 32 for character in value):
        raise ValueError("Semantic relationship delimiter is invalid.")
    return "'" + value.replace("'", "''") + "'"


def fit_pava(samples: list[tuple[float, int]]) -> tuple[tuple[float, float], ...]:
    if not samples:
        return DEFAULT_CALIBRATION
    blocks = [[score, score, float(label), 1] for score, label in sorted(samples)]
    index = 0
    while index < len(blocks) - 1:
        if blocks[index][2] / blocks[index][3] <= blocks[index + 1][2] / blocks[index + 1][3]:
            index += 1
            continue
        left, right = blocks[index], blocks[index + 1]
        blocks[index:index + 2] = [[left[0], right[1], left[2] + right[2], left[3] + right[3]]]
        index = max(index - 1, 0)
    return tuple((float(block[1]), float(block[2] / block[3])) for block in blocks)


def _weighted_confidence(scores: dict[str, float | None]) -> float:
    applicable = [(name, value) for name, value in scores.items() if value is not None and name in CONFIDENCE_WEIGHTS]
    total = sum(CONFIDENCE_WEIGHTS[name] for name, _ in applicable) or 1
    return max(0.0, min(1.0, sum(CONFIDENCE_WEIGHTS[name] * float(value) for name, value in applicable) / total))


def _calibrate(score: float, breakpoints: tuple[tuple[float, float], ...] = DEFAULT_CALIBRATION) -> float:
    previous = breakpoints[0]
    for current in breakpoints[1:]:
        if score <= current[0]:
            span = current[0] - previous[0] or 1
            ratio = (score - previous[0]) / span
            return max(0.0, min(1.0, previous[1] + ratio * (current[1] - previous[1])))
        previous = current
    return previous[1]


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.lower())


def _stable_semantic_id(scope_id: UUID, dataset_id: UUID, source_name: str, semantic_type: str) -> str:
    digest = hashlib.sha256(f"{scope_id}:{dataset_id}:{source_name}:{semantic_type}".encode()).hexdigest()[:16]
    return f"{semantic_type}_{digest}"


def _field_source(entity: dict[str, Any], field_id_or_name: str) -> str:
    for field in entity.get("fields") or []:
        if str(field.get("field_id")) == field_id_or_name:
            return str(field.get("source_name"))
    return field_id_or_name


def _definition_field_resolver(definition: dict[str, Any]):
    entities = {str(item.get("id")): item for item in definition.get("entities") or []}

    def resolve(entity_id: str, field_id: str) -> tuple[str, str]:
        entity = entities.get(entity_id)
        if entity is None:
            raise SemanticDslError(f"Unknown semantic entity: {entity_id}")
        return str(entity.get("sql_alias") or entity_id), _field_source(entity, field_id)

    return resolve


def _dimension_source_binding(definition: dict[str, Any], dimension: dict[str, Any]) -> tuple[str, str]:
    entity_id = str(dimension.get("entity_id") or dimension.get("entity") or "")
    field_id = str(dimension.get("field_id") or dimension.get("field") or "")
    entities = {str(item.get("id")): item for item in definition.get("entities") or []}
    entity = entities.get(entity_id)
    if entity is None:
        return entity_id, field_id
    return entity_id, _field_source(entity, field_id)
