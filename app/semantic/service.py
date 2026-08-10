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
from app.semantic.relationship_graph import (
    build_relationship_graph,
    plan_relationship_path,
)
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
        warnings: list[str] = []
        entity_items = [item for item in definition.get("entities") or [] if isinstance(item, dict)]
        metric_items = [item for item in definition.get("metrics") or [] if isinstance(item, dict)]
        dimensions = [item for item in definition.get("dimensions") or [] if isinstance(item, dict)]
        relationship_items = [
            item
            for item in definition.get("relationships") or []
            if isinstance(item, dict)
        ]
        for label, items in (
            ("Entity", entity_items),
            ("Metric", metric_items),
            ("Dimension", dimensions),
            ("Relationship", relationship_items),
        ):
            _append_semantic_id_errors(errors, label=label, items=items)
        entities = {str(item.get("id")): item for item in entity_items}
        metrics = {str(item.get("id")): item for item in metric_items}
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
        declared_fields = {
            entity_id: {
                str(value)
                for field in entity.get("fields") or ()
                if isinstance(field, dict)
                and str(field.get("source_name") or "")
                in actual_fields.get(entity_id, set())
                for value in (field.get("field_id"), field.get("source_name"))
                if value
            }
            for entity_id, entity in entities.items()
        }
        allowed_cardinalities = {
            "one_to_one",
            "one_to_many",
            "many_to_one",
            "many_to_many",
        }
        for relationship in relationship_items:
            relationship_id = str(relationship.get("id") or "")
            left = str(relationship.get("left_entity_id") or relationship.get("left_entity") or "")
            right = str(relationship.get("right_entity_id") or relationship.get("right_entity") or "")
            if left not in entities or right not in entities:
                errors.append(f"Relationship {relationship_id} references an unknown entity.")
            if left and left == right:
                errors.append(f"Relationship {relationship_id} cannot join an entity to itself.")
            left_field = str(relationship.get("left_field_id") or relationship.get("left_field") or "")
            right_field = str(relationship.get("right_field_id") or relationship.get("right_field") or "")
            if not left_field or left_field not in declared_fields.get(left, set()):
                errors.append(f"Relationship {relationship_id} references unknown left field {left}.{left_field}.")
            if not right_field or right_field not in declared_fields.get(right, set()):
                errors.append(f"Relationship {relationship_id} references unknown right field {right}.{right_field}.")
            cardinality = str(relationship.get("cardinality") or "")
            if cardinality == "unknown" and not relationship.get("enabled", True):
                warnings.append(
                    f"Relationship {relationship_id} is disabled because cardinality could not be proven."
                )
            elif cardinality not in allowed_cardinalities:
                errors.append(f"Relationship {relationship_id} has invalid cardinality: {cardinality or 'missing'}.")
            if (
                relationship.get("enabled", True)
                and cardinality == "many_to_many"
                and not relationship.get("deduplication_strategy")
            ):
                errors.append(f"Relationship {left}->{right} requires an explicit aggregation grain or deduplication strategy.")
        current_fingerprint = self.schema_fingerprint(model["scope_type"], model["scope_id"])
        if model.get("schema_fingerprint") and model["schema_fingerprint"] != current_fingerprint:
            errors.append("Dataset schema changed after this semantic model was created.")
        if definition.get("unresolved_bindings"):
            errors.append("Copied semantic model contains unresolved field bindings.")
        return {"valid": not errors, "errors": tuple(dict.fromkeys(errors)), "warnings": tuple(dict.fromkeys(warnings)), "schema_fingerprint": current_fingerprint}

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
        model = next((item for item in models if item["status"] == "published"), None)
        if model is None:
            return None
        current_fingerprint = self.schema_fingerprint(scope_type, scope_id)
        if model.get("schema_fingerprint") == current_fingerprint:
            return model
        validation = dict(model.get("validation") or {})
        validation["stale_reason"] = "Dataset schema changed after publication."
        validation["current_schema_fingerprint"] = current_fingerprint
        model.update({"status": "stale", "validation": validation})
        self.repository.save_semantic_model(model)
        return None

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

    def reconcile_planner_decision(
        self,
        decision: dict[str, Any],
        *,
        question: str,
    ) -> dict[str, Any]:
        """Re-resolve immutable semantic bindings before execution.

        Planner decisions may have been created by an older API process. Re-resolving
        from the pinned model version prevents stale dimensions or missing predicates
        from contradicting the deterministic analysis contract.
        """

        model_id = decision.get("semantic_model_id")
        if decision.get("semantic_source") != "published" or not model_id:
            return decision
        model = self.repository.get_semantic_model(UUID(str(model_id)))
        plan, scores = self._resolve(question, model)
        return {
            **decision,
            "question": question,
            "semantic_plan": plan,
            "component_scores": scores,
        }

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
        filters = [
            item
            for item in plan.get("filters") or []
            if isinstance(item, dict)
            and str(item.get("dimension_id") or "") in dimensions
            and str(item.get("operator") or "=") in {"=", "!=", ">", ">=", "<", "<="}
        ]
        filter_dimension_ids = tuple(
            dict.fromkeys(str(item["dimension_id"]) for item in filters)
        )
        if not metric_ids:
            raise ValueError("Semantic plan did not resolve a published metric.")
        grain_plan = plan.get("grain_plan") or plan_relationship_path(
            definition,
            metric_ids=tuple(metric_ids),
            dimension_ids=tuple(dict.fromkeys((*dimension_ids, *filter_dimension_ids))),
        )
        if not grain_plan.get("safe"):
            raise ValueError(
                "Semantic relationship path is not grain-safe: "
                + "; ".join(str(item) for item in grain_plan.get("warnings") or ())
            )
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
        root_candidates = tuple(
            str(item) for item in grain_plan.get("metric_entity_ids") or ()
        )
        root = next((item for item in root_candidates if item in entities), None)
        if root is None:
            raise ValueError("Semantic plan did not resolve the metric source entity.")
        root_alias = str(entities[root].get("sql_alias") or root)
        sql = "SELECT " + ", ".join(select_parts) + f" FROM {quote_identifier(root_alias)}"
        selected_relationship_ids = {
            str(item.get("relationship_id") or item.get("id") or "")
            for item in grain_plan.get("join_path") or ()
        }
        relationships = [
            item
            for item in definition.get("relationships") or []
            if item.get("enabled", True)
            and str(item.get("id") or "") in selected_relationship_ids
        ]
        steps_by_relationship = {
            str(item.get("relationship_id") or ""): item
            for item in grain_plan.get("steps") or ()
        }
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
                    target = _semantic_join_target(
                        alias=right_alias,
                        field=right_field,
                        strategy=str(
                            steps_by_relationship.get(
                                str(relationship.get("id") or ""),
                                {},
                            ).get("strategy") or "direct_join"
                        ),
                    )
                    sql += f" {str(relationship.get('join_type') or 'left').upper()} JOIN {target} ON {join_condition}"
                    connected.add(right)
                    remaining.remove(relationship)
                    progress = True
                elif right in connected and left not in connected:
                    target = _semantic_join_target(
                        alias=left_alias,
                        field=left_field,
                        strategy=str(
                            steps_by_relationship.get(
                                str(relationship.get("id") or ""),
                                {},
                            ).get("strategy") or "direct_join"
                        ),
                    )
                    sql += f" {str(relationship.get('join_type') or 'left').upper()} JOIN {target} ON {join_condition}"
                    connected.add(left)
                    remaining.remove(relationship)
                    progress = True
            if not progress:
                break
        where_parts: list[str] = []
        for item in filters:
            dimension = dimensions[str(item["dimension_id"])]
            entity_alias, source_name = resolver(
                str(dimension.get("entity_id") or dimension.get("entity")),
                str(dimension.get("field_id") or dimension.get("field")),
            )
            reference = f"{quote_identifier(entity_alias)}.{quote_identifier(source_name)}"
            where_parts.append(
                f"{reference} {item.get('operator') or '='!s} "
                f"{_semantic_sql_literal(item.get('value'))}"
            )
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)
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
        return {
            "sql": sql,
            "rows": rows,
            "explanation": (
                f"Deterministic semantic query using model "
                f"{model['name']} v{model['version']}."
            ),
            "relationship_graph": plan.get("relationship_graph") or {},
            "grain_plan": grain_plan,
        }

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
        dataset_fields: list[dict[str, Any]] = []
        semantic_name_counts: dict[str, int] = {}
        for dataset_id in dataset_ids:
            dataset = self.repository.get_dataset(dataset_id)
            records = self.repository.preview_analysis_records(dataset_id, limit=50)
            fields = sorted({str(key) for row in records for key in row})
            metadata = {
                str(item["column_name"]): item
                for item in self.repository.list_column_metadata(dataset_id)
            }
            profiles = []
            for field in fields:
                item = metadata.get(field, {})
                values = tuple(row.get(field) for row in records if row.get(field) is not None)
                inferred_type = _infer_semantic_field_type(field, values, item)
                role = _infer_semantic_field_role(field, values, item, inferred_type)
                profiles.append(
                    {
                        "field": field,
                        "metadata": item,
                        "inferred_type": inferred_type,
                        "role": role,
                    }
                )
                normalized_name = _normalized(field)
                semantic_name_counts[normalized_name] = (
                    semantic_name_counts.get(normalized_name, 0) + 1
                )
            dataset_fields.append(
                {
                    "dataset_id": dataset_id,
                    "dataset": dataset,
                    "profiles": profiles,
                }
            )

        used_semantic_names: set[str] = set()
        for context in dataset_fields:
            dataset_id = context["dataset_id"]
            dataset = context["dataset"]
            profiles = context["profiles"]
            entity_id = _stable_semantic_id(scope_id, dataset_id, dataset.name, "entity")
            alias = f"t_{dataset_id.hex[:12]}"
            entity_by_dataset[str(dataset_id)] = entity_id
            field_definitions = []
            for profile in profiles:
                field = profile["field"]
                item = profile["metadata"]
                role = profile["role"]
                field_definitions.append({"field_id": _stable_semantic_id(scope_id, dataset_id, field, role), "source_name": field, "type": profile["inferred_type"], "role": role, "description": str(item.get("description") or "")})
            entity_type = (
                "fact"
                if any(profile["role"] == "metric" for profile in profiles)
                else "dimension"
            )
            entities.append({"id": entity_id, "entity_id": entity_id, "name": dataset.name, "sql_alias": alias, "dataset_id": str(dataset_id), "entity_type": entity_type, "primary_key": None, "grain": "one row per source record", "fields": field_definitions})
            for profile in profiles:
                field = profile["field"]
                item = profile["metadata"]
                role = profile["role"]
                field_id = _stable_semantic_id(scope_id, dataset_id, field, role)
                if role not in {"dimension", "date", "text", "metric"}:
                    continue
                semantic_name = _unique_semantic_name(
                    dataset_name=dataset.name,
                    dataset_id=dataset_id,
                    field=field,
                    qualify=semantic_name_counts.get(_normalized(field), 0) > 1,
                    used=used_semantic_names,
                )
                if role in {"dimension", "date", "text"}:
                    dimensions.append({"id": _stable_semantic_id(scope_id, dataset_id, field, "dimension"), "name": semantic_name, "aliases": [], "entity_id": entity_id, "field_id": field_id, "type": "time" if role == "date" else "categorical", "time_grains": ["day", "week", "month", "quarter", "year"] if role == "date" else []})
                elif role == "metric":
                    metrics.append({"id": _stable_semantic_id(scope_id, dataset_id, field, "metric"), "name": semantic_name, "aliases": [], "description": str(item.get("description") or ""), "unit": "", "format": "number", "direction": "neutral", "formula": {"op": "sum", "expr": {"op": "field", "entity_id": entity_id, "field_id": field_id}}, "default_time_dimension": None, "allowed_dimensions": []})
        relationships = []
        relationship_cardinalities: dict[tuple[str, str], tuple[int, int]] = {}
        for relationship in (group.relationships if group else ()):
            left_dataset_text = str(relationship.get("left_dataset_id"))
            right_dataset_text = str(relationship.get("right_dataset_id"))
            left = entity_by_dataset.get(left_dataset_text)
            right = entity_by_dataset.get(right_dataset_text)
            if left and right:
                left_entity = next(item for item in entities if item["id"] == left)
                right_entity = next(item for item in entities if item["id"] == right)
                left_field_id = next((item["field_id"] for item in left_entity["fields"] if item["source_name"] == relationship.get("left_column")), "")
                right_field_id = next((item["field_id"] for item in right_entity["fields"] if item["source_name"] == relationship.get("right_column")), "")
                left_column = str(relationship.get("left_column") or "")
                right_column = str(relationship.get("right_column") or "")
                value_modes = {
                    str(relationship.get("left_value_mode") or "scalar"),
                    str(relationship.get("right_value_mode") or "scalar"),
                }
                if value_modes == {"scalar"}:
                    for key in (
                        (left_dataset_text, left_column),
                        (right_dataset_text, right_column),
                    ):
                        if key not in relationship_cardinalities:
                            try:
                                relationship_cardinalities[key] = (
                                    self.repository.analysis_column_cardinality(
                                        UUID(key[0]),
                                        column_name=key[1],
                                    )
                                )
                            except (RuntimeError, ValueError):
                                relationship_cardinalities[key] = (0, 0)
                    cardinality = _infer_semantic_relationship_cardinality(
                        left_counts=relationship_cardinalities[
                            (left_dataset_text, left_column)
                        ],
                        right_counts=relationship_cardinalities[
                            (right_dataset_text, right_column)
                        ],
                    )
                else:
                    cardinality = "unknown"
                deduplication = relationship.get("deduplication_strategy")
                enabled = bool(relationship.get("enabled", True))
                risk_note = str(relationship.get("risk_note") or "")
                if cardinality in {"unknown", "many_to_many"} and not deduplication:
                    enabled = False
                    risk_note = risk_note or (
                        f"Auto-disabled {cardinality} relationship until an explicit "
                        "aggregation grain or deduplication strategy is configured."
                    )
                relationships.append({"id": _semantic_relationship_id(scope_id, relationship), "left_entity_id": left, "right_entity_id": right, "left_field_id": left_field_id, "right_field_id": right_field_id, "join_type": relationship.get("join_type", "left"), "left_value_mode": relationship.get("left_value_mode", "scalar"), "right_value_mode": relationship.get("right_value_mode", "scalar"), "left_delimiter": relationship.get("left_delimiter"), "right_delimiter": relationship.get("right_delimiter"), "cardinality": cardinality, "enabled": enabled, "deduplication_strategy": deduplication, "risk_note": risk_note})
        return {"definition_schema_version": 2, "entities": entities, "relationships": relationships, "dimensions": dimensions, "metrics": metrics, "unresolved_bindings": []}

    def _transfer_semantic_labels(self, source: dict[str, Any], target: dict[str, Any]) -> list[dict[str, Any]]:
        unresolved: list[dict[str, Any]] = []
        ranker = SemanticCandidateRanker(self.embedding_provider)
        source_label_counts = _semantic_label_counts(source)
        target_items = [
            item
            for collection in ("metrics", "dimensions")
            for item in target.get(collection) or []
            if isinstance(item, dict)
        ]
        label_owners: dict[str, str] = {}
        for item in target_items:
            owner = str(item.get("id") or id(item))
            for label in _semantic_item_labels(item):
                normalized = _normalized(label)
                if normalized:
                    label_owners[normalized] = owner
        target_field_roles = _semantic_field_roles_by_binding(target)
        matched_target_ids: set[str] = set()
        for collection, expected_type in (("metrics", "metric"), ("dimensions", "dimension")):
            collection_targets = [
                item
                for item in target.get(collection) or []
                if isinstance(item, dict)
            ]
            targets_by_binding = {
                binding: item
                for item in collection_targets
                if (binding := _semantic_item_binding(target, item, expected_type))
            }
            for source_item in source.get(collection) or []:
                if not isinstance(source_item, dict):
                    continue
                source_binding = _semantic_item_binding(
                    source,
                    source_item,
                    expected_type,
                )
                exact_target = targets_by_binding.get(source_binding)
                exact_target_id = str(exact_target.get("id") or "") if exact_target else ""
                if exact_target_id in matched_target_ids:
                    exact_target = None
                if exact_target is not None:
                    selected = exact_target
                    binding_score = 1.0
                    ranked = ()
                elif (
                    source_binding
                    and target_field_roles.get(source_binding) in {"id", "ignored"}
                ):
                    # Old auto drafts exposed identifiers as dimensions. The new
                    # role inference intentionally removes them from the semantic
                    # surface, so they are a completed migration rather than an
                    # unresolved binding.
                    continue
                else:
                    available_targets = [
                        item
                        for item in collection_targets
                        if str(item.get("id") or "") not in matched_target_ids
                    ]
                    if not available_targets:
                        continue
                    query = " ".join([str(source_item.get("name") or ""), *(str(alias) for alias in source_item.get("aliases") or [])])
                    ranked = ranker.rank(
                        query,
                        available_targets,
                        expected_type=expected_type,
                    )
                    top = ranked[0] if ranked else None
                    gap = top.final_score - ranked[1].final_score if top and len(ranked) > 1 else 1.0
                    if not (top and top.final_score >= 0.85 and gap >= 0.08):
                        unresolved.append({"source_id": source_item.get("id"), "source_name": source_item.get("name"), "semantic_type": expected_type, "candidates": [candidate.evidence() for candidate in ranked[:3]]})
                        continue
                    selected = top.item
                    binding_score = top.final_score
                _transfer_unique_semantic_labels(
                    source_item=source_item,
                    target_item=selected,
                    source_label_counts=source_label_counts,
                    label_owners=label_owners,
                )
                if source_item.get("description"):
                    selected["description"] = source_item["description"]
                selected["binding_source"] = "auto_rebound"
                selected["binding_score"] = round(binding_score, 4)
                matched_target_ids.add(str(selected.get("id") or ""))
        return unresolved

    def _resolve(self, question: str, model: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float | None]]:
        # Import lazily so the standalone semantic package can initialize without
        # cycling through app.analysis.__init__ -> workflow -> agent_loop -> semantic.
        from app.analysis.query_intent import strip_negated_clauses

        definition = model["definition"]
        ranker = SemanticCandidateRanker(self.embedding_provider)
        semantic_question = strip_negated_clauses(question)
        dimension_items = list(definition.get("dimensions") or [])
        filters = _semantic_plan_filters(question, definition)
        filter_dimension_ids = {
            str(item["dimension_id"])
            for item in filters
        }
        metrics = ranker.rank(
            semantic_question,
            definition.get("metrics") or [],
            expected_type="metric",
        )
        dimensions = [
            item
            for item in ranker.rank(
                semantic_question,
                dimension_items,
                expected_type="dimension",
            )
            if str(item.item.get("id") or "") not in filter_dimension_ids
        ]
        selected_metric = metrics[0].item if metrics and metrics[0].final_score >= 0.25 else None
        explicitly_requested_dimension_ids = {
            str(item.get("id") or "")
            for item in dimension_items
            if str(item.get("id") or "") not in filter_dimension_ids
            and _semantic_item_is_explicitly_mentioned(
                semantic_question,
                _semantic_dimension_names(definition, item),
            )
        }
        selected_dimension = next(
            (
                candidate.item
                for candidate in dimensions
                if str(candidate.item.get("id") or "")
                in explicitly_requested_dimension_ids
            ),
            dimensions[0].item
            if dimensions and dimensions[0].final_score >= 0.25
            else None,
        )
        ambiguity = []
        if len(metrics) > 1 and metrics[0].final_score - metrics[1].final_score < 0.08:
            ambiguity.append("Ambiguous metric: " + ", ".join(str(item.item.get("name")) for item in metrics[:2]))
        evidence = [metrics[0].evidence() if metrics else {"reason": "No metric matched"}, dimensions[0].evidence() if dimensions else {"reason": "No dimension matched"}]
        metric_ids = (str(selected_metric["id"]),) if selected_metric else ()
        dimension_ids = (str(selected_dimension["id"]),) if selected_dimension else ()
        relationship_graph = build_relationship_graph(definition)
        grain_plan = plan_relationship_path(
            definition,
            metric_ids=metric_ids,
            dimension_ids=tuple(
                dict.fromkeys((*dimension_ids, *sorted(filter_dimension_ids)))
            ),
        )
        if selected_metric and not grain_plan["safe"]:
            ambiguity.extend(grain_plan["warnings"])
        plan = {
            "route": "hybrid" if selected_metric else "python",
            "metric_ids": list(metric_ids),
            "dimension_ids": list(dimension_ids),
            "time_dimension_id": (
                selected_dimension["id"]
                if selected_dimension and selected_dimension.get("type") == "time"
                else None
            ),
            "filters": filters,
            "join_path": list(grain_plan["join_path"]),
            "relationship_graph": relationship_graph,
            "grain_plan": grain_plan,
            "ambiguities": ambiguity,
            "evidence": evidence,
            "embedding_model_revision": self.embedding_provider.model_revision,
        }
        relationship_count = len(definition.get("relationships") or [])
        scores = {
            "intent": 0.86 if selected_metric else 0.65,
            "metric": metrics[0].final_score if selected_metric else 0.15,
            "dimension": dimensions[0].final_score if selected_dimension else None,
            "time": 0.9 if plan["time_dimension_id"] else None,
            "join": (
                0.9 if grain_plan["safe"] else 0.25
            ) if relationship_count else None,
            "data_quality": 0.82,
            "route": 0.86 if selected_metric else 0.45,
        }
        return plan, scores


def _semantic_plan_filters(
    question: str,
    definition: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resolve only explicit field predicates; sampled values never become filters."""

    resolved: list[dict[str, Any]] = []
    for dimension in definition.get("dimensions") or []:
        if not isinstance(dimension, dict) or not dimension.get("id"):
            continue
        names = _semantic_dimension_names(definition, dimension)
        match = _explicit_semantic_filter(question, names)
        if match is None:
            continue
        operator, value = match
        resolved.append(
            {
                "dimension_id": str(dimension["id"]),
                "operator": operator,
                "value": value,
            }
        )
    return resolved


def _semantic_dimension_names(
    definition: dict[str, Any],
    dimension: dict[str, Any],
) -> tuple[str, ...]:
    entity_id = str(dimension.get("entity_id") or dimension.get("entity") or "")
    field_id = str(dimension.get("field_id") or dimension.get("field") or "")
    source_name = ""
    for entity in definition.get("entities") or []:
        if str(entity.get("id") or "") != entity_id:
            continue
        source_name = next(
            (
                str(field.get("source_name") or "")
                for field in entity.get("fields") or []
                if str(field.get("field_id") or field.get("id") or "") == field_id
            ),
            "",
        )
        break
    names = [
        str(dimension.get("name") or ""),
        source_name,
        *(str(item) for item in dimension.get("aliases") or []),
    ]
    return tuple(dict.fromkeys(item for item in names if item))


def _explicit_semantic_filter(
    question: str,
    names: tuple[str, ...],
) -> tuple[str, str] | None:
    for raw_name in sorted(names, key=len, reverse=True):
        name = re.split(r"__|\.", raw_name.casefold())[-1]
        if not name:
            continue
        matched = re.search(
            rf"(?<![\w]){re.escape(name)}\s*"
            r"(==|=|!=|<>|>=|<=|>|<|:|：|为|是)\s*"
            r"['\"]?([^'\"\s,，;；。)）]+)",
            question,
            flags=re.IGNORECASE,
        )
        if not matched:
            continue
        operator = {
            "==": "=",
            ":": "=",
            "：": "=",
            "为": "=",
            "是": "=",
            "<>": "!=",
        }.get(matched.group(1), matched.group(1))
        return operator, matched.group(2).strip()
    return None


def _semantic_item_is_explicitly_mentioned(
    question: str,
    names: tuple[str, ...],
) -> bool:
    folded = question.casefold()
    normalized_question = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", folded)
    for raw_name in names:
        name = re.split(r"__|\.", raw_name.casefold())[-1].strip()
        if not name:
            continue
        if re.search(r"[\u3400-\u9fff]", name):
            normalized_name = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", name)
            if normalized_name and normalized_name in normalized_question:
                return True
            continue
        if re.search(rf"(?<![a-z0-9_]){re.escape(name)}(?![a-z0-9_])", folded):
            return True
    return False


def _semantic_sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


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


def _semantic_join_target(*, alias: str, field: str, strategy: str) -> str:
    from app.semantic.dsl import quote_identifier

    table = quote_identifier(alias)
    if strategy == "direct_join":
        return table
    if strategy == "deduplicate_before_join":
        key = quote_identifier(field)
        return (
            f"(SELECT * FROM {table} "
            f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {key} ORDER BY {key}) = 1) "
            f"AS {table}"
        )
    raise ValueError(f"Semantic join strategy is not executable yet: {strategy}")


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


def _semantic_item_labels(item: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(label)
            for label in (item.get("name"), *(item.get("aliases") or ()))
            if str(label or "").strip()
        )
    )


def _semantic_label_counts(definition: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for collection in ("metrics", "dimensions"):
        for item in definition.get(collection) or ():
            if not isinstance(item, dict):
                continue
            normalized_labels = {
                _normalized(label)
                for label in _semantic_item_labels(item)
                if _normalized(label)
            }
            for normalized in normalized_labels:
                counts[normalized] = counts.get(normalized, 0) + 1
    return counts


def _semantic_item_binding(
    definition: dict[str, Any],
    item: dict[str, Any],
    semantic_type: str,
) -> tuple[str, str] | None:
    entities = {
        str(entity.get("id") or entity.get("entity_id") or ""): entity
        for entity in definition.get("entities") or ()
        if isinstance(entity, dict)
    }
    if semantic_type == "dimension":
        entity_id = str(item.get("entity_id") or item.get("entity") or "")
        field_id = str(item.get("field_id") or item.get("field") or "")
    else:
        field_references = _semantic_formula_field_references(item.get("formula"))
        if len(field_references) != 1:
            return None
        entity_id, field_id = next(iter(field_references))
    entity = entities.get(entity_id)
    if not entity:
        return None
    dataset_id = str(entity.get("dataset_id") or "")
    source_name = _field_source(entity, field_id)
    if not dataset_id or not source_name:
        return None
    return dataset_id, _normalized(source_name)


def _semantic_formula_field_references(value: Any) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        if value.get("op") == "field":
            entity_id = str(value.get("entity_id") or value.get("entity") or "")
            field_id = str(value.get("field_id") or value.get("field") or "")
            if entity_id and field_id:
                result.add((entity_id, field_id))
        for child in value.values():
            result.update(_semantic_formula_field_references(child))
    elif isinstance(value, list | tuple):
        for child in value:
            result.update(_semantic_formula_field_references(child))
    return result


def _semantic_field_roles_by_binding(
    definition: dict[str, Any],
) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for entity in definition.get("entities") or ():
        if not isinstance(entity, dict):
            continue
        dataset_id = str(entity.get("dataset_id") or "")
        for field in entity.get("fields") or ():
            if not isinstance(field, dict):
                continue
            source_name = str(field.get("source_name") or "")
            if dataset_id and source_name:
                result[(dataset_id, _normalized(source_name))] = str(
                    field.get("role") or ""
                ).lower()
    return result


def _transfer_unique_semantic_labels(
    *,
    source_item: dict[str, Any],
    target_item: dict[str, Any],
    source_label_counts: dict[str, int],
    label_owners: dict[str, str],
) -> None:
    owner = str(target_item.get("id") or id(target_item))
    current_name = str(target_item.get("name") or target_item.get("id") or "")
    current_aliases = [str(alias) for alias in target_item.get("aliases") or ()]
    for label in (current_name, *current_aliases):
        normalized = _normalized(label)
        if label_owners.get(normalized) == owner:
            label_owners.pop(normalized, None)

    source_name = str(source_item.get("name") or "").strip()
    source_name_normalized = _normalized(source_name)
    if (
        source_name_normalized
        and source_label_counts.get(source_name_normalized, 0) == 1
        and source_name_normalized not in label_owners
    ):
        final_name = source_name
    else:
        final_name = current_name

    final_name_normalized = _normalized(final_name)
    label_owners[final_name_normalized] = owner
    aliases: list[str] = []
    for alias in (*current_aliases, *(source_item.get("aliases") or ()), source_name):
        text = str(alias or "").strip()
        normalized = _normalized(text)
        if (
            not normalized
            or normalized == final_name_normalized
            or source_label_counts.get(normalized, 0) > 1
            or normalized in label_owners
        ):
            continue
        aliases.append(text)
        label_owners[normalized] = owner
    target_item["name"] = final_name
    target_item["aliases"] = list(dict.fromkeys(aliases))


def _infer_semantic_field_type(
    field: str,
    values: tuple[Any, ...],
    metadata: dict[str, Any],
) -> str:
    declared = str(
        metadata.get("override_type") or metadata.get("inferred_type") or ""
    ).strip()
    if declared:
        return declared
    usable = tuple(
        value
        for value in values
        if not (isinstance(value, str) and not value.strip())
    )
    if not usable:
        return "text"
    if all(isinstance(value, bool) for value in usable):
        return "boolean"
    if all(_is_semantic_number(value) for value in usable):
        return "number"
    if _looks_like_time_field(field) or all(
        _is_semantic_datetime(value) for value in usable
    ):
        return "datetime"
    return "text"


def _infer_semantic_field_role(
    field: str,
    values: tuple[Any, ...],
    metadata: dict[str, Any],
    inferred_type: str,
) -> str:
    declared = str(metadata.get("role") or "").strip().lower()
    if declared:
        if declared in {"numeric", "number"}:
            return "metric" if _looks_like_additive_metric_field(field) else "dimension"
        return {"categorical": "dimension", "datetime": "date"}.get(
            declared,
            declared,
        )
    if _looks_like_identifier_field(field) or _values_look_like_identifiers(values):
        return "id"
    if _looks_like_time_field(field) or _is_semantic_date_type(inferred_type):
        return "date"
    if (
        _is_semantic_numeric_type(inferred_type)
        and _looks_like_additive_metric_field(field)
    ):
        return "metric"
    return "dimension"


def _looks_like_identifier_field(field: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", field.lower()).strip("_")
    tokens = tuple(filter(None, normalized.split("_")))
    identifier_tokens = {
        "id",
        "uuid",
        "guid",
        "hash",
        "key",
        "code",
        "编号",
        "编码",
    }
    return bool(identifier_tokens.intersection(tokens)) or normalized.endswith("_id")


def _looks_like_additive_metric_field(field: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", field.lower()).strip("_")
    tokens = set(filter(None, normalized.split("_")))
    non_additive_tokens = {
        "id",
        "uuid",
        "guid",
        "code",
        "zip",
        "sequential",
        "installments",
        "score",
        "length",
        "weight",
        "height",
        "width",
        # Counts attached to a descriptive entity are attributes, not
        # transaction-grain measures.  Treating product_photos_qty as SUM
        # would turn the products dimension into a fact and duplicate the
        # value when it is joined through order items.
        "photo",
        "photos",
    }
    additive_tokens = {
        "amount",
        "value",
        "price",
        "freight",
        "revenue",
        "sales",
        "cost",
        "discount",
        "quantity",
        "qty",
        "total",
        "金额",
        "价格",
        "收入",
        "销售额",
        "成本",
        "折扣",
        "数量",
        "总额",
        "运费",
    }
    return not tokens.intersection(non_additive_tokens) and bool(
        tokens.intersection(additive_tokens)
    )


def _values_look_like_identifiers(values: tuple[Any, ...]) -> bool:
    strings = [str(value).strip() for value in values if str(value).strip()]
    if not strings:
        return False
    uuid_like = re.compile(
        r"(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})",
        re.IGNORECASE,
    )
    return sum(bool(uuid_like.fullmatch(value)) for value in strings) / len(strings) >= 0.8


def _is_semantic_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int | float):
        return True
    try:
        float(str(value).strip())
    except (TypeError, ValueError):
        return False
    return True


def _is_semantic_datetime(value: Any) -> bool:
    if isinstance(value, datetime):
        return True
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or not any(marker in text for marker in ("-", "/", "T", ":")):
        return False
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00").replace("/", "-"))
    except ValueError:
        return False
    return True


def _looks_like_time_field(field: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", field.lower()).strip("_")
    tokens = set(filter(None, normalized.split("_")))
    return bool(
        tokens.intersection({"date", "time", "datetime", "timestamp", "日期", "时间"})
        or normalized.endswith(("_at", "_ts"))
    )


def _is_semantic_numeric_type(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {
        "number",
        "numeric",
        "integer",
        "int",
        "float",
        "double",
        "decimal",
    } or any(token in normalized for token in ("int", "float", "double", "decimal"))


def _is_semantic_date_type(value: str) -> bool:
    normalized = value.strip().lower()
    return any(token in normalized for token in ("date", "time", "timestamp"))


def _unique_semantic_name(
    *,
    dataset_name: str,
    dataset_id: UUID,
    field: str,
    qualify: bool,
    used: set[str],
) -> str:
    candidate = f"{dataset_name}.{field}" if qualify else field
    normalized = _normalized(candidate)
    if normalized in used:
        candidate = f"{dataset_name}.{dataset_id.hex[:8]}.{field}"
        normalized = _normalized(candidate)
    suffix = 2
    base = candidate
    while normalized in used:
        candidate = f"{base}.{suffix}"
        normalized = _normalized(candidate)
        suffix += 1
    used.add(normalized)
    return candidate


def _semantic_relationship_id(
    scope_id: UUID,
    relationship: dict[str, Any],
) -> str:
    existing = relationship.get("id") or relationship.get("relationship_id")
    if existing:
        return str(existing)
    left_dataset_id = UUID(str(relationship.get("left_dataset_id")))
    right_dataset_id = UUID(str(relationship.get("right_dataset_id")))
    binding = (
        f"{left_dataset_id}:{relationship.get('left_column')}->"
        f"{right_dataset_id}:{relationship.get('right_column')}"
    )
    return _stable_semantic_id(
        scope_id,
        left_dataset_id,
        binding,
        "relationship",
    )


def _infer_semantic_relationship_cardinality(
    *,
    left_counts: tuple[int, int],
    right_counts: tuple[int, int],
) -> str:
    left_value_count, left_distinct_count = left_counts
    right_value_count, right_distinct_count = right_counts
    if not left_value_count or not right_value_count:
        return "unknown"
    left_unique = left_value_count == left_distinct_count
    right_unique = right_value_count == right_distinct_count
    return {
        (True, True): "one_to_one",
        (False, True): "many_to_one",
        (True, False): "one_to_many",
        (False, False): "many_to_many",
    }[(left_unique, right_unique)]


def _append_semantic_id_errors(
    errors: list[str],
    *,
    label: str,
    items: list[dict[str, Any]],
) -> None:
    seen: set[str] = set()
    for item in items:
        semantic_id = str(item.get("id") or "").strip()
        if not semantic_id:
            errors.append(f"{label} is missing an id.")
        elif semantic_id in seen:
            errors.append(f"Duplicate {label.lower()} id: {semantic_id}")
        seen.add(semantic_id)


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
