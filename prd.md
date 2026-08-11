# Product Requirements Document (PRD)

# DataMind

**AI Data Analysis Copilot based on FastAPI, LangGraph, React and provider-routed LLM agents**

Version: v1.3 implementation-aligned (2026-07-30)

---

# 1. Vision

DataMind is an AI data analysis assistant. Users upload local datasets, describe an analysis goal in natural language, and receive governed data preparation, bounded autonomous analysis, evidence-validated reports, and exportable HTML/Markdown/PDF-ready outputs. Loop Engineering is the default execution path; deterministic rules and the legacy fixed SQL/Python path remain bounded fallbacks and compatibility options.

The product is positioned as an AI Agent engineering demo for data analysis workflows. It is not yet a full BI platform, enterprise auth system, or distributed data warehouse.

---

# 2. Current Product Goal

Build an end-to-end data analysis agent that can:

- Import CSV, Excel, JSON, and TXT datasets through the backend.
- Persist uploaded datasets, cleaned datasets, analysis records, and reports in the project database.
- Parse raw records, detect headers, infer schema, and run asynchronous autonomous cleaning with quality-gated version activation.
- Profile datasets and show raw/cleaned previews in the UI.
- Use natural language questions to plan an analysis route and execute a bounded decide/execute/observe/verify/repair Loop by default.
- Execute safe SQL against an internal DuckDB temporary table named `dataset`.
- Execute DeepSeek-generated Python analysis code for statistics, text analysis, and chart-ready outputs.
- Use Kimi for insight integration, adversarial review, multimodal context, final report writing, and a persistent evidence-backed DataMind assistant with scoped tool permissions.
- Generate structured web reports and export HTML/Markdown.
- Keep the React frontend in sync with backend-persisted state instead of temporary client-side state.
- Treat a batch of related uploaded files as a dataset group, automatically validate and persist relationships with compact schema context, and run joined analysis from the saved plan.

---

# 3. Out Of Scope For Current Version

The following are not current-version requirements:

- Enterprise RBAC, organization management, password reset, SSO, or centralized compliance/audit-log management. User-scoped Kimi action records are part of the current product.
- Multi-user collaboration inside the same report.
- Scheduler and recurring jobs.
- Knowledge graph, vector database, or RAG.
- Browser crawling.
- AutoML or model training.
- Multi-region orchestration and Kubernetes-native distributed execution.
- User-uploaded DuckDB database files as a product feature.

Local mode supports lightweight login. The production profile uses revocable Cookie sessions,
CSRF/Origin validation, Argon2id password hashes, Redis rate limits, and user-scoped records;
it is still not an enterprise identity or RBAC system.

---

# 4. Target Users

- Students learning data analysis.
- Data analysts who want fast exploratory reports.
- AI engineers exploring LLM agent workflows.
- Developers building data-analysis copilots.

---

# 5. User Workflow

```text
Login
  │
  ▼
Upload CSV / Excel / JSON / TXT
  │
  ▼
Backend import parser
  ├─ table/header detection
  ├─ raw records persistence
  └─ dataset/dataset-group creation
  │
  ▼
Autonomous cleaning job
  ├─ decide rules / LLM / hybrid strategy
  ├─ execute in rules engine or one-shot Python sandbox
  ├─ verify row/column/missing/duplicate/type quality gates
  ├─ repair or deterministic fallback
  └─ commit and activate one cleaning version idempotently
  │
  ▼
Ask analysis question
  │
  ▼
Planner + semantic decision
  └─ freeze AnalysisContract: scope, metric, grain, method, assumptions and budgets
  │
  ▼
Autonomous analysis Loop (default)
  ├─ select one job-scoped allowlisted tool per decision
  ├─ execute safe SQL / sandboxed Python / profile / chart tools
  ├─ observe and verify bounded evidence
  ├─ repair classified failures
  └─ deterministic fallback when budgets or providers fail
  │
  ▼
Chart formatting
  │
  ▼
Deterministic statistical verification
  ├─ numeric evidence coverage
  ├─ comparison sample size / effect size / confidence interval
  ├─ observational causal-language guard
  └─ Join grain and row-expansion validation
  │
  ▼
Adversarial review
  └─ failed verification may return once to the analysis Loop
  │
  ▼
Report Loop
  ├─ decide report strategy
  ├─ generate and verify evidence citations
  ├─ repair at most twice or request analysis evidence once
  └─ commit one report idempotently
  │
  ▼
Structured report page
  ├─ Executive Summary
  ├─ Key Findings
  ├─ Charts
  ├─ SQL Results
  ├─ Validation Issues
  ├─ Analysis Contract and Statistical Verification
  ├─ Field / metric / finding / chart / report lineage
  ├─ Analysis Trace
  └─ HTML / Markdown / browser-print PDF export

Kimi Assistant is a parallel entry point over the same persisted assets. In Ask mode it retrieves evidence only; in Execute mode it can invoke authorized DataMind cleaning, relationship, analysis, report, semantic-model, and recycle/restore tools without bypassing quality gates or system safety prompts.
```

---

# 6. Functional Requirements

## 6.1 Lightweight User Access

Implemented behavior:

- Login page in React.
- First-time local username login can create a user record.
- Production login creates a revocable server-side session and sets `datamind_session` as an HttpOnly, SameSite Cookie; production forces `Secure`.
- Mutating requests use CSRF tokens and Origin validation.
- Passwords use Argon2id; successful login upgrades legacy password hashes.
- Logout revokes the session and returns to the login page.
- Datasets, groups, jobs, reports, semantic models, conversations, attachments, and actions are scoped by the authenticated user.
- `X-DataMind-User` is available only when explicitly running local development with `DATAMIND_AUTH_MODE=legacy`; production configuration rejects legacy identity headers.

Limitations:

- No enterprise RBAC, organization hierarchy, SSO, password reset, delegated administration, or compliance-grade audit console.

## 6.2 Dataset Import And Persistence

Supported upload formats:

- CSV
- Excel `.xlsx`
- JSON `.json`
- Text `.txt`

Backend responsibilities:

- Parse files through one backend import path.
- Detect tabular structures where possible.
- Store raw records and cleaned records.
- Store schema, profile, source type, status, and timestamps.
- Persist local-development data in SQLite at `data/datamind.db`; production uses the configured PostgreSQL database.
- Create a dataset group for multi-file batches so related CSV/XLSX/JSON/TXT files can be managed together.
- In the UI, a batch-created dataset group is displayed as one top-level imported record while child tables remain available in the group workbench.

Explicitly removed:

- User-facing DuckDB local path import. A browser-based app cannot reliably read arbitrary local database paths after deployment. DuckDB remains an internal SQL execution engine only.

## 6.3 Data Cleaning

Import should create both:

- Raw imported dataset.
- Cleaned dataset.

Cleaning requirements:

- Remove fully empty rows.
- Normalize obvious blank cells.
- Infer usable header rows.
- Preserve original raw records for comparison.
- Produce cleaned records used by planner, SQL, Python, and reports.
- Allow a user-provided cleaning requirement text.
- Create an asynchronous cleaning job for both single-file and batch imports with `cleaning_strategy=auto` by default.
- In `auto`, the controller chooses `rules`, `llm`, or `hybrid` from compressed schema and aggregate quality evidence; deployments may explicitly request one strategy.
- Generated cleaning code runs only through the one-shot Python subprocess/container Runner, never through in-process `exec`.
- Explicit LLM cleaning gets up to three execution attempts; every repair prompt includes prior code and errors.
- Every candidate must pass conservative quality gates for empty output, row/column retention, missingness, duplicates, type stability, and unexpected expansion.
- Only a validated candidate creates and activates a new cleaning version. Cancellation, failure, or rejected candidates preserve the previously active version.
- Cleaning jobs persist leases, heartbeats, ordered SSE events, retry lineage, cancellation state, selected strategy, terminal reason, and idempotent result references.
- Cleaning prompts cap schema/sample size, redact common email/phone fields, and treat dataset names, columns, cells, and user requirements as untrusted data rather than instructions.
- Persist cleaning runs as versions.
- Show raw/previous/current summaries and sample diffs.
- Allow activating a previous cleaning version as the current cleaned dataset.

Current behavior:

- The upload queue starts `cleaning_strategy=auto`, displays the autonomous cleaning Loop, and remains mounted while users navigate to other pages.
- Cleaning diff/version UI exists, including rollback by activation.
- A manual cleaning rule editor can preview rules before applying them as a new cleaning version.
- Manual rules cannot be applied before a successful preview. Validation issues block commit, and high-impact changes such as deleting all rows, removing at least 10% of rows, dropping columns, increasing missingness, converting types, or filtering rows require an explicit impact acknowledgment before a reversible version is created.
- The older synchronous `cleaning-runs` endpoint remains available for compatibility; it is not the default upload path.
- Per-cell manual approval is not a product feature yet.

## 6.4 Dataset Detail Page

The React dataset page should show:

- Imported dataset list.
- Cleaned dataset list.
- Raw data preview.
- Cleaned data preview.
- Schema and field types.
- Editable field override type, role, and description.
- Cleaning version history and diff summary.
- Profiling summary.
- Related analysis records.
- Recycle/delete action with the documented 30-day soft-delete behavior.

## 6.5 Automatic Dataset Profiling

After upload/import, DataMind generates:

- Row count.
- Column count.
- Missing values.
- Duplicate rows.
- Numeric fields.
- Categorical/text fields.
- Field type summary.
- Basic descriptive statistics.

Data reliability monitoring:

- Every active raw/cleaned version has a bounded, privacy-safe profile snapshot with Schema fingerprint, row count, inferred types, missing rate, unique rate, numeric distribution summary, and hashed value signatures.
- New data is compared with the previous snapshot for added/removed/renamed fields, type changes, row-count changes, missing/unique-rate drift, and numeric distribution shifts.
- Dataset-group relationship keys are revalidated against current fields and bounded sample match rates. Material drops mark the relationship stale instead of silently continuing to Join.
- Schema-breaking drift marks affected published semantic models and reports stale while preserving their immutable historical content and execution-time references.
- The dataset-group workbench shows current reliability status, stale relationships, detected changes, and suggested cleaning/relationship/analysis actions. Suggested mutations require the existing user/Kimi authorization path; monitoring never silently rewrites data.

Example:

```text
Rows: 18,240
Columns: 12
Missing Values: 1.2%
Duplicate Rows: 15
Numeric Fields: 7
Text/Categorical Fields: 5
```

## 6.6 Natural Language Analysis

Users ask questions in natural language, for example:

```text
分析不同客户/产品/地区的销售额、利润和利润率表现，先用 SQL 汇总找出利润贡献最高和最低的分组，再用 Python 分析利润率分布、异常低利润客户或产品。
```

The system should:

- Understand intent.
- Inspect dataset schema and profile.
- Inspect automatically validated dataset group relationships when a multi-file package is selected.
- Choose SQL, Python, or hybrid route.
- Generate SQL/Python analysis.
- Produce charts and report content.
- Persist the analysis result and latest report.

## 6.7 Planner Agent

Responsibilities:

- Understand user intent.
- Inspect schema/profile.
- Determine route: SQL, Python, or hybrid.
- Prefer SQL for aggregation, grouping, filtering, and count/sum/avg questions.
- Prefer Python for EDA, distribution analysis, text analysis, statistical summaries, and chart generation.
- Use rules fallback when model planning fails.
- Use automatically validated dataset group relationship trees as the source of multi-file context.
- Surface and enforce join risk controls: directional match coverage, right-key uniqueness, estimated/actual row expansion, skipped unsafe joins, and unmatched rows.
- Prefer a published dataset/dataset-group semantic model when available and pin its model ID/version into the planner decision, job, report, and trace.
- Resolve intent, metric, dimension, time field, join path, data quality, and route as separately scored components.
- Calibrate raw confidence with the active user-level monotonic PAVA calibrator.
- Execute directly at high confidence (`>= 0.80`), allow an informed run at medium confidence (`0.55-0.79`), and require explicit confirmation below `0.55`.
- Preserve the legacy schema/profile planner when no published semantic model is available.

Provider:

- DeepSeek by default.

## 6.8 SQL Agent

Responsibilities:

- Generate SQL from the user question.
- Execute SQL through DuckDB against a temporary table named `dataset`.
- Return SQL, result rows, explanation, and validation issues.

Safety requirements:

- Only allow `SELECT`.
- Reject dangerous statements such as `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ATTACH`, and `COPY`.
- Restrict SQL to the `dataset` table.
- Fall back to rule SQL when model SQL is invalid.
- Avoid treating id/hash/code fields as numeric metrics for `SUM`/`AVG`.
- When a published semantic model is pinned, register only its declared entities as DuckDB tables and compile metric DSL deterministically into SQL.
- Validate semantic SQL with `sqlglot`; allow only declared tables, fields, relationships, and safe `SELECT`/CTE operations.
- Reject external reads, system tables, table functions, write operations, undeclared joins, `CROSS JOIN`, and `NATURAL JOIN`.
- Keep the single temporary `dataset` table path as the legacy compatibility route when semantic planning is unavailable.

Provider:

- DeepSeek by default.

## 6.9 Python Agent

Responsibilities:

- Generate and execute analysis code with Pandas-oriented logic.
- Perform EDA and statistical summaries.
- Analyze numeric, categorical, and text columns.
- Produce chart specs consumed by frontend reports.
- Return execution source, code, statistics, insights, charts, and text-analysis results.

Provider:

- DeepSeek by default.

Current execution posture:

- Python code runs locally for demo/development.
- Python code runs in a subprocess with timeout, output size limit, temporary working directory, and reduced environment variables.
- Common safe Python constructs such as `for`, `while`, Pandas usage, and allowlisted imports are supported.
- Dangerous modules, dangerous builtins, and common file-write calls are blocked before execution.
- The output shape is still validated.
- If LLM-generated Python code fails, the failed code and execution error are sent back to the LLM for repair.
- Python self-repair allows up to three execution attempts: initial code plus two repaired versions.
- Python code generation is split into a statistics/insights phase and a separate chart-construction phase to reduce single-response truncation risk.
- Normal chart generation is not hard-capped to two charts; prompts instead require concise code, short strings, and limited repetition.
- Python chart payloads are constrained before and after execution: row-level chart data is capped, large histograms are converted to fixed-size bins, box plots are converted to five-number summaries, and oversized nested statistics are compacted before subprocess stdout is parsed.
- If Python code failure looks like output truncation, such as unterminated strings or unclosed brackets, the repair prompt switches to a concise repair mode with shorter code and reduced output only as needed.
- If all three attempts fail, DataMind returns the three attempt errors to the user and falls back to rule-based analysis without failing the whole task.
- Local development can use the isolated subprocess executor. The production Compose profile routes generated code through the controlled `python-runner`, which creates one-shot no-network, non-root containers with a read-only root filesystem, dropped capabilities, PID/CPU/memory/output limits, and cooperative cancellation.
- Kubernetes-native scheduling, gVisor-grade isolation, and multi-host Runner orchestration remain future work.

Supported chart types:

- Bar chart.
- Line chart.
- Pie chart.
- Histogram.
- Box plot.
- Correlation heatmap.

Text analysis requirements:

- Detect text columns such as `review`, `comment`, `feedback`, `评价`, `评论`.
- Detect group columns such as `sentiment`, `label`, `category`, `情绪`.
- Compute text count, empty count, average/median/max length.
- Extract simple English keywords and Chinese bigram keywords.
- Compare group count, average text length, and group-level keywords.
- Return results through `python_result.text_analysis`.

## 6.10 Report Generation

The report page is the final product surface.

Report content:

- Executive Summary.
- Key Findings.
- Charts.
- SQL Results.
- Python Results.
- Data Gaps.
- Validation Issues.
- Analysis Trace.
- Chart explanations.
- Recommended next steps.

Exports:

- HTML report.
- Markdown report.
- Browser-print PDF.
- Per-chart SVG and 2x PNG image export.

Presentation modes:

- Brief, standard, and detailed templates are available when generating and viewing a report.
- Template selection changes bounded report/visualization preferences for new reports and deterministically limits findings, charts, validation details, SQL rows, and recommendations when rendering an existing structured report.
- Template selection is applied consistently to the web detail view, print view, HTML download, and Markdown download without changing persisted analytical evidence.

Provider:

- Kimi by default for report integration, review, chart explanation, and final structured report generation.
- Rule report fallback when Kimi is unavailable or the prompt is too large.

Current context-control requirement:

- Large SQL/Python/chart payloads must be compacted before being sent to LLM review/report prompts to avoid provider context-length or HTTP 413 errors.
- Dataset group relationship prompts must use compressed table schema, field statistics, sample values, and short previews only; full uploaded records must not be sent to the LLM.
- Every model call enforces a configurable text budget (`DATAMIND_LLM_PROMPT_MAX_CHARS`); inline image bytes are excluded from the text calculation while their descriptions and extracted text remain bounded.
- User questions, file names, column names, cell values, preview rows, multimodal descriptions, and extracted file text are untrusted prompt data. Agent system prompts explicitly reject instructions embedded in those values.
- Planner, SQL, and Python prompts receive compact multi-dataset provenance, join status, row expansion, key uniqueness, and a grain rule that prevents duplicated `SUM`/`AVG` after one-to-many joins.
- Experience-library context is selected per agent; execution agents receive priority guidance only, while review/report stages may receive concise historical summaries. Incompatible plan schemas are not injected into agent output contracts.

## 6.11 Dashboard

Dashboard should read real backend data and show:

- Dataset count.
- SQL query count.
- Python analysis count.
- Report count.
- Recent analyses.
- Dataset status summary.
- Delete recent analysis/report actions.

## 6.12 Multimodal Context

Current requirement:

- Analysis page can accept optional screenshot/image/PDF context.
- Image context can be passed to Kimi for later-stage report/review prompts.
- PDF context should extract text when possible.
- Multimodal context can support interpretation and data-gap notes, but cannot replace dataset-backed SQL/Python evidence.

## 6.13 Dataset Groups And Relationship Modeling

Implemented behavior:

- A successful multi-file batch upload is represented as a dataset group.
- Dataset groups show table list, row/column counts, inferred table entity type, field summaries, and sample previews.
- Relationship suggestions combine deterministic rules and optional LLM semantic assistance.
- Rule suggestions use field name similarity, normalized names, field roles, type compatibility, sample match rate, and duplicate-rate risk.
- LLM assistance receives compact schema/sample context only, returns structured relationship candidates, and is validated by the backend before display.
- Multi-file import automatically runs relationship inference after cleaning and persists relationships that pass confidence, sample-match, shape, and executable-plan validation.
- Automatic configuration chooses one reliable field pair per table connection, avoids indiscriminately saving duplicate candidates, rejects many-to-many candidates, and reports tables that cannot be connected reliably.
- The import card exposes one continuous `import/clean -> create package -> identify and save relationships` pipeline with elapsed time and rule/semantic/sample-validation stages.
- Relationship inference can be rerun from relationship management; the result is automatically validated and saved instead of requiring a non-editable manual confirmation step.
- If no relationship passes validation, analysis remains disabled and links directly to automatic re-identification without affecting imported or cleaned data.
- Automatic selection builds a maximum-confidence acyclic relationship tree, prefers `many_to_one` / `one_to_one` directions, and supports chained paths such as `order_items -> orders -> customers` and `order_items -> products -> translation`.
- Join execution resolves original fields through every chain level, prefixes all attached-table columns, and still exposes one internal DuckDB table named `dataset` to SQL and Python.
- A join estimated above 10x row expansion or 1,000,000 output rows is skipped before materialization; moderate expansion is executed with an explicit aggregation-duplication warning.
- Relationship plans are validated before persistence or async job creation for real columns, one root, one parent per table, reachability, duplicate edges, and cycles.
- Persisted relationships retain baseline/current match rates, drift amount, validation timestamp, freshness state, and the triggering drift event.
- Published semantic execution uses an entity relationship graph and selects only the shortest declared path needed by the chosen metric and dimensions.
- The Planner distinguishes fact-table analysis grain from metric-source entities. Many-to-one and one-to-one traversals are direct; one-to-many requires an explicit deduplication rule; unsupported pre-aggregation, semi-join, unknown-cardinality, and many-to-many paths are blocked before SQL execution.

Limitations:

- Relationship modeling and published semantic models provide a governed semantic-layer v1, not organization-wide semantic governance.
- Arbitrary user-authored multi-table SQL is intentionally not exposed; multi-table SQL must come from a validated published semantic model and metric DSL.
- Organization-wide lineage governance and interactive impact exploration remain future work; execution-scoped cross-dataset lineage is implemented.

## 6.14 Semantic Models, Metric Ontology, And Embeddings

Implemented semantic model lifecycle:

- Dataset and dataset-group scopes with versioned drafts, optimistic revision checks, validation, immutable publication, copy/rebinding, and history.
- Entities with fact/dimension roles, stable entity IDs, stable field IDs, source bindings, grain, dimensions, time dimensions, metrics, relationships, cardinality, join risk, and deduplication requirements.
- Published models are immutable; changes create the next draft version.
- Jobs, planner decisions, generated reports, metric formulas, field provenance, join paths, and semantic model versions remain pinned to the execution-time version.
- Planner decisions include the selected entity graph, fact grain, metric-source entities, Join strategies, safety verdict, and warnings.
- Analysis responses and report metadata persist a lineage graph from source fields and semantic metrics through findings/charts to the report artifact.
- The dataset-group workbench provides a visual editor for entity roles/grain, dimensions, metrics, aliases, source-field bindings, relationships, cardinality, and enabled state, plus a relationship/lineage view. Advanced JSON DSL editing remains available for expressions not yet covered by the visual controls.

Metric DSL requirements:

- DSL v2 uses `definition_schema_version: 2`, stable `entity_id`/`field_id`, ASCII SQL aliases, and consistently quoted source fields.
- DSL v1 `{entity, field}` definitions remain readable for backward compatibility.
- Supported expressions: `field`, `literal`, aggregate functions, arithmetic, `case`, `coalesce`, `nullif`, `abs`, `round`, `date_diff`, `date_trunc`, `metric_ref`, and structured filters.
- Validation rejects unknown fields, incompatible types, metric-reference cycles, invalid grain/join paths, unsafe many-to-many publication, and unprotected divide-by-zero behavior.
- Users cannot persist arbitrary SQL as a metric definition.

Chinese semantic compatibility:

- Stable IDs are derived from scope, dataset, source name, and semantic type; display names are never used as internal SQL identifiers.
- Chinese, mixed-language, spaced, slash, parenthesis, and quoted source columns execute through one quoting/resolution path.
- Metric/dimension ranking weights name/alias 45%, embedding 35%, type/role 10%, and question context 10%; exact metric names remain dominant and a Top-2 gap below `0.08` is surfaced as ambiguity.
- Relationship ranking keeps rules/type/sample/cardinality at 90% and embedding at 10%; embedding cannot create absent fields, override zero sample matching, or auto-publish many-to-many relationships.

Embedding behavior:

- Providers: disabled, mock, local SentenceTransformer, and persistent user-scoped cache wrapper.
- Default model is `BAAI/bge-small-zh-v1.5` at revision `4e17e244a0fb63bfb78fca8fcf95079fcc664f5c` with normalized CPU batch encoding.
- Request handling never downloads a model. Development can fall back to deterministic rules; production-required mode fails readiness when the local model is unavailable.
- Process LRU and `semantic_embedding_cache` store only user ID, model revision, text hash, vector, and timestamp; raw sample text is not persisted.

---

# 7. Layered Model Strategy

DataMind uses provider routing by task, not a single model for the full chain.

```text
DeepSeek: planning, cleaning guidance, Text-to-SQL, Python code generation, structured JSON
Kimi: insight integration, adversarial review, chart explanation, report writing, multimodal context
Mock: tests and non-Assistant no-key fallback; Kimi Assistant reports `not_configured` instead of silently changing providers
```

Default provider mapping:

```text
DATAMIND_DEFAULT_LLM_PROVIDER=deepseek
DATAMIND_PLANNER_LLM_PROVIDER=deepseek
DATAMIND_SQL_LLM_PROVIDER=deepseek
DATAMIND_REFLECTION_LLM_PROVIDER=deepseek
DATAMIND_REPORT_LLM_PROVIDER=kimi
DATAMIND_REVIEW_LLM_PROVIDER=kimi
DATAMIND_MULTIMODAL_LLM_PROVIDER=kimi
```

Default models:

```text
DeepSeek: deepseek-chat
Kimi report/review: moonshot-v1-32k
Kimi Assistant: kimi-k2.6
```

Business logic should call providers through the internal model router, not through provider SDKs directly.

---

# 8. LangGraph Workflow

Default analysis graph:

```text
START
  -> planner
  -> design_framework
  -> loop_bootstrap
  -> loop_decide
       ├─ one allowlisted tool call -> loop_execute -> loop_observe -> loop_verify
       │                                      ├─ repairable -> loop_repair -> loop_decide
       │                                      ├─ sufficient -> loop_decide / loop_finalize
       │                                      └─ exhausted/provider failure -> loop_fallback
       └─ finish -> loop_finalize
  -> integrate_insights
  -> format_charts
  -> statistical_verify
  -> adversarial_validate
       └─ one optional evidence repair -> loop_adversarial_repair -> loop_decide
  -> report_decide
  -> report_execute
  -> report_verify
       ├─ report issue -> report_repair -> report_execute (at most two revisions)
       ├─ evidence gap -> analysis Loop (at most once)
       └─ provider/validation failure -> report_fallback
  -> report_commit
  -> END
```

Current behavior:

- `agent_mode=auto` resolves to `loop` under the default deployment policy. The React analysis page presents “自主分析” as the default segmented mode and “兼容模式” as an explicit alternative.
- The Loop controller can choose one job-scoped, read-only analysis tool per decision. Identity, dataset scope, semantic decision, timeout, call budget, decision budget, token budget, and duplicate-action keys are injected server-side.
- Tool errors are classified as repairable, policy, provider, timeout, budget, or terminal failures. Repair must change the tool or arguments; repeated identical failures cannot create an unbounded retry cycle.
- Successful actions and report commits are idempotent across retries and checkpoint recovery. Ordered events expose decisions, tool execution, verification, repair, fallback, report validation, and commit without storing hidden reasoning or unbounded raw rows.
- SQL remains guarded by deterministic safety validation. Generated Python remains sandboxed, output-bounded, compacted, and eligible for up to two LLM repairs before deterministic fallback.
- Report findings with numeric claims must reference valid evidence IDs. The report Loop may revise twice and request one additional analysis pass before committing or falling back.
- Planner execution freezes a versioned `AnalysisContract` with data scope, population, metric, dimensions, time field, analysis grain, method, assumptions, acceptance criteria, stop conditions, causal policy, and server-owned budgets.
- `statistical_verify` deterministically checks every finding before report generation. Numeric evidence coverage must be 100%; comparison claims carry sample size and an effect size or 95% confidence interval; unsafe Join expansion without source-grain evidence and unqualified causal language fail validation.
- High-severity statistical failures reuse the existing one-pass adversarial evidence-repair route. Findings that still fail after bounded repair are excluded from the final report and remain visible as validation issues.
- `agent_mode=legacy` preserves the fixed planner -> SQL/Python -> iterative rounds -> review -> report graph for historical compatibility and operational diagnosis.

Related autonomous cleaning graph:

```text
select_cleaning_strategy
  -> execute rules / llm / hybrid
  -> verify quality gates
  -> repair or switch strategy
  -> commit one validated cleaning version
```

Remaining gaps:

- The visual debugger is read-only; users cannot edit graph topology or rerun an arbitrary node with modified state.
- Experience files and per-stage prompt preferences influence bounded prompts, but are not yet a complete non-developer workflow configuration system.
- Multi-round hypothesis exploration remains shallower than the referenced third-party agent.

---

# 9. Technology Stack

## Backend

- Python 3.12.
- FastAPI.
- LangGraph.
- Pydantic v2.
- SQLite for persisted application data.
- PostgreSQL for production persistence, with SQLAlchemy compatibility and Alembic migrations.
- Redis and Celery for production task delivery and workers.
- Persistent LangGraph SQLite/PostgreSQL checkpointers.
- DuckDB as internal SQL execution engine.
- `sqlglot` for semantic multi-table SQL AST validation.
- Pandas.
- Plotly-compatible chart specs.
- Optional local SentenceTransformers/Hugging Face runtime for Chinese semantic embeddings.

## Frontend

- React.
- Vite.
- Tailwind CSS.
- Lucide React icons.

## LLM

- DeepSeek.
- Kimi / Moonshot.
- Mock fallback.

## Deployment

- Production Docker Compose stack with a Caddy HTTPS gateway and an Nginx-served React bundle using same-origin `/api/v1` proxying, including unbuffered SSE and long-running request support.
- Separate API, Celery Worker, Beat, PostgreSQL, password-protected Redis, Python Runner, and one-shot sandbox image, plus explicit data-volume initialization and Alembic migration jobs.
- Persistent PostgreSQL, Redis, application data, Assistant attachment, Runner temp, and Caddy certificate volumes; backend services remain on an internal network and only the gateway is publicly exposed.
- Non-root application images, read-only frontend filesystem, service health checks, restart policies, graceful Worker shutdown, and dynamic Docker DNS resolution for rolling API replacement.
- Production startup requires an authenticated Python Runner. Each generated-code container has a controller-enforced wall timeout and is killed and removed in `finally`; production cannot fall back to a host subprocess.
- `/health/ready` verifies PostgreSQL, Redis, a live Celery Worker, Python Runner, MCP Registry, required semantic embedding, and Assistant configuration. Failed critical checks return HTTP 503, and Compose gates the frontend on this strict readiness endpoint.
- Optional OpenTelemetry OTLP export.
- Production Compose enables autonomous cleaning, analysis, and report Loops by default while local compatibility modes remain configurable.
- `.github/workflows/production-smoke.yml` provisions the real PostgreSQL/Redis/Celery/Python Runner stack and verifies Cookie login, asynchronous cleaning, autonomous analysis, persisted report generation, and result retrieval.

---

# 10. User Interface

## Login

- Animated login page.
- Local username/password entry with Argon2id migration.
- Production HttpOnly/SameSite Cookie session and CSRF protection.
- Logout revokes the server-side session and returns to login.

## Sidebar

- Home.
- Datasets.
- Analysis Tasks.
- Reports.
- Kimi Assistant.

## Dashboard

- Real backend counts.
- Recent analyses.
- Running and failed analysis jobs.
- Dataset status.
- Delete recent analysis/report.

## Dataset Page

- Batch upload CSV/XLSX/JSON/TXT files by file picker or drag-and-drop.
- Drag-and-drop stays enabled on the upload surface, but drag handling is temporarily ignored while the native file picker is open to avoid duplicate selection behavior.
- Multi-file batches are shown as dataset groups after successful batch import.
- In imported dataset lists, one batch-created dataset group is shown as one record instead of listing every child file as a separate top-level row.
- Dataset group cards show table entity type, columns, relationship candidates, automatically adopted relationships, and unresolved tables.
- Dataset-group workbench includes visual entity/metric/dimension/relationship editing, relationship lineage, advanced JSON DSL editing, validation, immutable publication, version selection, and validation messages; cross-scope copy/rebinding is available through the API.
- Users can rerun automatic relationship identification; validated relationships are persisted in the same operation.
- Preview Excel sheets per uploaded file and choose one sheet before importing.
- Process the upload queue by importing and cleaning each selected file sequentially.
- Batch processing shows that multi-file cleaning can take longer and reports per-file success/failure counts.
- Dataset upload and cleaning work remains mounted while navigating to other product sections, preserving queue items, per-file progress, Excel selections, and completion messages.
- Enter optional cleaning requirement.
- Imported dataset list.
- Cleaned dataset list.
- Raw preview.
- Cleaned preview.
- Schema/profile.
- Editable field type, role, and description metadata.
- Cleaning version history, diff preview, and activation/rollback.
- Manual cleaning rule editor with preview, validation blocking, high-impact acknowledgment, and apply-as-reversible-version.
- Related analysis records.

## Analysis Page

- Dataset selector.
- Optional dataset group selector.
- Automatically validated dataset group relationships can be used as the relationship plan for joined analysis.
- Optional additional dataset selector for multi-file analysis.
- Join recommendation and editable join configuration for uploaded datasets.
- Question input.
- Optional multimodal context.
- Agent plan.
- Workflow view.
- Event-driven realtime Workflow UI with waiting/running/completed/failed states, running logs, and expandable agent details.
- Global floating task-progress capsule remains visible outside the analysis page, showing the current Workflow stage, live percentage, task title, and a direct return action to the running session.
- Planner metadata: confidence, route reason, candidate fields, and clarifying questions.
- Semantic plan card shows selected metric/dimensions, model version, component confidence, evidence, ambiguities, and the calibrated confidence level.
- Low-confidence semantic plans cannot start analysis until the user explicitly confirms the proposed metric, dimensions, and join path.
- If semantic plan preview is unavailable, the page preserves the legacy analysis flow.
- Workflow timeline/debugger with node summaries and fallback/error details.
- Multi-dataset join summary in planner metadata and workflow debugger.
- Async job status, progress events, cancel, retry.
- ChatGPT-style analysis session history sidebar backed by `analysis_jobs`; every run creates and selects a new session record.
- Any queued/running/completed/failed/canceled/interrupted session can be reopened to restore its own Workflow nodes, ordered events, progress, logs, and errors.
- Completed sessions additionally restore their persisted SQL/Python/charts/report result instead of reusing a global latest-result view.
- New analysis opens an independent draft; navigating between product sections preserves the selected session.
- SQL result, Python result, charts, report summary, validation issues, trace.

## Reports Page

- Generate report from selected dataset and question.
- Report history with search and detail view.
- Report rename.
- Report version history.
- Report version comparison.
- Structured report preview.
- Charts with visible axis labels/ticks where applicable.
- SQL results.
- Validation issues.
- Analysis trace.
- Browser-print PDF export.
- HTML export.
- Markdown export.
- Brief, standard, and detailed report templates.
- Per-chart SVG and 2x PNG export.

## Kimi Assistant Page

- Persistent user-scoped conversation history with create, rename, soft-delete, and cross-page run recovery.
- Cursor-based conversation summaries compress older completed messages after 12 unsummarized messages or 24,000 characters while preserving the latest eight messages verbatim.
- User-scoped and asset-scoped long-term memory persists approved preferences, terminology, metric definitions, workflow preferences, and business context across conversations. Explicit durable statements are activated automatically; inferred candidates require confirmation in the Kimi workbench.
- Trustworthy Memory v2 stores immutable semantic-memory version chains. Explicit conflicts supersede the current version transactionally; inferred conflicts remain pending. Structured summaries retain source message IDs, while every actual recall is recorded with lexical, embedding, scope, recency, and selection-reason evidence.
- Memory retrieval applies hard user/asset/status/validity filters, lexical Top-100 preselection, BGE/scope/recency scoring, MMR diversity, and minimum relevance gates. Current instructions and published semantic models override memory, and memory cannot grant tools or widen asset scope.
- Validated analysis experience from ordinary or Assistant-started jobs is stored separately as episodic memory only after Statistical Verifier success and a validated report. Planner may use its compact contract, semantic version, Join/grain route, tool sequence, and result summary as read-only route evidence; Schema, cleaning, relationship, or semantic drift marks it stale before reuse.
- A user-level switch disables long-term memory read/write without deleting stored history or disabling current-conversation summaries and task Checkpoints. Kimi answers disclose only memories actually injected for that run.
- Unpinned active memory is recycled after 180 unused days; pending memory is recycled after 30 days; both remain recoverable for 30 additional days. Deleting a conversation does not delete independently stored long-term memory.
- Automatic evidence retrieval across the user's completed reports and analysis jobs, with optional dataset, dataset-group, or report scope pinning.
- JPEG/PNG/WebP upload with authenticated storage and native Kimi visual context.
- CSV/XLSX/JSON/TXT chat attachments support preview-first multi-file import, disk-backed upload staging, bounded batch parsing, read-only per-Sheet XLSX scanning, automatic cleaning, dataset-group creation, relationship inference, and explicit management authorization.
- Ask mode is read-only. Execute mode exposes cleaning, relationship, analysis, report, semantic draft/publication, recycle, and restore tools only when conversation scope and a long-lived asset Grant both allow the operation.
- `AssistantPermissionService` performs authorization independently from `NodeExecutionHarness`; the model cannot supply a user ID, Grant ID, or expanded scope.
- Every mutation uses a canonical action hash, idempotency key, before/after state, and user-scoped audit record. Reversible actions can be undone from the Kimi workbench.
- Dataset-group Grants inherit to member datasets and future reports/tasks/semantic versions; revocation applies on the next tool call.
- Grants are constrained by a server-side asset capability matrix. Report-scoped Grants can manage/recycle the report and run analysis only against its source dataset; they cannot clean, edit fields, manage relationships/semantic models, or recycle the source dataset even if an older stored Grant contains those capabilities.
- Assistant Celery Runs use atomic database claims, expiring leases, independent heartbeats, periodic recovery, and checkpoint-safe idempotency. Event sequence allocation is an atomic counter rather than concurrent `MAX(sequence)+1`.
- Final Kimi answers use the provider's native SSE stream; visible `message.delta` events are emitted as answer tokens arrive rather than splitting an already completed response.
- Ask-mode questions already covered by validated report evidence use a deterministic fast path that skips the redundant non-streaming tool-routing model call. Execute mode and requests involving status, schema, cleaning, relationships, semantics, or report mutation always retain the full permissioned tool route.
- Assistant completion telemetry records queue wait, retrieval, tool routing, provider first-token, first-answer, total latency, fast-path usage, and combined token usage. The privacy-safe Benchmark history summary aggregates only these metrics and never reads prompts, message bodies, dataset rows, or report content.
- Assistant Repository schema initialization is process-scoped in local SQLite mode. Production PostgreSQL performs a lightweight migrated-schema check and leaves all DDL to Alembic.
- Report revisions are presentation-only operations: they reuse the source report's frozen SQL rows, Python statistics, chart records, and evidence fingerprint instead of rerunning the analysis Workflow. Concise or visual revisions may select fewer findings/charts and change chart presentation metadata, but cannot change analytical values.
- Each Assistant message has at most one primary report deliverable. Historical or supporting reports remain clickable evidence, while the report created by the latest audited action is rendered as the single complete-report card.
- Chat images are loaded through the authenticated API client rather than direct browser URLs, so both secure Cookie sessions and local legacy user headers can display protected attachments.
- Cleaning and analysis tools accept bounded per-stage task preferences for cleaning, Planner, SQL, Python, visualization, review, and report generation. Preferences are persisted with the job and injected as untrusted user instructions below immutable safety prompts.
- New cleaning and analysis Workflow events appear inside the conversation; low-confidence semantic plans and every soft delete stop for explicit confirmation.
- Datasets, dataset groups, reports, and unpublished semantic drafts use API-compatible soft deletion. Normal reads return 404, the recycle bin restores the original UUID within 30 days, and only lifespan/Celery Beat cleanup can permanently purge expired assets.
- Final answers retain validated dataset/report/job citations and never expose raw paths, generated Python code, or cross-user assets.

---

# 11. API Surface

Current primary API groups:

- `/api/v1/auth`
- `/api/v1/store/datasets`
- `/api/v1/store/dataset-groups`
- `/api/v1/store/reports`
- `/api/v1/analysis/run`
- `/api/v1/analysis/jobs`
- `/api/v1/analysis/join-suggestions`
- `/api/v1/analysis/plans`
- `/api/v1/analysis/planner-decisions`
- `/api/v1/assistant/conversations`
- `/api/v1/assistant/runs`
- `/api/v1/assistant/attachments`
- `/api/v1/assistant/permission-grants`
- `/api/v1/assistant/actions`
- `/api/v1/assistant/memories`
- `/api/v1/assistant/import-batches`
- `/api/v1/assistant/recycle-bin`
- `/api/v1/health`
- `/api/v1/health/live`
- `/api/v1/health/ready`

Dataset store extensions:

- `POST /api/v1/store/files/xlsx-sheets`
- `POST /api/v1/store/dataset-groups`
- `GET /api/v1/store/dataset-groups`
- `GET /api/v1/store/dataset-groups/{group_id}`
- `POST /api/v1/store/dataset-groups/{group_id}/relationship-suggestions`
- `PATCH /api/v1/store/dataset-groups/{group_id}/relationships`
- `POST /api/v1/store/dataset-groups/{group_id}/relationships/auto-configure`
- `GET /api/v1/store/dataset-groups/{group_id}/drift`
- `POST /api/v1/store/dataset-groups/{group_id}/drift/scan`
- `DELETE /api/v1/store/dataset-groups/{group_id}?delete_datasets=`
- `POST /api/v1/store/datasets/{dataset_id}/cleaning-runs` with optional `use_llm`
- `GET /api/v1/store/datasets/{dataset_id}/cleaning-runs`
- `GET /api/v1/store/datasets/{dataset_id}/cleaning-runs/{run_id}`
- `POST /api/v1/store/datasets/{dataset_id}/cleaning-runs/{run_id}/activate`
- `POST /api/v1/store/datasets/{dataset_id}/cleaning-rules/preview`
- `POST /api/v1/store/datasets/{dataset_id}/cleaning-rules/apply`
- `GET /api/v1/store/datasets/{dataset_id}/columns`
- `POST /api/v1/store/datasets/{dataset_id}/columns`
- `PATCH /api/v1/store/datasets/{dataset_id}/columns/{column_name}`
- `GET /api/v1/store/datasets/{dataset_id}/drift`
- `POST /api/v1/store/datasets/{dataset_id}/drift/scan`
- `GET /api/v1/store/datasets/{dataset_id}/drift/history`

Autonomous cleaning job extensions:

- `POST /api/v1/store/datasets/{dataset_id}/cleaning-jobs`
- `GET /api/v1/store/datasets/{dataset_id}/cleaning-jobs/{job_id}`
- `GET /api/v1/store/datasets/{dataset_id}/cleaning-jobs/{job_id}/events` as resumable SSE
- `POST /api/v1/store/datasets/{dataset_id}/cleaning-jobs/{job_id}/cancel`
- `POST /api/v1/store/datasets/{dataset_id}/cleaning-jobs/{job_id}/retry`
- `GET /api/v1/store/datasets/{dataset_id}/cleaning-jobs/{job_id}/result`
- Cleaning requests accept `cleaning_strategy=auto|rules|llm|hybrid`, an optional requirement, and bounded per-stage `prompt_overrides`.

Semantic model extensions:

- `POST /api/v1/store/semantic-models/drafts`
- `GET /api/v1/store/semantic-models?scope_type=&scope_id=`
- `GET /api/v1/store/semantic-models/{model_id}`
- `PUT /api/v1/store/semantic-models/{model_id}`
- `POST /api/v1/store/semantic-models/{model_id}/validate`
- `POST /api/v1/store/semantic-models/{model_id}/publish`
- `POST /api/v1/store/semantic-models/{model_id}/copy`

Semantic planning extensions:

- `POST /api/v1/analysis/plans`
- `POST /api/v1/analysis/planner-decisions/{decision_id}/feedback`
- Analysis job requests may include `planner_decision_id` and `confirmed_low_confidence`.
- Planner responses expose semantic model ID/version, semantic plan, component scores, raw/calibrated confidence, decision band, ambiguities, evidence, and confirmation requirement.

Report store extensions:

- `GET /api/v1/store/reports?query=&dataset_id=&limit=`
- `GET /api/v1/store/reports/{report_id}`
- `PATCH /api/v1/store/reports/{report_id}`
- `GET /api/v1/store/datasets/{dataset_id}/report-versions`

Async analysis job extensions:

- `POST /api/v1/analysis/join-suggestions`
- `POST /api/v1/analysis/jobs`
- `GET /api/v1/analysis/jobs`
- `GET /api/v1/analysis/jobs/{job_id}`
- `GET /api/v1/analysis/jobs/{job_id}/events` as resumable SSE
- `GET /api/v1/analysis/jobs/{job_id}/result`
- `POST /api/v1/analysis/jobs/{job_id}/cancel`
- `POST /api/v1/analysis/jobs/{job_id}/retry`
- Analysis requests accept `agent_mode=auto|loop|legacy`; `auto` resolves to Loop by default.
- Analysis requests accept bounded per-stage `prompt_overrides`. Retries preserve the original overrides, semantic decision, dataset scope, join/relationship plan, and execution mode.

Authentication extensions:

- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`
- `POST /api/v1/auth/logout`

Kimi Assistant extensions:

- `POST/GET /api/v1/assistant/conversations`
- `GET/PATCH/DELETE /api/v1/assistant/conversations/{conversation_id}`
- `GET /api/v1/assistant/conversations/{conversation_id}/messages`
- `POST /api/v1/assistant/conversations/{conversation_id}/messages`
- `POST /api/v1/assistant/conversations/{conversation_id}/attachments`
- `GET /api/v1/assistant/attachments/{attachment_id}/content`
- `POST /api/v1/assistant/import-batches/preview`
- `POST /api/v1/assistant/import-batches/{batch_id}/commit`
- `GET /api/v1/assistant/runs/{run_id}`
- `GET /api/v1/assistant/runs/{run_id}/events` as resumable SSE
- `POST /api/v1/assistant/runs/{run_id}/confirm`
- `POST /api/v1/assistant/runs/{run_id}/cancel`
- `GET/POST /api/v1/assistant/permission-grants`
- `DELETE /api/v1/assistant/permission-grants/{grant_id}`
- `GET /api/v1/assistant/actions` and `POST /api/v1/assistant/actions/{action_id}/undo`
- `GET/POST /api/v1/assistant/memories`
- `PATCH/DELETE /api/v1/assistant/memories/{memory_id}`
- `POST /api/v1/assistant/memories/{memory_id}/confirm`
- `POST /api/v1/assistant/memories/{memory_id}/restore`
- `GET /api/v1/assistant/recycle-bin`
- `POST /api/v1/assistant/recycle-bin/{asset_type}/{asset_id}/restore`

Internal runtime and operational endpoints:

- `GET /api/v1/mcp/tools` requires an authenticated user.
- `POST /api/v1/mcp/invoke` is restricted to administrator/development policy and is not described as a standard external MCP transport.
- `POST /api/v1/mcp/model-stream` supports internal provider streaming.

API requirements:

- Frontend should read persisted records from FastAPI.
- Frontend should not rely on temporary session state for datasets, reports, or recent analysis records.
- Analysis endpoints should return both legacy-compatible fields and structured report fields.
- Analysis requests may include `additional_dataset_ids` and a user-confirmed `join_plan`; single-dataset requests remain compatible.
- Analysis requests may include `dataset_group_id` and `relationship_plan`; these are mapped to the existing joined dataframe v1 execution path.
- Model-supplied identities, Grant IDs, raw filesystem paths, arbitrary SQL, safety-policy changes, and system-prompt replacements are never accepted as user tool parameters.

---

# 12. Project Structure

```text
app/
  api/
    v1/
      auth.py
      analysis.py
      dataset_store.py
      deps.py
  analysis/
    agent_loop.py
    analysis_contract.py
    checkpoints.py
    cleaning_jobs.py
    cleaning_sandbox.py
    cleaning_workflow.py
    dataset_groups.py
    jobs.py
    lineage.py
    multidataset.py
    prompt_override_router.py
    prompt_utils.py
    python_execution.py
    workflow.py
    workflow_prompt_context.py
    services.py
    statistical_verifier.py
    model_router.py
    python_sandbox.py
    text_analysis.py
    validators.py
    experience_loader.py
    experience/
  assistant/
    jobs.py
    permissions.py
    report_revision.py
    tools.py
    workflow.py
  semantic/
    dsl.py
    embedding.py
    ranking.py
    service.py
    relationship_graph.py
    download_model.py
  data_reliability/
    drift.py
  schemas/
    analysis.py
    assistant.py
    auth.py
    dataset_store.py
    data_reliability.py
    prompt_overrides.py
    semantic.py
  services/
    tabular_import.py
  storage/
    auth_repository.py
    assistant_repository.py
    dataset_store.py
    models.py
    repositories.py
    row_mappers.py
    sqlalchemy_compat.py
    migrate_sqlite_to_postgres.py
  harness/
    node.py
  python_runner/
    main.py
    smoke.py
  task_queue.py
  observability.py
  main.py

frontend/
  react/
    e2e/
    src/
      api-client.ts
      workflow-ui.ts
      main.tsx
      styles.css
      features/
        assistant/
          AssistantPage.tsx
          AssistantControlPanel.tsx
          AssistantConversationHeader.tsx
          AssistantWorkflowCard.tsx
          AssistantEvidenceCards.tsx
          AssistantImportPreview.tsx
          AssistantAttachmentImage.tsx
          types.ts
        datasets/
          CleaningWorkspace.tsx
        data-reliability/
          DriftMonitorPanel.tsx
        analysis/
          AnalysisReliabilityPanel.tsx
        reports/
          chart-export.ts
          report-templates.ts
        semantic/
          SemanticModelWorkbench.tsx

data/
  datamind.db

tests/
docs/
scripts/
  production_smoke.py
migrations/
  versions/
    0001_production_foundation.py
    0002_semantic_layer.py
    0003_semantic_embedding_cache.py
    0004_agent_loop.py
    0005_full_loop_engineering.py
    0006_ai_assistant.py
    0007_kimi_capabilities.py
    0008_agent_prompt_overrides.py
    0009_p1_security_reliability.py
    0010_data_reliability_graph.py
    0011_assistant_memory.py
    0012_trustworthy_memory.py
    0013_memory_effectiveness.py
```

---

# 13. Non-functional Requirements

- Modular architecture.
- Strong type annotations where practical.
- Unit tests for backend analysis, import, routing, persistence, and API behavior.
- Frontend build must pass.
- Provider-routed LLM settings must be configurable through environment variables.
- No nested third-party application projects inside the repository.
- Large LLM prompts must use compact execution payloads.
- Uploaded data and generated reports must persist across browser refresh and process restart.

## Semantic Persistence And Deployment

- `semantic_models`, `planner_decisions`, `planner_feedback`, and `planner_calibrators` are introduced by migration `0002_semantic_layer`.
- `semantic_embedding_cache` is introduced by `0003_semantic_embedding_cache` and is included in SQLite-to-PostgreSQL migration.
- `analysis_jobs` stores `planner_decision_id`, `semantic_model_id`, and `semantic_model_version` without changing existing UUIDs.
- `data_snapshots` and `data_drift_events` are introduced by `0010_data_reliability_graph` and included in SQLite-to-PostgreSQL migration.
- `python -m app.semantic.download_model` downloads the configured fixed revision outside request handling; `--verify-only` validates an existing local model.
- Production API and Worker share the model image layer; the Python sandbox image does not contain the embedding model.
- Readiness reports `semantic_embedding=ready|disabled|fallback|failed` and fails when embedding is configured as required but is unavailable.

Embedding configuration:

- `DATAMIND_SEMANTIC_EMBEDDING_ENABLED`
- `DATAMIND_SEMANTIC_EMBEDDING_REQUIRED`
- `DATAMIND_SEMANTIC_EMBEDDING_MODEL`
- `DATAMIND_SEMANTIC_EMBEDDING_MODEL_PATH`
- `DATAMIND_SEMANTIC_EMBEDDING_REVISION`
- `DATAMIND_SEMANTIC_EMBEDDING_DEVICE`
- `DATAMIND_SEMANTIC_EMBEDDING_LOCAL_FILES_ONLY`
- `DATAMIND_SEMANTIC_EMBEDDING_BATCH_SIZE`
- `DATAMIND_SEMANTIC_EMBEDDING_CACHE_SIZE`

## Kimi Assistant Persistence And Deployment

- `assistant_conversations`, `assistant_messages`, `assistant_runs`, `assistant_run_events`, and `assistant_attachments` are introduced by `0006_ai_assistant` and included in SQLite-to-PostgreSQL migration.
- `assistant_permission_grants`, `assistant_action_log`, `assistant_import_batches`, execution-mode fields, attachment import metadata, and recycle columns are introduced by `0007_kimi_capabilities` and included in SQLite-to-PostgreSQL migration.
- `assistant_memories` and conversation summary cursors are introduced by `0011_assistant_memory` and included in SQLite-to-PostgreSQL migration. The dedicated Memory Repository keeps long-term context separate from run/checkpoint persistence.
- `0012_trustworthy_memory` adds immutable memory version chains, structured summaries, per-user settings, recall usage records, resumable background maintenance jobs, and validated episodic analysis experience. It reuses the application database and BGE cache rather than introducing a vector database.
- `0013_memory_effectiveness` adds typed entity/predicate/value facts, per-candidate selection and suppression audit, idempotent user feedback, utility signals, reversible dormancy, and effectiveness aggregation. Background Kimi extraction remains outside the first-token path and every model candidate is revalidated against its source user message.
- `analysis_jobs.prompt_overrides` and `cleaning_jobs.prompt_overrides` are introduced by `0008_agent_prompt_overrides`; retries preserve the original stage preferences and reports retain them as audit metadata.
- API and Worker share protected attachment storage; image and data-file bytes are never exposed as a public static directory.
- Local lifespan and production Celery Beat run daily expiry cleanup; permanent purge is not exposed to Kimi or public HTTP APIs.
- Readiness reports `assistant_model=ready|not_configured|disabled`. Kimi provider errors are visible and do not silently masquerade as another provider.

Assistant configuration:

- `DATAMIND_ASSISTANT_ENABLED`
- `DATAMIND_ASSISTANT_LLM_PROVIDER`
- `DATAMIND_ASSISTANT_LLM_MODEL`
- `DATAMIND_ASSISTANT_MAX_TOOL_CALLS`
- `DATAMIND_ASSISTANT_MAX_CONTEXT_CHARS`
- `DATAMIND_ASSISTANT_TIMEOUT_SECONDS`
- `DATAMIND_ASSISTANT_IMAGE_MAX_BYTES`
- `DATAMIND_ASSISTANT_DATA_FILE_MAX_BYTES`
- `DATAMIND_ASSISTANT_DATA_FILE_MAX_COUNT`
- `DATAMIND_ASSISTANT_DATA_BATCH_MAX_BYTES`
- `DATAMIND_ASSISTANT_RECYCLE_RETENTION_DAYS`
- `DATAMIND_ASSISTANT_RATE_LIMIT`
- `DATAMIND_ASSISTANT_MEMORY_ENABLED`
- `DATAMIND_ASSISTANT_MEMORY_SUMMARY_MESSAGES`
- `DATAMIND_ASSISTANT_MEMORY_SUMMARY_CHARS`
- `DATAMIND_ASSISTANT_MEMORY_SUMMARY_MAX_CHARS`
- `DATAMIND_ASSISTANT_MEMORY_RETRIEVAL_LIMIT`
- `DATAMIND_ASSISTANT_MEMORY_CONTEXT_CHARS`
- `DATAMIND_ASSISTANT_MEMORY_TTL_DAYS`
- `DATAMIND_ASSISTANT_MEMORY_RECYCLE_DAYS`
- `DATAMIND_ASSISTANT_MEMORY_TIMEOUT_SECONDS`
- `DATAMIND_ASSISTANT_MEMORY_RELEVANCE_THRESHOLD`
- `DATAMIND_ASSISTANT_MEMORY_PREFILTER_LIMIT`
- `DATAMIND_ASSISTANT_MEMORY_MMR_LAMBDA`
- `DATAMIND_ASSISTANT_MEMORY_EXPERIENCE_ENABLED`
- `DATAMIND_ASSISTANT_MEMORY_MODEL_EXTRACTION_ENABLED`
- `DATAMIND_ASSISTANT_MEMORY_AUTO_DORMANCY_ENABLED`
- `DATAMIND_ASSISTANT_MEMORY_DORMANCY_THRESHOLD`
- `DATAMIND_ASSISTANT_MEMORY_DORMANCY_MIN_FEEDBACK`
- `DATAMIND_ASSISTANT_MEMORY_WRONG_FEEDBACK_LIMIT`

### Bounded Loop Engineering (implemented, default)

The stable planner, insight integration, chart formatting, deterministic statistical verification, adversarial review and report nodes remain in place. The default LangGraph path replaces the fixed SQL/Python/iterative execution segment with `loop_bootstrap → loop_decide → loop_execute → loop_observe → loop_verify → loop_repair|loop_fallback → loop_finalize`. Statistical or adversarial validation may return to the Loop once for a final evidence repair. The legacy path remains available only as an explicit compatibility mode.

- The model receives an explicit read-only tool allowlist and may issue at most one Tool Call per decision.
- `AgentToolRuntime` injects the authenticated repository, job ID, dataset scope and semantic decision server-side; model arguments cannot override identity or scope.
- Successful actions are keyed by canonical tool arguments and reused after checkpoint recovery. Repeated identical failures, per-tool attempts, total calls, decisions, elapsed time and token use are bounded.
- Tool outputs are compressed into event payloads; large evidence is stored as an artifact. Events never persist raw hidden reasoning, credentials or unbounded result rows.
- `agent_mode=legacy` preserves the existing workflow; `auto` resolves to Loop under the default deployment policy; explicit deployment configuration can still disable Loop or select legacy compatibility mode.
- The frontend keeps the seven-stage workflow, labels the execution segment as an autonomous loop, and renders decision, execution, verification, repair, fallback and remaining-budget events in a separate component.
- The offline release evaluator records legal tool selection, repair success, simple-task call count and duplicate successful actions against the 95% / 90% / 4-call / zero-duplicate gates. Provider-specific benchmark outcomes can be fed into this evaluator without storing source data rows.
- The project-level Benchmark Harness executes frozen deterministic cases for cleaning, Chinese semantic ranking, SQL safety, relationship inference, Schema drift, relationship-grain safety, report evidence, statistical contracts, Assistant permissions and Loop reliability. It writes JSON, JSONL, Markdown and JUnit artifacts with environment/model identity and corpus checksums; missing latency/token telemetry is `metric_unavailable`, never zero.
- Push/PR CI runs the deterministic `release` benchmark. A separate weekly/manual workflow runs three-repeat DeepSeek/Kimi canaries plus explicit performance and resilience suites on the Docker production stack. Real-provider quality and performance remain observational until five valid calibration batches exist; safety gates are always blocking.

Configuration:

- `DATAMIND_AGENT_LOOP_ENABLED`
- `DATAMIND_AGENT_LOOP_DEFAULT_MODE=legacy|loop`
- `DATAMIND_AGENT_LOOP_ALLOW_REQUEST_OVERRIDE`
- `DATAMIND_AGENT_LOOP_PROVIDER`
- `DATAMIND_AGENT_LOOP_MODEL`
- `DATAMIND_AGENT_LOOP_MAX_TOOL_CALLS`
- `DATAMIND_AGENT_LOOP_MAX_DECISIONS`
- `DATAMIND_AGENT_LOOP_MAX_TOOL_ATTEMPTS`
- `DATAMIND_AGENT_LOOP_TIMEOUT_SECONDS`
- `DATAMIND_AGENT_LOOP_MAX_TOKENS`
- `DATAMIND_ANALYSIS_FAST_PATH_ENABLED`
- `DATAMIND_ANALYSIS_FAST_PATH_MAX_ROWS`

---

# 14. Current Completion Status

Implemented:

- React + Vite + Tailwind frontend.
- Configurable single-dataset SQL fast path preserves the bounded Agent Loop, statistical verification and final report while avoiding redundant intermediate LLM calls; complex, multi-dataset and multimodal analyses retain the full model path.
- React runtime and TypeScript declarations are aligned on React 18 (`react`/`react-dom` 18.3, `@types/react`/`@types/react-dom` 18.x).
- Frontend modularization phase 1: Workflow node definitions, status derivation, log translation, and task labels live in `frontend/react/src/workflow-ui.ts` instead of the application entrypoint.
- Frontend modularization phase 2: API base URL/fallback, Cookie/CSRF request headers, typed GET/POST/PATCH/DELETE helpers, cleaning compatibility calls, and auth cache live in `frontend/react/src/api-client.ts`.
- Frontend modularization phase 3: Kimi Assistant components, the visual semantic-model workbench, cleaning version/rule UI, field metadata editing, report template transforms, and chart export utilities live under `frontend/react/src/features`; the extracted dataset workspace is the active implementation and its former duplicates were removed from `main.tsx`.
- Frontend build tooling uses Vite 8.1.4 and `@vitejs/plugin-react` 6.0.3; the dependency audit is clean after replacing the vulnerable Vite 5/esbuild chain.
- FastAPI backend.
- SQLite persistence for users, datasets, reports, and analysis records.
- Storage modularization phase 1: persisted dataset/group/job domain models live in `app/storage/models.py`, separate from repository SQL and migrations.
- Storage modularization phase 2: password/session helpers and user-session persistence live in `app/storage/auth_repository.py` behind the existing `DatasetStoreRepository` facade.
- Storage modularization phase 3: SQLite compatibility row decoding for datasets, groups, cleaning/analysis jobs, ordered events, semantic models, cleaning versions, column metadata, and reports lives in `app/storage/row_mappers.py`; the repository facade retains query, transaction, and domain behavior.
- PostgreSQL production persistence through the SQLAlchemy compatibility layer and Alembic.
- SQLite-to-PostgreSQL UUID-preserving migration command.
- Revocable HttpOnly Cookie sessions, CSRF/Origin validation, Argon2id password migration, and user-scoped records.
- Redis rate limits for login, analysis job creation, and LLM calls in production.
- CSV/XLSX/JSON/TXT upload through disk-backed backend parsing. Uploads are capped before parsing, JSON and delimited text are consumed in bounded record batches, XLSX uses read-only per-Sheet scans, and database inserts retain only one bounded batch in memory.
- Batch upload queue in the React dataset page with drag-and-drop, file-picker duplicate-guarding, per-file status, Excel sheet selection, import, and cleaning.
- Cross-page dataset-import state preservation: active upload/cleaning work continues when the user navigates away and is restored unchanged on return.
- Single-file and batch imports create asynchronous cleaning jobs with `cleaning_strategy=auto`; the bounded controller selects rules/LLM/hybrid, validates candidate quality, repairs or falls back, and activates only a verified version.
- Dataset group v1 for multi-file batches, including table summaries, entity type hints, relationship suggestions, saved relationship plans, and dataset-group analysis entry.
- Imported dataset list collapses a batch-created dataset group into one top-level record while keeping child tables visible in the group workbench.
- Relationship recommendation v1 with rules first, manually refreshed compact-context LLM semantic supplementation, backend validation, and row-multiplication risk notes.
- Automatic relationship configuration integrated into batch import: rules and compact-context LLM suggestions are backend-validated, selected into an executable acyclic relationship tree, persisted automatically, and shown in a live import pipeline with unresolved-table fallback.
- Chained local multi-dataset execution with original-to-prefixed column lineage, directional match rates, vectorized row-count estimation, hard expansion limits, and per-join executed/skipped risk summaries.
- Version-to-version data reliability monitoring with bounded snapshots, Schema/type/missing/unique/distribution drift, relationship match-rate revalidation, and stale propagation to semantic models, reports, and saved relationships.
- Dataset-group reliability UI with live scan status, stale relationship reasons, drift changes, and authorization-gated recommended actions.
- Raw and cleaned dataset storage.
- Dataset detail previews and profile/schema display.
- Dashboard real backend statistics.
- Report page generation and persisted report history.
- Brief, standard, and detailed report presentation templates with consistent web, print, HTML, and Markdown rendering.
- HTML, Markdown, browser-print PDF, and per-chart SVG/2x PNG export.
- SQL safety harness and fallback rule SQL.
- DeepSeek-routed planner, SQL, reflection, and Python code generation.
- Kimi-routed report/review/multimodal stages with fallback.
- LangGraph Loop Engineering is the default analysis path with bounded decide/execute/observe/verify/repair/fallback behavior, visible trace events, and SQLite/PostgreSQL checkpoints; the fixed SQL/Python graph remains an explicit compatibility mode.
- Report generation uses a bounded decide/execute/verify/repair-or-fallback/commit sub-loop with evidence-ID validation, at most two revisions, at most one request for additional analysis evidence, and idempotent commit by job ID.
- Workflow modularization phase 1: reusable prompt trust/experience/multi-dataset context lives in `app/analysis/workflow_prompt_context.py`.
- LangGraph nodes execute through a unified Node Harness for transient retry classification, validation, timing, and trace events.
- Process-level internal MCP Runtime reuse for API and Worker model/tool calls.
- Chart support for bar, line, pie, histogram, box plot, and correlation heatmap.
- Text analysis support in Python agent/rules fallback.
- Cleaning diff, cleaning versioning, and rollback by activating a saved cleaning run.
- Manual cleaning rule editor with preview and apply-as-new-version.
- Manual cleaning applies only after preview; validation issues block commit and high-impact row/column/missingness/type/filter changes require explicit user acknowledgment before a reversible version is created.
- Manual field type correction, field roles, and field descriptions.
- Profile and planner behavior using user-overridden field metadata.
- Planner metadata with confidence, route reason, candidate fields, and non-blocking clarifying questions.
- Planner-frozen `AnalysisContract` plus deterministic `StatisticalVerifier` for numeric evidence coverage, comparison support, observational causal-language qualification, and multi-table grain/row-expansion validation. Failed checks reuse the bounded evidence-repair Loop; unresolved findings are excluded from reports while verdicts remain persisted and visible.
- Workflow timeline/debugger based on analysis trace and node summaries.
- Event-driven realtime workflow runner UI: agent plan chips, seven-step workflow state, running log auto-scroll, and expandable agent details driven by ordered SSE job events, with polling fallback.
- Cross-page floating task-progress capsule driven by the same live job updates; desktop and mobile layouts keep it clear of primary navigation and return directly to the active analysis session.
- Async analysis jobs with local development execution or Redis/Celery production workers, database leases, heartbeats, idempotent report creation, cooperative cancel, retry, and recovery.
- Unified analysis-session workspace in React: one `analysis_job` per run, searchable ChatGPT-style history sidebar, selected-session state shared across navigation, and complete Workflow/result restoration by `job_id`.
- Dashboard analysis records use `analysis_jobs` as the single source of truth; clicking any record opens that exact Workflow session, including running/failed jobs and completed outputs.
- Report detail view, search, rename, and version history.
- Report version comparison UI.
- Browser-print PDF export for report detail pages.
- Excel multi-sheet preview and single-sheet import selection per uploaded Excel file.
- Python subprocess execution with timeout, output limits, isolated working directory, allowlisted imports, and dangerous call/file-write blocking.
- Optional controlled Docker Runner creating one-shot generated-code containers with no network, read-only root, non-root execution, dropped capabilities, and CPU/memory/PID limits.
- OpenTelemetry node spans and live/ready health endpoints.
- Responsive mobile bottom navigation plus Playwright desktop/mobile login and Workflow smoke tests.
- Layered backend tests with strict `unit`, `workflow`, `integration`, `sandbox`, and explicit `benchmark` categories; plain `pytest` runs the subprocess-free unit suite.
- Project-level dual-track benchmarks with deterministic PR release gates, fixed-seed privacy-safe corpus generation, optional checksum-pinned external data, real-provider canaries, production performance/resilience workloads, baseline comparison, and read-only historical execution aggregation.
- Workflow and integration tests inject a deterministic in-memory Python executor while production continues to use the real 8-second subprocess sandbox.
- Single Push/PR CI workflow runs unit, real LangGraph workflow, FastAPI/SQLite/DuckDB integration, frontend build, and Playwright without external service containers.
- Reverified 2026-07-15 baseline: Ruff passed; 181 backend tests passed across 84 Unit, 28 Workflow, 60 Integration, and 9 Sandbox cases; the frontend production build and all 24 desktop/mobile Playwright cases passed after the feature-module extraction; Alembic `0001 -> 0008`, `0008 -> 0007`, and re-upgrade to `0008` passed on a fresh SQLite database; Docker Compose configuration passed.
- Reverified 2026-07-16 production containers: Caddy, Nginx frontend, FastAPI, PostgreSQL, password-protected Redis, Celery Worker/Beat and Python Runner started successfully; Alembic migration and data initialization jobs completed; HTTPS same-origin readiness, rolling API replacement through dynamic Docker DNS, semantic embedding readiness, and an actual isolated Python Runner execution passed.
- Python Agent code self-repair: failed generated code and errors are fed back to the LLM for up to two repairs, with three attempts surfaced to users.
- Python Agent split generation: statistics/insights code and chart code are generated separately, with concise chart prompts and truncation-aware repair prompts.
- Python Agent chart payload hardening: prompt-level chart data limits plus worker-side compaction for histograms, box plots, raw records, nested statistics, and long strings before stdout size validation.
- Prompt reliability hardening: bounded and redacted samples, untrusted-data instructions across cleaning/relationship/analysis agents, global prompt text budget, and compact multimodal summaries.
- LLM cleaning self-repair and quality gates: up to three attempts with accumulated error feedback, followed by conservative local fallback when execution or shape validation fails.
- Multi-dataset prompt provenance: Planner/SQL/Python receive compact field-source, join-expansion, key-uniqueness, skipped-join, and aggregation-grain context.
- Phase-specific Python repair contracts prevent statistics repairs from returning charts and chart repairs from returning statistics/insights; prompts list the same import allowlist enforced by the sandbox.
- Agent-specific experience context avoids injecting irrelevant thresholds or conflicting plan schemas into execution prompts.
- Python baseline field-shape hardening: model plans cannot reuse one column as metric/category/time in a way that creates a two-dimensional Pandas selection; duplicate-column access is reduced to a safe one-dimensional Series instead of failing the Workflow.
- Multi-file analysis v1: uploaded dataset multi-select, join key recommendations, user-confirmed left/inner joins, joined dataframe analysis through SQL/Python, persisted job/report join metadata.
- Dataset group analysis v1: automatically validated dataset group relationships can be submitted as relationship plans while preserving single-dataset API compatibility.
- Dataset-group primary-table alignment: analysis requests use the persisted relationship plan's left dataset as the locked primary table, preventing stale import selection from submitting an unrelated dataset with the saved Join plan.
- Versioned dataset/dataset-group semantic models with automatic drafts, immutable publication, copy/rebinding support, schema fingerprints, entity/dimension/relationship definitions, and a safe metric expression DSL.
- Semantic Planner preview and feedback APIs with component confidence scores, monotonic PAVA calibration, high/medium/low decision bands, low-confidence confirmation gates, and immutable decision references on analysis jobs.
- Deterministic semantic metric compilation and direct DuckDB multi-table execution; `sqlglot` AST validation restricts SQL to published entities, fields, relationships, and safe SELECT/CTE operations.
- Native semantic relationship-graph planning separates fact grain from metric-source entities, selects only required paths, compiles explicit deduplication, and blocks unsupported one-to-many/many-to-many expansion before execution.
- Execution-scoped lineage persists source fields, semantic metrics, findings, charts, relationship graph, grain plan, and the report artifact in both analysis responses and report metadata.
- React semantic-model workbench with visual entity/grain, dimension, metric binding, relationship/cardinality, and lineage editing; advanced JSON DSL editing, validation/publication, version selection, and the analysis-time semantic-plan confidence card remain available.
- Chinese-safe semantic DSL v2 with stable entity/field IDs, quoted source-column resolution, backward-compatible v1 reads, and direct execution for Chinese, mixed-language, spaced, slash, parenthesis, and quoted column names.
- Optional local `BAAI/bge-small-zh-v1.5` semantic embedding provider with process/database caches, deterministic fallback, Planner candidate evidence, relationship-score supplementation, and copy/rebinding candidate support.
- Production API/Worker image packages a pinned BGE revision and runs with local-files-only inference; readiness reports embedding availability and production can require it.
- Session identities use an independent UUID owner ID and a separately normalized unique login name. Existing owner IDs remain stable during migration, while punctuation-distinct login names no longer collapse into one data space and blank usernames are rejected.
- P1 production hardening reverified 2026-07-30: 111 Unit, 33 Workflow, and 70 Integration tests passed; Ruff, frontend production build, Alembic `0001 -> 0009`, PostgreSQL `0008 -> 0009`, strict Docker readiness, real container Runner smoke, and the public same-origin endpoint passed.
- P2/P3 reliability hardening reverified 2026-07-30: 114 Unit, 34 Workflow, 71 Integration, and 26 desktop/mobile Playwright tests passed; the frontend production build, zero-vulnerability npm audit, Docker rebuild/readiness, and a real Kimi Provider SSE smoke passed; ordinary and Assistant imports use bounded-memory parsing, Assistant DDL is process-scoped/Alembic-owned in production, Kimi final answers use native Provider SSE, and Playwright uses dedicated strict ports without silently reusing an unrelated local application.
- AnalysisContract/StatisticalVerifier v1 reverified 2026-07-30: Ruff passed; 119 Unit, 36 Workflow, 71 Integration, the deterministic release Benchmark, the frontend production build, and all 26 desktop/mobile Playwright cases passed. The Workflow tests cover unsupported numeric report summaries and a failed Join-grain verdict returning to the autonomous Loop and succeeding only after source-table native-grain evidence is collected.
- Data reliability and grain-aware relationship graph v1 reverified 2026-07-30: Ruff passed; 126 Unit, 36 Workflow, and 72 Integration tests passed together with the deterministic release Benchmark, frontend production build, and all 26 desktop/mobile Playwright cases. Alembic `0001 -> 0010`, `0010 -> 0009`, and re-upgrade to `0010` passed on a fresh SQLite database; coverage includes drift invalidation, relationship freshness, unsafe grain blocking, shortest-path planning, and field-to-report lineage persistence.
- Persistent Kimi data assistant with independent `kimi-k2.6` routing, user-scoped conversation/message history, protected image attachments, automatic report/result retrieval, and validated evidence citations.
- Trustworthy Memory v2 with sourced structured summaries, immutable conflict/version chains, user-level enablement, relevance/MMR retrieval, per-run recall audit, background maintenance leases, and stale-aware read-only analysis experience for Planner. Checkpoints remain task-local and do not masquerade as long-term memory.
- Memory v3 effectiveness loop with source-verified typed formation, relevance/utility separation, per-memory helpful/irrelevant/wrong feedback, selected and suppressed recall audit, reversible dormant versions, and a quality workbench. Automatic dormancy remains in shadow mode until five valid Memory benchmark batches establish the deployment baseline.
- The deterministic Memory benchmark blocks releases on isolation, current-instruction precedence, superseded-use rate, Precision@8, Recall@8, conflict correctness, and 500-memory local retrieval latency.
- Trustworthy Memory v2 reverified 2026-08-12: Ruff passed; 211 Unit, 82 Workflow, 89 Integration, and 9 Sandbox tests passed; deterministic release and Memory benchmarks passed; the frontend production build and all 60 desktop/mobile Playwright cases passed. Alembic `0001 -> 0012`, `0012 -> 0011`, and re-upgrade to `0012` passed on fresh SQLite and PostgreSQL 16 databases, including Repository conflict-chain and lease smoke tests.
- Memory v3 reverified 2026-08-12: Ruff passed; 213 Unit, 82 Workflow, and 89 Integration tests passed; deterministic release and Memory v3 benchmarks passed with `Precision@8=97.5%`, `Recall@8=97.5%`, harmful-memory adoption `0%`, and 500-memory retrieval P95 `79.3ms`; the frontend production build and all 60 desktop/mobile Playwright cases passed. Alembic `0001 -> 0013`, `0013 -> 0012`, and re-upgrade to `0013` passed on fresh SQLite; the running PostgreSQL stack upgraded from `0012` to `0013`, and Docker Compose readiness passed.
- Kimi ask-mode report fast path, database-level Top-N report retrieval, bounded summary-plus-cursor context, segmented first-answer latency telemetry, combined token accounting, and Benchmark P50/P95 aggregation.
- Kimi report revision with frozen analytical evidence, deterministic evidence fingerprints, no analysis rerun, and one audited primary report deliverable per message.
- Assistant LangGraph and local/Celery execution path with ordered SSE tool/analysis/message events, immediate terminal cancellation, checkpoint-safe pause/resume, low-confidence semantic-plan confirmation, and analysis-job reuse. Cancel and final-answer commits are atomic so a late model response cannot overwrite the user's stop action.
- Assistant execute mode exposes the user-facing cleaning, relationship, analysis, report, semantic-model, recycle, and restore tools through server-injected scope and capability checks. Kimi can refine individual Agent stages with bounded prompt overrides, but cannot replace system prompts, disable validation, invoke arbitrary SQL, expose raw paths, or bypass Python sandbox rules.

Not complete yet:

- A recorded successful execution of the real-stack production smoke workflow on a Docker-capable host.
- Enterprise authentication, RBAC, SSO, password reset, and audit log management.
- Per-cell manual cleaning approval workflow.
- Rich semantic-layer lifecycle tooling beyond the current dataset-group scoped v1, including visual formula authoring and organization-wide governance.
- Kubernetes/gVisor-grade sandbox isolation and multi-host Runner scheduling beyond the Docker single-host controller.
- Fully interactive editable workflow graph/debugger.
- Organization-wide lineage impact analysis, interactive graph traversal, and policy-driven automatic remediation beyond the current execution-scoped v1.
- Full parity with the referenced `data-analysis-report-agent` multi-round hypothesis workflow.

---

# 15. Success Criteria

The current product is successful when users can:

- Log in locally and see only their own datasets/reports.
- Upload one or multiple CSV, Excel, JSON, or TXT files in a batch by selecting or dragging files.
- See a successfully imported multi-file batch as one dataset group record with table summaries and child table details.
- Understand that multi-file batch cleaning runs sequentially and can take longer than single-file cleaning.
- Complete a multi-file import and have reliable dataset relationships identified, validated, and saved automatically.
- Understand why a dataset group with no validated relationship cannot run, move directly to automatic re-identification, and see predictable progress feedback while relationships are generated.
- Refresh the frontend and still see uploaded datasets, cleaned datasets, recent analyses, and reports.
- View raw and cleaned data previews.
- View schema, field types, and profiling summary.
- Edit field type, role, and description metadata and see the planner/profile respect it.
- Inspect cleaning version diffs and activate an earlier cleaning version.
- Preview and apply manual cleaning rules as a new cleaning version, with validation blocking and explicit acknowledgment for high-impact changes.
- Preview Excel sheets for each Excel file and import the selected sheet.
- Select multiple uploaded datasets, accept/edit join recommendations, and run joined analysis.
- Select a dataset group, use its automatically validated relationship tree, and run chained joined analysis.
- Ask questions in natural language.
- Start in autonomous Loop mode by default, switch explicitly to the legacy compatibility path when needed, and see the selected execution path before submission.
- Observe autonomous analysis decisions, allowlisted tool execution, evidence verification, classified repair, deterministic fallback, remaining budgets, and terminal reason without exposing hidden model reasoning.
- Run SQL, Python, or hybrid analysis.
- Track async analysis job progress, cancel or retry jobs, and inspect job history.
- Start each analysis as a new session record and reopen any dashboard/history record to inspect its complete Workflow, logs, errors, and persisted outputs.
- Inspect planner metadata and workflow debugger node details.
- Inspect the frozen analysis contract and per-check statistical verdict, including numeric evidence coverage, comparison sample size/effect or confidence interval, causal-language policy, and Join-grain result.
- Inspect multi-dataset join summaries, connected table counts, joined row/column counts, row-expansion ratios, key uniqueness, skipped relationships, and validation issues.
- Inspect data/package drift status, stale relationship match rates, affected reports/semantic models, and authorization-gated remediation suggestions.
- Generate SQL safely against internal DuckDB.
- Run Python analysis on numeric, categorical, and text datasets.
- See charts in the analysis result and report page.
- Generate a structured web report.
- Search and rename saved reports.
- Compare saved report versions.
- Select brief, standard, or detailed report presentation and export HTML, Markdown, browser-print PDF, and individual SVG/2x PNG charts.
- Inspect validation issues and analysis trace.
- Receive ordered live Workflow events with reconnect/polling fallback.
- Continue seeing the active task stage and progress after navigating to another product section, and return to the exact running session from the floating progress capsule.
- Resume a stale production task from its persisted LangGraph checkpoint.
- Use Cookie sessions without trusting a client-supplied user ID in production.
- Create, visually edit entity/metric/dimension/relationship bindings and lineage, validate, publish, and select immutable semantic model versions; advanced JSON remains available and API clients can copy/rebind a model to another dataset or data package.
- Preview a semantic plan with metric/dimension/time/join evidence and calibrated component confidence before execution.
- Explicitly confirm low-confidence plans while medium/high-confidence plans retain the documented execution policy.
- Execute published metric definitions safely across declared DuckDB entities, including Chinese and special-character source fields, without storing arbitrary SQL.
- Reopen a historical job/report and retain its original planner decision, semantic model version, metric formula, field provenance, and join path.
- Trace source fields and semantic metrics through verified findings/charts to the persisted report, including the selected relationship path and grain-safety decision.
- Run with semantic embedding disabled/fallback in development and fail production readiness when required local embeddings are unavailable.
- Ask Kimi about existing DataMind reports and completed analyses, upload an image for visual context, and receive answers backed by clickable report/job/dataset evidence.
- Tell Kimi to remember an explicit durable preference across conversations, confirm inferred memory candidates, and edit, pin, recycle, or restore user- and asset-scoped memory without changing tool permissions or deleting it with the source conversation.
- Inspect why a memory was used or suppressed, rate each recalled memory as helpful, irrelevant, or wrong, and wake a dormant version without losing its feedback and version history.
- Let Kimi start a DataMind analysis when evidence is insufficient, follow the Workflow inside the conversation, confirm low-confidence semantic plans, and continue the answer from the completed result.
- Grant or revoke Kimi access to a selected dataset, dataset group, or report; every tool call must satisfy both conversation scope and capability Grant checks.
- In Execute mode, let Kimi run authorized cleaning, relationship, analysis, report, semantic-model, recycle, restore, and reversible action workflows without bypassing quality, SQL, semantic, or sandbox validation.
- Ask Kimi to simplify or restyle an existing report and receive one audited, clickable, fully rendered report revision backed by the source report's frozen evidence.
- Upload multiple data files in a Kimi conversation, preview files/Sheets, commit a dataset or dataset group, start automatic cleaning and relationship inference, and automatically scope the conversation to the imported asset.

---

# 16. Roadmap

## Next

- Retain five valid Memory v3 benchmark batches, review harmful-memory adoption and feedback calibration, then decide whether production automatic dormancy can be enabled.
- Continue behavior-preserving module extraction: move frontend domain types and page modules out of `main.tsx`; move Workflow Prompt builders into `workflow_prompts.py` and node factories into a `workflow_nodes` package; move relationship-graph validation into its own storage helper and split dataset/group/job/report repositories behind the existing facade.
- Run and retain the real-stack production smoke artifact on a Docker-capable host before production acceptance.
- Add per-cell manual cleaning approval workflow.
- Expand the semantic-model visual editor from source-field/aggregation bindings to a complete expression-tree formula authoring experience.
- Add fully interactive editable workflow graph/debugger.
- Add richer report chart interactivity, cross-filtering, and drill-down.

## Later

- Organization-wide semantic governance, interactive cross-run impact lineage, and policy-managed remediation.
- More mature experience library editing.
- Extend deterministic statistical verification with experiment-design metadata, power analysis, multiple-comparison correction, missingness-mechanism checks, and time-series diagnostics.
- Cross-report filtering, drill-down, and reusable dashboard composition beyond the Next-phase report chart improvements.
- Standard external MCP stdio/Streamable HTTP client support; the current MCP Runtime remains internal.
- Kubernetes-native workers, gVisor-level sandboxing, and multi-host Runner scheduling.
- Enterprise auth/RBAC only if the project moves beyond local/demo usage.
