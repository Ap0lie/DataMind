# DataMind Benchmarks

The benchmark harness is separate from the normal pytest layers. The deterministic
release suite is safe for PR CI; provider, performance, resilience, and frontend
suites are explicit because they may consume credentials, Docker capacity, memory,
or several minutes of runtime.

```bash
python -m app.evaluation.cli run --suite release
python -m app.evaluation.cli run --suite memory
python -m app.evaluation.cli run --suite provider --repeats 3
python -m app.evaluation.cli run --suite performance --backend compose
python -m app.evaluation.cli run --suite resilience --backend compose
python -m app.evaluation.cli run --suite frontend
python -m app.evaluation.cli history --database data/datamind.db
python -m app.evaluation.cli compare --baseline baseline.json --candidate candidate.json
python -m app.evaluation.cli calibrate --runs run1.json run2.json run3.json run4.json run5.json --output baseline.json
```

Provider credentials are read only from the existing DataMind environment settings.
Do not put API keys in manifests or artifacts. External datasets are opt-in through
`BENCHMARK_DATA_ROOT` and must provide `benchmark-manifest.json` with a SHA-256 for
every file. No user database rows are copied into benchmark artifacts.

The deterministic `memory` suite is a blocking trust gate. It validates user
isolation, temporal supersession, current-instruction precedence, Precision@8,
Recall@8, and local retrieval P95 with 500 active memories. It uses generated
business topics and does not call an LLM provider.

Memory v3 additionally injects a misleading memory, records idempotent negative
feedback, verifies reversible dormancy, and requires harmful-memory adoption below
1%. Keep production automatic dormancy in shadow mode until five valid benchmark
batches have been retained for the current corpus and model configuration.
