# DepWatch — Dependency Abandonment Prediction Service

## What This Is

A stateless API that predicts whether open-source dependencies are at risk of abandonment. Users submit a manifest file (package.json, requirements.txt, go.mod) and receive a ranked risk report with abandonment probabilities and explanations.

## Prior Work

Feature design and model strategy are informed by Xu et al. (2025), ["Predicting Abandonment of Open Source Software Projects with An Integrated Feature Framework"](https://arxiv.org/abs/2507.21678). Key findings from the paper:

- **Best model**: XGBoost AFT (Accelerated Failure Time) — tree-based survival analysis, not deep learning. C-index 0.846. They did not test any neural approaches.
- **18 features in 4 categories**: Surface activity (stars, commits, issues, PRs, tags, comments), user-centric (HITS-based contributor importance), maintainer-centric (response time, response decay trend), project evolution (contributor diversity, balance index, activity deviation).
- **Biggest signal gains** came from maintainer-behavior features (response time trends, activity deviation) and contributor diversity — not raw activity counts. Surface features alone only reached C-index 0.748.
- **Dataset**: 115,466 repos (57,733 abandoned) from GH Archive, dual-criteria labeling (archived flag + semantic README analysis).
- **6-month prediction horizon**.

**Our approach vs. Xu et al.**:
- We tested a sliding-window Transformer on monthly snapshots (T=6) to capture temporal trajectories — Xu et al. only used point-in-time features. The Transformer failed to learn (see Phase 3B results below), confirming that point-in-time features capture the signal and temporal deep learning doesn't add value for this problem.
- We adopt their key insight: maintainer-centric features (response time trend, activity deviation, contributor Gini) matter more than raw counts.
- **Production model**: XGBoost classifier, C-index 0.826 (vs Xu et al.'s 0.846). Gap is primarily due to labeling methodology (inactivity heuristic vs archived flag).

## Architecture

### Training (one-time, BigQuery + local)
1. BigQuery SQL queries against GH Archive: select 110K candidate repos, extract 183M parsed events into intermediate BigQuery tables (2 GH Archive scans, ~$170)
2. Compute 24 features per repo directly in BigQuery SQL with exact timestamp windows (90d, 365d) — one row per repo, no monthly approximations
3. Export 110K-row feature matrix + per-author commit counts for local Gini/bus_factor computation
4. Train XGBoost classifier (production model, C-index 0.826) + baselines (logistic regression, XGBoost AFT)
5. Best model artifact saved to disk, baked into container image

### Inference (API on a VPS)
1. User POSTs a manifest file (requirements.txt, package.json, go.mod) to `POST /scan`
2. FastAPI parses it, extracts package names, resolves GitHub repos via PyPI/npm registries
3. Deduplicates packages sharing the same repo (e.g., react/react-dom → facebook/react)
4. Fetches data in parallel (5 concurrent max): GraphQL for repo data + commit/issue/PR totalCounts, REST for accurate contributor lists
5. Healthy-repo bypass: repos with 50+ commits in 90d, 20+ contributors, and last commit within 30d skip the model (always low risk)
6. For remaining repos: compute 24 features with count overrides from totalCount fields, run XGBoost model, generate SHAP explanations
7. Computed snapshots cached on disk (SQLite) keyed by `(owner, repo, month)` — repeat queries are instant

### Deployment
- Single Docker container on a VPS (Hetzner/Railway/Fly.io)
- GitHub Actions for CI/CD
- Model artifact baked into the container image

### Why this architecture
- **GitHub GraphQL API** for inference — a single GraphQL query fetches all data for a repo (metadata, commits, issues, PRs, releases) plus accurate `totalCount` fields for commits (30d/90d/365d) and issues (90d). This avoids the 100-node pagination limit that would otherwise truncate counts for active repos. REST `/contributors` endpoint supplements GraphQL for accurate contributor lists. Parallel fetching with a semaphore (5 concurrent) keeps scans fast without hitting rate limits.
- **SQLite snapshot cache** — computed feature snapshots are cached on disk keyed by `(owner, repo, month)`. Popular packages (express, flask, etc.) are served instantly after first computation. Cache survives server restarts. No external service — just a file on disk.
- **BigQuery for extraction and feature computation** — BigQuery selects candidate repos, extracts parsed events into intermediate tables (2 GH Archive scans), then computes 24 features per repo directly in SQL with exact timestamp windows (90d, 365d). Only Gini coefficient and bus factor are computed locally. This avoids downloading TBs of raw events and eliminates monthly approximation errors that degraded model performance in earlier iterations.
- **No monthly worker** — there are no stored snapshots to refresh. The model was trained on historical data and scores on-the-fly. The SQLite cache is populated lazily per request.
- **No retraining pipeline** — the model is trained once on 50K+ repos spanning a decade of GitHub history. Abandonment patterns are stable; manual retraining if needed.
- **VPS over cloud compute** — a single FastAPI server with a model in memory is the right tool. Cloud Run/Lambda/ECS add cost and complexity for no benefit at this scale.

## Tech Stack

- Python 3.14, FastAPI, XGBoost, PyTorch (temporal model experiments), httpx
- GitHub GraphQL API (inference only)
- BigQuery + google-cloud-bigquery + pyarrow (GH Archive extraction for training)
- SQLite (on-disk snapshot cache in production)
- ruff (lint/format), mypy (strict), pytest
- Docker, GitHub Actions

## Project Structure

```
depwatch/
├── common/              # Shared: config, feature definitions, types
├── core/                # GitHub REST + GraphQL clients, feature extraction, labeling, sentiment
├── model_training/      # XGBoost (production) + temporal experiments + evaluation
│   └── baselines/       # Logistic regression, XGBoost AFT comparisons
├── ingestion_function/  # Manifest parsing + package discovery
│   └── registry_clients/  # PyPI, npm API clients
├── inference_service/   # FastAPI: manifest upload → ranked risk report
│   ├── models/          # Pydantic request/response schemas
│   ├── routers/         # POST /scan, GET /health
│   └── services/        # Scanner (orchestrator), scorer (model + SHAP), cache, manifest parser
└── tests/
    ├── unit/
    └── fixtures/
```

## Development

```bash
source .venv/bin/activate    # Required — system Python is externally managed
make lint                    # ruff check + format check
make fmt                     # auto-fix lint issues
make typecheck               # mypy strict
make test                    # unit tests only
make run-api                 # uvicorn dev server on :8000

# Training (requires data/training/direct_features.parquet)
python -m depwatch.model_training.train_baselines  # trains + saves to artifacts/latest/model.json
```

### Environment variables
```
GITHUB_TOKEN=github_pat_...  # Required for inference (GitHub GraphQL + REST API)
GCP_PROJECT_ID=...           # Training only (BigQuery)
LOG_LEVEL=INFO
```

## Testing Conventions

### Two tiers

- **Unit tests** (`tests/unit/`) — Fast, no external services. Test individual functions with fake/fixture data. Run via `make test`. Every new module should have corresponding unit tests.
- **Model tests** — Verify the training pipeline runs mechanically on small synthetic data, evaluation metrics compute without errors, and exported checkpoints load correctly.

### External API mocking pattern

External API clients (GitHub, PyPI, npm) must be testable without hitting real services. The pattern:

1. Record real API responses as JSON files in `tests/fixtures/github_api_responses/` or `tests/fixtures/registry_responses/`.
2. In unit tests, use `respx` (already in dev deps) to intercept httpx requests and return fixture data. The code under test doesn't know it's not talking to the real API.

### Design for testability

- API clients should accept an `httpx.AsyncClient` (dependency injection) so tests can pass a mock transport.
- The feature extractor should accept pre-fetched raw data, not call clients directly. This makes unit testing trivial — pass fixture data in, assert on feature values.
- FastAPI tests use `httpx.AsyncClient` with `app` transport (no real server needed).

### What to test per phase

- **Data clients**: Rate limiting, Pydantic model parsing of real API response fixtures, edge cases (404s, rate limit 403s).
- **Feature extraction**: Correct computation from known inputs, edge cases (zero issues, zero contributors, brand new repo), division-by-zero guards.
- **Labeler**: Keyword regex catches "no longer maintained" but not "we maintain high standards", threshold boundaries.
- **Model**: Forward pass produces 3 probabilities in [0,1], sliding window shapes are correct, train/val/test split has no repo_id leakage.
- **API**: Correct JSON response structure, malformed manifest handling, scanner mock injection, health endpoint.
- **Manifest parser**: All three formats, edge cases (extras, comments, `-r` includes, `github:` shorthand, dedup across dependency sections).
- **GraphQL client**: Response parsing, commit deduplication, null author handling, totalCount extraction, auth headers.
- **Cache**: Put/get roundtrip, overwrites, case-insensitive keys, cross-instance persistence.
- **Scorer**: Risk level thresholds, missing model error, mock model scoring with SHAP values.

## Architecture Decisions

- **GitHub GraphQL over REST for inference**: REST requires ~7 calls per endpoint per month × 6 months = 42 calls per package. GraphQL fetches all data in 1-3 calls per package. A 50-dependency manifest goes from 25+ minutes to ~2 minutes. GraphQL has a separate rate budget (5,000 points/hour) that doesn't compete with REST. Training data comes entirely from BigQuery (GH Archive) — no GitHub API calls needed during training.
- **SQLite snapshot cache**: Computed feature snapshots are cached on disk, keyed by `(owner, repo, month)`. Eliminates redundant API calls for popular packages across users. No external service — just a file. Cache survives restarts, giving warm starts. Not a database in the traditional sense — no schema migrations, no queries beyond key-value lookups.
- **BigQuery for training data extraction + feature computation**: GH Archive lives on BigQuery. Selecting 110K repos and extracting their 183M events is 2 SQL queries (~$170 on GCP free trial credits). Features computed directly in BigQuery SQL with exact timestamp windows — one row per repo, no monthly approximations. Only Gini/bus_factor computed locally. Only cloud service used, and only during training.
- **Train once**: Model trained on 50K+ repos spanning 10+ years of GitHub history. Abandonment patterns are stable over time — monthly retraining adds complexity without meaningful accuracy gains. Retrain manually if needed.
- **XGBoost over Transformer**: Temporal deep learning (Transformer, GRU, MLP on sliding windows) was tested and failed — the signal is in point-in-time features, not trajectory patterns. Abandonment prediction from monthly feature snapshots doesn't benefit from sequence modeling because most of a repo's trajectory looks identical whether it eventually abandons or not. Only the final months differ, and point-in-time features already capture that.
- **SHAP explainability**: Precompute at scoring time using XGBoost's native TreeExplainer for speed.
- **Labeler detects, model predicts**: The labeler flags abandonment *after the fact* during training data prep (inactivity heuristic + README keywords). The model learns leading indicators from features of repos that were eventually flagged, then predicts abandonment before it happens.
- **24 training features informed by Xu et al.**: Trimmed from 42 to 24, dropping redundant time-window variants, weak static signals, and features unavailable in BigQuery (registry features: maintainer_count, weekly_downloads, is_deprecated; README sentiment — only a static snapshot, not historical; issue/commit sentiment — VADER features too expensive to export at scale, likely weak predictors). Added maintainer-behavior features (response time trend, activity deviation, contributor Gini) which drove the biggest accuracy gains in Xu et al.
- **110K training repos**: Xu et al. used 115K repos. With BigQuery access to GH Archive, we matched their scale (110K repos, 50/50 abandoned/active split).
- **Healthy-repo bypass over model recalibration**: The model was trained on repos where median contributor_count_total=3 and median commit_count_90d=0. Large, actively maintained repos (React, TypeScript, Next.js) produce out-of-distribution features that the model misinterprets as risk (e.g., high issue counts → overwhelmed maintainers, when it's actually normal activity). Rather than retraining with normalized features, a simple heuristic bypass (50+ commits/90d AND 20+ contributors AND last commit ≤30d → always low risk) eliminates false positives while preserving model accuracy for the small-to-medium repos where abandonment actually occurs. Only 1% of true abandonments in training data meet these criteria.
- **Repo deduplication + parallel fetching**: Multiple packages often map to the same GitHub repo (react/react-dom, @types/react/@types/react-dom, tailwindcss/@tailwindcss/postcss). The scanner resolves all packages to repos first, deduplicates, then fetches unique repos in parallel. A 31-package manifest with 20 unique repos makes 20 API calls instead of 31, and runs them concurrently (semaphore-limited to 5) instead of sequentially.
- **GraphQL totalCount over pagination**: GitHub GraphQL returns max 100 nodes per connection, but `totalCount` gives the true count without pagination. This is critical for commit counts — without it, `commit_frequency_trend` was always 12.0 for any repo with >100 recent commits, a value that correlates with 67% abandonment rate in training data (repos with a final burst before going quiet). The `filterBy: {since}` parameter on issues uses `DateTime!` type, not `GitTimestamp!` — a GraphQL schema quirk that caused silent failures.

## Implementation Phases

### Phase 0: Scaffolding ✅ COMPLETE
- pyproject.toml, Makefile, .pre-commit-config.yaml, .gitignore
- GitHub Actions CI (lint + typecheck + unit tests)
- All verified: `make lint`, `make typecheck`, `make test` pass

### Phase 1: Data Layer ✅ COMPLETE
- Config module (common/config.py) — Pydantic BaseSettings (GitHub token, GCP project ID, registry URLs, model path, log level)
- Shared Pydantic types (common/types.py) — GitHubRepo, GitHubCommit, GitHubIssue, GitHubPullRequest, GitHubContributor, GitHubRelease, PackageMetadata
- GitHub REST API client (core/github_client.py) — rate-limited, typed responses, pagination, dependency-injectable httpx.AsyncClient
- Registry clients (ingestion_function/registry_clients/) — PyPI + npm → PackageMetadata

### Phase 2A: Feature Extraction ✅ COMPLETE
- Feature definitions (common/features.py) — 24-dim `FeatureVector` Pydantic model with `to_list()`, `feature_names()`, `dim()` serialization helpers
- Feature extractor (core/feature_extractor.py) — accepts pre-fetched raw data (no HTTP calls), computes all 24 features with division-by-zero guards, uses median (not mean) for response times
- Sentiment analysis (core/sentiment.py) — VADER `compound_score()` and `mean_compound()`, lazy-loaded singleton analyzer
- 24 features in 7 groups: repo metadata (4), commit activity (3), issue activity (3), PR activity (3), maintainer responsiveness (2), contributor health (6), release metrics (3)

### Phase 2B: Ground Truth Labeling ✅ COMPLETE
- Labeler (core/labeler.py) — 3-signal priority: archived flag → README keyword regex → inactivity heuristics (configurable `LabelThresholds`, default requires 2+ signals)
- Dataset utilities (model_training/dataset.py) — `load_snapshot_df()` for Parquet/CSV, `split_by_repo()` with no data leakage, `create_sliding_windows()` for temporal model (T=6, D=24, 3 horizon labels)
- Data quality checks — `compute_class_balance()` (abandoned ratio, signal distribution), `compute_feature_stats()` (per-feature mean/std/min/max/nulls)
- 95 unit tests total, all passing. `make lint`, `make typecheck`, `make test` pass. 96% coverage.

### Phase 3A: Training Data Pipeline ✅ COMPLETE
- BigQuery pipeline (model_training/bq_pipeline.py) — orchestration, export, CLI with --skip-readme / --extract-only / --export-only flags
- SQL queries (model_training/bq_queries.py) — candidate_repos (110K balanced), repo_events (183M parsed events), repo_labels (inactivity + README keywords)
- Direct feature computation (model_training/bq_direct_features.py) — computes all 24 features per repo directly in BigQuery SQL with exact timestamp windows, exports 110K-row dataset
- Monthly aggregation pipeline also available (model_training/compute_features.py) — used for temporal model experiments, superseded by direct features for production

### Phase 3B: Model Training ✅ COMPLETE
- **Production model: XGBoost classifier, C-index 0.826** (vs Xu et al.'s 0.846)
- Point-in-time baselines (model_training/train_baselines.py): LogReg AUROC=0.753, XGBoost AUROC=0.825, XGBoost AFT C-index=0.822
- Hyperparameter tuning (model_training/tune_xgboost.py): 50-trial random search, marginal gain (+0.001)
- Temporal models tested and rejected (model_training/train_temporal.py): Transformer (27K params), GRU, flattened MLP all failed to learn — sliding window formulation doesn't work because most windows from abandoned repos look identical to active repo windows (abandonment signal only appears in final months)
- Evaluation module (model_training/evaluate.py): C-index, AUROC, AUC-PR, Brier score
- Key finding: monthly approximations (3-month ≈ 90d) degraded C-index by ~0.10 vs exact timestamp features. Per-repo evaluation (not per-snapshot) was critical for meaningful C-index comparison with Xu et al.
- Remaining C-index gap (0.020) attributed to labeling methodology: our inactivity heuristic vs Xu et al.'s archived flag + semantic README analysis

### Phase 4: Inference API ✅ COMPLETE
- GitHub GraphQL client (core/github_graphql.py) — single query fetches repo metadata + commit/issue/PR nodes + totalCount fields for accurate counts across 30d/90d/365d windows. Uses `GitTimestamp` for commit history and `DateTime` for issue filterBy (different GraphQL types).
- SQLite snapshot cache (inference_service/services/cache.py) — WAL mode, keyed by `(owner, repo, month)`, case-insensitive keys
- Manifest parsers (inference_service/services/manifest_parser.py) — requirements.txt, package.json, go.mod with edge case handling (extras, comments, environment markers, GitHub shorthand repos)
- Feature extractor updated with `CountOverrides` — accepts accurate totalCount values to override len()-based counting from truncated API data. Caps `commit_frequency_trend` ≤ 12, `activity_deviation` ≤ 3, `response_time_trend` ≤ 100 to match BigQuery training bounds.
- Scorer (inference_service/services/scorer.py) — XGBoost model loading, inference, SHAP TreeExplainer for top-5 risk factors. Healthy-repo bypass for repos with 50+ commits/90d, 20+ contributors, last commit within 30d.
- Scanner (inference_service/services/scanner.py) — orchestrates full pipeline: parallel registry lookups → repo deduplication → parallel GraphQL+REST fetching (semaphore-throttled, 5 concurrent) → feature computation → scoring → ranked response
- FastAPI app (inference_service/main.py) — `POST /scan` (file upload), `GET /health`, model loaded at startup via lifespan handler
- npm client handles string repository fields and `github:` shorthand
- 142 unit tests, all passing. `make lint`, `make typecheck`, `make test` pass.

#### Train/serve skew mitigations
The model was trained on complete BigQuery data but inference uses the GitHub API with pagination limits. Mitigations applied:
- **Commit counts**: GraphQL `totalCount` fields give accurate 30d/90d/365d counts (not capped at 100 nodes). Without this, `commit_frequency_trend` was always 12.0 for active repos.
- **Issue counts**: `filterBy: {since}` totalCount for accurate `issues_opened_90d`.
- **Contributors**: REST `/contributors` endpoint instead of approximating from ~200 commit authors. Gives accurate `contributor_count_total`, `bus_factor`, `contributor_gini`.
- **Feature capping**: Trend/ratio features capped to training data bounds to prevent OOD values.
- **Healthy-repo bypass**: Repos clearly not at risk (high activity + many contributors + recent commits) skip the model entirely. Only ~1% of true abandonments in training data meet these criteria, nearly all mislabeled.

### Phase 5: Deployment
- [ ] Dockerfile
- [ ] GitHub Actions CI/CD → build container → deploy to VPS
- [ ] Health check endpoint
- [ ] structlog for structured logging

## Key Model Details

### Production model: XGBoost classifier
- **Input**: 24-dim feature vector (one per repo, computed at last observed event)
- **Output**: P(abandoned) — binary classification
- **Performance**: C-index 0.826, AUROC 0.825, AUC-PR 0.806 (vs Xu et al. C-index 0.846)
- **Training**: 77K repos train, 16.5K val, 16.5K test. 50/50 abandoned/active split. Early stopping on val AUC-PR.
- **Top features**: age_months, stars, new_contributor_count_90d, days_since_last_commit, commit_frequency_trend
- **Evaluation**: Per-repo (one prediction per repo), matching Xu et al.'s methodology

### Models tested (Phase 3B results)
| Model | AUROC | C-index | Notes |
|---|---|---|---|
| **XGBoost classifier** | **0.825** | **0.826** | Production model |
| XGBoost AFT | 0.821 | 0.822 | Xu et al.'s architecture |
| Logistic regression | 0.753 | 0.752 | Linear baseline |
| Transformer (T=6) | 0.455 | 0.330 | Failed — temporal signal not learnable |
| GRU | 0.464 | 0.563 | Failed — same issue as Transformer |
| Flattened MLP | 0.464 | 0.563 | Failed — confirms temporal formulation issue |
| Xu et al. reference | — | 0.846 | Different labeling (archived flag) |

### Why temporal models failed
Most sliding windows from abandoned repos come from years before abandonment when the repo was still active. These "healthy" windows are indistinguishable from active repo windows, producing ~95% negative labels. The abandonment signal only appears in the final few months, which point-in-time features already capture. Oversampling and class weighting couldn't overcome the fundamental label sparsity in the temporal formulation.

## Feature Set (v4, 24 dimensions)

All 24 features are computable from GH Archive via BigQuery — no GitHub API calls needed for training.

| Group | Count | Features |
|---|---|---|
| Repo metadata | 4 | stars, forks, open_issues, age_months |
| Commit activity | 3 | commit_count_90d, days_since_last_commit, commit_frequency_trend |
| Issue activity | 3 | issues_opened_90d, issue_close_ratio_90d, median_issue_close_time_days |
| PR activity | 3 | prs_opened_90d, pr_merge_ratio_90d, median_pr_merge_time_days |
| Maintainer responsiveness | 2 | response_time_trend, activity_deviation |
| Contributor health | 6 | contributor_count_total, contributor_count_90d, top1_contributor_ratio, bus_factor, new_contributor_count_90d, contributor_gini |
| Release metrics | 3 | release_count_365d, days_since_last_release, has_recent_release |

**Removed from model**:
- `issue_sentiment_avg`, `commit_message_sentiment_avg` — VADER sentiment on issue titles and commit messages. Too expensive to export at scale (64M+ rows of text), and likely weak predictors. Xu et al. achieved C-index 0.846 without any sentiment features. Sentiment data remains in BigQuery if needed later.
- `readme_sentiment` — `github_repos.contents` only has a static snapshot, not historical per month. Same value across all 6 windows is useless for the temporal model.
- `maintainer_count`, `weekly_downloads`, `is_deprecated` — registry-only (PyPI/npm APIs). Not in GH Archive, not in the model. Registry clients are still used for manifest parsing (package name → GitHub repo lookup).
