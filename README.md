# DepWatch

A dependency abandonment prediction API. Upload a manifest file (`package.json`, `requirements.txt`, or `go.mod`) and get a ranked risk report with abandonment probabilities and explanations for each dependency.

## How It Works

DepWatch uses an XGBoost model trained on 110K GitHub repositories to predict which dependencies are at risk of being abandoned. For each package in your manifest, it:

1. Resolves the GitHub repository via PyPI/npm registries
2. Fetches repository activity via the GitHub GraphQL API
3. Computes 24 behavioral features (commit patterns, contributor health, maintainer responsiveness)
4. Scores with the model and generates SHAP-based explanations of the top risk factors

Packages sharing the same repo are deduplicated, and all fetches run in parallel. A SQLite cache ensures repeat queries for the same package are instant.

## Sample Output

```
POST /scan  (upload package.json)
```

```json
{
  "packages_scanned": 12,
  "packages_scored": 12,
  "packages_errored": 0,
  "results": [
    {
      "package": "some-lib",
      "ecosystem": "npm",
      "github_repo": "owner/some-lib",
      "abandonment_probability": 0.69,
      "risk_level": "high",
      "top_risk_factors": [
        {
          "feature": "age_months",
          "value": 22.0,
          "impact": 0.96,
          "description": "Repository age (months)"
        },
        {
          "feature": "days_since_last_commit",
          "value": 91.8,
          "impact": -0.41,
          "description": "Days since last commit"
        }
      ]
    },
    {
      "package": "react",
      "ecosystem": "npm",
      "github_repo": "facebook/react",
      "abandonment_probability": 0.005,
      "risk_level": "low",
      "top_risk_factors": []
    }
  ],
  "errors": []
}
```

Results are sorted highest risk first. Each package includes the top 5 SHAP-derived risk factors explaining why the model scored it that way.

## Architecture

The API parses the manifest, resolves each package to its GitHub repo via PyPI/npm, deduplicates repos, then fetches activity data in parallel using the GitHub GraphQL API (with REST for contributor lists). A 24-feature vector is computed per repo and scored by the XGBoost model. SHAP TreeExplainer generates the risk factor explanations. Computed snapshots are cached in SQLite keyed by `(owner, repo, month)` so repeat queries skip the API entirely.

### Training Pipeline

The model was trained on 110K repos (50/50 abandoned/active) extracted from GH Archive via BigQuery. Features are computed directly in SQL with exact timestamp windows. The XGBoost classifier achieves a C-index of 0.826, compared to 0.846 reported by [Xu et al. (2025)](https://arxiv.org/abs/2507.21678). The gap is primarily due to labeling methodology differences.

Temporal models (Transformer, GRU, MLP on sliding windows) were tested and rejected — the abandonment signal lives in point-in-time features, not trajectory patterns.

## Feature Set (24 dimensions)

| Group | Features |
|---|---|
| Repo metadata | stars, forks, open_issues, age_months |
| Commit activity | commit_count_90d, days_since_last_commit, commit_frequency_trend |
| Issue activity | issues_opened_90d, issue_close_ratio_90d, median_issue_close_time_days |
| PR activity | prs_opened_90d, pr_merge_ratio_90d, median_pr_merge_time_days |
| Maintainer responsiveness | response_time_trend, activity_deviation |
| Contributor health | contributor_count_total, contributor_count_90d, top1_contributor_ratio, bus_factor, new_contributor_count_90d, contributor_gini |
| Release metrics | release_count_365d, days_since_last_release, has_recent_release |

Feature design informed by Xu et al. (2025), which found that maintainer-behavior features (response time trends, activity deviation, contributor Gini) drive the biggest accuracy gains over raw activity counts.

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
make install

# Set up environment
cp .env.example .env
# Add your GitHub token to .env (needed for API access)

# Train the model (requires data/training/direct_features.parquet)
python -m depwatch.model_training.train_baselines

# Start the API
make run-api
# Server runs at http://localhost:8000

# Scan a manifest
curl -X POST http://localhost:8000/scan -F "file=@package.json"
```

## Development

```bash
make lint        # ruff check + format check
make fmt         # auto-fix lint issues
make typecheck   # mypy strict
make test        # 142 unit tests
make run-api     # dev server with hot reload
```

## Tech Stack

- **API**: Python 3.12+, FastAPI, uvicorn
- **Model**: XGBoost, SHAP (TreeExplainer)
- **Data**: GitHub GraphQL + REST API, httpx (async)
- **Training**: BigQuery (GH Archive), pandas, scikit-learn
- **Quality**: ruff, mypy (strict), pytest, respx, GitHub Actions CI
