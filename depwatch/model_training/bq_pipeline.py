"""BigQuery pipeline for extracting training data from GH Archive.

Usage:
    python -m depwatch.model_training.bq_pipeline [--skip-readme] [--output-dir data/training]

Pipeline steps:
    1. Create BigQuery dataset (if needed)
    2. Select ~110K candidate repos (expensive GH Archive scan)
    3. Extract parsed events for candidates (expensive GH Archive scan)
    4. Label repos (inactivity + optional README keywords)
    5. Export aggregated data to local Parquet files

After export, run compute_features.py to assemble the final 24-dim training dataset.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.cloud.bigquery import Client, QueryJob

from depwatch.model_training.bq_queries import (
    author_commits_query,
    candidate_repos_query,
    create_dataset_ddl,
    issue_durations_query,
    monthly_stats_query,
    pr_durations_query,
    repo_events_query,
    repo_labels_no_readme_query,
    repo_labels_query,
)

logger = logging.getLogger(__name__)

BQ_DATASET = "depwatch"


class BQPipeline:
    """Orchestrates BigQuery extraction pipeline."""

    def __init__(self, project_id: str, dataset: str = BQ_DATASET) -> None:
        from google.cloud import bigquery

        if not project_id:
            msg = "GCP_PROJECT_ID is not set. Add it to your .env file."
            raise ValueError(msg)
        self.project = project_id
        self.dataset = dataset
        self.client: Client = bigquery.Client(project=project_id)

    def _run_query(self, name: str, sql: str) -> QueryJob:
        """Execute a query and block until completion, logging progress."""
        logger.info("Starting: %s", name)
        start = time.time()
        job = self.client.query(sql)
        job.result()  # blocks until done
        elapsed = time.time() - start
        bytes_billed = job.total_bytes_billed or 0
        gb_billed = bytes_billed / (1024**3)
        cost_estimate = max(gb_billed / 1024, 0) * 6.25  # $6.25/TB after free tier
        logger.info(
            "Completed: %s (%.1fs, %.2f GB scanned, ~$%.2f)",
            name,
            elapsed,
            gb_billed,
            cost_estimate,
        )
        return job

    def ensure_dataset(self) -> None:
        """Create the BigQuery dataset if it doesn't exist."""
        self._run_query(
            "create_dataset",
            create_dataset_ddl(self.project, self.dataset),
        )

    def select_candidates(self) -> None:
        """Step 1: Select ~110K candidate repos from GH Archive."""
        self._run_query(
            "candidate_repos",
            candidate_repos_query(self.project, self.dataset),
        )

    def extract_events(self) -> None:
        """Step 2: Extract parsed events for candidate repos."""
        self._run_query(
            "repo_events",
            repo_events_query(self.project, self.dataset),
        )

    def label_repos(self, *, skip_readme: bool = False) -> None:
        """Step 3: Label repos as abandoned/active."""
        if skip_readme:
            sql = repo_labels_no_readme_query(self.project, self.dataset)
        else:
            sql = repo_labels_query(self.project, self.dataset)
        self._run_query("repo_labels", sql)

    def run_extraction(self, *, skip_readme: bool = False) -> None:
        """Run all extraction steps (Steps 1-3)."""
        self.ensure_dataset()
        self.select_candidates()
        self.extract_events()
        self.label_repos(skip_readme=skip_readme)
        logger.info("Extraction complete. Tables created in %s.%s", self.project, self.dataset)

    def _export_query(self, name: str, sql: str, out: Path) -> None:
        """Export a query result to Parquet."""
        parquet_path = out / f"{name}.parquet"
        logger.info("Exporting %s → %s", name, parquet_path)
        start = time.time()
        df = self.client.query(sql).to_dataframe()
        df.to_parquet(parquet_path, index=False)
        elapsed = time.time() - start
        logger.info("Exported %s: %d rows (%.1fs)", name, len(df), elapsed)

    def export_all(self, output_dir: str | Path) -> None:
        """Export all data needed for local feature computation."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        exports = [
            ("monthly_stats", monthly_stats_query),
            ("issue_durations", issue_durations_query),
            ("pr_durations", pr_durations_query),
            ("author_commits", author_commits_query),
        ]

        for name, query_fn in exports:
            parquet_path = out / f"{name}.parquet"
            if parquet_path.exists():
                logger.info("Skipping %s (already exists)", name)
                continue
            sql = query_fn(self.project, self.dataset)
            self._export_query(name, sql, out)

        # Export labels
        labels_path = out / "repo_labels.parquet"
        if not labels_path.exists():
            logger.info("Exporting repo_labels → %s", labels_path)
            table_ref = f"{self.project}.{self.dataset}.repo_labels"
            df_labels = self.client.list_rows(table_ref).to_dataframe()
            df_labels.to_parquet(labels_path, index=False)
            logger.info("Exported repo_labels: %d rows", len(df_labels))
        else:
            logger.info("Skipping repo_labels (already exists)")

        logger.info("All exports complete → %s", out)


def main() -> None:
    """CLI entry point for the BigQuery extraction pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract training data from GH Archive via BigQuery"
    )
    parser.add_argument(
        "--output-dir",
        default="data/training",
        help="Directory for exported Parquet files (default: data/training)",
    )
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Skip README keyword check (cheaper, uses inactivity-only labeling)",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Run extraction steps only (no export)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export only (assumes extraction already done)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from depwatch.common.config import get_settings

    settings = get_settings()
    pipeline = BQPipeline(settings.gcp_project_id)

    if args.export_only:
        pipeline.export_all(args.output_dir)
    elif args.extract_only:
        pipeline.run_extraction(skip_readme=args.skip_readme)
    else:
        pipeline.run_extraction(skip_readme=args.skip_readme)
        pipeline.export_all(args.output_dir)


if __name__ == "__main__":
    main()
