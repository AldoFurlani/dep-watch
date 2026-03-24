"""Compute 24 features directly from repo_events using exact timestamps.

One row per repo, no monthly approximations. Features computed at each
repo's last event date as the reference point, matching Xu et al.'s
point-in-time methodology.

Usage:
    python -m depwatch.model_training.bq_direct_features
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def direct_features_query(project: str, dataset: str) -> str:
    """Compute all 24 features per repo directly from events.

    Reference date = last_event_at per repo. All windows (90d, 365d, etc.)
    computed with exact TIMESTAMP_DIFF, not monthly approximations.
    """
    return f"""
WITH repo_ref AS (
  -- Reference date per repo (last event)
  SELECT
    repo_name,
    MAX(created_at) AS ref_date,
    MIN(created_at) AS first_event
  FROM `{project}.{dataset}.repo_events`
  GROUP BY repo_name
),

-- === Repo metadata ===
stars AS (
  SELECT repo_name, COUNT(*) AS stars
  FROM `{project}.{dataset}.repo_events`
  WHERE event_type = 'WatchEvent'
  GROUP BY repo_name
),
forks AS (
  SELECT repo_name, COUNT(*) AS forks
  FROM `{project}.{dataset}.repo_events`
  WHERE event_type = 'ForkEvent'
  GROUP BY repo_name
),
open_issues AS (
  SELECT repo_name,
    GREATEST(
      COUNTIF(issue_action = 'opened') - COUNTIF(issue_action = 'closed'),
      0
    ) AS open_issues
  FROM `{project}.{dataset}.repo_events`
  WHERE event_type = 'IssuesEvent'
  GROUP BY repo_name
),

-- === Commit activity ===
commit_stats AS (
  SELECT
    e.repo_name,
    SUM(CASE WHEN TIMESTAMP_DIFF(r.ref_date, e.created_at, DAY) <= 90
      THEN COALESCE(e.push_distinct_size, 1) ELSE 0 END) AS commit_count_90d,
    SUM(CASE WHEN TIMESTAMP_DIFF(r.ref_date, e.created_at, DAY) <= 30
      THEN COALESCE(e.push_distinct_size, 1) ELSE 0 END) AS commit_count_30d,
    SUM(CASE WHEN TIMESTAMP_DIFF(r.ref_date, e.created_at, DAY) <= 365
      THEN COALESCE(e.push_distinct_size, 1) ELSE 0 END) AS commit_count_365d,
    TIMESTAMP_DIFF(r.ref_date, MAX(e.created_at), DAY) AS days_since_last_commit
  FROM `{project}.{dataset}.repo_events` e
  JOIN repo_ref r ON e.repo_name = r.repo_name
  WHERE e.event_type = 'PushEvent'
  GROUP BY e.repo_name, r.ref_date
),

-- === Issue activity ===
issue_stats AS (
  SELECT
    e.repo_name,
    COUNTIF(e.issue_action = 'opened'
      AND TIMESTAMP_DIFF(r.ref_date, e.created_at, DAY) <= 90) AS issues_opened_90d,
    COUNTIF(e.issue_action = 'closed'
      AND TIMESTAMP_DIFF(r.ref_date, e.created_at, DAY) <= 90) AS issues_closed_90d
  FROM `{project}.{dataset}.repo_events` e
  JOIN repo_ref r ON e.repo_name = r.repo_name
  WHERE e.event_type = 'IssuesEvent'
  GROUP BY e.repo_name, r.ref_date
),

-- Issue close durations (for median, within 90d window)
issue_durations AS (
  SELECT
    opened.repo_name,
    TIMESTAMP_DIFF(closed.created_at, COALESCE(opened.issue_created_at, opened.created_at), SECOND)
      / 86400.0 AS close_days,
    -- Split for response_time_trend: recent (0-45d) vs older (45-90d)
    TIMESTAMP_DIFF(r.ref_date, closed.created_at, DAY) AS days_before_ref
  FROM `{project}.{dataset}.repo_events` opened
  JOIN `{project}.{dataset}.repo_events` closed
    ON opened.repo_name = closed.repo_name
    AND opened.issue_number = closed.issue_number
    AND opened.issue_action = 'opened'
    AND closed.issue_action = 'closed'
  JOIN repo_ref r ON opened.repo_name = r.repo_name
  WHERE opened.event_type = 'IssuesEvent'
    AND closed.event_type = 'IssuesEvent'
    AND TIMESTAMP_DIFF(r.ref_date, closed.created_at, DAY) <= 90
    AND closed.created_at > COALESCE(opened.issue_created_at, opened.created_at)
),
issue_medians AS (
  SELECT
    repo_name,
    PERCENTILE_CONT(close_days, 0.5) OVER (PARTITION BY repo_name) AS median_close_days,
    PERCENTILE_CONT(
      IF(days_before_ref <= 45, close_days, NULL), 0.5
    ) OVER (PARTITION BY repo_name) AS median_close_recent,
    PERCENTILE_CONT(
      IF(days_before_ref > 45, close_days, NULL), 0.5
    ) OVER (PARTITION BY repo_name) AS median_close_older
  FROM issue_durations
),
issue_medians_dedup AS (
  SELECT DISTINCT repo_name, median_close_days, median_close_recent, median_close_older
  FROM issue_medians
),

-- === PR activity ===
pr_stats AS (
  SELECT
    e.repo_name,
    COUNTIF(e.pr_action = 'opened'
      AND TIMESTAMP_DIFF(r.ref_date, e.created_at, DAY) <= 90) AS prs_opened_90d,
    COUNTIF(e.pr_action = 'closed' AND e.pr_merged IS TRUE
      AND TIMESTAMP_DIFF(r.ref_date, e.created_at, DAY) <= 90) AS prs_merged_90d
  FROM `{project}.{dataset}.repo_events` e
  JOIN repo_ref r ON e.repo_name = r.repo_name
  WHERE e.event_type = 'PullRequestEvent'
  GROUP BY e.repo_name, r.ref_date
),
pr_durations AS (
  SELECT
    opened.repo_name,
    TIMESTAMP_DIFF(closed.pr_merged_at, COALESCE(opened.pr_created_at, opened.created_at), SECOND)
      / 86400.0 AS merge_days
  FROM `{project}.{dataset}.repo_events` opened
  JOIN `{project}.{dataset}.repo_events` closed
    ON opened.repo_name = closed.repo_name
    AND opened.pr_number = closed.pr_number
    AND opened.pr_action = 'opened'
    AND closed.pr_action = 'closed'
    AND closed.pr_merged IS TRUE
  JOIN repo_ref r ON opened.repo_name = r.repo_name
  WHERE opened.event_type = 'PullRequestEvent'
    AND closed.event_type = 'PullRequestEvent'
    AND TIMESTAMP_DIFF(r.ref_date, closed.pr_merged_at, DAY) <= 90
    AND closed.pr_merged_at > COALESCE(opened.pr_created_at, opened.created_at)
),
pr_medians AS (
  SELECT repo_name,
    PERCENTILE_CONT(merge_days, 0.5) OVER (PARTITION BY repo_name) AS median_merge_days
  FROM pr_durations
),
pr_medians_dedup AS (
  SELECT DISTINCT repo_name, median_merge_days
  FROM pr_medians
),

-- === Release metrics ===
release_stats AS (
  SELECT
    e.repo_name,
    COUNTIF(TIMESTAMP_DIFF(r.ref_date, e.created_at, DAY) <= 365) AS release_count_365d,
    MIN(CASE WHEN e.create_ref_type = 'tag'
      THEN TIMESTAMP_DIFF(r.ref_date, e.created_at, DAY) END) AS days_since_last_release
  FROM `{project}.{dataset}.repo_events` e
  JOIN repo_ref r ON e.repo_name = r.repo_name
  WHERE e.event_type = 'CreateEvent' AND e.create_ref_type = 'tag'
  GROUP BY e.repo_name, r.ref_date
),

-- === Contributor health ===
-- All-time per-author commit counts
author_commits_all AS (
  SELECT
    repo_name,
    actor_login,
    SUM(COALESCE(push_distinct_size, 1)) AS commits
  FROM `{project}.{dataset}.repo_events`
  WHERE event_type = 'PushEvent' AND actor_login IS NOT NULL
  GROUP BY repo_name, actor_login
),
contributor_agg AS (
  SELECT
    repo_name,
    COUNT(*) AS contributor_count_total,
    MAX(commits) AS top1_commits,
    SUM(commits) AS total_commits,
    -- Gini coefficient components
    SUM(commits * commits) AS sum_sq,
    ARRAY_AGG(commits ORDER BY commits) AS sorted_commits
  FROM author_commits_all
  GROUP BY repo_name
),
-- 90-day unique authors
authors_90d AS (
  SELECT
    e.repo_name,
    COUNT(DISTINCT e.actor_login) AS contributor_count_90d
  FROM `{project}.{dataset}.repo_events` e
  JOIN repo_ref r ON e.repo_name = r.repo_name
  WHERE e.event_type = 'PushEvent'
    AND e.actor_login IS NOT NULL
    AND TIMESTAMP_DIFF(r.ref_date, e.created_at, DAY) <= 90
  GROUP BY e.repo_name
),
-- New contributors: first commit ever is within 90d of ref
first_commits AS (
  SELECT repo_name, actor_login, MIN(created_at) AS first_commit_at
  FROM `{project}.{dataset}.repo_events`
  WHERE event_type = 'PushEvent' AND actor_login IS NOT NULL
  GROUP BY repo_name, actor_login
),
new_contribs AS (
  SELECT
    fc.repo_name,
    COUNT(*) AS new_contributor_count_90d
  FROM first_commits fc
  JOIN repo_ref r ON fc.repo_name = r.repo_name
  WHERE TIMESTAMP_DIFF(r.ref_date, fc.first_commit_at, DAY) <= 90
  GROUP BY fc.repo_name
),

-- === Activity deviation ===
activity_dev AS (
  SELECT
    e.repo_name,
    SUM(CASE WHEN TIMESTAMP_DIFF(r.ref_date, e.created_at, DAY) <= 30
      THEN 1 ELSE 0 END) AS activity_30d,
    SUM(CASE WHEN TIMESTAMP_DIFF(r.ref_date, e.created_at, DAY) <= 90
      THEN 1 ELSE 0 END) AS activity_90d
  FROM `{project}.{dataset}.repo_events` e
  JOIN repo_ref r ON e.repo_name = r.repo_name
  WHERE e.event_type IN ('PushEvent', 'IssuesEvent', 'PullRequestEvent')
  GROUP BY e.repo_name, r.ref_date
),

-- === Labels ===
labels AS (
  SELECT repo_name, is_abandoned, abandonment_signal, estimated_abandonment_date
  FROM `{project}.{dataset}.repo_labels`
)

-- === Assemble final dataset ===
SELECT
  r.repo_name,
  -- Repo metadata
  COALESCE(s.stars, 0) AS stars,
  COALESCE(f.forks, 0) AS forks,
  COALESCE(oi.open_issues, 0) AS open_issues,
  TIMESTAMP_DIFF(r.ref_date, r.first_event, DAY) / 30.44 AS age_months,
  -- Commit activity
  COALESCE(cs.commit_count_90d, 0) AS commit_count_90d,
  COALESCE(cs.days_since_last_commit,
    TIMESTAMP_DIFF(r.ref_date, r.first_event, DAY)) AS days_since_last_commit,
  CASE WHEN SAFE_DIVIDE(cs.commit_count_365d, 12.0) > 0
    THEN LEAST(cs.commit_count_30d / (cs.commit_count_365d / 12.0), 100.0)
    ELSE 1.0 END AS commit_frequency_trend,
  -- Issue activity
  COALESCE(ist.issues_opened_90d, 0) AS issues_opened_90d,
  CASE WHEN COALESCE(ist.issues_opened_90d, 0) > 0
    THEN LEAST(SAFE_DIVIDE(ist.issues_closed_90d, ist.issues_opened_90d), 1.0)
    ELSE 0.0 END AS issue_close_ratio_90d,
  LEAST(COALESCE(imd.median_close_days, 0.0), 365.0) AS median_issue_close_time_days,
  -- PR activity
  COALESCE(pst.prs_opened_90d, 0) AS prs_opened_90d,
  CASE WHEN COALESCE(pst.prs_opened_90d, 0) > 0
    THEN LEAST(SAFE_DIVIDE(pst.prs_merged_90d, pst.prs_opened_90d), 1.0)
    ELSE 0.0 END AS pr_merge_ratio_90d,
  LEAST(COALESCE(pmd.median_merge_days, 0.0), 365.0) AS median_pr_merge_time_days,
  -- Maintainer responsiveness
  CASE
    WHEN imd.median_close_older IS NOT NULL AND imd.median_close_older > 0.01
    THEN LEAST(GREATEST(SAFE_DIVIDE(imd.median_close_recent, imd.median_close_older), 0.01), 100.0)
    ELSE 1.0
  END AS response_time_trend,
  CASE WHEN SAFE_DIVIDE(ad.activity_90d, 3.0) > 0
    THEN LEAST(ad.activity_30d / (ad.activity_90d / 3.0), 100.0)
    ELSE 1.0 END AS activity_deviation,
  -- Contributor health
  COALESCE(ca.contributor_count_total, 0) AS contributor_count_total,
  COALESCE(a90.contributor_count_90d, 0) AS contributor_count_90d,
  CASE WHEN COALESCE(ca.total_commits, 0) > 0
    THEN ca.top1_commits / ca.total_commits
    ELSE 1.0 END AS top1_contributor_ratio,
  -- bus_factor: computed from sorted_commits array
  COALESCE(ca.contributor_count_total, 0) AS _contributor_count_for_bus,
  COALESCE(ca.total_commits, 0) AS _total_commits_for_bus,
  COALESCE(nc.new_contributor_count_90d, 0) AS new_contributor_count_90d,
  -- contributor_gini placeholder (computed locally from sorted_commits)
  0.0 AS contributor_gini,
  0.0 AS bus_factor,
  -- Release metrics
  COALESCE(rs.release_count_365d, 0) AS release_count_365d,
  COALESCE(rs.days_since_last_release,
    TIMESTAMP_DIFF(r.ref_date, r.first_event, DAY)) AS days_since_last_release,
  CASE WHEN COALESCE(rs.days_since_last_release, 9999) <= 90
    THEN 1.0 ELSE 0.0 END AS has_recent_release,
  -- Labels
  COALESCE(lb.is_abandoned, FALSE) AS is_abandoned,
  COALESCE(lb.abandonment_signal, 'not_abandoned') AS abandonment_signal,
  lb.estimated_abandonment_date
FROM repo_ref r
LEFT JOIN stars s ON r.repo_name = s.repo_name
LEFT JOIN forks f ON r.repo_name = f.repo_name
LEFT JOIN open_issues oi ON r.repo_name = oi.repo_name
LEFT JOIN commit_stats cs ON r.repo_name = cs.repo_name
LEFT JOIN issue_stats ist ON r.repo_name = ist.repo_name
LEFT JOIN issue_medians_dedup imd ON r.repo_name = imd.repo_name
LEFT JOIN pr_stats pst ON r.repo_name = pst.repo_name
LEFT JOIN pr_medians_dedup pmd ON r.repo_name = pmd.repo_name
LEFT JOIN release_stats rs ON r.repo_name = rs.repo_name
LEFT JOIN contributor_agg ca ON r.repo_name = ca.repo_name
LEFT JOIN authors_90d a90 ON r.repo_name = a90.repo_name
LEFT JOIN new_contribs nc ON r.repo_name = nc.repo_name
LEFT JOIN activity_dev ad ON r.repo_name = ad.repo_name
LEFT JOIN labels lb ON r.repo_name = lb.repo_name
"""


def contributor_details_query(project: str, dataset: str) -> str:
    """Export per-repo per-author commit counts for local Gini/bus_factor."""
    return f"""
SELECT repo_name, actor_login, SUM(COALESCE(push_distinct_size, 1)) AS commits
FROM `{project}.{dataset}.repo_events`
WHERE event_type = 'PushEvent' AND actor_login IS NOT NULL
GROUP BY repo_name, actor_login
ORDER BY repo_name, commits DESC
"""


def main() -> None:
    """Export direct features from BigQuery and compute Gini/bus_factor locally."""
    parser = argparse.ArgumentParser(
        description="Compute features directly from repo_events (no monthly approximations)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/training",
        help="Output directory (default: data/training)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from google.cloud import bigquery

    from depwatch.common.config import get_settings

    settings = get_settings()
    project = settings.gcp_project_id
    dataset = "depwatch"
    client = bigquery.Client(project=project)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Export main features
    logger.info("Computing direct features from repo_events...")
    start = time.time()
    sql = direct_features_query(project, dataset)
    df = client.query(sql).to_dataframe()
    elapsed = time.time() - start
    logger.info("Features computed: %d repos (%.1fs)", len(df), elapsed)

    # Step 2: Export contributor details for Gini/bus_factor
    logger.info("Exporting contributor details...")
    start = time.time()
    contrib_sql = contributor_details_query(project, dataset)
    contrib_df = client.query(contrib_sql).to_dataframe()
    elapsed = time.time() - start
    logger.info("Contributors exported: %d rows (%.1fs)", len(contrib_df), elapsed)

    # Step 3: Compute Gini and bus_factor locally
    import numpy as np

    logger.info("Computing Gini coefficients and bus factors...")
    repo_gini: dict[str, float] = {}
    repo_bus: dict[str, float] = {}

    for repo_name, group in contrib_df.groupby("repo_name"):
        counts = np.sort(group["commits"].values.astype(int))
        n = len(counts)
        total = counts.sum()

        # Gini
        if n <= 1 or total == 0:
            repo_gini[str(repo_name)] = 0.0
        else:
            index = np.arange(1, n + 1)
            repo_gini[str(repo_name)] = float(
                (2.0 * (index * counts).sum()) / (n * total) - (n + 1.0) / n
            )

        # Bus factor
        if total == 0:
            repo_bus[str(repo_name)] = 0.0
        else:
            cumsum = np.cumsum(counts[::-1])
            bf = int(np.searchsorted(cumsum, total / 2, side="left") + 1)
            repo_bus[str(repo_name)] = float(bf)

    df["contributor_gini"] = df["repo_name"].map(repo_gini).fillna(0.0)
    df["bus_factor"] = df["repo_name"].map(repo_bus).fillna(0.0)

    # Drop temp columns
    df = df.drop(columns=["_contributor_count_for_bus", "_total_commits_for_bus"], errors="ignore")

    # Step 4: Save
    output_path = out / "direct_features.parquet"
    df.to_parquet(output_path, index=False)
    logger.info("Saved to %s", output_path)

    # Summary
    logger.info("")
    logger.info("Dataset summary:")
    logger.info("  Repos: %d", len(df))
    abandoned = df["is_abandoned"].sum()
    logger.info("  Abandoned: %d (%.1f%%)", abandoned, abandoned / len(df) * 100)
    logger.info("  Active: %d", len(df) - abandoned)

    feature_cols = [
        "stars",
        "forks",
        "open_issues",
        "age_months",
        "commit_count_90d",
        "days_since_last_commit",
        "commit_frequency_trend",
        "issues_opened_90d",
        "issue_close_ratio_90d",
        "median_issue_close_time_days",
        "prs_opened_90d",
        "pr_merge_ratio_90d",
        "median_pr_merge_time_days",
        "response_time_trend",
        "activity_deviation",
        "contributor_count_total",
        "contributor_count_90d",
        "top1_contributor_ratio",
        "bus_factor",
        "new_contributor_count_90d",
        "contributor_gini",
        "release_count_365d",
        "days_since_last_release",
        "has_recent_release",
    ]
    logger.info("  Features: %d", len(feature_cols))
    for f in feature_cols:
        if f in df.columns:
            s = df[f]
            logger.info(
                "    %-30s mean=%.2f std=%.2f zero%%=%.1f%%",
                f,
                s.mean(),
                s.std(),
                (s == 0).mean() * 100,
            )


if __name__ == "__main__":
    main()
