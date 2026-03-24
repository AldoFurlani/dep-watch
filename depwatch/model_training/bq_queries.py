"""BigQuery SQL queries for training data extraction from GH Archive.

Pipeline overview (2 expensive GH Archive scans, everything else is cheap):

Step 1: candidate_repos — Select ~110K repos balanced between abandoned/active.
        Scans GH Archive columns: repo.name, created_at, type, actor.login (~1-2 TB).

Step 2: repo_events — Extract and parse events for candidate repos.
        Scans GH Archive including payload column (~3-5 TB).

Step 3: repo_labels — Label repos using inactivity + README keywords (optional).
        Reads our tables + github_repos.contents (no GH Archive scan).

Export queries read from repo_events only (cheap, <$1 per query):
  - monthly_stats: event counts per repo per month
  - issue_durations: issue open→close timestamps
  - pr_durations: PR open→merge timestamps
  - author_commits: per-author commit counts per repo per month
  - sentiment_texts: issue titles + commit messages for VADER
"""


def create_dataset_ddl(project: str, dataset: str) -> str:
    """Create the BigQuery dataset if it doesn't exist."""
    return f"CREATE SCHEMA IF NOT EXISTS `{project}.{dataset}`"


# ---------------------------------------------------------------------------
# Step 1: Candidate repo selection
# ---------------------------------------------------------------------------


def candidate_repos_query(project: str, dataset: str) -> str:
    """Select ~110K repos with balanced abandoned/active split.

    Criteria:
    - 10+ stars (WatchEvent proxy), 20+ pushes, 50+ total events
    - 2+ unique push authors (filters bots/personal backups)
    - 1+ year of activity span
    - Balanced: ~55K abandoned (no events in 2+ years) + ~55K active (events in last 6 months)
    """
    return f"""
CREATE OR REPLACE TABLE `{project}.{dataset}.candidate_repos` AS
WITH repo_stats AS (
  SELECT
    repo.name AS repo_name,
    MIN(created_at) AS first_event_at,
    MAX(created_at) AS last_event_at,
    COUNT(*) AS total_events,
    COUNTIF(type = 'WatchEvent') AS star_count,
    COUNTIF(type = 'PushEvent') AS push_count,
    COUNTIF(type = 'IssuesEvent') AS issue_count,
    COUNTIF(type = 'PullRequestEvent') AS pr_count,
    COUNT(DISTINCT CASE WHEN type = 'PushEvent' THEN actor.login END) AS unique_authors
  FROM `githubarchive.day.20*`
  WHERE _TABLE_SUFFIX BETWEEN '150101' AND '241231'
  GROUP BY repo.name
),
qualified AS (
  SELECT
    *,
    TIMESTAMP_DIFF(last_event_at, first_event_at, DAY) AS active_span_days,
    TIMESTAMP_DIFF(TIMESTAMP('2025-01-01'), last_event_at, DAY) AS days_since_last_event,
    CASE
      WHEN TIMESTAMP_DIFF(TIMESTAMP('2025-01-01'), last_event_at, DAY) > 730
        THEN 'abandoned'
      WHEN TIMESTAMP_DIFF(TIMESTAMP('2025-01-01'), last_event_at, DAY) <= 180
        THEN 'active'
      ELSE 'ambiguous'
    END AS activity_status
  FROM repo_stats
  WHERE
    star_count >= 10
    AND push_count >= 20
    AND total_events >= 50
    AND unique_authors >= 2
    AND TIMESTAMP_DIFF(last_event_at, first_event_at, DAY) >= 365
),
-- Deterministic sampling: ~55K abandoned + ~55K active
sampled AS (
  (
    SELECT * FROM qualified
    WHERE activity_status = 'abandoned'
    ORDER BY FARM_FINGERPRINT(repo_name)
    LIMIT 55000
  )
  UNION ALL
  (
    SELECT * FROM qualified
    WHERE activity_status = 'active'
    ORDER BY FARM_FINGERPRINT(repo_name)
    LIMIT 55000
  )
)
SELECT * FROM sampled
"""


# ---------------------------------------------------------------------------
# Step 2: Event extraction with parsed payload fields
# ---------------------------------------------------------------------------


def repo_events_query(project: str, dataset: str) -> str:
    """Extract all events for candidate repos with parsed payload fields.

    This is the most expensive query (~3-5 TB) because it reads the payload
    column. Run once; all subsequent queries read from this table.
    """
    return f"""
CREATE OR REPLACE TABLE `{project}.{dataset}.repo_events` AS
SELECT
  repo.name AS repo_name,
  type AS event_type,
  actor.login AS actor_login,
  created_at,
  DATE_TRUNC(DATE(created_at), MONTH) AS event_month,
  -- IssuesEvent fields
  CASE WHEN type = 'IssuesEvent'
    THEN JSON_EXTRACT_SCALAR(payload, '$.action') END AS issue_action,
  CASE WHEN type = 'IssuesEvent'
    THEN JSON_EXTRACT_SCALAR(payload, '$.issue.number') END AS issue_number,
  CASE WHEN type = 'IssuesEvent'
    THEN JSON_EXTRACT_SCALAR(payload, '$.issue.title') END AS issue_title,
  CASE WHEN type = 'IssuesEvent' THEN SAFE.PARSE_TIMESTAMP(
    '%Y-%m-%dT%H:%M:%SZ',
    JSON_EXTRACT_SCALAR(payload, '$.issue.created_at')
  ) END AS issue_created_at,
  CASE WHEN type = 'IssuesEvent' THEN SAFE.PARSE_TIMESTAMP(
    '%Y-%m-%dT%H:%M:%SZ',
    JSON_EXTRACT_SCALAR(payload, '$.issue.closed_at')
  ) END AS issue_closed_at,
  -- PullRequestEvent fields
  CASE WHEN type = 'PullRequestEvent'
    THEN JSON_EXTRACT_SCALAR(payload, '$.action') END AS pr_action,
  CASE WHEN type = 'PullRequestEvent'
    THEN JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') END AS pr_number,
  CASE WHEN type = 'PullRequestEvent' THEN SAFE.PARSE_TIMESTAMP(
    '%Y-%m-%dT%H:%M:%SZ',
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at')
  ) END AS pr_created_at,
  CASE WHEN type = 'PullRequestEvent' THEN SAFE.PARSE_TIMESTAMP(
    '%Y-%m-%dT%H:%M:%SZ',
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at')
  ) END AS pr_merged_at,
  CASE WHEN type = 'PullRequestEvent' THEN SAFE_CAST(
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') AS BOOL
  ) END AS pr_merged,
  -- PushEvent fields
  CASE WHEN type = 'PushEvent' THEN SAFE_CAST(
    JSON_EXTRACT_SCALAR(payload, '$.distinct_size') AS INT64
  ) END AS push_distinct_size,
  -- Commit messages: concatenate first 5 per push (for sentiment)
  CASE WHEN type = 'PushEvent' THEN ARRAY_TO_STRING(
    ARRAY(
      SELECT JSON_EXTRACT_SCALAR(c, '$.message')
      FROM UNNEST(JSON_EXTRACT_ARRAY(payload, '$.commits')) AS c
      WITH OFFSET AS pos
      WHERE pos < 5
    ), ' ||| '
  ) END AS commit_messages,
  -- CreateEvent fields
  CASE WHEN type = 'CreateEvent'
    THEN JSON_EXTRACT_SCALAR(payload, '$.ref_type') END AS create_ref_type
FROM `githubarchive.day.20*`
WHERE _TABLE_SUFFIX BETWEEN '150101' AND '241231'
  AND repo.name IN (SELECT repo_name FROM `{project}.{dataset}.candidate_repos`)
  AND type IN (
    'PushEvent', 'IssuesEvent', 'PullRequestEvent',
    'CreateEvent', 'WatchEvent', 'ForkEvent'
  )
"""


# ---------------------------------------------------------------------------
# Step 3: Labeling
# ---------------------------------------------------------------------------


def repo_labels_query(project: str, dataset: str) -> str:
    """Label repos using inactivity heuristics + optional README keyword check.

    Labels:
    - 'inactivity': no events in 2+ years (from candidate_repos.activity_status)
    - 'readme_keyword': README contains abandonment language (github_repos snapshot)
    - 'not_abandoned': active repos

    The README check uses bigquery-public-data.github_repos which is a snapshot
    (circa 2016-2017), so coverage is limited. Inactivity is the primary signal.
    """
    return f"""
CREATE OR REPLACE TABLE `{project}.{dataset}.repo_labels` AS
WITH readme_check AS (
  SELECT
    f.repo_name,
    REGEXP_CONTAINS(
      c.content,
      r'(?i)(no\\s+longer\\s+maintained'
      r'|this\\s+project\\s+is\\s+(now\\s+)?archived'
      r'|this\\s+project\\s+is\\s+(now\\s+)?deprecated'
      r'|no\\s+longer\\s+(actively\\s+)?developed'
      r'|unmaintained'
      r'|this\\s+project\\s+has\\s+been\\s+abandoned'
      r'|not\\s+(being\\s+)?actively\\s+maintained'
      r'|consider\\s+this\\s+project\\s+dead)'
    ) AS has_abandonment_keywords
  FROM `bigquery-public-data.github_repos.files` f
  JOIN `bigquery-public-data.github_repos.contents` c ON f.id = c.id
  WHERE LOWER(f.path) LIKE 'readme%'
    AND c.binary = false
    AND f.repo_name IN (SELECT repo_name FROM `{project}.{dataset}.candidate_repos`)
)
SELECT
  cr.repo_name,
  cr.first_event_at,
  cr.last_event_at,
  cr.days_since_last_event,
  cr.activity_status,
  COALESCE(rc.has_abandonment_keywords, FALSE) AS has_abandonment_readme,
  -- Final label: abandoned if inactive OR readme signals abandonment
  CASE
    WHEN cr.activity_status = 'abandoned' THEN TRUE
    WHEN COALESCE(rc.has_abandonment_keywords, FALSE) THEN TRUE
    ELSE FALSE
  END AS is_abandoned,
  -- Signal that triggered the label
  CASE
    WHEN cr.activity_status = 'abandoned' THEN 'inactivity'
    WHEN COALESCE(rc.has_abandonment_keywords, FALSE) THEN 'readme_keyword'
    ELSE 'not_abandoned'
  END AS abandonment_signal,
  -- Estimated abandonment date (last activity for abandoned repos)
  CASE
    WHEN cr.activity_status = 'abandoned' THEN cr.last_event_at
    WHEN COALESCE(rc.has_abandonment_keywords, FALSE) THEN cr.last_event_at
    ELSE NULL
  END AS estimated_abandonment_date
FROM `{project}.{dataset}.candidate_repos` cr
LEFT JOIN readme_check rc ON cr.repo_name = rc.repo_name
"""


def repo_labels_no_readme_query(project: str, dataset: str) -> str:
    """Fallback labeling using inactivity only (skips README check).

    Use this if the github_repos.contents join is too expensive or slow.
    """
    return f"""
CREATE OR REPLACE TABLE `{project}.{dataset}.repo_labels` AS
SELECT
  repo_name,
  first_event_at,
  last_event_at,
  days_since_last_event,
  activity_status,
  FALSE AS has_abandonment_readme,
  CASE
    WHEN activity_status = 'abandoned' THEN TRUE
    ELSE FALSE
  END AS is_abandoned,
  CASE
    WHEN activity_status = 'abandoned' THEN 'inactivity'
    ELSE 'not_abandoned'
  END AS abandonment_signal,
  CASE
    WHEN activity_status = 'abandoned' THEN last_event_at
    ELSE NULL
  END AS estimated_abandonment_date
FROM `{project}.{dataset}.candidate_repos`
"""


# ---------------------------------------------------------------------------
# Export queries (all read from repo_events — cheap)
# ---------------------------------------------------------------------------


def monthly_stats_query(project: str, dataset: str) -> str:
    """Monthly event counts per repo. Used for most feature computation."""
    return f"""
SELECT
  repo_name,
  event_month AS snapshot_month,
  SUM(CASE WHEN event_type = 'PushEvent'
    THEN COALESCE(push_distinct_size, 1) ELSE 0 END) AS commit_count,
  COUNTIF(event_type = 'PushEvent') AS push_events,
  COUNTIF(event_type = 'IssuesEvent' AND issue_action = 'opened') AS issues_opened,
  COUNTIF(event_type = 'IssuesEvent' AND issue_action = 'closed') AS issues_closed,
  COUNTIF(event_type = 'PullRequestEvent' AND pr_action = 'opened') AS prs_opened,
  COUNTIF(
    event_type = 'PullRequestEvent'
    AND pr_action = 'closed'
    AND pr_merged IS TRUE
  ) AS prs_merged,
  COUNTIF(
    event_type = 'CreateEvent' AND create_ref_type = 'tag'
  ) AS releases,
  COUNTIF(event_type = 'WatchEvent') AS stars,
  COUNTIF(event_type = 'ForkEvent') AS forks,
  COUNT(DISTINCT CASE WHEN event_type = 'PushEvent'
    THEN actor_login END) AS unique_authors,
  MAX(CASE WHEN event_type = 'PushEvent'
    THEN created_at END) AS last_push_at,
  MAX(CASE WHEN event_type = 'CreateEvent' AND create_ref_type = 'tag'
    THEN created_at END) AS last_release_at
FROM `{project}.{dataset}.repo_events`
GROUP BY repo_name, event_month
ORDER BY repo_name, event_month
"""


def issue_durations_query(project: str, dataset: str) -> str:
    """Issue open→close durations for median close time computation."""
    return f"""
WITH issue_events AS (
  SELECT
    repo_name,
    issue_number,
    MIN(CASE WHEN issue_action = 'opened' THEN COALESCE(issue_created_at, created_at) END)
      AS opened_at,
    MIN(CASE WHEN issue_action = 'closed' THEN created_at END) AS closed_at
  FROM `{project}.{dataset}.repo_events`
  WHERE event_type = 'IssuesEvent'
    AND issue_number IS NOT NULL
  GROUP BY repo_name, issue_number
)
SELECT
  repo_name,
  issue_number,
  opened_at,
  closed_at,
  TIMESTAMP_DIFF(closed_at, opened_at, SECOND) / 86400.0 AS close_time_days
FROM issue_events
WHERE closed_at IS NOT NULL
  AND opened_at IS NOT NULL
  AND closed_at > opened_at
ORDER BY repo_name, closed_at
"""


def pr_durations_query(project: str, dataset: str) -> str:
    """PR open→merge durations for median merge time computation."""
    return f"""
WITH pr_events AS (
  SELECT
    repo_name,
    pr_number,
    MIN(CASE WHEN pr_action = 'opened' THEN COALESCE(pr_created_at, created_at) END)
      AS opened_at,
    MIN(pr_merged_at) AS merged_at
  FROM `{project}.{dataset}.repo_events`
  WHERE event_type = 'PullRequestEvent'
    AND pr_number IS NOT NULL
  GROUP BY repo_name, pr_number
)
SELECT
  repo_name,
  pr_number,
  opened_at,
  merged_at,
  TIMESTAMP_DIFF(merged_at, opened_at, SECOND) / 86400.0 AS merge_time_days
FROM pr_events
WHERE merged_at IS NOT NULL
  AND opened_at IS NOT NULL
  AND merged_at > opened_at
ORDER BY repo_name, merged_at
"""


def author_commits_query(project: str, dataset: str) -> str:
    """Per-author commit counts per repo per month.

    Used for: contributor_count_total, contributor_count_90d,
    top1_contributor_ratio, bus_factor, new_contributor_count_90d,
    contributor_gini.
    """
    return f"""
SELECT
  repo_name,
  event_month AS month,
  actor_login,
  SUM(COALESCE(push_distinct_size, 1)) AS commit_count
FROM `{project}.{dataset}.repo_events`
WHERE event_type = 'PushEvent'
  AND actor_login IS NOT NULL
GROUP BY repo_name, event_month, actor_login
ORDER BY repo_name, event_month, commit_count DESC
"""


def sentiment_texts_query(project: str, dataset: str) -> str:
    """Issue titles and commit messages for VADER sentiment computation."""
    return f"""
-- Issue titles (one per opened issue)
SELECT
  repo_name,
  event_month AS month,
  'issue' AS text_type,
  issue_title AS text_content
FROM `{project}.{dataset}.repo_events`
WHERE event_type = 'IssuesEvent'
  AND issue_action = 'opened'
  AND issue_title IS NOT NULL

UNION ALL

-- Commit messages (concatenated per push, split locally by ' ||| ')
SELECT
  repo_name,
  event_month AS month,
  'commit' AS text_type,
  commit_messages AS text_content
FROM `{project}.{dataset}.repo_events`
WHERE event_type = 'PushEvent'
  AND commit_messages IS NOT NULL
ORDER BY repo_name, month
"""
