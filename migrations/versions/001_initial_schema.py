"""Initial schema: repos, packages, snapshots, risk scores, scans.

Revision ID: 001
Revises: None
Create Date: 2026-03-21

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "repos",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("github_owner", sa.Text(), nullable=False),
        sa.Column("github_name", sa.Text(), nullable=False),
        sa.Column("primary_language", sa.Text(), nullable=True),
        sa.Column("created_at_github", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_archived", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("is_abandoned", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("abandonment_date", sa.Date(), nullable=True),
        sa.Column("abandonment_signal", sa.Text(), nullable=True),
        sa.Column("latest_snapshot_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "monitoring_priority",
            sa.Text(),
            server_default=sa.text("'normal'"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("github_owner", "github_name"),
    )

    op.create_table(
        "packages",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("ecosystem", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("repo_id", sa.Uuid(), sa.ForeignKey("repos.id"), nullable=True),
        sa.Column("registry_data", sa.JSON(), nullable=True),
        sa.Column("is_deprecated", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("weekly_downloads", sa.BigInteger(), nullable=True),
        sa.Column("dependent_count", sa.Integer(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("ecosystem", "name"),
    )

    op.create_table(
        "repo_snapshots",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column(
            "repo_id",
            sa.Uuid(),
            sa.ForeignKey("repos.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("snapshot_month", sa.Date(), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("feature_schema_version", sa.Integer(), server_default=sa.text("1")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("repo_id", "snapshot_month"),
    )
    op.create_index(
        "ix_repo_snapshots_repo_month",
        "repo_snapshots",
        ["repo_id", sa.text("snapshot_month DESC")],
    )

    op.create_table(
        "risk_scores",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column(
            "repo_id",
            sa.Uuid(),
            sa.ForeignKey("repos.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("p_abandon_3mo", sa.Float(), nullable=False),
        sa.Column("p_abandon_6mo", sa.Float(), nullable=False),
        sa.Column("p_abandon_12mo", sa.Float(), nullable=False),
        sa.Column("risk_tier", sa.Text(), nullable=False),
        sa.Column("top_risk_factors", sa.JSON(), nullable=True),
        sa.Column(
            "scored_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("repo_id", "model_version"),
    )
    op.create_index(
        "ix_risk_scores_repo_model",
        "risk_scores",
        ["repo_id", "model_version"],
    )

    op.create_table(
        "scans",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("ecosystem", sa.Text(), nullable=False),
        sa.Column("manifest_hash", sa.Text(), nullable=False),
        sa.Column("package_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "scan_results",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column(
            "scan_id",
            sa.Uuid(),
            sa.ForeignKey("scans.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("package_id", sa.Uuid(), sa.ForeignKey("packages.id"), nullable=True),
        sa.Column("risk_score_id", sa.Uuid(), sa.ForeignKey("risk_scores.id"), nullable=True),
        sa.Column("pinned_version", sa.Text(), nullable=True),
    )

    op.create_index("ix_packages_ecosystem_name", "packages", ["ecosystem", "name"])
    op.create_index(
        "ix_repos_abandoned_priority",
        "repos",
        ["is_abandoned", "monitoring_priority"],
    )


def downgrade() -> None:
    op.drop_table("scan_results")
    op.drop_table("scans")
    op.drop_table("risk_scores")
    op.drop_table("repo_snapshots")
    op.drop_table("packages")
    op.drop_table("repos")
