"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RiskFactorResponse(BaseModel):
    """A feature's contribution to the abandonment risk."""

    feature: str = Field(description="Feature name")
    value: float = Field(description="Feature value for this package")
    impact: float = Field(description="SHAP value (positive = increases risk)")
    description: str = Field(description="Human-readable explanation")


class PackageRiskResponse(BaseModel):
    """Risk assessment for a single dependency."""

    package: str = Field(description="Package name from manifest")
    ecosystem: str = Field(description="Package ecosystem (pypi, npm, go)")
    github_repo: str = Field(description="GitHub owner/repo")
    abandonment_probability: float = Field(
        description="Predicted probability of abandonment (0-1)",
        ge=0.0,
        le=1.0,
    )
    risk_level: str = Field(description="Risk level: low, medium, high, critical")
    top_risk_factors: list[RiskFactorResponse] = Field(
        description="Top features driving the risk score",
    )


class PackageErrorResponse(BaseModel):
    """Error details for a package that couldn't be scored."""

    package: str
    ecosystem: str
    error: str


class ScanResponse(BaseModel):
    """Response from the /scan endpoint."""

    packages_scanned: int = Field(description="Total packages in manifest")
    packages_scored: int = Field(description="Packages successfully scored")
    packages_errored: int = Field(description="Packages that failed")
    results: list[PackageRiskResponse] = Field(
        description="Scored packages, sorted by risk (highest first)",
    )
    errors: list[PackageErrorResponse] = Field(
        default_factory=list,
        description="Packages that could not be scored",
    )
