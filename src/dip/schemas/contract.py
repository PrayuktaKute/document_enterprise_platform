from __future__ import annotations

from pydantic import Field

from dip.schemas.base import ExtractionBase


class Contract(ExtractionBase):
    """Legal agreement. Eval fields derived from CUAD annotation categories."""

    document_name: str | None = Field(default=None, description="Title of the agreement")
    contract_type: str | None = Field(
        default=None, description="e.g. NDA, Master Services Agreement, License, Supply"
    )
    parties: list[str] = Field(default_factory=list, description="Legal names of the parties")
    agreement_date: str | None = Field(
        default=None, description="Date the agreement is dated / signed, ISO 8601"
    )
    effective_date: str | None = Field(default=None, description="Effective date, ISO 8601")
    expiration_or_term: str | None = Field(
        default=None, description="Expiration date or stated term (e.g. '3 years from Effective Date')"
    )
    governing_law: str | None = Field(default=None, description="Governing law / jurisdiction")
    renewal_term: str | None = Field(
        default=None, description="Auto-renewal / extension term if any"
    )
