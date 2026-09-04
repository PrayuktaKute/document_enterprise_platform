from __future__ import annotations

from pydantic import Field

from dip.schemas.base import ExtractionBase


class MedicalReport(ExtractionBase):
    """Synthetic radiology / laboratory report. No real PHI -- all fields fabricated."""

    patient_id: str | None = Field(default=None, description="Patient identifier (fabricated)")
    patient_name: str | None = Field(default=None, description="Patient name (fabricated)")
    report_date: str | None = Field(default=None, description="Report date in ISO 8601 (YYYY-MM-DD)")
    ordering_physician: str | None = Field(default=None, description="Physician who ordered the study")
    modality: str | None = Field(
        default=None, description="Study type, e.g. X-ray, CT, MRI, Ultrasound, Lab panel"
    )
    body_site: str | None = Field(default=None, description="Anatomical region examined")
    findings: str | None = Field(default=None, description="Findings section text")
    impression: str | None = Field(default=None, description="Impression / conclusion text")
    diagnoses: list[str] = Field(default_factory=list, description="Named diagnoses or conditions")
