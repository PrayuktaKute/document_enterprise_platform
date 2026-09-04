from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ExtractionBase(BaseModel):
    """Base for all extraction schemas.

    ``extra="ignore"`` keeps parsing resilient when a small model emits a stray
    key; type validation on the declared fields still applies. The JSON schema
    handed to the model for constrained decoding is derived from subclasses via
    ``model_json_schema()``.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    @classmethod
    def extraction_fields(cls) -> list[str]:
        return list(cls.model_fields.keys())
