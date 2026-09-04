"""Configuration loading: environment settings + YAML pipeline / doc-type configs.

The whole "configurable pipeline" story lives here. Adding a document type means
dropping a new YAML into ``config/doc_types/`` and a Pydantic model into
``dip.schemas`` -- no changes to pipeline code.
"""
from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (src/dip/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
DOC_TYPES_DIR = CONFIG_DIR / "doc_types"


# --------------------------------------------------------------------------- #
# Environment settings
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen2.5:3b-instruct-q4_K_M"

    embed_model: str = "BAAI/bge-m3"

    database_url: str = "postgresql+psycopg://dip:dip@localhost:5432/dip"
    qdrant_url: str = "http://localhost:6333"
    qdrant_path: str = ""          # if set, use embedded on-disk Qdrant (no server) -- for Colab
    qdrant_collection: str = "documents"

    confidence_method: str = "logprob_min"

    data_dir: Path = REPO_ROOT / "data"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    cache_dir: Path = REPO_ROOT / "data" / "cache"


# --------------------------------------------------------------------------- #
# pipeline.yaml
# --------------------------------------------------------------------------- #
class LLMCfg(BaseModel):
    temperature: float = 0.0
    max_output_tokens: int = 800
    logprobs: bool = True
    top_logprobs: int = 5
    use_json_schema: bool = False   # False -> json_object mode + schema in prompt (safest on Ollama)
    text_budget_default: int = 12000
    text_budget_contract: int = 18000
    request_timeout_s: int = 180
    max_retries: int = 2


class EmbeddingCfg(BaseModel):
    model: str = "BAAI/bge-m3"
    batch_size: int = 12
    dense_dim: int = 1024
    normalize: bool = True


class QdrantCfg(BaseModel):
    collection: str = "documents"
    distance: str = "Cosine"


class ChunkingCfg(BaseModel):
    strategy: str = "layout_section"
    max_tokens: int = 512
    overlap: int = 64


class ClassificationCfg(BaseModel):
    first_n_chars: int = 1500
    low_confidence_fallback: float = 0.45


class ConfidenceDefaults(BaseModel):
    method: str = "logprob_min"
    default_field_threshold: float = 0.55
    default_doc_threshold: float = 0.70
    max_low_conf_fields: int = 1
    self_consistency_samples: int = 2
    self_consistency_temperature: float = 0.3


class ReviewCfg(BaseModel):
    route_low_confidence: bool = True


class PipelineConfig(BaseModel):
    llm: LLMCfg = LLMCfg()
    embedding: EmbeddingCfg = EmbeddingCfg()
    qdrant: QdrantCfg = QdrantCfg()
    chunking: ChunkingCfg = ChunkingCfg()
    classification: ClassificationCfg = ClassificationCfg()
    confidence: ConfidenceDefaults = ConfidenceDefaults()
    review: ReviewCfg = ReviewCfg()


# --------------------------------------------------------------------------- #
# config/doc_types/*.yaml
# --------------------------------------------------------------------------- #
class DocConfidenceCfg(BaseModel):
    method: str | None = None
    field_threshold: float | None = None
    doc_threshold: float | None = None


class DocExtractionCfg(BaseModel):
    strategy: str = "single_pass"          # single_pass | field_by_field
    max_output_tokens: int = 800
    few_shot: str | None = None

    def few_shot_examples(self) -> list[dict[str, Any]]:
        if not self.few_shot:
            return []
        path = REPO_ROOT / self.few_shot
        if not path.exists():
            return []
        return yaml.safe_load(path.read_text(encoding="utf-8")) or []


class DocValidationCfg(BaseModel):
    critical_rules: list[str] = Field(default_factory=list)
    rules: list[Any] = Field(default_factory=list)
    confidence: DocConfidenceCfg = DocConfidenceCfg()


class DocTypeConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    doc_type: str
    schema_path: str = Field(alias="schema")
    classifier_hint: str
    extraction: DocExtractionCfg = DocExtractionCfg()
    validation: DocValidationCfg = DocValidationCfg()
    chunking: ChunkingCfg = ChunkingCfg()

    def schema_cls(self) -> type[BaseModel]:
        module_path, cls_name = self.schema_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)


# --------------------------------------------------------------------------- #
# Loaders (cached)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_pipeline_config() -> PipelineConfig:
    raw = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text(encoding="utf-8"))
    return PipelineConfig(**(raw or {}))


@lru_cache(maxsize=1)
def get_doc_type_configs() -> dict[str, DocTypeConfig]:
    configs: dict[str, DocTypeConfig] = {}
    for path in sorted(DOC_TYPES_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg = DocTypeConfig(**raw)
        configs[cfg.doc_type] = cfg
    return configs


def get_doc_type_config(doc_type: str) -> DocTypeConfig:
    try:
        return get_doc_type_configs()[doc_type]
    except KeyError as exc:
        known = ", ".join(get_doc_type_configs())
        raise KeyError(f"Unknown doc_type {doc_type!r}; known types: {known}") from exc


def resolve_confidence(doc_type: str) -> tuple[str, float, float]:
    """Return (method, field_threshold, doc_threshold) with pipeline defaults applied."""
    pipe = get_pipeline_config().confidence
    dc = get_doc_type_config(doc_type).validation.confidence
    return (
        dc.method or pipe.method,
        dc.field_threshold if dc.field_threshold is not None else pipe.default_field_threshold,
        dc.doc_threshold if dc.doc_threshold is not None else pipe.default_doc_threshold,
    )
