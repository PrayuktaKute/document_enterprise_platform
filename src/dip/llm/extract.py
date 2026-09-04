"""Schema-driven extraction: Docling text + Qwen2.5 -> validated JSON + confidence."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from dip.config import get_doc_type_config, get_pipeline_config, resolve_confidence
from dip.llm.client import LLMClient
from dip.parsing.docling_parser import ParsedDoc
from dip.validation.confidence import (
    aggregate_doc_confidence,
    field_confidences_from_logprobs,
    field_confidences_from_samples,
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass
class ExtractionResult:
    doc_type: str
    data: dict
    field_confidences: dict[str, float] = field(default_factory=dict)
    doc_confidence: float = 0.0
    confidence_method: str = "none"
    raw_response: str = ""
    parse_ok: bool = False
    schema_ok: bool = False
    error: str | None = None


def _extract_json(text: str) -> dict:
    m = _FENCE_RE.search(text)
    candidate = m.group(1) if m else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in response")
    return json.loads(candidate[start : end + 1])


def _budget(doc_type: str) -> int:
    llm = get_pipeline_config().llm
    return llm.text_budget_contract if doc_type == "contract" else llm.text_budget_default


def _build_messages(doc_type: str, schema_cls: type[BaseModel], text: str) -> list[dict]:
    schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
    system = (
        f"You extract structured data from a {doc_type.replace('_', ' ')}.\n"
        "Return ONLY one JSON object that conforms to the schema below - no prose, no code fences.\n"
        "Use null for any field not stated in the document. Do not invent values.\n"
        "All dates must be ISO 8601 (YYYY-MM-DD).\n\n"
        f"JSON schema:\n{schema_json}"
    )
    messages: list[dict] = [{"role": "system", "content": system}]

    for ex in get_doc_type_config(doc_type).extraction.few_shot_examples():
        if "document" in ex and "output" in ex:
            messages.append({"role": "user", "content": f"Document:\n\"\"\"\n{ex['document']}\n\"\"\""})
            messages.append({"role": "assistant", "content": json.dumps(ex["output"])})

    messages.append(
        {
            "role": "user",
            "content": f"Document:\n\"\"\"\n{text[: _budget(doc_type)]}\n\"\"\"\n\n"
            "Return the JSON object now.",
        }
    )
    return messages


def extract_fields(
    parsed: ParsedDoc, doc_type: str, client: LLMClient | None = None
) -> ExtractionResult:
    client = client or LLMClient.from_config()
    pipe = get_pipeline_config()
    cfg = get_doc_type_config(doc_type)
    schema_cls = cfg.schema_cls()
    fields = schema_cls.extraction_fields()
    method, _, _ = resolve_confidence(doc_type)
    want_logprobs = method in ("logprob_min", "logprob_mean")

    messages = _build_messages(doc_type, schema_cls, parsed.text)
    json_schema = schema_cls.model_json_schema() if pipe.llm.use_json_schema else None

    resp = client.chat(
        messages,
        temperature=pipe.llm.temperature,
        max_tokens=cfg.extraction.max_output_tokens,
        json_object=json_schema is None,
        json_schema=json_schema,
        logprobs=want_logprobs,
        top_logprobs=pipe.llm.top_logprobs,
    )

    result = ExtractionResult(doc_type=doc_type, data={}, raw_response=resp.text)

    # Parse (one retry with error feedback).
    raw: dict | None = None
    for attempt in range(2):
        try:
            raw = _extract_json(resp.text)
            result.parse_ok = True
            break
        except (ValueError, json.JSONDecodeError) as exc:
            result.error = f"parse: {exc}"
            if attempt == 0:
                retry_msgs = messages + [
                    {"role": "assistant", "content": resp.text},
                    {"role": "user", "content": f"That was not valid JSON ({exc}). "
                                                "Return ONLY the JSON object."},
                ]
                resp = client.chat(
                    retry_msgs,
                    temperature=0.0,
                    max_tokens=cfg.extraction.max_output_tokens,
                    json_object=json_schema is None,
                    json_schema=json_schema,
                    logprobs=want_logprobs,
                    top_logprobs=pipe.llm.top_logprobs,
                )
                result.raw_response = resp.text
    if raw is None:
        return result

    # Schema-coerce.
    try:
        model = schema_cls.model_validate(raw)
        result.data = model.model_dump()
        result.schema_ok = True
    except ValidationError as exc:
        result.data = {k: raw.get(k) for k in fields}
        result.error = f"schema: {exc.error_count()} error(s)"

    # Confidence.
    if want_logprobs and resp.has_logprobs:
        fc = field_confidences_from_logprobs(resp.tokens, fields, method=method)
        used = method
    elif method == "self_consistency" or (want_logprobs and not resp.has_logprobs):
        samples = [result.data]
        for _ in range(max(1, pipe.confidence.self_consistency_samples - 1)):
            r2 = client.chat(
                messages,
                temperature=pipe.confidence.self_consistency_temperature,
                max_tokens=cfg.extraction.max_output_tokens,
                json_object=json_schema is None,
                json_schema=json_schema,
            )
            try:
                samples.append(_extract_json(r2.text))
            except (ValueError, json.JSONDecodeError):
                pass
        fc = field_confidences_from_samples(samples, fields)
        used = "self_consistency"
    else:
        fc = {f: 0.75 for f in fields}
        used = "none"

    result.field_confidences = {k: round(v, 4) for k, v in fc.items()}
    result.doc_confidence = aggregate_doc_confidence(fc)
    result.confidence_method = used
    return result
