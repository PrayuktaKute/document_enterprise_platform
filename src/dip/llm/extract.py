"""Schema-driven extraction: Docling text + Qwen2.5 -> validated JSON + confidence.

Two strategies (per doc-type config):
  * single_pass    -- one call, whole schema, truncated document
  * field_by_field -- one call per field group, each with focused context
                      (document head, or windows around keyword hits). Keeps each
                      call short so a 3B model stays reliable on long documents.
"""
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


# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> dict:
    m = _FENCE_RE.search(text)
    candidate = m.group(1) if m else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in response")
    blob = candidate[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # tolerate trailing commas / stray control chars
        blob2 = re.sub(r",(\s*[}\]])", r"\1", blob)
        return json.loads(blob2)


def _budget(doc_type: str) -> int:
    llm = get_pipeline_config().llm
    return llm.text_budget_contract if doc_type == "contract" else llm.text_budget_default


def _sub_schema(schema_cls: type[BaseModel], fields: list[str]) -> dict:
    full = schema_cls.model_json_schema()
    props = {k: v for k, v in full.get("properties", {}).items() if k in fields}
    return {"type": "object", "properties": props}


def _keyword_context(text: str, keywords: list[str], max_chars: int, window: int = 700) -> str:
    low = text.lower()
    spans: list[tuple[int, int]] = []
    for kw in keywords:
        i = 0
        k = kw.lower()
        while True:
            j = low.find(k, i)
            if j == -1:
                break
            spans.append((max(0, j - window), min(len(text), j + len(k) + window)))
            i = j + len(k)
    if not spans:
        return text[:max_chars]
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + 60:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    out, total = [], 0
    for s, e in merged:
        piece = text[s:e]
        out.append(piece)
        total += len(piece)
        if total >= max_chars:
            break
    return "\n...\n".join(out)[:max_chars]


# --------------------------------------------------------------------------- #
def _call_group(
    client: LLMClient,
    doc_type: str,
    schema_cls: type[BaseModel],
    fields: list[str],
    context: str,
    *,
    want_logprobs: bool,
    method: str,
) -> tuple[dict, dict, str | None]:
    """One extraction call for a group of fields. Returns (data, field_conf, error)."""
    pipe = get_pipeline_config()
    sub = _sub_schema(schema_cls, fields)
    hint = get_doc_type_config(doc_type).extraction.hint
    system = (
        f"Extract fields from this {doc_type.replace('_', ' ')}.\n"
        f"Return ONLY a JSON object with EXACTLY these keys: {', '.join(fields)}.\n"
        "Use null when the value is not stated in the text. Do not invent values.\n"
        "Dates must be ISO 8601 (YYYY-MM-DD).\n"
        + (f"{hint}\n" if hint else "")
        + f"\nField schema:\n{json.dumps(sub, indent=1)}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f'Document:\n"""\n{context}\n"""\n\nJSON object now.'},
    ]
    json_schema = sub if pipe.llm.use_json_schema else None

    err: str | None = None
    resp = client.chat(
        messages,
        temperature=pipe.llm.temperature,
        max_tokens=pipe.llm.max_output_tokens,
        json_object=json_schema is None,
        json_schema=json_schema,
        logprobs=want_logprobs,
        top_logprobs=pipe.llm.top_logprobs,
    )
    data: dict = {}
    for attempt in range(2):
        try:
            raw = _extract_json(resp.text)
            data = {k: raw.get(k) for k in fields}
            err = None
            break
        except (ValueError, json.JSONDecodeError) as exc:
            err = f"parse[{','.join(fields[:2])}...]: {exc}"
            if attempt == 0:
                resp = client.chat(
                    messages + [
                        {"role": "assistant", "content": resp.text},
                        {"role": "user", "content": f"Not valid JSON ({exc}). Return ONLY the JSON object."},
                    ],
                    temperature=0.0,
                    max_tokens=pipe.llm.max_output_tokens,
                    json_object=json_schema is None,
                    json_schema=json_schema,
                    logprobs=want_logprobs,
                    top_logprobs=pipe.llm.top_logprobs,
                )

    if want_logprobs and resp.has_logprobs and data:
        fconf = field_confidences_from_logprobs(resp.tokens, fields, method=method)
    else:
        fconf = {f: (0.55 if data.get(f) not in (None, "", []) else 0.0) for f in fields}
    return data, fconf, err


# --------------------------------------------------------------------------- #
def extract_fields(
    parsed: ParsedDoc, doc_type: str, client: LLMClient | None = None
) -> ExtractionResult:
    client = client or LLMClient.from_config()
    pipe = get_pipeline_config()
    cfg = get_doc_type_config(doc_type)
    schema_cls = cfg.schema_cls()
    all_fields = schema_cls.extraction_fields()
    method, _, _ = resolve_confidence(doc_type)
    want_logprobs = method in ("logprob_min", "logprob_mean")

    result = ExtractionResult(doc_type=doc_type, data={})

    # Build the list of (fields, context) groups.
    groups: list[tuple[list[str], str]] = []
    if cfg.extraction.strategy == "field_by_field" and cfg.extraction.field_groups:
        for g in cfg.extraction.field_groups:
            gf = [f for f in g.get("fields", []) if f in all_fields]
            max_chars = int(g.get("max_chars", 6000))
            if g.get("source") == "keywords" and g.get("keywords"):
                ctx = _keyword_context(
                    parsed.text, list(g["keywords"]), max_chars, window=int(g.get("window", 700))
                )
            else:
                ctx = parsed.text[:max_chars]
            if gf:
                groups.append((gf, ctx))
        covered = {f for gf, _ in groups for f in gf}
        leftover = [f for f in all_fields if f not in covered]
        if leftover:
            groups.append((leftover, parsed.text[: _budget(doc_type)]))
    else:
        groups = [(all_fields, parsed.text[: _budget(doc_type)])]

    merged: dict = {}
    fconf: dict = {}
    errors: list[str] = []
    for gf, ctx in groups:
        data, fc, err = _call_group(
            client, doc_type, schema_cls, gf, ctx, want_logprobs=want_logprobs, method=method
        )
        merged.update(data)
        fconf.update(fc)
        if err:
            errors.append(err)

    result.parse_ok = len(errors) < len(groups)
    result.raw_response = json.dumps(merged, ensure_ascii=False)

    # Coerce through the full schema (dates, numbers, list shapes).
    try:
        result.data = schema_cls.model_validate(
            {k: merged.get(k) for k in all_fields}
        ).model_dump()
        result.schema_ok = True
    except ValidationError as exc:
        result.data = {k: merged.get(k) for k in all_fields}
        errors.append(f"schema: {exc.error_count()} error(s)")

    # self-consistency fallback only when logprobs unavailable and requested
    if not want_logprobs and method == "self_consistency":
        samples = [result.data]
        for _ in range(max(1, pipe.confidence.self_consistency_samples - 1)):
            d2, _, _ = _call_group(
                client, doc_type, schema_cls, all_fields,
                parsed.text[: _budget(doc_type)], want_logprobs=False, method=method,
            )
            samples.append(d2)
        fconf = field_confidences_from_samples(samples, all_fields)
        result.confidence_method = "self_consistency"
    else:
        result.confidence_method = method if want_logprobs else "heuristic"

    result.field_confidences = {k: round(float(fconf.get(k, 0.0)), 4) for k in all_fields}
    result.doc_confidence = aggregate_doc_confidence(result.field_confidences)
    result.error = "; ".join(errors) or None
    return result
