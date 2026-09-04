"""Document-type classification: Qwen zero-shot over an excerpt, keyword fallback."""
from __future__ import annotations

import json
import re

from dip.config import get_doc_type_configs, get_pipeline_config
from dip.llm.client import LLMClient

_KEYWORDS: dict[str, list[str]] = {
    "invoice": ["invoice", "invoice no", "bill to", "amount due", "tax invoice", "gst", "receipt"],
    "purchase_order": ["purchase order", "po number", "po-", "ship to", "vendor", "buyer", "ordered by"],
    "medical_report": ["patient", "findings", "impression", "radiology", "clinical", "diagnosis",
                       "physician", "mrn", "modality"],
    "contract": ["agreement", "party", "parties", "hereby", "governing law", "term of this",
                 "witnesseth", "shall", "confidentiality"],
}


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no json")
    return json.loads(m.group(0))


def _keyword_scores(text: str) -> dict[str, int]:
    low = text.lower()
    return {t: sum(low.count(k) for k in kws) for t, kws in _KEYWORDS.items()}


def classify_document(text: str, client: LLMClient | None = None) -> tuple[str | None, float]:
    cfg = get_pipeline_config().classification
    types = get_doc_type_configs()
    excerpt = text[: cfg.first_n_chars]

    client = client or LLMClient.from_config()
    labels = "\n".join(f'- {name}: {c.classifier_hint.strip()}' for name, c in types.items())
    messages = [
        {
            "role": "system",
            "content": "Classify the document as exactly one of the given type ids. "
            'Respond with JSON only: {"doc_type": "<id>", "confidence": <0..1>}.',
        },
        {"role": "user", "content": f"Type ids:\n{labels}\n\nDocument excerpt:\n\"\"\"\n{excerpt}\n\"\"\""},
    ]

    dt: str | None = None
    conf = 0.0
    try:
        d = _extract_json(client.chat(messages, json_object=True, max_tokens=60, temperature=0.0).text)
        dt = str(d.get("doc_type", "")).strip()
        conf = float(d.get("confidence", 0.5))
    except Exception:  # noqa: BLE001
        dt, conf = None, 0.0

    if dt not in types or conf < cfg.low_confidence_fallback:
        scores = _keyword_scores(text)
        best, best_score = max(scores.items(), key=lambda kv: kv[1])
        if best_score > 0:
            return best, max(conf, 0.55) if dt == best else 0.55
    return dt, round(conf, 3)
