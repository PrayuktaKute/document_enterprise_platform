"""Field-level confidence scoring.

Primary signal: token log-probabilities from the extraction call. For each schema
field we locate its value span in the generated JSON, map that span to the
covering output tokens, and aggregate ``exp(logprob)`` over the non-structural
tokens (``logprob_min`` = least confident token in the value, ``logprob_mean`` =
average).

Fallback: self-consistency -- agreement of a field's value across resampled
extractions.
"""
from __future__ import annotations

import math
from statistics import fmean

from dip.llm.client import TokenLogprob

_STRUCTURAL = set(' \t\r\n{}[]:,"')
_MISSING_KEY_CONF = 0.0
_NULL_VALUE_CONF = 0.6


def _value_span(text: str, field: str) -> tuple[int, int] | None:
    """Char span of ``field``'s value inside a JSON object string."""
    key = f'"{field}"'
    k = text.find(key)
    if k == -1:
        return None
    i = text.find(":", k + len(key))
    if i == -1:
        return None
    i += 1
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    if i >= len(text):
        return None

    start = i
    depth = 0
    in_str = False
    esc = False
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in "[{":
                depth += 1
            elif ch in "]}":
                if depth == 0:
                    break
                depth -= 1
            elif ch == "," and depth == 0:
                break
        i += 1
    return start, i


def _token_char_offsets(tokens: list[TokenLogprob]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for t in tokens:
        spans.append((pos, pos + len(t.token)))
        pos += len(t.token)
    return spans


def field_confidences_from_logprobs(
    tokens: list[TokenLogprob],
    fields: list[str],
    *,
    method: str = "logprob_min",
    rebuilt_text: str | None = None,
) -> dict[str, float]:
    """Map value spans -> covering tokens -> aggregated probability, per field."""
    text = rebuilt_text if rebuilt_text is not None else "".join(t.token for t in tokens)
    offsets = _token_char_offsets(tokens)
    out: dict[str, float] = {}

    for field in fields:
        span = _value_span(text, field)
        if span is None:
            out[field] = _MISSING_KEY_CONF
            continue
        s, e = span
        if text[s:e].strip() in ("null", ""):
            out[field] = _NULL_VALUE_CONF
            continue

        probs: list[float] = []
        struct_probs: list[float] = []
        for (ts, te), tok in zip(offsets, tokens):
            if te <= s or ts >= e:
                continue
            p = math.exp(tok.logprob)
            if tok.token.strip("".join(_STRUCTURAL)) == "":
                struct_probs.append(p)
            else:
                probs.append(p)
        use = probs or struct_probs
        if not use:
            out[field] = _NULL_VALUE_CONF
            continue
        out[field] = min(use) if method == "logprob_min" else fmean(use)

    return out


def field_confidences_from_samples(
    samples: list[dict], fields: list[str]
) -> dict[str, float]:
    """Fraction of resamples whose value for a field matches the primary sample."""
    if not samples:
        return {f: 0.5 for f in fields}
    primary, rest = samples[0], samples[1:]
    out: dict[str, float] = {}
    for field in fields:
        base = _norm(primary.get(field))
        if not rest:
            out[field] = 0.75 if base not in ("", "none") else 0.4
            continue
        agree = sum(1 for s in rest if _norm(s.get(field)) == base)
        out[field] = (agree + 1) / (len(rest) + 1)
    return out


def aggregate_doc_confidence(field_conf: dict[str, float]) -> float:
    vals = [v for v in field_conf.values() if v is not None]
    return round(fmean(vals), 4) if vals else 0.0


def _norm(v) -> str:
    if v is None:
        return "none"
    if isinstance(v, (list, dict)):
        return str(v).lower().strip()
    return str(v).lower().strip()
