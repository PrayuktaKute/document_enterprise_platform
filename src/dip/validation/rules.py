"""Rule-based business validation.

Config lists rules as either a bare name (``total_is_positive``) or a single-key
mapping with arguments (``{required_fields: [company, total]}``). Rules whose name
appears in the doc type's ``critical_rules`` gate auto-acceptance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable

from dip.config import get_doc_type_config

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_TOL_REL = 0.02
_TOL_ABS = 0.05


@dataclass
class RuleOutcome:
    rule_name: str
    passed: bool
    is_critical: bool = False
    message: str | None = None


# --------------------------------------------------------------------------- #
def _num(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = _NUM_RE.search(str(v))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _parse_date(v: Any) -> date | None:
    if not v or not isinstance(v, str):
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y",
                "%d/%m/%y", "%m/%d/%y"):
        try:
            return datetime.strptime(v.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(_TOL_ABS, _TOL_REL * max(abs(a), abs(b)))


def _empty(v: Any) -> bool:
    return v is None or (isinstance(v, (str, list, dict)) and len(v) == 0)


# --------------------------------------------------------------------------- #
def rule_required_fields(ext: dict, fields: list[str]) -> RuleOutcome:
    missing = [f for f in fields if _empty(ext.get(f))]
    return RuleOutcome(
        "required_fields", not missing,
        message=None if not missing else f"missing/empty: {', '.join(missing)}",
    )


def rule_total_is_positive(ext: dict) -> RuleOutcome:
    t = _num(ext.get("total"))
    if t is None:
        return RuleOutcome("total_is_positive", False, message="total not numeric")
    return RuleOutcome("total_is_positive", t > 0, message=None if t > 0 else f"total={t}")


def rule_date_is_valid(ext: dict) -> RuleOutcome:
    cands = [k for k in ext if "date" in k.lower()]
    present = [(k, ext[k]) for k in cands if ext.get(k)]
    if not present:
        return RuleOutcome("date_is_valid", True, message="no date fields present")
    bad = [k for k, v in present if _parse_date(v) is None]
    return RuleOutcome("date_is_valid", not bad,
                       message=None if not bad else f"unparseable: {', '.join(bad)}")


def rule_line_items_sum_matches_subtotal(ext: dict) -> RuleOutcome:
    items = ext.get("line_items") or []
    subtotal = _num(ext.get("subtotal"))
    if not items or subtotal is None:
        return RuleOutcome("line_items_sum_matches_subtotal", True, message="not applicable")
    s = sum(_num(it.get("amount")) or 0.0 for it in items if isinstance(it, dict))
    ok = _close(s, subtotal)
    return RuleOutcome("line_items_sum_matches_subtotal", ok,
                       message=None if ok else f"sum(items)={s:.2f} vs subtotal={subtotal:.2f}")


def rule_subtotal_plus_tax_matches_total(ext: dict) -> RuleOutcome:
    sub, tax, tot = _num(ext.get("subtotal")), _num(ext.get("tax")), _num(ext.get("total"))
    if sub is None or tot is None:
        return RuleOutcome("subtotal_plus_tax_matches_total", True, message="not applicable")
    ok = _close(sub + (tax or 0.0), tot)
    return RuleOutcome("subtotal_plus_tax_matches_total", ok,
                       message=None if ok else f"{sub:.2f}+{(tax or 0):.2f} vs total={tot:.2f}")


def rule_delivery_after_order_date(ext: dict) -> RuleOutcome:
    o, d = _parse_date(ext.get("order_date")), _parse_date(ext.get("delivery_date"))
    if o is None or d is None:
        return RuleOutcome("delivery_after_order_date", True, message="not applicable")
    return RuleOutcome("delivery_after_order_date", d >= o,
                       message=None if d >= o else f"delivery {d} < order {o}")


def rule_at_least_two_parties(ext: dict) -> RuleOutcome:
    p = ext.get("parties") or []
    if isinstance(p, str):
        p = [x for x in re.split(r";|\band\b|,", p) if x.strip()]
    return RuleOutcome("at_least_two_parties", len(p) >= 2,
                       message=None if len(p) >= 2 else f"{len(p)} party(ies)")


def rule_impression_present(ext: dict) -> RuleOutcome:
    return RuleOutcome("impression_present", not _empty(ext.get("impression")),
                       message=None if not _empty(ext.get("impression")) else "impression empty")


_NOARG: dict[str, Callable[[dict], RuleOutcome]] = {
    "total_is_positive": rule_total_is_positive,
    "date_is_valid": rule_date_is_valid,
    "line_items_sum_matches_subtotal": rule_line_items_sum_matches_subtotal,
    "subtotal_plus_tax_matches_total": rule_subtotal_plus_tax_matches_total,
    "delivery_after_order_date": rule_delivery_after_order_date,
    "at_least_two_parties": rule_at_least_two_parties,
    "impression_present": rule_impression_present,
}


# --------------------------------------------------------------------------- #
def run_rules(doc_type: str, extraction: dict) -> list[RuleOutcome]:
    cfg = get_doc_type_config(doc_type)
    critical = set(cfg.validation.critical_rules)
    outcomes: list[RuleOutcome] = []

    for spec in cfg.validation.rules:
        if isinstance(spec, str):
            name, args = spec, None
        elif isinstance(spec, dict) and len(spec) == 1:
            name, args = next(iter(spec.items()))
        else:
            continue

        if name == "required_fields":
            outcome = rule_required_fields(extraction, list(args or []))
        elif name in _NOARG:
            outcome = _NOARG[name](extraction)
        else:
            outcome = RuleOutcome(name, True, message=f"unknown rule {name!r} (skipped)")

        outcome.is_critical = name in critical
        outcomes.append(outcome)

    return outcomes


def critical_failures(outcomes: list[RuleOutcome]) -> list[str]:
    return [o.rule_name for o in outcomes if o.is_critical and not o.passed]
