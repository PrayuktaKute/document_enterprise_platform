"""Streamlit front-end. Talks to the FastAPI service over HTTP."""
from __future__ import annotations

import json
import os
import time

import httpx
import streamlit as st

API = os.environ.get("DIP_API_URL", "http://localhost:8000")
DOC_TYPES = ["invoice", "purchase_order", "medical_report", "contract"]

st.set_page_config(page_title="Document Intelligence Platform", layout="wide")
st.title("Enterprise Document Intelligence Platform")


def api(method: str, path: str, **kw):
    r = httpx.request(method, f"{API}{path}", timeout=120, **kw)
    r.raise_for_status()
    return r.json()


def conf_badge(v: float | None) -> str:
    if v is None:
        return "-"
    color = "🟢" if v >= 0.7 else ("🟡" if v >= 0.5 else "🔴")
    return f"{color} {v:.2f}"


tab_up, tab_review, tab_search, tab_metrics = st.tabs(
    ["Upload & Process", "Review Queue", "Search", "Metrics"]
)

# --------------------------------------------------------------------------- #
with tab_up:
    col1, col2 = st.columns([1, 1])
    with col1:
        up = st.file_uploader("Document", type=["pdf", "jpg", "jpeg", "png", "txt"])
        forced = st.selectbox("Force document type (optional)", ["(auto-classify)"] + DOC_TYPES)
        if st.button("Process", disabled=up is None):
            files = {"file": (up.name, up.getvalue())}
            data = {} if forced == "(auto-classify)" else {"doc_type": forced}
            resp = api("POST", "/documents", files=files, data=data)
            st.session_state["last_doc"] = resp["doc_id"]
            st.success(f"Submitted {resp['doc_id']}")

    with col2:
        doc_id = st.text_input("Document id", st.session_state.get("last_doc", ""))
        if st.button("Refresh", disabled=not doc_id):
            pass
        if doc_id:
            for _ in range(2):
                try:
                    view = api("GET", f"/documents/{doc_id}")
                    break
                except Exception as exc:  # noqa: BLE001
                    st.warning(str(exc))
                    time.sleep(1)
                    view = None
            if view:
                st.metric("Status", view["status"])
                st.write(
                    f"**Type:** {view['doc_type']} ({conf_badge(view['doc_type_confidence'])}) "
                    f"| **Doc confidence:** {conf_badge(view['doc_confidence'])}"
                )
                st.json(view["extraction"])
                if view["field_confidences"]:
                    st.caption("Field confidence")
                    st.dataframe(
                        [
                            {"field": f["field"], "confidence": round(f["confidence"], 3),
                             "flag": "" if f["confidence"] >= 0.55 else "review"}
                            for f in view["field_confidences"]
                        ],
                        use_container_width=True, hide_index=True,
                    )
                if view["validation"]:
                    st.caption("Validation")
                    st.dataframe(
                        [
                            {"rule": v["rule"], "passed": v["passed"],
                             "critical": v["critical"], "message": v["message"]}
                            for v in view["validation"]
                        ],
                        use_container_width=True, hide_index=True,
                    )

# --------------------------------------------------------------------------- #
with tab_review:
    st.subheader("Documents needing review")
    try:
        queue = api("GET", "/documents", params={"status": "needs_review"})
    except Exception as exc:  # noqa: BLE001
        queue = []
        st.error(str(exc))
    if not queue:
        st.info("Review queue is empty.")
    for item in queue:
        with st.expander(f"{item['doc_id']}  —  {item['doc_type']}  ({item['filename']})"):
            view = api("GET", f"/documents/{item['doc_id']}")
            low = {f["field"] for f in view["field_confidences"] if f["confidence"] < 0.55}
            if low:
                st.warning("Low-confidence fields: " + ", ".join(sorted(low)))
            edited = st.text_area(
                "Extraction JSON", json.dumps(view["extraction"], indent=2),
                height=280, key=f"edit_{item['doc_id']}",
            )
            c1, c2 = st.columns(2)
            if c1.button("Approve corrected", key=f"appr_{item['doc_id']}"):
                try:
                    payload = json.loads(edited)
                    res = api("PUT", f"/documents/{item['doc_id']}/extraction",
                              json={"payload": payload, "note": "streamlit review"})
                    st.success(f"Re-indexed ({res['chunks_indexed']} chunks)")
                except json.JSONDecodeError as exc:
                    st.error(f"Invalid JSON: {exc}")
            if c2.button("Approve as-is", key=f"asis_{item['doc_id']}"):
                api("POST", f"/documents/{item['doc_id']}/approve")
                st.success("Approved")

# --------------------------------------------------------------------------- #
with tab_search:
    q = st.text_input("Semantic query", "contract governed by the laws of New York")
    c1, c2 = st.columns([1, 3])
    dt = c1.selectbox("Type filter", ["(any)"] + DOC_TYPES)
    k = c2.slider("Top K", 1, 15, 5)
    if st.button("Search", disabled=not q):
        body = {"query": q, "top_k": k}
        if dt != "(any)":
            body["doc_type"] = dt
        res = api("POST", "/search", json=body)
        for h in res["results"]:
            st.markdown(
                f"**#{h['rank']}** · `{h['doc_id']}` · {h['doc_type']} · score {h['score']}"
                f"{'  · ' + h['section'] if h.get('section') else ''}"
            )
            st.write(h["text"][:600] + ("…" if len(h["text"]) > 600 else ""))
            st.divider()

# --------------------------------------------------------------------------- #
with tab_metrics:
    try:
        m = api("GET", "/metrics")
    except Exception as exc:  # noqa: BLE001
        m = {"extraction": {}, "retrieval": {}}
        st.error(str(exc))
    ex, rt = m.get("extraction", {}), m.get("retrieval", {})
    if not ex:
        st.info("Run scripts/eval_extraction.py to populate metrics.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Field accuracy", f"{ex.get('field_accuracy_overall', 0):.1%}")
        b.metric("Classification", f"{ex.get('classification_accuracy', 0):.1%}")
        c.metric("Manual-verif. reduction", f"{ex.get('manual_verification_reduction_pct', 0):.0f}%")
        d.metric("Top-5 retrieval", f"{rt.get('recall_at_5', 0):.1%}")
        st.caption(
            f"Accuracy in auto-accepted: {ex.get('field_accuracy_auto_accepted', 0):.1%} "
            f"· in review queue: {ex.get('field_accuracy_needs_review', 0):.1%} "
            f"· calibration ECE: {ex.get('calibration_ece', 0):.3f}"
        )
        st.write("**Field accuracy by type**")
        st.dataframe(
            [{"type": k, "accuracy": f"{v:.1%}"} for k, v in
             ex.get("field_accuracy_by_type", {}).items()],
            use_container_width=True, hide_index=True,
        )
        cal = os.path.join(os.environ.get("DIP_ARTIFACTS", "artifacts"), "calibration.png")
        if os.path.exists(cal):
            st.image(cal, caption="Confidence calibration", width=420)
