# Enterprise Document Intelligence Platform — Build Plan

**Timeline:** 1 day, solo. **Compute:** Google Colab (T4 GPU) for heavy stages, HP EliteBook 840 G6 for the Dockerized app/demo.

---

## 1. Guiding decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| LLM serving | **Ollama + Qwen2.5-3B-Instruct (Q4_K_M)** | CPU-capable local story; run on Colab GPU for speed during eval — identical model/quantization, so numbers transfer honestly |
| Dev model | Hosted **Qwen2.5** (Together / Fireworks / OpenRouter) OR Colab-Ollama | Same model family as local → prompt engineering stays valid |
| LLM interface | One OpenAI-compatible client, 3 env vars (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`) | No branching code between hosted / Colab / laptop |
| Doc types | Invoice, Purchase Order, Medical Report, Contract (30 each = 120) | Matches the spec; invoices carry the richest eval |
| Data | SROIE (invoices), CUAD (contracts), synthetic (PO + medical) | All ship or generate ground truth — no manual labeling |
| Privacy | Synthetic / public only | No real PHI or confidential contracts |
| Confidence signal | **Token log-probability aggregation** per field (`min` / `mean` of `exp(logprob)` over value tokens); fallback = 2-sample self-consistency | Defensible, sophisticated, enables a calibration plot |
| Human-in-the-loop | Status flag + Streamlit review/edit → re-run `validate → index` | Real HITL without LangGraph checkpointer time sink |
| Orchestration | LangGraph `StateGraph`, conditional routing on confidence + rules | Genuine graph, keyword satisfied |
| Retrieval | BGE-M3 **dense** → Qdrant; hybrid sparse + RAG Q&A = stretch | Core path first |
| Infra | `docker-compose`: Postgres + Qdrant on laptop; FastAPI + Streamlit on host (full-stack compose documented) | Docker story real, iteration fast |
| Async | FastAPI `BackgroundTasks` | No Celery in a day |

---

## 2. Target metrics (report whatever they actually are)

| Claim | Definition | Method |
|---|---|---|
| ~89% field-level extraction accuracy | normalized match vs ground truth (`correct / (correct+incorrect+missing)`), averaged over fields; per-type + overall | one Ollama pass over 120 docs |
| ~55% manual-verification reduction | `auto_accepted / total`, where auto-accepted = all critical rules pass AND `doc_confidence ≥ doc_threshold` AND no field `< field_threshold` | tune thresholds so rate ≈ 0.5–0.6 while auto-accepted accuracy > overall |
| ~84% Top-5 retrieval accuracy | recall@5 over ~35 hand-written `{query, relevant_doc_ids}` pairs | `eval_retrieval.py` |
| 1,500+ indexed chunks | 120 docs × ~12 structure-aware chunks | BGE-M3 → Qdrant |

Plus a **confidence calibration plot** (confidence decile vs. actual field accuracy).

---

## 3. Repository layout

```
document_enterprise_platform/
├── docker-compose.yml            # postgres, qdrant (+ api, ui profiles)
├── .env.example
├── requirements.txt
├── README.md
├── BUILD_PLAN.md
├── config/
│   ├── pipeline.yaml             # LLM/embed endpoints, thresholds, chunking defaults
│   ├── examples/                 # few-shot files per type
│   └── doc_types/
│       ├── invoice.yaml
│       ├── purchase_order.yaml
│       ├── medical_report.yaml
│       └── contract.yaml
├── src/dip/
│   ├── config.py                 # load + validate YAML + env
│   ├── schemas/                  # Pydantic model per doc type (+ base.py)
│   ├── llm/
│   │   ├── client.py             # OpenAI-compatible wrapper
│   │   └── extract.py            # constrained JSON + logprob capture + retry
│   ├── parsing/
│   │   ├── docling_parser.py     # parse → structured doc + text, cache
│   │   └── chunker.py            # layout-section chunks
│   ├── validation/
│   │   ├── rules.py              # per-type rule functions
│   │   └── confidence.py         # logprob → field confidence; self-consistency fallback
│   ├── pipeline/
│   │   ├── state.py              # DocState TypedDict
│   │   ├── graph.py              # StateGraph wiring
│   │   └── nodes/               ingest, parse, classify, extract, confidence, validate, route, index
│   ├── retrieval/
│   │   ├── embed.py             # BGE-M3
│   │   ├── store.py             # Qdrant client, collection, snapshot/restore
│   │   └── search.py
│   ├── db/  models.py, session.py     # SQLAlchemy, create_all (no Alembic today)
│   ├── api/  main.py                  # FastAPI
│   └── ui/   app.py                   # Streamlit (4 tabs)
├── scripts/
│   ├── fetch_data.py            # SROIE + CUAD subsets
│   ├── gen_synthetic.py        # PO + medical PDFs + ground-truth JSON
│   ├── prepare_eval.py        # → data/eval/manifest.jsonl
│   ├── run_pipeline_batch.py  # run graph over corpus (Colab)
│   ├── build_index.py         # embed + upsert + snapshot (Colab)
│   ├── eval_extraction.py     # field accuracy + review-reduction + calibration
│   └── eval_retrieval.py      # recall@5 / MRR@5
├── notebooks/colab_runner.ipynb
├── data/  raw/  processed/  cache/  eval/
└── artifacts/  metrics.json  eval_report.md  calibration.png  qdrant_snapshot/
```

---

## 4. Data plan (120 docs, 30 per type)

| Type | Source | Fields extracted | Ground truth |
|---|---|---|---|
| Invoice | SROIE v2 test subset (30) | `company, date, address, total` (+ optional line items from CORD) | SROIE `entities` files |
| Contract | CUAD subset (30 agreements) | `document_name, parties, agreement_date, effective_date, expiration_or_term, governing_law, renewal_term` | CUAD master CSV |
| Purchase Order | synthetic (`gen_synthetic.py`, Faker + reportlab/HTML→PDF) | `po_number, order_date, vendor, buyer, line_items[], subtotal, tax, total, delivery_date` | generation params → JSON |
| Medical Report | synthetic radiology/lab template | `patient_id (fake), report_date, ordering_physician, modality, findings, impression, diagnoses[]` | generation params → JSON |

Eval manifest: `data/eval/manifest.jsonl` → `{doc_id, doc_type, file_path, ground_truth: {...}}`.
Retrieval queries: `data/eval/retrieval_queries.jsonl` → `{query, relevant_doc_ids: [...]}` (~35, spanning all types).

---

## 5. Config-driven pipeline (the "configurable" story)

Adding a doc type = **new YAML + new Pydantic class + optional few-shot file. No pipeline code.**

`config/doc_types/invoice.yaml` (shape):
```yaml
doc_type: invoice
schema: dip.schemas.invoice.Invoice
classifier_hint: "Vendor invoice / bill requesting payment; has invoice number, totals, line items"
extraction:
  strategy: single_pass          # or field_by_field for long docs (contracts)
  max_output_tokens: 700
  few_shot: config/examples/invoice.json
validation:
  critical_rules: [required_fields, total_is_positive]
  rules:
    - required_fields: [company, date, total]
    - line_items_sum_matches_subtotal
    - subtotal_plus_tax_matches_total
    - date_is_valid
  confidence:
    method: logprob_min          # logprob_min | logprob_mean | self_consistency
    field_threshold: 0.55
    doc_threshold: 0.70
chunking: { strategy: layout_section, max_tokens: 512, overlap: 64 }
```

`config/pipeline.yaml`: LLM endpoint/model, embed model, Qdrant collection/URL, `top_logprobs`, default thresholds, review routing defaults.

---

## 6. LangGraph pipeline

**State** (`state.py`):
```python
class DocState(TypedDict):
    doc_id: str; file_path: str
    raw_text: str | None
    structured_doc: dict | None          # Docling output
    doc_type: str | None; doc_type_confidence: float | None
    extraction: dict | None              # schema-conformant
    field_confidences: dict | None       # {field: prob}
    doc_confidence: float | None
    validation: dict | None              # {rule: {passed, message}}
    status: Literal["processing","auto_accepted","needs_review","indexed","failed"]
    errors: list[str]
```

**Nodes:**
1. `ingest` — register `documents` row; load parse cache if present.
2. `parse` — Docling → structured doc + text; cache `data/cache/{doc_id}.json`.
3. `classify` — Qwen zero-shot over first ~1500 chars + all `classifier_hint`s → `{type, confidence}`; keyword-rule fallback if confidence low.
4. `extract` — load type's Pydantic → JSON schema → LLM call with constrained decoding (`format`/JSON schema), few-shot, `logprobs=true, top_logprobs=5`; parse → Pydantic validate; **1 retry** with error feedback on failure.
5. `confidence` — map generated tokens → JSON value spans → per field `min` and `mean` of `exp(logprob)` (exclude structural tokens); `doc_confidence = mean(field_confidences)`.
6. `validate` — run type's rule functions; collect pass/fail + messages.
7. `route` (conditional edge) — `auto_accepted` iff all `critical_rules` pass AND `doc_confidence ≥ doc_threshold` AND `min(field_confidences) ≥ field_threshold`; else `needs_review`.
8. `index` — chunk (structure-aware) → BGE-M3 → Qdrant upsert, payload `{doc_id, doc_type, section, text, status}`. Runs for `auto_accepted` and post-review docs.
9. Persist all artifacts to Postgres at each step.

**HITL:** `needs_review` docs appear in Streamlit → user edits fields → `PUT /documents/{id}/extraction` re-runs `validate → index` only. Status flag; no checkpointer.

---

## 7. Confidence via logprobs (detail)

- Ollama build **with logprobs support**; OpenAI-compatible `/v1/chat/completions` with `logprobs=true, top_logprobs=5`. **Verify in Phase 0.**
- Reconstruct JSON from returned tokens with offsets → for each schema field locate value substring → covering token indices → `field_conf_min = min(exp(logprob))`, `field_conf_mean = mean(exp(logprob))`.
- `confidence.method` in config selects `logprob_min` / `logprob_mean` / `self_consistency`.
- **Fallback** (`self_consistency`): 2 runs at temp 0.3, field agreement fraction.
- Eval reuses stored logprobs → threshold tuning needs **no re-inference**.
- README: calibration plot (confidence decile vs. field accuracy) + short note that logprob confidence tracks correctness.

---

## 8. Retrieval

- **Embed:** BGE-M3 dense (1024-d), `FlagEmbedding` or `sentence-transformers`, batched on Colab GPU.
- **Qdrant:** collection `documents`, cosine, payload as above. 120 × ~12 ≈ 1,400–1,800 chunks.
- **Search:** embed query → top-k → return chunks + parent doc + extraction snapshot. Optional `doc_type` filter.
- **Snapshot** collection on Colab → **restore** on laptop (skip re-embed).
- **Stretch:** BGE-M3 sparse → Qdrant hybrid (named vectors + fusion); `/ask` RAG endpoint (retrieve top-5 → Qwen answer + citations).

---

## 9. Postgres schema

- `documents(id, filename, doc_type, doc_type_confidence, status, file_path, created_at)`
- `extractions(id, document_id, schema_version, payload jsonb, doc_confidence, source['model'|'human'], created_at)`
- `field_confidences(id, extraction_id, field_name, confidence, method)`
- `validation_results(id, extraction_id, rule_name, passed, message)`
- `review_queue(id, document_id, reason, resolved, resolved_at)`
- `audit_log(id, document_id, event, detail jsonb, ts)`

`create_all` on startup — no migrations today.

---

## 10. FastAPI endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/documents` | upload → background graph run → `202 {doc_id}` |
| GET | `/documents/{id}` | status + extraction + field confidences + validation |
| GET | `/documents?status=needs_review` | review queue |
| PUT | `/documents/{id}/extraction` | human correction → re-validate + re-index |
| POST | `/documents/{id}/approve` | approve as-is |
| POST | `/search` | `{query, top_k, doc_type?}` → hits |
| POST | `/ask` | *(stretch)* RAG answer + citations |
| GET | `/metrics` | serve `artifacts/metrics.json` |
| GET | `/health` | liveness |

---

## 11. Streamlit UI (4 tabs)

1. **Upload & Process** — drop file, live status, extracted JSON with per-field confidence (color-coded), validation results.
2. **Review Queue** — `needs_review` table → editable form pre-filled with model output + flagged low-confidence fields → Approve → re-index.
3. **Search** — query box + `doc_type` filter → ranked chunks with parent doc + extraction preview (RAG answer box if built).
4. **Metrics** — field accuracy (overall / per type / per field), auto-accept & review-reduction rate, retrieval recall@5, calibration plot.

---

## 12. Execution order

**Phase 0 — repo + infra (~45 min)**
scaffold · `requirements.txt` · `docker-compose.yml` · `.env.example` · `config.py` · base schemas · LLM client · `docker compose up postgres qdrant` · `create_all` · **verify Ollama logprobs on a hosted/Colab endpoint**.

**Phase 1 — data (~60 min)**
`fetch_data.py` (SROIE 30, CUAD 30 + CSV) · `gen_synthetic.py` (PO 30, medical 30 + GT) · `prepare_eval.py` → `manifest.jsonl` · sanity-check counts + GT keys.

**Phase 2 — parsing + pipeline (~2.5 h)**
`docling_parser.py` + cache · `chunker.py` · 4 Pydantic schemas + few-shot files · LangGraph nodes + `graph.py` · `confidence.py` (logprob) · `rules.py` · run graph on 3–4 sample docs end-to-end, fix.

**Phase 3 — Colab heavy run (~45 min)**
`colab_runner.ipynb`: clone repo · `pip install` · install + start Ollama · `ollama pull qwen2.5:3b-instruct-q4_K_M` · mount Drive · `run_pipeline_batch.py` over 120 → results JSON to Drive · `build_index.py` → BGE-M3 → Qdrant → snapshot to Drive.

**Phase 4 — eval (~30 min)**
`eval_extraction.py` + `eval_retrieval.py` → `metrics.json` + `eval_report.md` + `calibration.png` · tune thresholds · re-score (reuse stored logprobs).

**Phase 5 — API + UI (~1.5 h)**
FastAPI endpoints · restore Qdrant snapshot on laptop · load artifacts · Streamlit 4 tabs · live-process 2 fresh docs via **laptop Ollama** to prove the local path.

**Phase 6 — package (~45 min)**
README (architecture diagram, quickstart, results tables, screenshots, "add a doc type" guide, honest limitations) · full-stack compose profile · screenshots via the app · push repo.

**Buffer ~1 h.**

---

## 13. Cut list if behind (in order)

1. `/ask` RAG endpoint → drop (retrieval-only).
2. Hybrid sparse retrieval → dense only.
3. PO + medical → 20 each (100 docs total).
4. Streamlit Metrics tab → `eval_report.md` in README only.
5. Full-stack compose → pg + qdrant only, app on host (documented).
6. Calibration plot → table only.

---

## 14. Definition of done

- `docker compose up` + documented commands bring up a working demo on the laptop.
- Upload → classify → extract → validate → route → search works end-to-end against **local Ollama**.
- `artifacts/metrics.json` holds real numbers: field accuracy (overall + per type), auto-accept / review-reduction rate, retrieval recall@5, chunk count.
- README with architecture, results, screenshots, add-a-type guide, limitations.
- Repo initialized and pushed.

---

## 15. Known risks

| Risk | Mitigation |
|---|---|
| Colab session timeout / GPU quota | checkpoint every stage to Drive; Colab Pro if possible; reduced-eval fallback on laptop Ollama |
| Ollama logprobs unsupported/misaligned | `self_consistency` fallback already in config |
| Qwen2.5-3B weak on long contracts | `field_by_field` strategy + retrieval-guided prompts for contract type |
| Docling slow on scanned SROIE images | parse once, cache; use SROIE-provided OCR text if Docling OCR too slow |
| Tunnel dies before demo | pre-cache all results; laptop Ollama serves live uploads |
