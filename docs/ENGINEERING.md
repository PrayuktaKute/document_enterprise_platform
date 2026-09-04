# Engineering Deep-Dive

Enterprise Document Intelligence Platform — a configurable pipeline that ingests business
documents, classifies them, extracts structured data against a per-type schema, validates
the extraction with business rules and a model-confidence signal, routes low-confidence
results to human review, and indexes everything for semantic retrieval.

This document explains what the system does, **why every non-trivial decision was made**,
what the alternatives were, how it is evaluated, where it falls short, and what a
production version would add. It is written to be read cover-to-cover before a technical
interview at a document-AI company.

- **Repo:** https://github.com/PrayuktaKute/document_enterprise_platform
- **Stack:** Python 3.11 · FastAPI · LangGraph · Qwen2.5-3B (Ollama) · Docling · BAAI BGE-M3 · Qdrant · PostgreSQL · Docker · Streamlit
- **Headline results (120-doc eval, local Qwen2.5-3B):** 82.7% field-level extraction accuracy · 94.2% classification · 65% documents auto-accepted (91% field accuracy inside that set) · 82.2% Top-5 retrieval recall · confidence-calibration ECE 0.078

---

## 1. Problem statement and scope

**Document intelligence** = turning unstructured business documents into structured,
queryable data with enough reliability that downstream automation can trust it. The hard
parts are:

1. **Heterogeneity** — invoices, purchase orders, medical reports and contracts have
   nothing in common structurally. A single extraction prompt cannot serve all of them.
2. **Layout** — the information is 2-D (tables, columns, headers, key-value blocks), but
   LLMs consume 1-D token streams. Something has to linearise layout without destroying it.
3. **Trust** — an extraction that is 85% correct is useless for straight-through
   processing unless you know *which* 15% is wrong. The system needs a per-field
   reliability signal, not just an answer.
4. **Long documents** — a 40-page contract does not fit a small model's context usefully;
   naive truncation loses the governing-law clause on page 30.
5. **Change** — a real deployment adds a new document type every few weeks. That must not
   require touching pipeline code.

**Scope of this build:** four document types, a 120-document labelled evaluation set, a
local small model (Qwen2.5-3B) so the whole thing runs offline, a working API + UI, and a
reproducible evaluation harness. Out of scope: multi-tenancy, auth, horizontal scaling,
active learning, and a fine-tuned extraction model — all discussed in §10.

---

## 2. High-level architecture

```
                 ┌──────────────┐        ┌──────────────────────────────────────────────┐
  upload  ─────► │  FastAPI     │ ─────► │  LangGraph StateGraph (one run per document)  │
  (PDF/img/txt)  │  + Background │        │                                              │
                 │    Tasks     │        │  ingest → parse → classify → extract →        │
                 └──────┬───────┘        │            validate → route ──┐              │
                        │                │                               │              │
                        │                │      auto_accept ┌────────────┴─────┐ needs_review
                        │                │                  ▼                  ▼          │
                        │                │               index            (mark)         │
                        │                └────────────────┬─────────────────┬────────────┘
                        │                                 │                 │
                        ▼                                 ▼                 ▼
                 ┌──────────────┐                  ┌────────────┐   ┌──────────────┐
   Streamlit ◄── │  PostgreSQL  │                  │   Qdrant   │   │ review_queue │
   (4 tabs)      │  6 tables    │                  │  BGE-M3    │   │  (Postgres)  │
                 └──────────────┘                  │  vectors   │   └──────────────┘
                                                   └────────────┘
```

**Control flow:** `POST /documents` writes the file, inserts a `documents` row with
`status="processing"`, and schedules a background task. The task calls
`run_document(doc_id, path)` which invokes the compiled LangGraph. The graph's terminal
node returns a `DocState`; `persist_state(state)` writes one `extractions` row plus its
`field_confidences`, `validation_results`, an optional `review_queue` entry, and an
`audit_log` line. The UI polls `GET /documents/{id}`.

**Why a graph and not a function call chain.** The pipeline has a real branch (auto-accept
vs. review) and each node has a uniform contract — `DocState -> partial DocState` — which
makes nodes independently testable, independently swappable, and trivially observable
(every transition is a checkpoint). LangGraph was chosen over:

| Alternative | Why not |
|---|---|
| Plain Python functions | Works, but you re-implement conditional routing, state merging, and per-step tracing by hand; no standard place to add checkpointing later. |
| Celery / RQ task chain | Right tool for distributed scale, wrong tool for a single-document DAG that finishes in seconds. Adds a broker dependency for no benefit at this size. |
| Airflow / Prefect / Dagster | Batch/ETL orchestrators; per-document latency and dynamic branching are not their model. |
| An agent loop (ReAct etc.) | The pipeline is deterministic — fixed stages, no tool-selection reasoning. An agent would add nondeterminism and cost with zero upside. |

LangGraph also gives a clean upgrade path: the human-in-the-loop step (§4.8) is currently
a status flag, but swapping it for LangGraph's `interrupt()` + a Postgres checkpointer is
a localised change because the graph is already the unit of execution.

---

## 3. The configuration system — "configurable pipeline"

**`src/dip/config.py`.** Three layers, all Pydantic-validated:

1. **`Settings`** (`pydantic-settings`, read from `.env`) — endpoints and paths only:
   `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `EMBED_MODEL`, `DATABASE_URL`, `QDRANT_URL`,
   `QDRANT_PATH`, `QDRANT_COLLECTION`, `CONFIDENCE_METHOD`, `DATA_DIR`, `ARTIFACTS_DIR`,
   `CACHE_DIR`.
2. **`pipeline.yaml`** → `PipelineConfig` — global call parameters: LLM temperature / max
   tokens / `top_logprobs` / retry budget / text budgets, embedding model + batch size +
   dimension, Qdrant collection + distance, default chunking params, classification
   excerpt length + fallback threshold, confidence defaults (`method`,
   `default_field_threshold`, `default_doc_threshold`, `max_low_conf_fields`,
   self-consistency knobs), review routing toggle.
3. **`config/doc_types/<type>.yaml`** → `DocTypeConfig` (one per document type):
   - `schema:` — a **dotted path** (e.g. `dip.schemas.invoice.Invoice`) resolved at
     runtime via `importlib.import_module` + `getattr`. The YAML key is `schema` but the
     Pydantic field is `schema_path` with `alias="schema"` and `populate_by_name=True`,
     because `schema` collides with a deprecated `BaseModel` method name.
   - `classifier_hint:` — a natural-language description injected into the zero-shot
     classifier prompt.
   - `extraction:` — `strategy` (`single_pass` | `field_by_field`), `max_output_tokens`,
     `few_shot` (path to an examples file), `hint` (extra system-prompt guidance), and
     `field_groups` (for `field_by_field` — see §4.4).
   - `validation:` — `critical_rules` (names that gate auto-accept), `rules` (each a bare
     name or a single-key mapping `{rule_name: args}`), and a per-type `confidence`
     override block.
   - `chunking:` — `strategy`, `max_tokens`, `overlap`.

`resolve_confidence(doc_type)` merges the per-type `confidence` block over the pipeline
defaults and returns `(method, field_threshold, doc_threshold)`.

**Adding a document type is:** drop a `<type>.yaml`, add an `ExtractionBase` subclass in
`dip/schemas/`, optionally add a few-shot file. **No pipeline code changes.** The graph,
the confidence scorer, the validator, the chunker, the API and the UI all read the type
from config.

**Why YAML + Pydantic rather than a database of configs or a rules DSL.** For four types
edited by the developer, a file per type that lives in git (reviewable, diffable,
rollback-able) beats a config table. Pydantic gives schema validation and typo detection
on load. A full rules DSL would be over-engineering — the "rules" here are a fixed library
of Python functions selected and parameterised by name (§4.6), which covers the real
business checks without inventing a language.

---

## 4. Component deep-dive

### 4.1 LLM client — one interface, three back-ends

**`src/dip/llm/client.py`.** `LLMClient` wraps `openai.OpenAI` pointed at
`LLM_BASE_URL`. All three back-ends we use speak the OpenAI Chat Completions protocol:

| Back-end | When | Notes |
|---|---|---|
| Ollama on a Colab T4 GPU (via tunnel) | building + running the eval | identical GGUF model to local, ~10× faster |
| Ollama on the laptop (`localhost:11434/v1`) | the "runs fully local" demo | ~75 s/doc on CPU |
| OpenRouter free tier | documented fallback | if no GPU is available |

Switching back-ends is three environment variables and **zero code**. This is deliberate:
prompt engineering done against a hosted 7B would be invalid when you drop to a local 3B,
so development uses the *same model family* the eval and the local demo use.

**Retry.** The SDK's own retry is disabled (`max_retries=0`) and `tenacity`
(`stop_after_attempt(3)`, exponential backoff 1–20 s) wraps the call, so retry policy is
in one place and observable.

**Log-probabilities.** `chat(..., logprobs=True, top_logprobs=N)` sets the OpenAI-compat
fields; the response's `choices[0].logprobs.content` is a list of
`{token, logprob, top_logprobs}`. We parse it into `list[TokenLogprob]` on the
`LLMResponse`. Ollama 0.33 returns this; if a back-end does not, `LLMResponse.has_logprobs`
is `False` and the confidence layer falls back (§4.5).

**Structured output.** `pipeline.yaml: llm.use_json_schema` toggles between
`response_format={"type": "json_schema", ...}` (strict, not universally supported) and
`response_format={"type": "json_object"}` + the schema pasted into the prompt (default —
most compatible with Ollama). We do **not** rely on constrained decoding / grammars
because Ollama's support is uneven; instead the parser is defensive (§4.4).

### 4.2 Parsing — Docling, with a cache

**`src/dip/parsing/docling_parser.py`.** `parse_document(file_path, doc_id)` returns a
`ParsedDoc(doc_id, source_path, source_format, parser, text, sections, page_count)` and
**caches it as JSON** at `data/cache/{doc_id}.json` — Docling's layout + OCR models are
expensive to run and the eval re-parses the corpus several times.

- **PDF / image** → Docling `DocumentConverter` (lazily constructed, `lru_cache`d because
  it loads models). We take `document.export_to_markdown()` — Markdown preserves the
  useful layout signal (headings become `#`, tables become pipe tables) in a form a
  language model reads well. Sections are split on Markdown headings.
- **`.txt`** (CUAD contracts) → bypass Docling entirely; a regex heading detector
  (`ARTICLE X`, `SECTION N`, numbered clauses, ALL-CAPS lines) splits the text into
  sections.

**Why Docling.** It is layout-aware (table structure recovery, reading-order), ships its
own OCR (RapidOCR / EasyOCR), handles PDF + images + office formats behind one API, and
outputs Markdown directly. Alternatives considered:

| Alternative | Trade-off |
|---|---|
| `unstructured` | Similar surface; Docling's table-structure model and Markdown export were cleaner for this corpus. |
| PyMuPDF / pdfplumber + Tesseract | Fast and light, but you rebuild reading-order and table reconstruction yourself, and there is no image pipeline. Kept as the mental fallback for a 3.14-only environment. |
| LayoutLMv3 / Donut / a VLM (e.g. Qwen2-VL) | A document-understanding model that consumes pixels directly is the *right* long-term answer (no OCR-error compounding), but it is a much bigger model and a different serving story. Noted in §10. |

**Known cost:** OCR on the SROIE receipt images is the slowest stage and its errors
propagate into extraction — this is the main reason invoice accuracy is capped at ~64%
(§6).

### 4.3 Classification

**`src/dip/pipeline/classify.py`.** `classify_document(text)`:

1. Zero-shot LLM call: system prompt lists every type id with its `classifier_hint`; user
   message is the first `classification.first_n_chars` (1500) characters. Model returns
   `{"doc_type": "<id>", "confidence": <0..1>}`.
2. If the returned type is unknown **or** confidence `< low_confidence_fallback` (0.45),
   fall back to keyword scoring: count occurrences of a curated keyword list per type and
   take the argmax.

**Why LLM-first with a keyword safety net rather than an embedding classifier or a
fine-tuned head.** The four types are lexically and structurally distinct; a 3B model
nails this at **94.2%** zero-shot with no training data and no separate model to maintain.
The keyword fallback costs nothing and catches the rare confident-but-wrong or
low-confidence case. An embedding + logistic-regression classifier would need labelled
training data and a model artifact for a problem the LLM already solves; it becomes worth
it only at dozens of types with subtle boundaries (e.g. "invoice" vs. "credit note" vs.
"statement").

The classifier's own confidence is stored on `documents.doc_type_confidence` and shown in
the UI.

### 4.4 Schema-driven extraction

**`src/dip/schemas/*.py`.** One Pydantic model per type, subclassing `ExtractionBase`
(`ConfigDict(extra="ignore", str_strip_whitespace=True)` — tolerate stray keys, don't let
one bad field fail the whole parse). Each field carries a `description` used both for the
JSON-schema prompt and as human documentation. `extraction_fields()` returns the ordered
field list.

**`src/dip/llm/extract.py`.** `extract_fields(parsed, doc_type)`:

1. Resolve schema class, confidence method (`logprob_min` / `logprob_mean` /
   `self_consistency`), and whether logprobs are wanted.
2. **Build extraction groups:**
   - `single_pass` (invoice, PO, medical): one group `(all_fields, text[:budget])`.
     Budget is 12 k chars default, 18 k for contracts (`pipeline.yaml`).
   - `field_by_field` (contract): one group per `field_groups` entry in the YAML. A group
     is `{fields, source, ...}`:
     - `source: head` → context = `text[:max_chars]` (title, parties, dates live at the
       top).
     - `source: keywords` → `_keyword_context(text, keywords, max_chars, window)`: find
       every occurrence of each keyword, take a `±window`-char slice around it, merge
       overlapping slices, concatenate until `max_chars`. This is a **cheap
       retrieval-guided prompt** — it puts the governing-law and renewal clauses in front
       of the model even when they are on page 30.
   - Any fields not covered by a group get a final catch-all group.
3. **Per group** (`_call_group`): build a *sub-schema* (`_sub_schema` filters the full
   JSON schema to the group's fields); system prompt states the exact key list, "use null
   when not stated, do not invent", ISO-8601 dates, plus the type's `hint`; call the LLM
   with logprobs. Parse with `_extract_json` (strip ```` ```json ```` fences, take the
   outermost `{...}`, and if `json.loads` fails, retry after stripping trailing commas).
   On failure, **one** retry with the bad output and the parser error fed back. Compute
   per-field confidence for that group's fields.
4. **Merge** all groups' dicts and confidence dicts.
5. **Coerce** the merged dict through the *full* Pydantic model (`model_validate`) so
   dates, numbers and list shapes are normalised; on `ValidationError`, keep the raw
   values and record the error.
6. `doc_confidence = mean(field_confidences)`.

**Why `field_by_field` for contracts.** Single-pass on 18 k characters of legalese made
Qwen2.5-3B return all-nulls or malformed JSON — contract field accuracy was **~13%**.
Splitting into a small header call (5 fields, 6 k chars) plus keyword-windowed calls (3–4
fields, ~6 k chars each) took it to **~54%**. Each call is short, the JSON is small, and
the model sees exactly the clause it needs. The generalisation — groups declared in YAML
with `head` or `keyword` context — means any long-document type can opt in without code.

**Why not constrained decoding / function-calling for the JSON.** Reliability of
grammar-constrained output on Ollama is inconsistent across model/version; a robust
extractor + a lenient parser + a one-shot repair retry proved more dependable and is
back-end-agnostic. In production with vLLM you would add Outlines/XGrammar and delete the
repair path.

### 4.5 Confidence scoring — the differentiator

**`src/dip/validation/confidence.py`.** The core question: *for each extracted field, how
much should we trust it?* The primary signal is **token log-probability aggregation over
the field's value span**.

**Algorithm (`field_confidences_from_logprobs`):**

1. The LLM response carries a token list with a `logprob` per generated token.
   Reconstruct the output string by concatenating the token strings; build a parallel
   list of `(char_start, char_end)` offsets by accumulating token lengths.
2. For each schema field, locate its **value span** in the JSON string
   (`_value_span`): find `"field"`, skip to the `:`, then walk forward tracking string
   state and `[]`/`{}` nesting depth until a top-level `,` or `}` — this correctly
   captures scalars, arrays and nested objects.
3. Collect every token whose `[start,end)` overlaps the value span. Drop tokens that are
   purely structural (`{ } [ ] : , "` and whitespace). Convert the survivors to
   probabilities with `exp(logprob)`.
4. Aggregate: `logprob_min` = the least-confident token in the value (a single uncertain
   token — a mis-OCR'd digit, a hallucinated name — drags the field down); `logprob_mean`
   = the average. `min` is the default because for extraction one wrong token usually
   means the whole field is wrong.
5. Special cases: **key missing** from the output → `0.0` (the model declined to answer);
   **value is literal `null`** → `0.6` (weak-but-not-damning — the field may genuinely be
   absent).

`doc_confidence` = mean of the field confidences. Per-field confidences and the method
name are persisted in `field_confidences`.

**Why this signal.** It is:
- **Free** — no extra model calls, no extra latency; the logprobs come back with the
  answer.
- **Per-field, not per-document** — it tells you *which* fields to check, which is what
  makes selective human review possible.
- **Calibrated** — on the eval set the confidence deciles track observed accuracy well
  (Expected Calibration Error **0.078**): fields the model reports at ~0.99 are right ~95%
  of the time; the 0.5–0.7 band is where errors concentrate. That relationship is what the
  auto-accept gate exploits.

**Fallback: self-consistency.** If a back-end returns no logprobs (or the config asks for
it), `field_confidences_from_samples` runs the extraction 2× more at temperature 0.3 and
sets each field's confidence to the fraction of samples that agree with the primary
answer. More expensive (3× the calls) but back-end-independent.

**Alternatives considered:**

| Approach | Why not primary |
|---|---|
| "Ask the model for a confidence score" | LLM self-reported confidence is poorly calibrated and trivially gamed; adds output tokens. |
| Verifier / LLM-as-judge second pass | Doubles cost and latency; the judge has the same blind spots as the extractor. Useful as an *additional* gate, not the base signal. |
| Ensemble / temperature sampling only | This is the self-consistency fallback — kept, but 3× cost. |
| Entropy of `top_logprobs` instead of chosen-token prob | Roughly equivalent signal; chosen-token `exp(logprob)` is simpler to explain and aggregate. |

### 4.6 Rule-based validation

**`src/dip/validation/rules.py`.** A fixed library of business checks, each a function
`extraction -> RuleOutcome(rule_name, passed, is_critical, message)`. `run_rules` reads
the type's `rules:` list (bare name, or `{name: args}` for parameterised rules like
`required_fields`) and dispatches. Implemented rules:

- `required_fields([...])` — listed fields present and non-empty.
- `total_is_positive` — numeric `total > 0`.
- `date_is_valid` — every `*date*` field parses under one of ~10 formats (or none present).
- `line_items_sum_matches_subtotal` — `sum(item.amount) ≈ subtotal`, tolerance
  `max(0.05, 2%)`.
- `subtotal_plus_tax_matches_total`.
- `delivery_after_order_date`.
- `at_least_two_parties` — contract has ≥ 2 parties (string is split on `;`/`,`/`and`).
- `impression_present` — medical report has a non-empty impression.

Number parsing strips currency symbols and thousands separators; date parsing is
multi-format. Rules named in the type's `critical_rules` are marked `is_critical`; a failed
critical rule blocks auto-accept regardless of confidence.

**Why a curated function library, not a DSL or an LLM validator.** These checks are
arithmetic and format invariants — deterministic, fast, explainable, and exactly what
domain experts ask for ("do the line items add up?"). An LLM validator would be slower,
non-deterministic, and no more capable here. A rules DSL would be a language to maintain
for ~8 rules. The library is extended by adding a function and referencing it by name in
YAML.

### 4.7 Routing — the auto-accept gate

**`validate_node` in `src/dip/pipeline/nodes.py`.** A document is **auto-accepted** iff:

```
no critical rule failed
AND doc_confidence >= doc_threshold            (per-type, default 0.70)
AND (# fields with confidence < field_threshold) <= max_low_conf_fields   (default 1)
```

The third clause replaced an earlier `min(field_confidence) >= field_threshold`, which was
too strict: any optional field the model correctly left null cratered to 0.0 and forced
every document into review, making the "reduce manual verification" goal unreachable.
Allowing a small budget of low-confidence fields (default 1) is the knob that trades
**auto-accept rate** against **error rate inside the auto-accepted set**. On the eval:
65% auto-accepted, and field accuracy inside that set is **90.8%** vs **56.5%** in the
review queue — the gate is doing real work separating reliable from unreliable.

`route_decision` maps the boolean to the `"index"` or `"review"` edge.

### 4.8 Human-in-the-loop

Documents that fail the gate get `status="needs_review"` and a `review_queue` row with a
reason. The Streamlit **Review Queue** tab lets a reviewer pick a document, see its
per-field confidences (low ones flagged), the rule outcomes, and an **editable JSON**
extraction. "Approve corrected" → `PUT /documents/{id}/extraction` → `resolve_review`
writes a new `extractions` row with `source="human"`, closes the queue item, sets
`status="indexed"`, and re-indexes the document.

**Design choice: status flag, not LangGraph `interrupt()` + checkpointer.** A true
suspend/resume of the graph mid-run (persist the `DocState`, halt, resume on human input)
is more elegant and is the production answer, but it needs a Postgres checkpointer, a
resume endpoint, and careful state-versioning. For this build the flag + re-validate/-index
path delivers the same user-visible behaviour (queue, edit, approve, re-index) with far
less surface area. The graph being the unit of execution keeps the upgrade localised.

### 4.9 Chunking

**`src/dip/parsing/chunker.py`.** `chunk_parsed(parsed, strategy, max_tokens, overlap)`:

- `layout_section` (default) — iterate `ParsedDoc.sections`; within each section, slide a
  word window of `max_tokens / 1.3` words (rough tokens→words) with `overlap` carry-over.
  Section boundaries are respected so a chunk never straddles two clauses / two report
  sections.
- `fixed_window` — same windowing over the whole document, ignoring structure.

`max_tokens` is 300–400 per type. Chunk ids are `"{doc_id}::{order:04d}"`.

**Why structure-first, fixed-size within.** Pure fixed-size chunking splits mid-sentence
and mid-table; pure "one chunk per section" produces 50-page chunks for a contract's
recitals. Section-bounded windows get retrieval-sized units that still align to the
document's own structure. A learned semantic chunker (embedding-similarity boundaries) is
a reasonable upgrade but adds a model and tuning for a marginal gain at this corpus size.

### 4.10 Embeddings and vector store

**Embeddings — `src/dip/retrieval/embed.py`.** `BAAI/bge-m3` via `FlagEmbedding`
(`BGEM3FlagModel`, `use_fp16=False` for CPU), dense vectors only (1024-d), lazily loaded
and `lru_cache`d. Batched encode.

**Why BGE-M3.** Strong on MTEB retrieval, multilingual (the SROIE receipts are
Malay/English), a single model that *also* produces sparse and ColBERT vectors so hybrid
search is a later config change not a model swap, and it runs locally with no API. The
project uses dense-only today; hybrid (dense + BGE-M3 sparse, fused in Qdrant) is the
documented next step. Alternatives: OpenAI `text-embedding-3` (API dependency, breaks
"fully local"), `e5-large` / `bge-large-en` (English-only), `gte` (comparable, no built-in
sparse).

**Vector store — `src/dip/retrieval/store.py`.** Qdrant. `VectorStore` supports two modes
behind one class:

- **Server** (`QDRANT_URL`) — the laptop demo, Qdrant in Docker.
- **Embedded on-disk** (`QDRANT_PATH`) — the Colab eval. The Colab environment could not
  reliably fetch a Qdrant server binary, so `QdrantClient(path=...)` runs an embedded
  store; snapshot methods no-op in this mode and the laptop simply re-embeds.

Collection: cosine distance, 1024-d, payload indexes on `doc_type` and `doc_id`. Points
are upserted with a **deterministic `uuid5(namespace, chunk_id)`** id so re-indexing a
document is idempotent (delete-by-`doc_id` filter, then upsert). Payload carries
`chunk_id, doc_id, order, section, text, doc_type`. Search is a single `query_points` with
an optional `doc_type` filter.

**Why Qdrant over pgvector / FAISS / Weaviate.** Payload filtering + metadata indexes out
of the box (needed for the `doc_type` facet), an embedded mode that made the Colab run
possible, snapshots for portability, and a simple `pip` client. pgvector would collapse
the stack to one datastore (attractive) but its filtering + index story was heavier to set
up here; FAISS has no payloads/filtering; Weaviate is a bigger operational footprint.

### 4.11 Orchestration internals

**`src/dip/pipeline/`.** `DocState` is a `TypedDict(total=False)` — nodes return partial
dicts, LangGraph merges. Nodes are defensive: any node short-circuits if
`status == "failed"`, so a parse failure cleanly skips the rest instead of throwing deep
in extraction. `build_pipeline(index=True)` is `lru_cache`d and wires
`ingest→parse→classify→extract→validate`, then `add_conditional_edges("validate",
route_decision, {"index": "index", "review": "review"})`, both terminating at `END`.
`index=False` (used by the batch eval) routes everything through `review` so the eval
measures extraction without paying for embedding.

**DB writes are not in the nodes.** Nodes are pure w.r.t. Postgres; `persist_state` is
called once by the API's background task (or the batch runner) after the graph returns.
This keeps nodes unit-testable without a database and makes the persistence schema a
single well-understood function.

### 4.12 Persistence — PostgreSQL schema

**`src/dip/db/models.py`** (SQLAlchemy 2.0 declarative). Six tables:

| Table | Purpose | Notable columns |
|---|---|---|
| `documents` | one row per ingested document | `id` (string PK, 160 chars — CUAD filenames are long), `status` (indexed), `doc_type`, `doc_type_confidence` |
| `extractions` | one row per extraction attempt (model **or** human) | `payload` **JSONB**, `doc_confidence`, `source` (`model`/`human`), FK `documents.id` `ON DELETE CASCADE` |
| `field_confidences` | per-field score for an extraction | `field_name`, `confidence`, `method` |
| `validation_results` | per-rule outcome for an extraction | `rule_name`, `passed`, `is_critical`, `message` |
| `review_queue` | open + resolved review items | `reason`, `resolved` (indexed), `resolved_at` |
| `audit_log` | append-only event trail | `event`, `detail` **JSONB**, `ts` |

**Design decisions:**

- **`payload` as JSONB, not a wide typed table.** The extraction shape differs per type
  and evolves with the schema; JSONB stores it verbatim, is queryable
  (`payload->>'total'`), and needs no migration when a schema field is added. The typed,
  per-field detail that *is* worth normalising (confidence, validation) lives in child
  tables.
- **Extractions are append-only.** A human correction inserts a new row with
  `source="human"` rather than mutating the model's output — you keep the full provenance
  chain (what the model said, what the human changed) which is exactly what you need for
  error analysis and, later, for building a fine-tuning / active-learning set.
- **`audit_log`** records every `pipeline_run` and `human_review` with a JSONB detail
  blob. Cheap insurance for debugging "why did this document end up here".
- **`schema_version`** on `extractions` so results produced under different schema
  revisions are distinguishable.
- Tables are created with `Base.metadata.create_all` (no Alembic) — a deliberate scope
  cut; production wants migrations.

### 4.13 API — FastAPI

**`src/dip/api/main.py`.**

| Endpoint | Behaviour |
|---|---|
| `POST /documents` | save upload, insert `documents(processing)`, schedule background `_process` (runs the graph, persists), return `{doc_id, status}` immediately |
| `GET /documents/{id}` | assemble a view from the latest extraction + its confidences + validations |
| `GET /documents?status=` | list (filtered) |
| `PUT /documents/{id}/extraction` | human correction → `resolve_review` + re-index |
| `POST /documents/{id}/approve` | accept current extraction as-is |
| `POST /search` | `{query, top_k, doc_type?}` → dense semantic search |
| `GET /metrics` | serve `artifacts/metrics.json` + `retrieval_metrics.json` |
| `GET /health` | liveness |

**Async model: FastAPI `BackgroundTasks`, not Celery/Redis.** For a single-node demo where
a document takes seconds–minutes, an in-process background task is enough and adds no
infrastructure. The moment you need retries, visibility, back-pressure, or multiple
workers, this is the seam where a real queue goes in — the task body is already a single
function call (`run_document` + `persist_state`).

Document ids are URL-encoded on the path (`quote(doc_id, safe="")`) because CUAD ids
contain `#`, `,`, `(`, `)`; an un-encoded `#` truncated the request URL and 404'd.

### 4.14 UI — Streamlit

**`src/dip/ui/app.py`**, four tabs, talks to the API over HTTP:

1. **Upload & Process** — upload, optional forced type, submit; poll a document id for its
   result (status, extracted JSON, per-field confidence table, validation table).
2. **Review Queue** — a `selectbox` over `needs_review` documents (not an eager per-row
   fetch — that made every rerun do 42 HTTP calls); selected document shows confidence
   metrics, the rule table, and an editable JSON box with Approve buttons. `get_doc` is
   `@st.cache_data(ttl=30)`.
3. **Search** — query + type filter + `top_k`; results show rank, score, section, snippet.
4. **Metrics** — the four headline numbers, in/out-of-auto-accept accuracy, ECE, per-type
   table, and the calibration plot.

Streamlit was chosen because the UI is an internal review/inspection tool, not a product
surface — four tabs of forms and tables is exactly its sweet spot and it is ~200 lines. A
React/Next front-end would be the right call for a real reviewer product with keyboard
workflows, bounding-box overlays on the source document, and role-based views.

---

## 5. Evaluation methodology

### 5.1 Datasets and ground truth (120 documents, 30 per type)

| Type | Source | Ground truth | Notes |
|---|---|---|---|
| Invoice | **SROIE** (ICDAR-2019) receipt images | `company, date, address, total` from the dataset's key files | real scanned receipts → real OCR noise |
| Contract | **CUAD v1** full-text contracts + `master_clauses.csv` | `document_name, parties, agreement_date, effective_date, expiration_or_term, governing_law, renewal_term` | professionally annotated; messy GT (redactions, `[]/[]/2004` partial dates, `;`-joined party strings) |
| Purchase order | **synthetic** (`scripts/gen_synthetic.py`, Faker + ReportLab) | the generation parameters, saved verbatim | exact GT; native-text PDFs (no OCR) |
| Medical report | **synthetic** radiology/lab templates | the generation parameters | exact GT; templated findings/impression banks per modality |

**Why this mix.** Two real datasets with independent ground truth (SROIE, CUAD) test the
system on genuine layout and genuine annotation noise; two synthetic sets give *exact*
ground truth and a clean upper bound on what the pipeline can do when parsing is not the
bottleneck. Synthetic medical data also side-steps PHI entirely.

`scripts/prepare_eval.py` unifies everything into `data/eval/manifest.jsonl` —
`{doc_id, doc_type, file_path, ground_truth}` per line.

### 5.2 Running the eval

`scripts/run_pipeline_batch.py` — LLM preflight (a 1-token call; abort loudly if the
endpoint is down rather than burn 120 × retry budget), then a `ThreadPoolExecutor` over
the manifest calling `run_document(index=False)`, writing one JSON line per document with
the prediction, the ground truth, and wall-clock seconds.

`scripts/build_index.py` — parse + chunk + BGE-M3 embed + upsert the whole corpus (this is
independent of extraction quality; every document is retrievable).

Both run on a **Colab T4** (`notebooks/colab_runner.ipynb`) because the laptop has no GPU:
Qwen2.5-3B on the EliteBook CPU is ~130 s/doc and BGE-M3 indexing of 844 chunks took 50
minutes. On the T4 the full pipeline pass is ~10 s/doc. The model is the *same* Ollama
GGUF, so the numbers are the honest "local" numbers, just produced faster.

### 5.3 Scoring — `scripts/eval_extraction.py`

Per (document, field), `compare(field, gt, pred)` returns one of
**correct / incorrect / missing / spurious / skip**:

- both empty → `skip` (not counted).
- GT empty, prediction present → `spurious`.
- GT present, prediction empty → `missing`.
- both present → type-specific comparison:
  - **number** — parse both (strip `$`, `,`), match within `max(0.01, 1%)`.
  - **date** — `_date_match`: `_date_components` parses a value into
    `(year, {n1, n2})` trying *both* `M/D` and `D/M` orderings, or `(year, None)` for a
    year-only value like `[]/[]/2004`; two dates match iff years are equal and the
    `{month, day}` sets are equal (or one side is year-only). This is **order-agnostic on
    purpose** — see §5.5.
  - **list** (`parties`, `diagnoses`) — `_list_f1`: normalise each side to a set of names
    (split strings on `;`/`,`/`and`, drop parenthetical defined-terms and quotes), fuzzy
    set F1 with a `token_sort_ratio ≥ 85` / `partial_ratio ≥ 80` match, correct iff F1 ≥
    0.7.
  - **string** — `max(token_sort_ratio, partial_ratio) ≥ 85` (case-insensitive). Handles
    "Pennsylvania" vs "Commonwealth of Pennsylvania".

**Field accuracy = correct / (correct + incorrect + missing + spurious)** — a missed or
hallucinated field counts against you.

**Metrics produced:**

- `field_accuracy_overall`, `_by_type`, `_by_field`.
- `classification_accuracy` — predicted vs. ground-truth `doc_type`.
- `auto_accept_rate` (= "manual-verification reduction"), plus
  `field_accuracy_auto_accepted` vs `field_accuracy_needs_review` and
  `auto_accepted_doc_precision` (fraction of auto-accepted docs with zero wrong fields).
- **Calibration** — bucket every `(field_confidence, is_correct)` pair into deciles;
  `ECE = Σ (bucket_size / N) · |mean_confidence − accuracy|`; render `calibration.png`.
- `mean_latency_s`.

`scripts/eval_retrieval.py` — for `scripts/make_retrieval_queries.py`'s ~2 queries/doc
(generated from ground truth: "invoice or receipt from {company}", "purchase order sent to
vendor {vendor}", "{modality} report and its impression", "contract governed by the laws
of {law}", …) plus 6 thematic queries, run `semantic_search(top_k=5)` and compute
**recall@5** (any relevant doc in the top 5) and **MRR@5**, overall and per type.

### 5.4 Why these metrics

- **Field-level, not document-level** accuracy — a document with 9/10 fields right is not
  "wrong"; the operational unit is the field.
- **Auto-accept rate + accuracy-inside-the-set** together — either alone is gameable
  (accept nothing → 100% precision; accept everything → 100% automation). The pair is the
  real "how much work did we remove, and at what quality" statement.
- **Calibration / ECE** — the entire selective-review design rests on the confidence
  signal meaning something. ECE 0.078 is the evidence that it does.
- **recall@k, not nDCG** — the downstream use is "did the reviewer's query surface the
  right document", a recall question.

### 5.5 Bugs found *in the evaluation harness* (worth calling out)

The first eval run reported contract accuracy at 8%. Investigation showed roughly half was
a genuine model failure and half was **the scorer being wrong**:

1. **Date order ambiguity** — the model output `2003-03-01` (correct); the scorer parsed
   the CUAD ground truth `3/1/03` with a `%d/%m/%y`-first format list and got January 3.
   Fixed with the order-agnostic `{month, day}` set comparison.
2. **Year-only ground truth** — CUAD renders partially-known dates as `[]/[]/2004`. Now
   matched on the year alone.
3. **List-vs-string** — `parties` ground truth is a single `;`-delimited string; the model
   returns a JSON array. The scorer string-compared the array's `repr` to the string and
   always failed. Now both sides are normalised to name sets.
4. **Date field not detected as a date** — `expiration_or_term` holds a date but its name
   lacks "date". Added an explicit date-field set.

**Lesson:** an extraction eval is only as good as its normalisation layer. The scorer
needs the same care as the extractor, and you should sanity-check low scores by reading
the actual (prediction, ground-truth) pairs before believing them.

---

## 6. Results and failure analysis

| Type | Field accuracy | Why |
|---|---|---|
| purchase_order | **99.1%** | native-text PDF, exact synthetic GT, short structured doc — the ceiling case |
| medical_report | **89.6%** | synthetic + exact GT; `findings`/`diagnoses` are free-text/list fields where fuzzy matching is stricter |
| invoice | **64.2%** | **OCR-bound** — SROIE images produce garbled merchant names, split addresses, mis-read digits; a 3B model cannot recover text that OCR destroyed |
| contract | **53.8%** | long, dense, real annotation noise; `field_by_field` fixed the catastrophic-null problem; `governing_law` (15%) and `renewal_term` (0%) remain weak because those clauses are scattered and the keyword windows still miss some |

- **Classification 94.2%** — the ~7 misses are mostly invoice↔purchase_order (they share
  vocabulary).
- **Auto-accept 65%**, and inside that set field accuracy is **90.8%** vs **56.5%** in the
  review queue — the gate separates reliable from unreliable well. `auto_accepted_doc_precision`
  (fully-correct docs among auto-accepted) is lower (~0.47) because "one wrong field in a
  10-field PO" still counts as an imperfect doc; tightening `max_low_conf_fields` to 0
  trades auto-accept rate down for that number up.
- **Retrieval recall@5 82.2%**, MRR 0.61, over **844 chunks** / 157 graded queries. PO
  (0.93) and contract (0.92) are strong; invoice (0.65) lags because receipts are short
  and lexically similar to each other.
- **Calibration ECE 0.078.**

**Honest gaps vs. an idealised spec:** ~83% is not ~90%; contracts and OCR'd invoices are
the drag. The manual-verification reduction (65%) actually exceeds a "~55%" target. Chunk
count (~850) is below a "1,500+" aspiration because the corpus is small and receipts are
one chunk each.

---

## 7. Performance characteristics

| Stage | Colab T4 | Laptop CPU (EliteBook 840 G6, no GPU) |
|---|---|---|
| Docling parse (native PDF) | ~1–2 s | ~5 s |
| Docling parse (image + OCR) | ~5–10 s | ~30–60 s |
| Classify (1 LLM call) | ~1–2 s | ~15–30 s |
| Extract — single_pass | ~3–8 s | ~40–90 s |
| Extract — contract (3 calls) | ~10–20 s | ~2–4 min |
| BGE-M3 embed (per doc, few chunks) | ~1 s | ~10–30 s (+ model load) |
| **Full pipeline, per doc** | **~10 s** | **~75 s (PDF) – 4 min (contract)** |
| Full 120-doc eval pass | ~30–40 min | 4–8 h (not run) |
| Index build (844 chunks) | ~5 min | ~50 min |

**Bottlenecks, in order:** (1) LLM decode on CPU, (2) OCR on CPU, (3) BGE-M3 model
reload per invocation in the single-doc path. Mitigations already in place: aggressive
parse caching, batch embedding in the eval path, GPU for the eval. The parsed-document
cache is committed to the repo so the Colab run skips re-OCR.

---

## 8. Infrastructure and environment decisions

- **Docker Compose** runs Postgres 16 + Qdrant; FastAPI and Streamlit run on the host for
  fast iteration. An `app` compose profile containerises them too (large image — Docling
  pulls Torch).
- **Python 3.11 via `uv`.** The machine's only interpreter was 3.14, which has no wheels
  for Torch/Docling/FlagEmbedding. `uv` installs a project-local 3.11 without touching the
  system Python. Requirements are split: `requirements-app.txt` (light: FastAPI, Streamlit,
  SQLAlchemy, Qdrant client, OpenAI client, LangGraph — installs on 3.11–3.14) and
  `requirements-ml.txt` (heavy: Docling, FlagEmbedding, Torch — 3.11/3.12 only).
- **Qdrant client pinned to the server minor** to silence the client/server compat
  warning.
- **Colab** for GPU: Ollama started from a Python `subprocess.Popen` in the notebook (a
  backgrounded `ollama serve` inside a `!bash` cell gets reaped when the cell exits);
  embedded on-disk Qdrant (no server binary to fetch); `zstd` installed before the Ollama
  installer (its tarball is zstd-compressed and Colab's image lacks it).
- **Repo hygiene:** the 120 raw documents are `.gitignore`d and regenerated deterministically
  (`fetch_data.py` + `gen_synthetic.py --seed 7`); the eval artifacts (`metrics.json`,
  `eval_report.md`, `calibration.png`) and the parsed-doc cache *are* committed for
  reproducibility.

---

## 9. Security and privacy considerations

- **Fully local by default** — Qwen2.5-3B and BGE-M3 run on-box; no document text leaves
  the machine. This is the main reason a 3B local model was chosen over a frontier API:
  for medical reports and contracts, "the data never leaves" is often a hard requirement.
- **Synthetic-only sensitive data** — the medical and PO sets are generated; no real PHI
  or confidential agreements are in the repo or the eval.
- **Append-only audit log** and append-only extractions give a tamper-evident trail of
  what the model produced and what a human changed.
- **Not yet present** (see §10): authN/Z, tenant isolation, encryption at rest,
  PII detection/redaction before indexing, and access controls on retrieval.

---

## 10. Known limitations and a production roadmap

**Limitations today:**

- No auth, no multi-tenancy, no RBAC.
- `create_all` instead of migrations.
- Human-in-the-loop is a status flag, not a resumable graph interrupt.
- Retrieval is dense-only; no RAG question-answering endpoint.
- Extraction is prompt-based on a general 3B model — no fine-tuning, no
  bounding-box grounding, no confidence from a trained calibrator.
- OCR errors compound into extraction errors with no feedback loop.
- Single-node, in-process background processing.

**What a production version adds, roughly in priority order:**

1. **Grounding** — return a source span / bounding box per extracted field so reviewers
   see *where* a value came from and the UI can highlight it. Requires keeping Docling's
   layout coordinates through the pipeline.
2. **A document-understanding model** — replace OCR→text→LLM with a VLM (Qwen2-VL,
   or a fine-tuned Donut/LayoutLM) that reads pixels, killing the OCR-error class.
3. **Constrained decoding** — vLLM + Outlines/XGrammar for guaranteed-valid JSON; delete
   the repair retry.
4. **A real queue** — Redis + RQ/Celery (or Temporal) for retries, back-pressure,
   multi-worker throughput, and DLQ.
5. **LangGraph checkpointer + `interrupt()`** for true suspend/resume human review, with
   the `DocState` persisted per run.
6. **Hybrid retrieval** — BGE-M3 dense + sparse fused in Qdrant; add a `/ask` RAG endpoint
   with citations.
7. **Active learning** — mine the `source="human"` corrections into a fine-tuning /
   few-shot set; measure whether swapping the general 3B for a small fine-tuned extractor
   moves accuracy and cost.
8. **A trained confidence calibrator** — logistic regression on
   `[token_prob_min, token_prob_mean, entropy, field_type, rule_flags]` → P(correct),
   instead of a raw threshold on `exp(logprob)`.
9. **Migrations (Alembic), auth, tenant isolation, encryption, PII redaction, metrics/tracing
   (OpenTelemetry), CI.**
10. **Schema/versioning UI** so non-engineers can add a document type.

---

## 11. Interview Q&A appendix

**Q: Why a 3B model? Extraction quality would be higher with a bigger one.**
Correct, and the code makes the swap a one-line env change. 3B was chosen for two reasons:
(a) it runs fully local on commodity hardware, which is a hard requirement for medical and
legal documents in many shops; (b) it is an honest stress test — if the *architecture*
(schema-driven prompts, field-by-field for long docs, logprob confidence, hybrid gate)
works at 3B, it works better at 7B/70B. The eval numbers are the 3B floor.

**Q: How do you know the confidence signal is trustworthy?**
The calibration curve. Bucketed by confidence decile, predicted confidence tracks observed
field accuracy with ECE 0.078; the 0.5–0.7 band is where errors concentrate and the
0.95+ band is ~95% correct. That measured relationship is what the auto-accept threshold
exploits — it is not a vibe, it is a plotted curve in `artifacts/calibration.png`.

**Q: What's the failure mode of token-logprob confidence?**
It measures the model's *certainty*, not its *correctness* — a confidently wrong
hallucination (fluent, in-distribution) can score high. That is why confidence is one of
*three* gate conditions alongside deterministic rule checks and a low-confidence-field
budget, and why the fallback is self-consistency (disagreement across samples catches some
confident-but-unstable answers).

**Q: Why not just fine-tune an extraction model?**
It is roadmap item 7. It needs a labelled corpus; the system is *designed to produce one*
(every human correction is an append-only `source="human"` row with the model's original
output alongside). Fine-tuning before you have that data and a baseline to beat is
premature.

**Q: How does this handle a brand-new document type?**
One YAML file (schema path, classifier hint, rules, thresholds, chunking) + one Pydantic
model + an optional few-shot file. No pipeline, API, or UI code changes. The
`field_by_field` mechanism (YAML-declared field groups with `head` or `keyword` context)
means even a new long-document type is config-only.

**Q: Contract accuracy is 54%. Is the approach wrong?**
The approach recovered it from 13%. The residual is two things: (a) genuinely hard,
scattered clauses (`governing_law`, `renewal_term`) where keyword windows still miss —
fixable with tighter, clause-specific keyword groups or a retrieval step over the chunk
index; (b) CUAD's own annotation noise (redactions, partial dates). Long-document
extraction is where a grounded VLM would help most.

**Q: Why Docling and not a VLM already?**
Time and serving simplicity — Docling is one `pip` install with OCR included and a
Markdown output an LLM reads well. The OCR-error compounding it introduces is the single
biggest accuracy drag (invoices), and replacing the OCR→text→LLM path with a
pixels→structure VLM is the top roadmap item.

**Q: Walk me through what happens to one uploaded invoice.**
`POST /documents` writes the file and a `documents(status=processing)` row, schedules a
background task. The task runs the graph: `ingest` (file exists), `parse` (Docling OCRs
the image → Markdown, cached), `classify` (LLM zero-shot on the first 1500 chars →
`invoice`, 0.98), `extract` (single_pass: full Invoice JSON schema + OCR text → JSON;
parsed, repaired if needed, coerced through the Pydantic model; per-field confidence from
the value-span token logprobs), `validate` (`required_fields`, `total_is_positive`,
`subtotal_plus_tax_matches_total`, …; then the gate: no critical failure ∧
doc_confidence ≥ 0.70 ∧ ≤ 1 low-confidence field). If it passes → `index` (chunk, BGE-M3
embed, upsert to Qdrant), `status=indexed`. If not → `status=needs_review` + a
`review_queue` row. `persist_state` writes the `extractions` row, `field_confidences`,
`validation_results`, and an `audit_log` line. The UI polls and renders the JSON, the
confidence table, and the rule outcomes.
