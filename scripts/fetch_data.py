"""Phase 1a: download the public datasets.

  * Invoices  -> SROIE (ICDAR-2019) receipt images + key JSON
                 https://github.com/zzzDavid/ICDAR-2019-SROIE
  * Contracts -> CUAD v1 full-text contracts + master_clauses.csv
                 https://zenodo.org/records/4595826

Each type lands in ``data/raw/<type>/`` with a sidecar ``ground_truth.json``
mapping ``doc_id -> {field: value}``.

    python scripts/fetch_data.py --invoices 30 --contracts 30
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import httpx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw"
CACHE = RAW / "_cache"

SROIE_BASE = "https://raw.githubusercontent.com/zzzDavid/ICDAR-2019-SROIE/master/data"
CUAD_ZIP_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"

# CUAD master_clauses.csv column name candidates (schema varies slightly by release).
CUAD_COLUMNS: dict[str, list[str]] = {
    "document_name": ["Document Name-Answer", "Document Name"],
    "parties": ["Parties-Answer", "Parties"],
    "agreement_date": ["Agreement Date-Answer", "Agreement Date"],
    "effective_date": ["Effective Date-Answer", "Effective Date"],
    "expiration_or_term": ["Expiration Date-Answer", "Expiration Date"],
    "governing_law": ["Governing Law-Answer", "Governing Law"],
    "renewal_term": ["Renewal Term-Answer", "Renewal Term"],
}


def _client() -> httpx.Client:
    return httpx.Client(timeout=120.0, follow_redirects=True, headers={"User-Agent": "dip/0.1"})


# --------------------------------------------------------------------------- #
def fetch_sroie(n: int) -> None:
    out = RAW / "invoices"
    out.mkdir(parents=True, exist_ok=True)
    gt: dict[str, dict] = {}
    got = 0
    idx = 0
    with _client() as c:
        while got < n and idx < 400:
            stem = f"{idx:03d}"
            idx += 1
            try:
                key = c.get(f"{SROIE_BASE}/key/{stem}.json")
                img = c.get(f"{SROIE_BASE}/img/{stem}.jpg")
            except httpx.HTTPError as exc:
                print(f"  sroie {stem}: {exc}")
                continue
            if key.status_code != 200 or img.status_code != 200:
                continue
            try:
                fields = json.loads(key.text)
            except json.JSONDecodeError:
                continue
            if not fields.get("company") or not fields.get("total"):
                continue
            doc_id = f"invoice_{stem}"
            (out / f"{doc_id}.jpg").write_bytes(img.content)
            gt[doc_id] = {
                "company": fields.get("company"),
                "date": fields.get("date"),
                "address": fields.get("address"),
                "total": fields.get("total"),
            }
            got += 1
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
    print(f"invoices: wrote {got} images + ground_truth.json -> {out}")


# --------------------------------------------------------------------------- #
def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def fetch_cuad(n: int) -> None:
    out = RAW / "contracts"
    out.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE / "CUAD_v1.zip"

    if not zip_path.exists() or zip_path.stat().st_size < 50_000_000:
        print("downloading CUAD_v1.zip (~106 MB)...")
        with _client() as c, c.stream("GET", CUAD_ZIP_URL) as resp:
            resp.raise_for_status()
            with zip_path.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    fh.write(chunk)
        print(f"  saved {zip_path.stat().st_size / 1e6:.1f} MB")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        csv_name = next(x for x in names if x.endswith("master_clauses.csv"))
        txt_names = sorted(x for x in names if "/full_contract_txt/" in x and x.endswith(".txt"))
        df = pd.read_csv(io.BytesIO(zf.read(csv_name)))
        cols = {f: _pick_column(df, cands) for f, cands in CUAD_COLUMNS.items()}
        print("  CUAD columns resolved:", {k: v for k, v in cols.items()})
        fname_col = _pick_column(df, ["Filename", "filename"])
        df = df.set_index(df[fname_col].astype(str).str.strip())

        gt: dict[str, dict] = {}
        written = 0
        for tn in txt_names:
            if written >= n:
                break
            stem = Path(tn).stem
            row = None
            for key in (stem, stem + ".pdf", stem + ".PDF", stem + ".txt"):
                if key in df.index:
                    row = df.loc[key]
                    break
            if row is None:
                continue
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            doc_id = "contract_" + stem[:60].replace(" ", "_")
            (out / f"{doc_id}.txt").write_bytes(zf.read(tn))
            record: dict = {}
            for field, col in cols.items():
                val = None if col is None else row.get(col)
                if pd.isna(val):
                    val = None
                record[field] = None if val is None else str(val).strip()
            gt[doc_id] = record
            written += 1

    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
    print(f"contracts: wrote {written} txt files + ground_truth.json -> {out}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--invoices", type=int, default=30)
    ap.add_argument("--contracts", type=int, default=30)
    ap.add_argument("--skip-invoices", action="store_true")
    ap.add_argument("--skip-contracts", action="store_true")
    args = ap.parse_args()

    if not args.skip_invoices:
        fetch_sroie(args.invoices)
    if not args.skip_contracts:
        fetch_cuad(args.contracts)


if __name__ == "__main__":
    main()
