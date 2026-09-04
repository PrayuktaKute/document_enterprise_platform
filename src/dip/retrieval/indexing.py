"""Single-document indexing helper shared by the pipeline node and the API."""
from __future__ import annotations

from dip.config import get_doc_type_config
from dip.parsing.chunker import chunk_parsed
from dip.parsing.docling_parser import ParsedDoc, parse_document
from dip.retrieval.embed import embed_texts
from dip.retrieval.store import VectorStore


def index_parsed(doc_id: str, parsed: ParsedDoc, doc_type: str) -> int:
    ck = get_doc_type_config(doc_type).chunking
    chunks = chunk_parsed(parsed, strategy=ck.strategy, max_tokens=ck.max_tokens, overlap=ck.overlap)
    if not chunks:
        return 0
    vectors = embed_texts([c.text for c in chunks])
    store = VectorStore.from_config()
    store.ensure_collection()
    store.delete_doc(doc_id)
    return store.upsert_chunks(chunks, vectors, extra_payload={"doc_type": doc_type})


def index_document(doc_id: str, file_path: str, doc_type: str) -> int:
    return index_parsed(doc_id, parse_document(file_path, doc_id), doc_type)
