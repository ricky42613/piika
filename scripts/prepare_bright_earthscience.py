#!/usr/bin/env python3
"""Convert pinned BRIGHT earth_science Parquet into Piika's BCP-style TSV/qrels.

Requires pyarrow. Source files are the official xlangai/BRIGHT examples and
standard documents (not long_documents), pinned at the revision below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

REVISION = "3066d29c9651a576c8aba4832d249807b181ecae"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/bright-earthscience"))
    args = parser.parse_args()
    root: Path = args.root
    examples_path = root / "source/examples-earth_science.parquet"
    documents_path = root / "source/documents-earth_science.parquet"

    examples = pq.read_table(examples_path).to_pylist()
    doc_table = pq.read_table(documents_path, columns=["id"])
    doc_ids = doc_table.column("id").to_pylist()
    corpus_ids = set(doc_ids)
    if len(examples) != 116 or len(doc_ids) != 121249 or len(corpus_ids) != len(doc_ids):
        raise ValueError("Unexpected BRIGHT earth_science query/corpus counts or duplicate docids")

    queries: list[str] = []
    qrels: list[str] = []
    qids: set[str] = set()
    for example in examples:
        qid = example["id"]
        query = re.sub(r"\s+", " ", example["query"]).strip()
        gold_ids = example["gold_ids"]  # NOT gold_ids_long: the API serves standard documents.
        if not qid or any(c.isspace() for c in qid) or qid in qids or not query or not gold_ids:
            raise ValueError(f"Invalid or duplicate query: {qid!r}")
        qids.add(qid)
        queries.append(f"{qid}\t{query}\n")
        if len(set(gold_ids)) != len(gold_ids):
            raise ValueError(f"Duplicate qrel docid for {qid}")
        for docid in gold_ids:
            if docid not in corpus_ids or any(c.isspace() for c in docid):
                raise ValueError(f"Gold docid not present in standard corpus: {qid} {docid!r}")
            qrels.append(f"{qid} Q0 {docid} 1\n")

    query_path = root / "queries/test.tsv"
    qrels_path = root / "qrels/qrels.test.txt"
    query_path.parent.mkdir(parents=True, exist_ok=True)
    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    query_path.write_text("".join(queries), encoding="utf-8")
    qrels_path.write_text("".join(qrels), encoding="utf-8")

    provenance = {
        "dataset": "xlangai/BRIGHT",
        "revision": REVISION,
        "split": "earth_science",
        "corpus": "documents (not long_documents)",
        "index_api": "http://127.0.0.1:8083/v1",
        "index_name": "bright-earth-science",
        "query_count": len(examples),
        "document_count": len(doc_ids),
        "qrel_count": len(qrels),
        "qrels_field": "gold_ids",
        "query_normalization": "collapse whitespace to one space for single-line TSV",
        "sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in (examples_path, documents_path, query_path, qrels_path)
        },
    }
    (root / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
