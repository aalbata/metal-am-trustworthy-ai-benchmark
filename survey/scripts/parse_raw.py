"""Flatten the stored raw responses into records_raw.csv and apply the per-stream caps.

The bulk endpoint returns every match in identifier order without a relevance score,
so each stream is ranked here by a deterministic term-match score and the top `cap`
records are flagged retained=1.
  Focused and review streams:
    score = 3 x (distinct dimension/review terms in the title)
          + 1 x (distinct dimension/review terms in the abstract)
          + 1 if a process term is in the title + 1 if a method term is in the title
  General streams (one process term p, one method term m):
    score = 3 x [p in title] + 3 x [m in title] + 1 x [p in abstract] + 1 x [m in abstract]
Ties are broken by citation count (descending) and then paper identifier (ascending).
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, RAW_DIR, norm_doi, write_csv  # noqa: E402
from streams import METHOD_TERMS, PROCESS_TERMS, REVIEW_METHOD_TERMS, all_streams  # noqa: E402

FIELDS = ["stream_id", "stream_kind", "rank_in_stream", "retained", "local_score",
          "s2_paperId", "doi", "arxiv_id", "title", "authors", "year", "publication_date",
          "venue", "journal_name", "s2_publication_types", "abstract", "cited_by_count",
          "is_oa", "oa_url", "retrieved_utc"]


def norm_text(s):
    return re.sub(r"\s+", " ", re.sub(r"[-‐‑–]", " ", (s or "").lower()))


def has_term(term, text):
    t = norm_text(term)
    return re.search(r"(?<![a-z0-9])" + re.escape(t), text) is not None


def local_score(rec, terms, method_terms):
    title = norm_text(rec.get("title"))
    abstract = norm_text(rec.get("abstract"))
    s = 3 * sum(has_term(t, title) for t in terms) + sum(has_term(t, abstract) for t in terms)
    s += 1 if any(has_term(t, title) for t in PROCESS_TERMS) else 0
    s += 1 if any(has_term(t, title) for t in method_terms) else 0
    return s


def flatten(p, stream, rank, retrieved_utc):
    ext = p.get("externalIds") or {}
    oa = p.get("openAccessPdf") or {}
    return {
        "stream_id": stream["stream_id"], "stream_kind": stream["kind"], "rank_in_stream": rank,
        "s2_paperId": p.get("paperId") or "", "doi": norm_doi(ext.get("DOI")),
        "arxiv_id": ext.get("ArXiv") or "", "title": (p.get("title") or "").strip(),
        "authors": "; ".join(a.get("name") or "" for a in (p.get("authors") or [])),
        "year": p.get("year") or "", "publication_date": p.get("publicationDate") or "",
        "venue": p.get("venue") or "", "journal_name": (p.get("journal") or {}).get("name") or "",
        "s2_publication_types": ";".join(p.get("publicationTypes") or []),
        "abstract": re.sub(r"\s+", " ", p.get("abstract") or "").strip(),
        "cited_by_count": p.get("citationCount") if p.get("citationCount") is not None else "",
        "is_oa": int(bool(p.get("isOpenAccess"))), "oa_url": oa.get("url") or "",
        "retrieved_utc": retrieved_utc,
    }


def main():
    if not os.path.isdir(RAW_DIR):
        sys.exit("raw_json/ not found in records/; run search_s2.py first")
    rows = []
    for s in all_streams():
        path = os.path.join(RAW_DIR, s["stream_id"] + ".json")
        if not os.path.exists(path):
            print("missing", s["stream_id"])
            continue
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        recs = []
        for page in raw["pages"]:
            if page["status"] != 200:
                continue
            for p in page["payload"].get("data") or []:
                recs.append(flatten(p, s, 0, page["requested_utc"]))
        if s["kind"] == "general":
            for r in recs:
                ti, ab = norm_text(r["title"]), norm_text(r["abstract"])
                r["local_score"] = (3 * has_term(s["process"], ti) + 3 * has_term(s["method"], ti)
                                    + has_term(s["process"], ab) + has_term(s["method"], ab))
        else:
            mterms = REVIEW_METHOD_TERMS if s["kind"] == "review" else METHOD_TERMS
            for r in recs:
                r["local_score"] = local_score(r, s["dim_terms"], mterms)
        recs.sort(key=lambda r: (-r["local_score"], -(r["cited_by_count"] or 0), r["s2_paperId"]))
        for i, r in enumerate(recs, 1):
            r["rank_in_stream"] = i
            r["retained"] = 1 if i <= s["cap"] else 0
        rows.extend(recs)
    write_csv(os.path.join(OUT, "records_raw.csv"), rows, FIELDS)
    print("records_raw rows", len(rows), "retained", sum(r["retained"] for r in rows))


if __name__ == "__main__":
    main()
