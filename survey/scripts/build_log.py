"""Write search_log.md. Every count is computed here from the CSV/JSON outputs."""
import collections
import json
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, QUERY_LOG, read_csv  # noqa: E402
from screen import EDITORIAL, REVIEW_ABS, REVIEW_TITLE  # noqa: E402
from streams import (DIMENSIONS, FOCUSED_CAP, GENERAL_CAP, METHOD_TERMS, PROCESS_TERMS,  # noqa: E402
                     REVIEW_CAP, all_streams)

CODES = [("E1", "not metal AM"), ("E2", "no implemented AI/ML model"),
         ("E3", "review/survey/editorial/perspective (routed to review list)"),
         ("E4", "not a peer-reviewed article or full conference paper"), ("E5", "outside date window"),
         ("E6", "not English"), ("E7", "simulation-only without ML"),
         ("E8", "insufficient information in title/abstract (full-text check)")]

# Command-line probes of candidate endpoints on 2026-10-15 (not logged by the scripts).
MANUAL_PROBES = [
    ("2026-10-15 10:51:16 UTC", "OpenAlex /works, title_and_abstract.search, no API key (curl)",
     "HTTP 429: free daily budget for the network IP address used up; resets at midnight UTC; "
     "an API key (account) would be required"),
    ("2026-10-15 10:51:36 UTC", "Semantic Scholar /paper/search/bulk (curl)", "HTTP 200"),
    ("2026-10-15 10:51:36 UTC", "Crossref /works query.bibliographic (curl)",
     "HTTP 200; free-text ranking without boolean operators, so not used for searching"),
    ("2026-10-15 approx. 10:55 to 11:01 UTC", "Semantic Scholar relevance endpoint /paper/search (curl)",
     "HTTP 429 on all five requests; the relevance-ranked endpoint was not used"),
]


def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(x).replace("|", "/") for x in r) + " |\n"
    return s


def main():
    qlog = [json.loads(l) for l in open(QUERY_LOG, encoding="utf-8")]
    raw = read_csv(os.path.join(OUT, "records_raw.csv"))
    dedup = read_csv(os.path.join(OUT, "records_dedup.csv"))
    ledger = read_csv(os.path.join(OUT, "screening_ledger.csv"))
    ext = read_csv(os.path.join(OUT, "extraction_set.csv"))
    rev = read_csv(os.path.join(OUT, "reviews.csv"))
    crm = read_csv(os.path.join(OUT, "crossref_meta.csv"))
    dstats = json.load(open(os.path.join(OUT, "dedup_stats.json"), encoding="utf-8"))
    sstats = json.load(open(os.path.join(OUT, "selection_stats.json"), encoding="utf-8"))
    spot_path = os.path.join(OUT, "spotcheck.csv")
    spot = read_csv(spot_path) if os.path.exists(spot_path) else []

    s2 = {e["stream_id"]: e for e in qlog if e.get("source") == "Semantic Scholar Graph API"}
    probe = [e for e in qlog if e.get("source") == "OpenAlex"]
    enrich = [e for e in qlog if e.get("stream_id") == "ENRICH"]
    pre = [e for e in qlog if e.get("stream_id") == "PREPRINT"]
    raw_by = collections.defaultdict(list)
    for r in raw:
        raw_by[r["stream_id"]].append(r)
    times = sorted(e["requested_utc"] for e in s2.values())

    L = []
    L.append("# Literature search and screening log\n")
    L.append("Structured critical survey for the article Trustworthy Artificial Intelligence in Metal "
             "Additive Manufacturing through a Critical Survey and a Controlled Benchmark. This is a documented, "
             "reproducible structured search with rule-based title/abstract screening. It is not a systematic "
             "review, and no independent second screener was involved.\n")
    L.append("## 1. Execution environment\n")
    L.append("- OS: %s; Python %s" % (platform.platform(), platform.python_version()))
    L.append("- Output folder: %s" % OUT.replace("\\", "/"))
    L.append("- Search requests (Semantic Scholar) ran from %s to %s." % (times[0], times[-1]))
    L.append("- The search and screening scripts in scripts/ use the Python standard library only (run with `python -I`).\n")

    L.append("## 2. Sources and a deviation from the planned design\n")
    L.append("The planned primary source was OpenAlex (title_and_abstract.search with boolean operators). "
             "It was not available on the search date: requests without an API key returned HTTP 429 "
             "because the free daily budget shared by the network IP address was exhausted, and obtaining a key "
             "requires an account, which was not used. The search was therefore run on "
             "the Semantic Scholar Graph API boolean bulk endpoint (/graph/v1/paper/search/bulk), which matches "
             "terms in titles and abstracts, applies stemming, and supports `+` (AND), `|` (OR) and quoted phrases. "
             "Its relevance-ranked endpoint (/paper/search) returned HTTP 429 on every attempt and supports no "
             "boolean syntax, so it was not used. Crossref was used only to enrich metadata (publication type, "
             "container title, abstract, preprint relations) by DOI.\n")
    L.append("Logged OpenAlex probe(s) (scripts/openalex_probe.py):\n")
    L.append(md_table(["UTC", "HTTP status", "query"], [(p["requested_utc"], p["http_status"][0], p["query"]) for p in probe]))
    L.append("Command-line probes of candidate endpoints (not logged by the scripts):\n")
    L.append(md_table(["time", "request", "outcome"], MANUAL_PROBES))

    L.append("## 3. Search design\n")
    L.append("- Block A (process): " + "; ".join('"%s"' % t for t in PROCESS_TERMS))
    L.append("- Block B (method): " + "; ".join('"%s"' % t for t in METHOD_TERMS))
    for sid, name, terms in DIMENSIONS:
        L.append("- Block C, %s (%s): %s" % (sid, name, "; ".join('"%s"' % t for t in terms)))
    L.append("- Review stream terms: (review | survey | \"state of the art\") AND (\"machine learning\" | "
             "\"deep learning\" | \"artificial intelligence\"); \"deep learning\" and \"artificial intelligence\" "
             "were added to \"machine learning\" so that surveys using those labels are found.")
    L.append("- Filter on every request: publicationDateOrYear=2015-01-01:2026-10-15. No language or "
             "document-type filter was applied at query time; those criteria were applied in screening.")
    L.append("- Fields requested: paperId, externalIds, title, abstract, year, publicationDate, venue, journal, "
             "publicationTypes, authors, citationCount, isOpenAccess, openAccessPdf.")
    L.append("- Caps (deliberate scope limit): %d records per general stream, %d per focused stream, %d for the "
             "review stream. The bulk endpoint returns all matches without a relevance score, so every match was "
             "downloaded and the cap was applied by a deterministic term-match score (scripts/parse_raw.py): "
             "general streams score 3 per query term in the title and 1 per query term in the abstract; focused "
             "and review streams score 3 per distinct dimension (or review) term in the title, 1 per distinct "
             "term in the abstract, plus 1 each for a process and a method term in the title; ties are broken by "
             "citation count and then paper identifier. Records beyond the cap are kept in records_raw.csv with "
             "retained=0 and were not screened.\n" % (GENERAL_CAP, FOCUSED_CAP, REVIEW_CAP))
    L.append("Endpoint: https://api.semanticscholar.org/graph/v1/paper/search/bulk\n")
    rows = []
    for s in all_streams():
        e = s2.get(s["stream_id"])
        rr = raw_by.get(s["stream_id"], [])
        label = s.get("dimension") or ("%s x %s" % (s.get("process"), s.get("method")) if s["kind"] == "general" else "reviews")
        rows.append((s["stream_id"], label, "`%s`" % s["query"], e["requested_utc"] if e else "not run",
                     ",".join(str(x) for x in e["http_status"]) if e else "", e["total_reported"] if e else "",
                     len(rr), sum(1 for r in rr if r["retained"] == "1"), s["cap"]))
    L.append(md_table(["stream", "label", "query string", "requested (UTC)", "HTTP", "hits reported",
                       "retrieved", "retained after cap", "cap"], rows))
    tot_hits = sum(int(e["total_reported"] or 0) for e in s2.values())
    L.append("Totals: %d streams run; hits reported %d (sum over streams, with overlap); records retrieved %d; "
             "records retained after caps %d; streams in which the cap was binding: %d.\n" % (
                 len(s2), tot_hits, len(raw), sum(1 for r in raw if r["retained"] == "1"),
                 sum(1 for s in all_streams() if len(raw_by.get(s["stream_id"], [])) > s["cap"])))

    L.append("## 4. Metadata enrichment (Crossref)\n")
    for e in enrich:
        L.append("- %s: %d DOIs requested from https://api.crossref.org/works (filter=doi:..., batches of up to 40); "
                 "%d new requests in this run%s; %d DOIs with a Crossref record after this run." % (
                     e["requested_utc"], e["dois_requested"], e["new_calls"],
                     (", %d failed" % e["failed_calls"]) if "failed_calls" in e else
                     " (failed requests not logged)", e["dois_found"]))
    L.append("- Crossref records with an abstract: %d of %d." % (sum(1 for r in crm if r["cr_abstract"]), len(crm)))
    for e in pre:
        L.append("- %s: published-version lookup for %d preprint records by exact title "
                 "(query.title, filter=type:journal-article,type:proceedings-article); %d new requests; "
                 "%d published versions accepted (preprint_links.csv)." % (
                     e["requested_utc"], e["preprints_checked"], e["new_calls"], e["published_versions_found"]))
    src = collections.Counter(r["abstract_source"] for r in dedup)
    L.append("- Abstract source after deduplication: Semantic Scholar %d, Crossref %d, none %d.\n" % (
        src.get("S2", 0), src.get("Crossref", 0), src.get("none", 0)))

    L.append("## 5. Deduplication\n")
    L.append("- Retained raw rows: %d" % dstats["retained_raw_rows"])
    L.append("- Unique after DOI (or Semantic Scholar paper id when no DOI): %d" % dstats["unique_after_doi_or_paperid_key"])
    L.append("- Unique after linking the same paper id across streams: %d" % dstats["unique_after_paperid_link"])
    L.append("- Additional merges by normalised title (length >= 30 characters, years within 2): %d" % dstats["title_links"])
    L.append("- Additional merges by Crossref preprint/version relations: %d" % dstats["relation_links"])
    L.append("- Unique records (records_dedup.csv, R0001 to R%04d): %d" % (len(dedup), dstats["unique_records"]))
    L.append("- Records carrying linked versions (preprint and journal or other duplicates): %d\n" % dstats["records_with_linked_versions"])

    L.append("## 6. Screening (title/abstract, single screener, rule-based)\n")
    L.append("Rules are coded in scripts/screen.py and applied in the order E5, E6, E4, E1, E3, I1, E8, E2/E7 "
             "(see the module docstring). Records without an abstract were screened on the title and coded E8 "
             "when the title alone could not establish eligibility; a record that would otherwise be I1 or E3 "
             "but has neither a DOI nor a venue was coded E8 for a publication-status check.\n")
    cnt = collections.Counter(l["reason_code"] for l in ledger)
    L.append(md_table(["code", "meaning", "records"], [(c, m, cnt.get(c, 0)) for c, m in CODES] + [("I1", "primary empirical metal-AM ML study", cnt.get("I1", 0))]))
    L.append("- Total screened: %d (sum of codes: %d)." % (len(ledger), sum(cnt.values())))
    basis = collections.Counter(l["screening_basis"] for l in ledger)
    L.append("- Screening basis: title+abstract %d, title only %d." % (basis.get("title+abstract", 0), basis.get("title only", 0)))
    ov = sum(1 for l in ledger if l["decision_source"].startswith("manual"))
    L.append("- Decisions set by a recorded manual override (screening_overrides.csv): %d." % ov)
    i1_title_only = sum(1 for l in ledger if l["reason_code"] == "I1" and l["screening_basis"] == "title only")
    L.append("- I1 decisions made on the title alone: %d." % i1_title_only)
    if spot:
        agree = sum(1 for s in spot if s["agree"] == "yes")
        si1 = [s for s in spot if s["rule_code_at_check"] == "I1"]
        L.append("- Check of the screening rules by the authors: a seeded (20261015), code-stratified random sample drawn by "
                 "scripts/spotcheck_sample.py (spotcheck_sample.csv) was read; %d records, rule decision agreed in %d "
                 "(%.0f%%); for rule code I1, %d of %d agreed. Disagreements led to two rule additions (multi-task learning "
                 "counted as ML; whole proceedings volumes coded E4) and three recorded overrides (spotcheck.csv). Because "
                 "the rules were refined after this sample, the agreement rate describes the rules before refinement. This "
                 "is a check of the screening rules on a sample, not a second screening of every record." % (
                     len(spot), agree, 100.0 * agree / len(spot), sum(1 for s in si1 if s["agree"] == "yes"), len(si1)))
        recd = {r["ID"]: r for r in dedup}
        n_s2_only = sum(1 for l in ledger if l["rule_code"] == "E3" and not (
            EDITORIAL.search(recd[l["ID"]]["title"]) or REVIEW_TITLE.search(recd[l["ID"]]["title"])
            or REVIEW_ABS.search(recd[l["ID"]]["abstract"])))
        n_gen = sum(1 for l in ledger if l["justification"].startswith("metal AM named only in the abstract"))
        L.append("- A further rule was added after inspecting the highest-ranked studies of the extraction set: when the "
                 "only metal-AM evidence is a generic mention in the abstract and the title names no AM process, the "
                 "record is coded E8 (%d records), because the ML may not be applied to an AM process." % n_gen)
        L.append("- Targeted check: every E3 decision that rested only on the Semantic Scholar 'Review' publication type "
                 "(no review wording in the title or abstract) was read; %d such records, %d recoded to I1 by override." % (
                     n_s2_only, sum(1 for l in ledger if l["decision_source"].startswith("manual") and "S2 review label only" in l["justification"])))
    L.append("")

    L.append("## 7. Flow (PRISMA-style counts)\n")
    excl = sum(cnt.get(c, 0) for c in ("E1", "E2", "E4", "E5", "E6", "E7"))
    L.append(md_table(["stage", "records"], [
        ("Records identified (hits reported, summed over %d streams)" % len(s2), tot_hits),
        ("Records retrieved", len(raw)),
        ("Records retained after per-stream caps", sum(1 for r in raw if r["retained"] == "1")),
        ("Records after deduplication (screened)", len(dedup)),
        ("Excluded E1 / E2 / E4 / E5 / E6 / E7", "%d / %d / %d / %d / %d / %d (total %d)" % tuple(
            [cnt.get(c, 0) for c in ("E1", "E2", "E4", "E5", "E6", "E7")] + [excl])),
        ("Routed to review list E3", cnt.get("E3", 0)),
        ("Pending full-text check E8 (counted separately)", cnt.get("E8", 0)),
        ("Included I1 (primary empirical metal-AM ML studies)", cnt.get("I1", 0)),
        ("Selected for evidence extraction", len(ext)),
        ("Review/survey articles on AI/ML for metal AM, 2019-2026 (reviews.csv)", len(rev)),
    ]))

    L.append("## 8. Selection of the evidence-extraction set\n")
    L.append("Rule: selection_rule.md (written before the rule was applied). Applied by scripts/select_extraction.py. "
             "The resulting set was replaced by the stratified random sample of sampling_rule.md before any study was drawn.\n")
    L.append("- Eligible I1 studies: %d" % sstats["eligible_I1"])
    L.append("- Step 1 core (any focused stream): %d%s" % (sstats["core_step1"],
             " (exceeds 110; all core studies kept as the rule requires)" if sstats["core_exceeds_110"] else ""))
    L.append("- Step 2 process-family fill (%s): %d added" % (sstats["fill_mode"], sstats["added_step2"]))
    L.append("- Step 3 I1 studies from 2025-2026 with open-access full text: %d added" % sstats["added_step3"])
    L.append("- Final extraction set: %d\n" % sstats["final"])
    fam = collections.Counter()
    for r in ext:
        for f in r["process_families"].split(";"):
            if f:
                fam[f] += 1
    L.append("By process family (a study naming several families counts in each):\n")
    L.append(md_table(["family", "studies"], sorted(fam.items(), key=lambda x: -x[1])))
    dim = collections.Counter()
    for r in ext:
        for f in r["focused_streams"].split(";"):
            if f:
                dim[f] += 1
    L.append("By focused stream (a study retrieved by several streams counts in each):\n")
    L.append(md_table(["stream", "dimension", "studies"], [(sid, name, dim.get(sid, 0)) for sid, name, _ in DIMENSIONS]))
    step = collections.Counter(r["selection_step"] for r in ext)
    L.append("By selection step: step 1 %d, step 2 %d, step 3 %d. With open-access full-text URL: %d.\n" % (
        step.get("1", 0), step.get("2", 0), step.get("3", 0), sum(1 for r in ext if r["oa_fulltext_url"])))

    L.append("## 9. Review and survey articles\n")
    L.append("- reviews.csv: %d review/survey articles on AI/ML for metal AM published 2019-2026 (E3 records with an "
             "AI/ML term, editorials and prefaces removed); %d carry at least one trustworthiness-related flag "
             "(trustworthiness, uncertainty, explainability, generalisation, robustness).\n" % (
                 len(rev), sum(1 for r in rev if int(r["n_flags"]) > 0)))

    L.append("## 10. Limitations\n")
    L.append("- No access to Scopus, Web of Science, IEEE Xplore or other subscription indexes; OpenAlex, the planned "
             "primary source, was unavailable on the search date (HTTP 429 without an API key). The search therefore "
             "rests on one index, Semantic Scholar.")
    L.append("- Semantic Scholar does not return abstracts for many publishers (notably Elsevier); %d deduplicated "
             "records had no abstract from Semantic Scholar or Crossref, so matching and screening for those records "
             "relied on titles. Recall for title/abstract matching is limited by the same gap." % src.get("none", 0))
    L.append("- Per-stream caps were applied with a transparent term-match ranking, not a search-engine relevance "
             "score; records beyond the caps were downloaded but not screened.")
    L.append("- Screening was rule-based keyword matching on titles and abstracts by a single screener. Keyword rules "
             "can misclassify, for example a study that names machine learning only in passing. Records coded E8 "
             "need full-text checks, and the I1 set needs confirmation at full-text extraction.")
    ki = read_csv(os.path.join(OUT, "known_items.csv"))
    dset = {r["doi"] for r in dedup}
    found = [k for k in ki if k["doi"] in dset]
    L.append("- Known-item check: %d widely cited reviews of machine learning for additive manufacturing in general "
             "(known_items.csv, DOIs verified in Crossref on 2026-10-15) were tested for presence; %d of them were "
             "retrieved (%s). Their absence indicates a recall gap for records whose titles/abstracts do not name a "
             "Block A process term or whose abstracts Semantic Scholar does not hold." % (
                 len(ki), len(found), "; ".join(k["doi"] for k in ki if k["doi"] not in dset) + " not retrieved"))
    L.append("- Citation counts are Semantic Scholar counts on the search date and favour older papers in the "
             "family fill step.")
    L.append("- Process-family labels come from title/abstract keywords and may be incomplete.\n")

    L.append("## 11. Files and rerun order\n")
    L.append("```\npython -I scripts/openalex_probe.py\npython -I scripts/search_s2.py\npython -I scripts/parse_raw.py\n"
             "python -I scripts/enrich_crossref.py\npython -I scripts/dedup.py\npython -I scripts/preprint_check.py\n"
             "python -I scripts/dedup.py\npython -I scripts/screen.py\npython -I scripts/spotcheck_sample.py\n"
             "python -I scripts/select_extraction.py\npython -I scripts/reviews.py\npython -I scripts/build_log.py\n```\n")
    L.append("Raw responses: raw_json/<stream>.json and raw_json/crossref_cache.jsonl; request log: query_log.jsonl.")
    with open(os.path.join(OUT, "search_log.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("search_log.md written")


if __name__ == "__main__":
    main()
