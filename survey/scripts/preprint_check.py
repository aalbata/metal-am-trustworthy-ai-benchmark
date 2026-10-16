"""Look up a peer-reviewed version of each preprint record in Crossref by exact title.

For every record in records_dedup.csv with is_preprint=1 or a preprint-server DOI, Crossref /works is queried with
query.title=<title> restricted to journal articles and proceedings articles. A version is
accepted only when its normalised title equals the preprint's normalised title. Matches
are written to preprint_links.csv; dedup.py then uses the published DOI as the canonical
record. Run order: dedup.py, preprint_check.py, dedup.py."""
import json
import os
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dedup import PREPRINT_PREFIXES  # noqa: E402
from common import (MAILTO, OUT, RAW_DIR, http_get_json, log_query, norm_doi, norm_title,  # noqa: E402
                    read_csv, utc_now, write_csv)

CACHE = os.path.join(RAW_DIR, "crossref_title_cache.jsonl")
PAUSE = 1.0


def main():
    recs = read_csv(os.path.join(OUT, "records_dedup.csv"))
    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE, encoding="utf-8"):
            d = json.loads(line)
            cache[d["key"]] = d
    links_path = os.path.join(OUT, "preprint_links.csv")
    prev = read_csv(links_path) if os.path.exists(links_path) else []
    links = {(l["preprint_doi"], l["preprint_s2_paperId"]): l for l in prev}
    n_calls = 0
    pending = [r for r in recs if r["is_preprint"] == "1" or r["doi"].startswith(PREPRINT_PREFIXES)]
    with open(CACHE, "a", encoding="utf-8") as cf:
        for r in pending:
            key = norm_title(r["title"])
            if key not in cache:
                url = "https://api.crossref.org/works?" + urllib.parse.urlencode({
                    "query.title": r["title"], "rows": 5, "mailto": MAILTO,
                    "filter": "type:journal-article,type:proceedings-article",
                    "select": "DOI,title,type,container-title,issued"})
                t = utc_now()
                status, payload, _ = http_get_json(url)
                n_calls += 1
                items = payload.get("message", {}).get("items", []) if status == 200 else []
                cache[key] = {"key": key, "requested_utc": t, "status": status, "items": items}
                cf.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
                time.sleep(PAUSE)
            for it in cache[key]["items"]:
                if norm_title(" ".join(it.get("title") or [])) == key:
                    links[(r["doi"], r["s2_paperId"])] = {
                        "preprint_doi": r["doi"], "preprint_s2_paperId": r["s2_paperId"], "title": r["title"],
                        "published_doi": norm_doi(it.get("DOI")), "published_type": it.get("type", ""),
                        "published_container": "; ".join(it.get("container-title") or []),
                        "published_year": ((it.get("issued") or {}).get("date-parts") or [[""]])[0][0],
                        "match": "exact normalised title"}
                    break
    write_csv(links_path, list(links.values()),
              ["preprint_doi", "preprint_s2_paperId", "title", "published_doi", "published_type",
               "published_container", "published_year", "match"])
    log_query({"source": "Crossref REST API", "stream_id": "PREPRINT", "kind": "published-version lookup",
               "endpoint": "https://api.crossref.org/works", "query": "query.title=<preprint title>, filter=type:journal-article,type:proceedings-article",
               "requested_utc": utc_now(), "preprints_checked": len(pending), "new_calls": n_calls,
               "published_versions_found": len(links)})
    print("preprints", len(pending), "published versions found", len(links))


if __name__ == "__main__":
    main()
