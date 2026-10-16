"""Fetch Crossref metadata (type, container, abstract, relations) for every DOI in the
retained raw records. Batches of DOIs are requested with the multi-value doi filter;
DOIs containing a comma are requested individually. Output: crossref_meta.csv."""
import json
import os
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (MAILTO, OUT, RAW_DIR, http_get_json, log_query, norm_doi, read_csv,  # noqa: E402
                    strip_jats, utc_now, write_csv)

BATCH = 40
PAUSE = 1.0
SELECT = "DOI,type,container-title,abstract,relation,issued,title"
CACHE = os.path.join(RAW_DIR, "crossref_cache.jsonl")


def load_cache():
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                cache[d["doi"]] = d
    return cache


def main():
    raw = read_csv(os.path.join(OUT, "records_raw.csv"))
    dois = sorted({r["doi"] for r in raw if r["retained"] == "1" and r["doi"]})
    cache = load_cache()
    pending = [d for d in dois if d not in cache]
    batches = [[d] for d in pending if "," in d]
    plain = [d for d in pending if "," not in d]
    batches += [plain[i:i + BATCH] for i in range(0, len(plain), BATCH)]
    n_calls = 0
    n_failed = 0
    with open(CACHE, "a", encoding="utf-8") as cf:
        for b in batches:
            if len(b) == 1 and "," in b[0]:
                url = "https://api.crossref.org/works/" + urllib.parse.quote(b[0]) + "?mailto=" + MAILTO
            else:
                url = ("https://api.crossref.org/works?" + urllib.parse.urlencode(
                    {"filter": ",".join("doi:" + d for d in b), "rows": len(b), "select": SELECT,
                     "mailto": MAILTO}))
            t = utc_now()
            status, payload, attempts = http_get_json(url)
            n_calls += 1
            if status != 200:
                n_failed += 1
            items = []
            if status == 200:
                msg = payload.get("message", {})
                items = msg.get("items", [msg]) if "items" in msg else [msg]
            found = set()
            for it in items:
                d = norm_doi(it.get("DOI"))
                found.add(d)
                rec = {"doi": d, "status": status, "requested_utc": t,
                       "cr_type": it.get("type") or "", 
                       "cr_container": "; ".join(it.get("container-title") or []),
                       "cr_abstract": strip_jats(it.get("abstract")),
                       "cr_relation": json.dumps(it.get("relation") or {}, ensure_ascii=False),
                       "cr_issued": "-".join(str(x) for x in ((it.get("issued") or {}).get("date-parts") or [[None]])[0] if x)}
                cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cache[d] = rec
            if status == 200:
                for d in b:
                    if d not in found:
                        rec = {"doi": d, "status": 404, "requested_utc": t, "cr_type": "",
                               "cr_container": "", "cr_abstract": "", "cr_relation": "{}", "cr_issued": ""}
                        cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        cache[d] = rec
            cf.flush()
            time.sleep(PAUSE)
    log_query({"source": "Crossref REST API", "stream_id": "ENRICH", "kind": "metadata enrichment",
               "endpoint": "https://api.crossref.org/works", "query": "filter=doi:<batch of up to %d DOIs>" % BATCH,
               "requested_utc": utc_now(), "dois_requested": len(dois), "new_calls": n_calls,
               "failed_calls": n_failed,
               "dois_found": sum(1 for d in dois if cache.get(d, {}).get("cr_type"))})
    rows = [cache[d] for d in dois if d in cache]
    write_csv(os.path.join(OUT, "crossref_meta.csv"), rows,
              ["doi", "status", "requested_utc", "cr_type", "cr_container", "cr_abstract",
               "cr_relation", "cr_issued"])
    print("DOIs", len(dois), "with Crossref record", sum(1 for r in rows if r["cr_type"]))


if __name__ == "__main__":
    main()
