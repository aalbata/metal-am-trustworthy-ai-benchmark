"""Run every search stream against the Semantic Scholar Graph API and store raw responses.

Every stream uses the boolean bulk endpoint (/paper/search/bulk), which returns every
match (paged by token); the per-stream cap is applied afterwards by parse_raw.py.

Usage: python -I search_s2.py [--only STREAM_ID ...] [--force]
"""
import json
import os
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RAW_DIR, WINDOW_END, WINDOW_START, http_get_json, log_query, utc_now)  # noqa: E402
from streams import all_streams  # noqa: E402

BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = ("paperId,externalIds,title,abstract,year,publicationDate,venue,journal,"
          "publicationTypes,authors,citationCount,isOpenAccess,openAccessPdf")
DATE_FILTER = "%s:%s" % (WINDOW_START, WINDOW_END)
PAUSE = 4.0


def run_stream(s):
    pages = []
    token = None
    while True:
        params = {"query": s["query"], "fields": FIELDS, "publicationDateOrYear": DATE_FILTER}
        if token:
            params["token"] = token
        url = BASE + "/paper/search/bulk?" + urllib.parse.urlencode(params)
        t = utc_now()
        status, payload, attempts = http_get_json(url)
        pages.append({"requested_utc": t, "url": url, "status": status, "attempts": attempts,
                      "payload": payload})
        token = payload.get("token") if status == 200 else None
        if not token:
            break
        time.sleep(PAUSE)
    return pages


def main():
    args = sys.argv[1:]
    force = "--force" in args
    only = [a for a in args if not a.startswith("--")]
    os.makedirs(RAW_DIR, exist_ok=True)
    for s in all_streams():
        if only and s["stream_id"] not in only:
            continue
        path = os.path.join(RAW_DIR, s["stream_id"] + ".json")
        if os.path.exists(path) and not force:
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
            if all(p["status"] == 200 for p in old["pages"]):
                continue
        pages = run_stream(s)
        ok = all(p["status"] == 200 for p in pages)
        total = pages[0]["payload"].get("total") if pages and pages[0]["status"] == 200 else None
        retrieved = sum(len(p["payload"].get("data") or []) for p in pages if p["status"] == 200)
        rec = {"stream": s, "pages": pages}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        log_query({"source": "Semantic Scholar Graph API", "stream_id": s["stream_id"],
                   "kind": s["kind"], "endpoint": BASE + "/paper/search/bulk",
                   "query": s["query"], "filters": {"publicationDateOrYear": DATE_FILTER},
                   "requested_utc": pages[0]["requested_utc"], "http_status": [p["status"] for p in pages],
                   "attempts": [p["attempts"] for p in pages], "total_reported": total,
                   "retrieved": retrieved, "ok": ok})
        print(s["stream_id"], "ok" if ok else "FAILED", total, retrieved, flush=True)
        time.sleep(PAUSE)


if __name__ == "__main__":
    main()
