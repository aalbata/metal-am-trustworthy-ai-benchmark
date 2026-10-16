"""Shared helpers for the literature search pipeline (standard library only)."""
import csv
import datetime as dt
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "records"))
RAW_DIR = os.path.join(OUT, "raw_json")
QUERY_LOG = os.path.join(OUT, "query_log.jsonl")
MAILTO = "aalbataineh@hct.ac.ae"
UA = "AM-survey-literature-search/1.0 (mailto:%s)" % MAILTO

WINDOW_START = "2015-01-01"
WINDOW_END = "2026-10-15"

csv.field_size_limit(10_000_000)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def http_get_json(url, max_attempts=8, base_sleep=3.0):
    """GET a JSON document with exponential back-off on HTTP 429/5xx.

    Returns (status, payload, attempts)."""
    attempt = 0
    while True:
        attempt += 1
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, json.load(r), attempt
        except urllib.error.HTTPError as e:
            status = e.code
            body = e.read().decode("utf-8", "replace")[:500]
            if status in (429, 500, 502, 503, 504) and attempt < max_attempts:
                time.sleep(base_sleep * (2 ** (attempt - 1)))
                continue
            return status, {"error": body}, attempt
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < max_attempts:
                time.sleep(base_sleep * (2 ** (attempt - 1)))
                continue
            return -1, {"error": str(e)}, attempt


def log_query(entry):
    with open(QUERY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def norm_doi(doi):
    if not doi:
        return ""
    d = doi.strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    d = re.sub(r"^doi:\s*", "", d)
    return d


def norm_title(t):
    t = (t or "").lower()
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def strip_jats(s):
    if not s:
        return ""
    s = re.sub(r"<jats:title>.*?</jats:title>", " ", s, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()
