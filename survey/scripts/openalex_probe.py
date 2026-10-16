"""Probe OpenAlex availability with one title/abstract query and log the response status."""
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MAILTO, UA, log_query, utc_now  # noqa: E402

params = {"filter": 'title_and_abstract.search:"laser powder bed fusion" AND "machine learning",'
                    "from_publication_date:2015-01-01,to_publication_date:2026-10-15",
          "per-page": 1, "mailto": MAILTO}
url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
t = utc_now()
try:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
        status, body = r.status, r.read(400).decode("utf-8", "replace")
except urllib.error.HTTPError as e:
    status, body = e.code, e.read(600).decode("utf-8", "replace")
log_query({"source": "OpenAlex", "stream_id": "PROBE", "kind": "availability probe",
           "endpoint": "https://api.openalex.org/works", "query": params["filter"],
           "requested_utc": t, "http_status": [status], "response_excerpt": body})
print(status, body)
