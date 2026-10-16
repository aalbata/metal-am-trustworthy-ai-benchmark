"""Build reviews.csv: review/survey articles on AI/ML for metal AM (2019-2026).

Source: records coded E3 in screening_ledger.csv (editorials and prefaces excluded) whose
title or abstract contains an AI/ML term, published 2019-2026. Flags mark explicit
attention to trustworthiness, uncertainty, explainability, generalisation or robustness.
Closeness score = sum over flags of (2 if the flag term is in the title else 1 if only in
the abstract); records are ordered by closeness, then citation count, then ID."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, read_csv, write_csv  # noqa: E402
from screen import EDITORIAL, ML_STRONG, ML_WEAK, metal_status  # noqa: E402

FLAGS = [
    ("flag_trustworthiness", re.compile(r"trustworth|\btrust\b|trusted|certif|qualification", re.I)),
    ("flag_uncertainty", re.compile(r"uncertaint|bayesian|probabilistic|calibrat|confidence interval", re.I)),
    ("flag_explainability", re.compile(r"explainab|interpretab|\bxai\b|\bshap\b|black[- ]box", re.I)),
    ("flag_generalisation", re.compile(r"generali[sz]|transferab|domain (shift|adaptation)|out[- ]of[- ]distribution|transfer learning", re.I)),
    ("flag_robustness", re.compile(r"robust|reliab", re.I)),
]
TOPICS = [("monitoring", r"monitor|in[- ]situ|sensing|sensor"), ("defects", r"defect|porosity|anomal|crack"),
          ("process optimisation", r"optimi[sz]|parameter"), ("properties", r"propert|fatigue|mechanical|tensile|hardness"),
          ("microstructure", r"microstructur|grain|texture"), ("melt pool", r"melt[- ]?pool"),
          ("control", r"control"), ("digital twin", r"digital twin"), ("design", r"design"),
          ("physics-informed", r"physics[- ]informed|physics[- ]based"), ("data/datasets", r"dataset|data management|database")]


def main():
    ledger = read_csv(os.path.join(OUT, "screening_ledger.csv"))
    recs = {r["ID"]: r for r in read_csv(os.path.join(OUT, "records_dedup.csv"))}
    out = []
    for l in ledger:
        if l["reason_code"] != "E3":
            continue
        r = recs[l["ID"]]
        title, ab = r["title"] or "", r["abstract"] or ""
        text = title + " " + ab
        if EDITORIAL.search(title) or not r["year"] or not (2019 <= int(r["year"]) <= 2026):
            continue
        if not (ML_STRONG.search(text) or ML_WEAK.search(text)):
            continue
        row = {"ID": r["ID"], "doi": r["doi"], "title": title, "year": r["year"], "venue": r["venue"],
               "cited_by_count": r["cited_by_count"], "streams": r["streams"],
               "process_families": ";".join(metal_status(text)[1]),
               "abstract_available": "yes" if ab else "no"}
        close = 0
        for name, rx in FLAGS:
            if rx.search(title):
                row[name], close = "title", close + 2
            elif rx.search(ab):
                row[name], close = "abstract", close + 1
            else:
                row[name] = ""
        row["n_flags"] = sum(1 for n, _ in FLAGS if row[n])
        row["closeness_score"] = close
        topics = [t for t, rx in TOPICS if re.search(rx, text, re.I)]
        first = re.split(r"(?<=[.!?])\s+", ab)[0] if ab else ""
        row["scope_note"] = ("topics: " + ", ".join(topics) if topics else "topics: not stated") + \
            ((" | abstract opening: " + first[:220]) if first else " | no abstract in records")
        out.append(row)
    out.sort(key=lambda r: (-r["closeness_score"], -int(r["cited_by_count"] or 0), r["ID"]))
    fields = ["ID", "doi", "title", "year", "venue", "cited_by_count", "process_families", "scope_note",
              "abstract_available"] + [n for n, _ in FLAGS] + ["n_flags", "closeness_score", "streams"]
    write_csv(os.path.join(OUT, "reviews.csv"), out, fields)
    print("reviews", len(out), "with any flag", sum(1 for r in out if r["n_flags"]))


if __name__ == "__main__":
    main()
