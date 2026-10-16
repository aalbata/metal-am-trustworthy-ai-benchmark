"""Draw a seeded, code-stratified random sample of screening decisions for a check of the
screening rules. Writes spotcheck_sample.csv (records to read); the reading is recorded in
spotcheck.csv (ID, rule_code_at_check, screener_code, agree, note, action)."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, read_csv, write_csv  # noqa: E402

SEED = 20261015
QUOTA = {"I1": 40, "E3": 15, "E8": 10, "E1": 10, "E2": 5, "E7": 5, "E4": 10}


def main():
    ledger = read_csv(os.path.join(OUT, "screening_ledger.csv"))
    recs = {r["ID"]: r for r in read_csv(os.path.join(OUT, "records_dedup.csv"))}
    rng = random.Random(SEED)
    out = []
    for code, n in QUOTA.items():
        pool = sorted([l for l in ledger if l["rule_code"] == code], key=lambda l: l["ID"])
        for l in rng.sample(pool, min(n, len(pool))):
            r = recs[l["ID"]]
            out.append({"ID": l["ID"], "rule_code": code, "justification": l["justification"],
                        "title": r["title"], "venue": r["venue"], "year": r["year"],
                        "crossref_type": r["crossref_type"], "abstract": r["abstract"]})
    write_csv(os.path.join(OUT, "spotcheck_sample.csv"), out,
              ["ID", "rule_code", "justification", "title", "venue", "year", "crossref_type", "abstract"])
    print("sample", len(out))


if __name__ == "__main__":
    main()
