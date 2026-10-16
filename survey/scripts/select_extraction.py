"""Apply the prespecified rule in selection_rule.md and write extraction_set.csv."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, read_csv, write_csv  # noqa: E402
from streams import DIMENSIONS  # noqa: E402

TARGET = 100
CORE_LIMIT = 110
FAMILIES = ["LPBF", "DED", "WAAM", "EB-PBF"]
MIN_PER_FAMILY = 5
FOCUSED = [d[0] for d in DIMENSIONS]


def main():
    ledger = {r["ID"]: r for r in read_csv(os.path.join(OUT, "screening_ledger.csv"))}
    recs = {r["ID"]: r for r in read_csv(os.path.join(OUT, "records_dedup.csv"))}
    pool = []
    for i, l in ledger.items():
        if l["reason_code"] != "I1":
            continue
        r = recs[i]
        streams = r["streams"].split(";")
        pool.append({"ID": i, "rec": r, "fams": [f for f in l["process_families"].split(";") if f],
                     "focused": [s for s in streams if s in FOCUSED],
                     "cites": int(r["cited_by_count"] or 0), "year": int(r["year"] or 0),
                     "oa": r["oa_url"]})
    selected = {}

    def add(p, step, reason):
        if p["ID"] not in selected:
            selected[p["ID"]] = dict(p, step=step, reason=reason)

    core = [p for p in pool if p["focused"]]
    for p in sorted(core, key=lambda p: p["ID"]):
        add(p, 1, "core: retrieved by focused stream(s) " + ",".join(p["focused"]))
    n_core = len(selected)

    cands = [p for p in pool if not p["focused"]]

    def best(fam):
        c = [p for p in cands if fam in p["fams"] and p["ID"] not in selected]
        c.sort(key=lambda p: (-p["cites"], p["ID"]))
        return c[0] if c else None

    fill_mode = ""
    if n_core < TARGET:
        fill_mode = "fill to %d in family rounds" % TARGET
        while len(selected) < TARGET:
            added = False
            for fam in FAMILIES:
                if len(selected) >= TARGET:
                    break
                b = best(fam)
                if b:
                    add(b, 2, "family fill (%s): highest-cited general-stream study not yet selected" % fam)
                    added = True
            if not added:
                break
    else:
        fill_mode = "minimum %d per family" % MIN_PER_FAMILY
        for fam in FAMILIES:
            while sum(1 for s in selected.values() if fam in s["fams"]) < MIN_PER_FAMILY:
                b = best(fam)
                if not b:
                    break
                add(b, 2, "family minimum (%s): highest-cited general-stream study" % fam)
    n_after_fill = len(selected)

    for p in sorted(pool, key=lambda p: p["ID"]):
        if p["year"] in (2025, 2026) and p["oa"]:
            add(p, 3, "published %d with open-access full text" % p["year"])
    n_final = len(selected)

    rows = sorted(selected.values(), key=lambda s: (-len(s["focused"]), s["step"], -s["cites"], s["ID"]))
    out = []
    for rank, s in enumerate(rows, 1):
        r = s["rec"]
        out.append({"ID": s["ID"], "priority_rank": rank, "selection_step": s["step"],
                    "selection_reason": s["reason"], "focused_streams": ";".join(s["focused"]),
                    "process_families": ";".join(s["fams"]), "doi": r["doi"], "title": r["title"],
                    "year": r["year"], "venue": r["venue"], "cited_by_count": s["cites"],
                    "oa_fulltext_url": s["oa"], "all_streams": r["streams"]})
    write_csv(os.path.join(OUT, "extraction_set.csv"), out,
              ["ID", "priority_rank", "selection_step", "selection_reason", "focused_streams",
               "process_families", "doi", "title", "year", "venue", "cited_by_count", "oa_fulltext_url",
               "all_streams"])
    stats = {"eligible_I1": len(pool), "core_step1": n_core, "core_exceeds_110": n_core > CORE_LIMIT,
             "fill_mode": fill_mode, "after_step2": n_after_fill, "added_step2": n_after_fill - n_core,
             "added_step3": n_final - n_after_fill, "final": n_final}
    with open(os.path.join(OUT, "selection_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    print(stats)


if __name__ == "__main__":
    main()
