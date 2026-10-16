"""Deduplicate retained raw records: (1) by DOI, (2) by normalised title, linking
preprint and peer-reviewed versions. Writes records_dedup.csv with stable IDs R0001...

IDs follow the order of first appearance (stream order G01..G81, F1..F6, RV, then rank).
Title merging requires a normalised title of at least 30 characters and publication
years no more than two years apart. Crossref has-preprint / is-preprint-of relations are
also used to link versions whose titles differ."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, norm_doi, norm_title, read_csv, write_csv  # noqa: E402
from streams import all_streams  # noqa: E402

PREPRINT_PREFIXES = ("10.48550/", "10.2139/", "10.21203/", "10.20944/", "10.36227/", "10.31224/",
                     "10.26434/", "10.1101/", "10.22541/", "10.31219/", "10.32388/")
PREPRINT_VENUE = re.compile(r"arxiv|ssrn|research square|preprints|techrxiv|engrxiv|chemrxiv|biorxiv|medrxiv|authorea", re.I)


def is_preprint(r, cr):
    name = r["journal_name"] or r["venue"] or ""
    if r["doi"].startswith(PREPRINT_PREFIXES):
        return not (name and not PREPRINT_VENUE.search(name))
    if cr.get("cr_type") == "posted-content":
        return True
    if not r["doi"] and (r["arxiv_id"] or PREPRINT_VENUE.search(r["venue"] or "")):
        return True
    if PREPRINT_VENUE.search(r["venue"] or "") and not r["journal_name"]:
        return True
    return False


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main():
    raw = [r for r in read_csv(os.path.join(OUT, "records_raw.csv")) if r["retained"] == "1"]
    if not raw:
        sys.exit("records_raw.csv holds no retained records; run search_s2.py and parse_raw.py first")
    crpath = os.path.join(OUT, "crossref_meta.csv")
    cr = {r["doi"]: r for r in read_csv(crpath)} if os.path.exists(crpath) else {}
    lpath = os.path.join(OUT, "preprint_links.csv")
    plinks = {}
    if os.path.exists(lpath):
        for l in read_csv(lpath):
            plinks[l["preprint_s2_paperId"] or ("doi:" + l["preprint_doi"])] = l
    for r in raw:
        l = plinks.get(r["s2_paperId"]) or plinks.get("doi:" + r["doi"])
        r["_pub_type"] = ""
        if l and l["published_doi"]:
            r["preprint_doi"] = r["doi"] or ("s2:" + r["s2_paperId"])
            r["doi"] = l["published_doi"]
            r["journal_name"] = r["venue"] = l["published_container"]
            r["_pub_type"] = l["published_type"]
    order = {s["stream_id"]: i for i, s in enumerate(all_streams())}
    raw.sort(key=lambda r: (order[r["stream_id"]], int(r["rank_in_stream"])))

    # Each raw row is first keyed by DOI, else by S2 paper id.
    keys, first_pos = [], {}
    for i, r in enumerate(raw):
        k = "doi:" + r["doi"] if r["doi"] else "s2:" + r["s2_paperId"]
        keys.append(k)
        first_pos.setdefault(k, i)
    uf = UF()
    for k in keys:
        uf.find(first_pos[k])
    n_after_doi = len(set(first_pos.values()))
    # S2 paper id links rows of the same paper that lack a DOI in one stream.
    by_pid = {}
    for i, r in enumerate(raw):
        if r["s2_paperId"]:
            if r["s2_paperId"] in by_pid:
                uf.union(first_pos[keys[i]], first_pos[keys[by_pid[r["s2_paperId"]]]])
            else:
                by_pid[r["s2_paperId"]] = i
    n_after_pid = len({uf.find(first_pos[k]) for k in keys})
    # Normalised title.
    title_links = 0
    by_title = {}
    for i, r in enumerate(raw):
        nt = norm_title(r["title"])
        if len(nt) < 30:
            continue
        if nt in by_title:
            j = by_title[nt]
            yi, yj = r["year"], raw[j]["year"]
            if yi and yj and abs(int(yi) - int(yj)) > 2:
                continue
            a, b = uf.find(first_pos[keys[i]]), uf.find(first_pos[keys[j]])
            if a != b:
                uf.union(a, b)
                title_links += 1
        else:
            by_title[nt] = i
    # Crossref preprint relations.
    rel_links = 0
    doi_pos = {r["doi"]: first_pos[keys[i]] for i, r in enumerate(raw) if r["doi"]}
    for d, m in cr.items():
        try:
            rel = json.loads(m.get("cr_relation") or "{}")
        except ValueError:
            continue
        for rtype in ("has-preprint", "is-preprint-of", "has-version", "is-version-of"):
            for it in rel.get(rtype, []) or []:
                other = norm_doi(it.get("id")) if it.get("id-type") == "doi" else ""
                if other and other in doi_pos and d in doi_pos:
                    a, b = uf.find(doi_pos[d]), uf.find(doi_pos[other])
                    if a != b:
                        uf.union(a, b)
                        rel_links += 1

    groups = {}
    for i in range(len(raw)):
        groups.setdefault(uf.find(first_pos[keys[i]]), []).append(i)
    out = []
    for n, root in enumerate(sorted(groups, key=lambda g: min(groups[g])), 1):
        members = [raw[i] for i in groups[root]]
        variants = {}
        for m in members:
            vk = m["doi"] or ("s2:" + m["s2_paperId"])
            variants.setdefault(vk, m)
        vlist = list(variants.values())

        def rank(m):
            c = cr.get(m["doi"], {})
            return (is_preprint(m, c), not m["doi"], not (m["abstract"] or c.get("cr_abstract")),
                    -int(m["cited_by_count"] or 0))
        vlist.sort(key=rank)
        canon = vlist[0]
        c = cr.get(canon["doi"], {})
        abstracts = [(m["abstract"], "S2") for m in members if m["abstract"]]
        abstracts += [(cr.get(m["doi"], {}).get("cr_abstract"), "Crossref") for m in vlist
                      if cr.get(m["doi"], {}).get("cr_abstract")]
        abstracts.sort(key=lambda a: -len(a[0]))
        streams = []
        for m in members:
            if m["stream_id"] not in streams:
                streams.append(m["stream_id"])
        oa = next((m["oa_url"] for m in [canon] + members if m["oa_url"]), "")
        linked = [m["doi"] or ("s2:" + m["s2_paperId"]) for m in vlist[1:]]
        basis = "single record"
        if any(m.get("preprint_doi") for m in members):
            basis = "preprint linked to published version (Crossref exact-title match)"
            linked += sorted({m["preprint_doi"] for m in members if m.get("preprint_doi")})
        elif len(vlist) > 1:
            basis = "title or relation link"
        elif len(members) > 1:
            basis = "same DOI/paper id across streams"
        out.append({
            "ID": "R%04d" % n, "doi": canon["doi"], "s2_paperId": canon["s2_paperId"],
            "title": canon["title"], "authors": canon["authors"], "year": canon["year"],
            "publication_date": canon["publication_date"],
            "venue": canon["journal_name"] or canon["venue"] or c.get("cr_container", ""),
            "s2_publication_types": canon["s2_publication_types"], "crossref_type": c.get("cr_type", "") or canon.get("_pub_type", ""),
            "is_preprint": int(is_preprint(canon, c)),
            "abstract": abstracts[0][0] if abstracts else "",
            "abstract_source": abstracts[0][1] if abstracts else "none",
            "cited_by_count": max(int(m["cited_by_count"] or 0) for m in members),
            "oa_url": oa, "streams": ";".join(streams), "n_streams": len(streams),
            "n_raw_rows": len(members), "linked_versions": ";".join(linked), "dedup_basis": basis,
        })
    fields = ["ID", "doi", "s2_paperId", "title", "authors", "year", "publication_date", "venue",
              "s2_publication_types", "crossref_type", "is_preprint", "abstract", "abstract_source",
              "cited_by_count", "oa_url", "streams", "n_streams", "n_raw_rows", "linked_versions", "dedup_basis"]
    write_csv(os.path.join(OUT, "records_dedup.csv"), out, fields)
    stats = {"retained_raw_rows": len(raw), "unique_after_doi_or_paperid_key": n_after_doi,
             "unique_after_paperid_link": n_after_pid, "title_links": title_links,
             "relation_links": rel_links, "unique_records": len(out),
             "records_with_linked_versions": sum(1 for o in out if o["linked_versions"])}
    with open(os.path.join(OUT, "dedup_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    print(stats)


if __name__ == "__main__":
    main()
