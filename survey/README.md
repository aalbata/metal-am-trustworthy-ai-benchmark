# Survey search, screening, sampling and evidence coding

This folder contains the scripts and records behind the structured critical survey in the article. The scripts use the Python standard library (search, screening), pandas (sampling, evidence map) and openpyxl (Online Resource 2).

## Records

| File | Content |
|---|---|
| `records/query_log.csv` | The 88 queries with request time (UTC), HTTP status and number of records retrieved, tabulated from the request log that `search_s2.py` writes (`query_log.jsonl`, not redistributed). |
| `records/dedup_stats.json`, `records/preprint_links.csv` | Deduplication counts; links between preprints and their published versions. |
| `records/known_items.csv` | Widely cited reviews used to test the recall of the search. |
| `records/selection_stats.json` | Counts of the selection rule that the stratified sample replaced. |
| `records/records_dedup.csv` | 1,650 unique records retrieved by the 88 queries (identifier, DOI, title, authors, year, venue, publication type, citation count, open-access link, retrieving queries). Abstracts are not redistributed. |
| `records/screening_ledger.csv` | Screening decision, rule code and justification for every record. Codes: I1 eligible primary study; E1 not metal AM; E2 no implemented ML model; E3 review article; E4 not a peer-reviewed article or full conference paper; E7 simulation without ML; E8 insufficient information in title and abstract. |
| `records/screening_overrides.csv`, `records/spotcheck.csv` | Decisions set by recorded override, and the check by the authors of a random sample of 94 screening decisions against the rules. |
| `records/survey_coverage.csv` | Coding of the surveys compared in Table 1 of the article (scope, method, code per dimension, version read, and the sections on which each code rests). |
| `records/reviews.csv` | Review and survey articles on AI or ML for metal AM published 2019 to 2026 (screening code E3 with an AI or ML term; editorials and prefaces removed). |
| `records/selection_rule.md`, `records/sampling_rule.md` | The rule that defined the candidate set and the stratified random sample that replaced its trimming step. |
| `records/extraction_sample_order.csv` | All 1,206 eligible studies with their stratum (F: retrieved by a dimension query; G: general queries only), random position and sample flag (seed 20261015). |
| `records/evidence_extraction.csv` | Extracted evidence for every sampled study, with access level and source locations. |
| `records/eligibility_overrides.csv` | Sampled studies judged ineligible on reading. |
| `records/verification.csv` | Second check, against the source, of a fixed set of coded fields (named in the field column) of every eligible study coded from full text and of two sampled studies later found ineligible (field, first value, checked value, agreement, source location, version read). The checks were made by the authors. |
| `records/coding_corrections.csv` | Corrections applied to the coded indicators, with reasons and source locations. A reason beginning Verification comes from the second check, Rule from the coding rules of Online Resource 1, Section S3, applied to all full-text studies, and Follows from marks an indicator that changes with a corrected one. |
| `records/evidence_coded.csv` | Extracted evidence with the coded indicators used in the evidence map. |
| `records/evidence_map.csv`, `records/landscape.csv` | Indicator counts per stratum and stratum-weighted proportions; counts of processes, tasks, inputs and models. |

## Scripts

| Script | Step |
|---|---|
| `scripts/streams.py`, `scripts/search_s2.py` | Query definitions and retrieval from the Semantic Scholar bulk search interface. |
| `scripts/parse_raw.py` | Parsing of responses and per-query ranking and limits. |
| `scripts/enrich_crossref.py`, `scripts/preprint_check.py`, `scripts/dedup.py` | Metadata completion, linking of preprints to published versions, deduplication. |
| `scripts/screen.py`, `scripts/spotcheck_sample.py`, `scripts/reviews.py` | Rule-based screening, sampling of decisions for checking, review list. |
| `scripts/select_extraction.py`, `scripts/sample_extraction.py` | Candidate set and stratified random sample. |
| `scripts/evidence_map.py` | Coding of indicators and the evidence map. |
| `scripts/build_log.py` | Search and screening log (needs the raw responses of a rerun search). |
| `scripts/build_online_resource2.py` | Online Resource 2 of the article, built from `records/`. |

All scripts read and write the files in `records/`. Raw search responses and abstracts are not redistributed, so the steps from search to screening run only after a new search, which returns the index content at the time of the rerun; counts can then differ from those reported for the search of 15 October 2026. The query log is in `records/query_log.csv`. The evidence map is reproduced exactly, and Online Resource 2 with identical cell contents, from the shipped records:

```powershell
python survey\scripts\evidence_map.py
python survey\scripts\build_online_resource2.py Online_Resource_2.xlsx
```
