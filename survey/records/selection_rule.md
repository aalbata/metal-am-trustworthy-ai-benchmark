# Prespecified rule for the evidence-extraction set

Written on 2026-10-15 before the final screening and before select_extraction.py was run. At that
time only trial screening runs on a partial download (used to debug the screening rules) had
been made, and no selection counts had been computed.
The rule is applied mechanically by scripts/select_extraction.py to screening_ledger.csv (code I1 only)
joined to records_dedup.csv. No record is added or removed by hand after the rule is applied.

## Definitions

- Eligible pool: records with final screening code I1 (primary empirical metal additive
  manufacturing study with an implemented AI/ML model, title/abstract screening).
- Focused streams: F1 generalisation and evaluation validity, F2 calibration and uncertainty,
  F3 robustness, F4 explainability, F5 reproducibility, F6 computational efficiency.
- General streams: G01 to G81 (one process term crossed with one method term).
- Process family of a study: every family the screening rules matched in its title or
  abstract (LPBF, EB-PBF, DED, WAAM, BJ, MEX, PBF-unspecified, Metal-AM-unspecified,
  Other-metal-AM). A study that names two families counts for both.
- Open-access full text: a non-empty openAccessPdf URL in the Semantic Scholar record.
- Target size: 100 studies (midpoint of the intended 90 to 110).

## Steps

1. Core set. Select every eligible study retrieved by at least one focused stream (F1 to F6).
   If the core set alone exceeds 110 studies, every core study is kept and this is reported.
2. Process-family fill from the general streams. Candidates are eligible studies retrieved
   only by general streams (not by any focused stream).
   a. If the core set has fewer than 100 studies, the remaining slots up to 100 are filled in
      rounds over the four families in the fixed order LPBF, DED, WAAM, EB-PBF; in each round
      every family receives its highest-cited not yet selected candidate (Semantic Scholar
      citation count; ties broken by record ID). A family with no remaining candidate is
      skipped. Rounds stop when 100 studies are selected.
   b. If the core set has 100 or more studies, each of the four families that has fewer than
      five selected studies receives its highest-cited candidates until it has five or the
      candidates run out.
3. Recent open-access studies. Every eligible study published in 2025 or 2026 with an
   open-access full-text URL is added if not already selected, whatever the size reached.
4. Priority rank (informative only; nothing is removed). Every selected study receives a rank
   ordered by (i) number of focused streams that retrieved it, descending; (ii) selection
   step, 1 before 2 before 3; (iii) citation count, descending; (iv) record ID. If the final
   set is larger than can be extracted, the rank gives a transparent order for trimming, and
   any trimming must be reported as a deviation from this rule.

## Outputs

extraction_set.csv with ID, DOI, title, year, venue, process families, focused streams,
selection step and reason, priority rank, citation count, and open-access URL.
