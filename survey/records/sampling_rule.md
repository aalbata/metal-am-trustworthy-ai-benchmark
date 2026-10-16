# Stratified random sample for full-text evidence extraction

Written on 2026-10-15 after the selection rule in selection_rule.md produced 681 studies
(too many for full-text extraction) and before any sample was drawn. This rule replaces the
priority-rank trimming foreseen in step 4 of selection_rule.md; this is a deviation from that
rule and is reported as such.

Population: all records with final screening decision "include" (code I1) in
screening_ledger.csv.

Strata:
- F (trust-focused): I1 records retrieved by at least one focused stream F1 to F6.
- G (general): I1 records retrieved only by general streams G01 to G81.

Draw: simple random sample without replacement of 60 records per stratum, using
numpy.random.default_rng(20261015) applied to each stratum sorted by record ID (stratum F
first, then G). The full random order of each stratum is saved, so that the records after
position 60 form an ordered reserve list.

Replacement: a sampled record that proves ineligible at full text (criteria E1 to E7 of
the screening rules) is recorded with its exclusion reason and replaced by the next record
of the reserve list of its stratum. A record whose full text cannot be obtained is NOT
replaced; it is extracted from the abstract and marked "abstract only".

Estimation: proportions are reported per stratum and, for the whole I1 population, as a
stratum-weighted proportion with weights equal to stratum sizes. Counts of papers describe
reporting practice, not effectiveness.
