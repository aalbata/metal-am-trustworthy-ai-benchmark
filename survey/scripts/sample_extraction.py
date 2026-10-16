"""Draw the stratified random extraction sample defined in sampling_rule.md."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

L = Path(__file__).resolve().parents[1] / "records"
SEED = 20261015
N = 60

led = pd.read_csv(L / "screening_ledger.csv")
inc = led[led.decision == "include"].copy()
streams = inc.streams.fillna("").astype(str)
inc["stratum"] = np.where(streams.str.contains(r"(?:^|;)F[1-6](?:;|$)", regex=True), "F", "G")
dedup = pd.read_csv(L / "records_dedup.csv")
keep = [c for c in ["ID", "abstract", "oa_url", "oa_fulltext_url", "venue", "authors"] if c in dedup.columns]
inc = inc.merge(dedup[keep], on="ID", how="left")

rng = np.random.default_rng(SEED)
parts = []
for s in ["F", "G"]:
    st = inc[inc.stratum == s].sort_values("ID").reset_index(drop=True)
    order = rng.permutation(len(st))
    st = st.iloc[order].reset_index(drop=True)
    st["stratum_size"] = len(st)
    st["random_position"] = np.arange(1, len(st) + 1)
    st["in_sample"] = st.random_position <= N
    parts.append(st)
out = pd.concat(parts, ignore_index=True)
out["study_id"] = [f"{s}{p:03d}" for s, p in zip(out.stratum, out.random_position)]
out.to_csv(L / "extraction_sample_order.csv", index=False, lineterminator="\n")
smp = out[out.in_sample]
smp.to_csv(L / "extraction_sample.csv", index=False, lineterminator="\n")
h = hashlib.sha256((L / "extraction_sample.csv").read_bytes()).hexdigest()
print("stratum sizes:", out.groupby("stratum").size().to_dict())
print("sample:", smp.groupby("stratum").size().to_dict(), "sha256", h)
print(smp.groupby("stratum").process_families.value_counts().to_string())
