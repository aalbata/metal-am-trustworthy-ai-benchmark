"""Frozen-embedding reference baselines (majority, linear SVM, random forest) for the eight tasks.

Outputs (below the output root):
  derived/resnet18_embeddings.npy and derived/embedding_rows.csv  (CPU feature cache)
  derived/baseline_fits.csv  one row per fit (40 fits)
  results/baselines.csv      mean and SD over random-forest seeds per task and baseline
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import _setup
from amtrust import baselines, data, splits
from amtrust.models import pretrained_file


def main() -> int:
    p = _setup.parser(__doc__.splitlines()[0])
    p.add_argument("--weights-dir", default=None)
    args = p.parse_args()
    layout, config = _setup.context(args)
    manifest = data.load_manifest(layout.manifest_csv)
    master = splits.load_master(layout.master_split_csv)
    layout.derived_dir.mkdir(parents=True, exist_ok=True)
    cache = layout.derived_dir / "resnet18_embeddings.npy"
    rows_csv = layout.derived_dir / "embedding_rows.csv"
    primary = splits.primary_cohort(manifest)
    order = baselines.embedding_order(primary)
    if cache.is_file() and rows_csv.is_file():
        X = np.load(cache)
        order = pd.read_csv(rows_csv, dtype={"benchmark_sample_id": str})
    else:
        weights = pretrained_file("resnet18", config,
                                  Path(args.weights_dir) if args.weights_dir else layout.pretrained)
        paths = [data.image_path(layout.data, r) for r in order["image_relpath"]]
        X = baselines.extract_embeddings(order, paths, weights)
        np.save(cache, X)
        order[["benchmark_sample_id", "modality", "original_member"]].to_csv(rows_csv, index=False, encoding="utf-8")
    row_of = {sid: i for i, sid in enumerate(order["benchmark_sample_id"])}
    fits = []
    for _, task in splits.task_table(master).iterrows():
        sub = master[(master["protocol"] == task.protocol) & (master["modality"] == task.modality)
                     & (master["fold_id"] == task.fold_id)]
        idx = {}
        for name in ("train", "test"):
            part = sub[sub["partition"] == name]
            idx[name] = np.sort(np.asarray([row_of[s] for s in part["benchmark_sample_id"]], dtype=np.int64))
        label_of = dict(zip(sub["benchmark_sample_id"], sub["binary_label"]))
        ids = order["benchmark_sample_id"].to_numpy()
        y_train = np.asarray([label_of[s] for s in ids[idx["train"]]], dtype=np.int8)
        y_test = np.asarray([label_of[s] for s in ids[idx["test"]]], dtype=np.int8)
        name = splits.task_key(task.protocol, task.modality, task.fold_id)
        for record in baselines.fit_task(X[idx["train"]].astype(np.float32), y_train,
                                         X[idx["test"]].astype(np.float32), y_test, config):
            fits.append({"task": name, "protocol": task.protocol, "modality": task.modality, "fold_id": task.fold_id,
                         "test_n": int(len(y_test)), **record})
        print(f"{name}: done")
    fit_frame = pd.DataFrame(fits)
    fit_frame.to_csv(layout.derived_dir / "baseline_fits.csv", index=False, encoding="utf-8")
    layout.results_dir.mkdir(parents=True, exist_ok=True)
    baselines.summarise(fit_frame, int(config["metrics"]["group_sd_ddof"])).to_csv(
        layout.results_dir / "baselines.csv", index=False, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
