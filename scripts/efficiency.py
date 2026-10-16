"""Inference latency, throughput and peak memory for eight architecture x precision contexts.

For each architecture the trained run with the lowest training-manifest index
(Protocol R, OT, seed 20260916) is measured under FP32_STRICT and AMP_FP16,
each context in a fresh Python process.  The reported protocol requires CUDA.
``--device cpu`` runs an FP32 CPU timing that is labelled as not comparable
with the reported CUDA measurements.

Outputs (below the output root):
  derived/efficiency/<architecture>__<precision>.json
  results/efficiency.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import subprocess

import pandas as pd

import _setup
from amtrust import efficiency, splits, train


def child(argv) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true")
    p.add_argument("--architecture", required=True)
    p.add_argument("--precision", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmups", type=int, default=30)
    p.add_argument("--repeats", type=int, default=100)
    a = p.parse_args(argv)
    if a.device == "cuda":
        result = efficiency.measure_cuda(a.architecture, a.state, a.precision, a.warmups, a.repeats)
    else:
        result = efficiency.measure_cpu(a.architecture, a.state, a.warmups, a.repeats)
    result["environment"] = efficiency.environment()
    Path(a.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    if "--child" in sys.argv:
        return child(sys.argv[1:])
    p = _setup.parser(__doc__.splitlines()[0])
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()
    layout, config = _setup.context(args)
    ecfg = config["efficiency"]
    master = splits.load_master(layout.master_split_csv)
    runs = train.run_manifest(master, config)
    out_dir = layout.derived_dir / "efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)
    precisions = ecfg["precisions"] if args.device == "cuda" else ["FP32_CPU"]
    rows = []
    for architecture in config["training"]["architecture_order"]:
        chosen = min((r for r in runs if r["architecture"] == architecture), key=lambda r: r["manifest_index"])
        state = layout.run_dir(chosen["run_id"]) / "model_state.pt"
        if not state.is_file():
            print(f"Missing model state for {chosen['run_id']}")
            continue
        for precision in precisions:
            target = out_dir / f"{architecture}__{precision}.json"
            command = [sys.executable, "-I", str(Path(__file__).resolve()), "--child", "--architecture", architecture,
                       "--precision", precision, "--state", str(state), "--json", str(target),
                       "--device", args.device, "--warmups", str(ecfg["warmup_iterations"]),
                       "--repeats", str(ecfg["timed_iterations"])]
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                print(f"Context failed: {architecture} {precision}")
                continue
            result = json.loads(target.read_text(encoding="utf-8"))
            rows.append({**efficiency.flatten(result), "run_id": chosen["run_id"],
                         "comparable_with_reported": args.device == "cuda"})
            print(f"{architecture} {precision}: done")
    if not rows:
        return 1
    layout.results_dir.mkdir(parents=True, exist_ok=True)
    name = "efficiency.csv" if args.device == "cuda" else "efficiency_cpu_not_comparable.csv"
    pd.DataFrame(rows).to_csv(layout.results_dir / name, index=False, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
