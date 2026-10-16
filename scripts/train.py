"""Train and evaluate one benchmark run (or a smoke-test subset of one run).

Select the run by ``--run-id`` or by its 1-based ``--index`` in the 96-run
training manifest (architecture, then task, then seed).  Outputs go to
``<output root>/runs/<run id>/``: model_state.pt, training_history.csv,
test_predictions.csv and test_metrics.json.

The reported runs require CUDA.  ``--allow-cpu`` permits a CPU run without
mixed precision; together with ``--max-epochs`` and the ``--limit-*`` options
it serves as a functional smoke test whose numbers are not comparable with the
reported results.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import _setup
from amtrust import splits, train
from amtrust.models import pretrained_file


def main() -> int:
    p = _setup.parser(__doc__.splitlines()[0])
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id")
    group.add_argument("--index", type=int)
    p.add_argument("--weights-dir", default=None, help="Folder with the torchvision ImageNet weights files.")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--run-dir", default=None, help="Output folder for this run (default: <output root>/runs/<run id>).")
    args = p.parse_args()
    layout, config = _setup.context(args)

    master = splits.load_master(layout.master_split_csv)
    manifest = train.run_manifest(master, config)
    if args.run_id:
        matches = [r for r in manifest if r["run_id"] == args.run_id]
    else:
        matches = [r for r in manifest if r["manifest_index"] == args.index]
    if not matches:
        print("Run not found in the training manifest.")
        return 1
    run = matches[0]
    limits = {k: v for k, v in (("train", args.limit_train), ("val", args.limit_val), ("test", args.limit_test)) if v}
    weights_dir = Path(args.weights_dir) if args.weights_dir else layout.pretrained
    weights = pretrained_file(run["architecture"], config, weights_dir)
    out_dir = Path(args.run_dir) if args.run_dir else layout.run_dir(run["run_id"])
    print(f"Run {run['manifest_index']}: {run['run_id']}")
    record = train.train_run(run, master, layout.data, out_dir, weights, config, allow_cpu=args.allow_cpu,
                             max_epochs=args.max_epochs, limits=limits or None)
    shown = {k: record[k] for k in ("run_id", "device", "amp", "train_n", "val_n", "test_n", "epochs_run",
                                    "best_epoch", "macro_f1", "balanced_accuracy", "auroc", "auprc", "nll", "brier")}
    print(json.dumps(shown, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
