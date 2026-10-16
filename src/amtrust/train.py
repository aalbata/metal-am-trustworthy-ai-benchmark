"""Training and test evaluation of one benchmark run.

One run is (architecture, protocol, modality, fold, seed).  The procedure is:

1. Seed ``random``, NumPy and torch (CPU and CUDA) with the run seed, enable
   deterministic algorithms, disable cuDNN benchmarking and enable cuDNN
   determinism.  ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` must be set before CUDA
   is initialised (``scripts/run_all.py`` starts every run in a fresh process
   with this variable set).
2. Rows of each partition are sorted by sample identifier.  The training loader
   shuffles with a ``torch.Generator`` seeded with the run seed; batch size 8,
   ``num_workers=0``, ``drop_last=False``; no augmentation.
3. The network is built with ImageNet weights and a new two-class head
   (:func:`amtrust.models.create_model`), then trained end to end with AdamW
   (lr 1e-4, weight decay 1e-4, no scheduler) on unweighted cross-entropy,
   with float16 autocast and gradient scaling on CUDA.
4. After every epoch the validation NLL (mean cross-entropy, FP32, no
   autocast) is computed.  The model state is saved whenever the validation NLL
   is strictly lower than the best so far.  Early stopping uses a separate
   reference: an epoch counts as an improvement only if its NLL is lower than
   the reference minus 1e-4, in which case the reference is updated; training
   stops after 6 consecutive epochs without such an improvement or after 30
   epochs.
5. The best model state is restored and the test set is evaluated in FP32;
   Defect is predicted when P(Defect) >= 0.5.

Running on CPU (``allow_cpu=True``) disables autocast and gradient scaling.
CPU runs are functional checks and are not numerically comparable with the
reported CUDA runs.
"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data import image_path
from .metrics import classification_metrics
from .models import create_model, model_state_sha256
from .preprocessing import preprocess_image

PREDICTION_FIELDS = ["run_id", "sample_id", "y_true", "strict_eligible", "logit_good", "logit_defect",
                     "p_good", "p_defect", "y_pred"]
HISTORY_FIELDS = ["epoch", "train_nll", "validation_nll", "best_updated", "significant_improvement",
                  "bad_epochs", "early_stop_triggered"]


def run_id(architecture: str, protocol: str, modality: str, fold_id: str, seed: int) -> str:
    return f"{architecture}__{protocol}__{modality}__{fold_id}__seed_{seed}"


def run_manifest(master: pd.DataFrame, config: dict) -> list[dict]:
    """The 96 runs in training order: architecture, then task, then seed."""
    tasks = (master[["protocol", "modality", "fold_id"]].drop_duplicates()
             .sort_values(["protocol", "modality", "fold_id"]).reset_index(drop=True))
    runs = []
    for architecture in config["training"]["architecture_order"]:
        for _, task in tasks.iterrows():
            for seed in config["training"]["seeds"]:
                runs.append({
                    "manifest_index": len(runs) + 1,
                    "run_id": run_id(architecture, task.protocol, task.modality, task.fold_id, seed),
                    "architecture": architecture, "protocol": task.protocol, "modality": task.modality,
                    "fold_id": task.fold_id, "seed": int(seed),
                })
    return runs


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def partition_rows(master: pd.DataFrame, protocol: str, modality: str, fold_id: str) -> dict:
    """Rows of each partition of one task, sorted by sample identifier."""
    task = master[(master["protocol"] == protocol) & (master["modality"] == modality) & (master["fold_id"] == fold_id)]
    out = {}
    for name, short in (("train", "train"), ("validation", "val"), ("test", "test")):
        part = task[task["partition"] == name].copy()
        part = part.sort_values("benchmark_sample_id", key=lambda s: s.astype(str), kind="mergesort")
        out[short] = part.reset_index(drop=True)
    return out


class BenchmarkDataset(Dataset):
    """Preprocessed images with labels, sample identifiers and strict-cohort flags."""

    def __init__(self, rows: pd.DataFrame, data_root: Path):
        self.rows = rows.to_dict("records")
        self.data_root = Path(data_root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "image": preprocess_image(image_path(self.data_root, row["image_relpath"]), row["modality"]),
            "label": int(row["binary_label"]),
            "sample_id": str(row["benchmark_sample_id"]),
            "strict_eligible": bool(row["cohort_strict_sensitivity"]),
        }


def make_loader(rows, data_root, batch_size, shuffle=False, generator=None, num_workers=0) -> DataLoader:
    return DataLoader(BenchmarkDataset(rows, data_root), batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=num_workers, drop_last=False, pin_memory=False)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """FP32 inference returning labels, logits, probabilities and identifiers."""
    model.eval()
    labels, logits, probs, ids, strict = [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            out = model(images)
            labels.extend(batch["label"].tolist())
            logits.extend(out.detach().cpu().numpy().tolist())
            probs.extend(torch.softmax(out, dim=1).detach().cpu().numpy().tolist())
            ids.extend(list(batch["sample_id"]))
            strict.extend(batch["strict_eligible"].tolist())
    return {"labels": np.asarray(labels, dtype=np.int64), "logits": np.asarray(logits, dtype=np.float32),
            "probabilities": np.asarray(probs, dtype=np.float32), "sample_ids": ids, "strict": strict}


def write_predictions(path: Path, run: str, result: dict, threshold: float = 0.5) -> None:
    rows = []
    for i, sid in enumerate(result["sample_ids"]):
        p = result["probabilities"][i]
        rows.append({
            "run_id": run, "sample_id": sid, "y_true": int(result["labels"][i]),
            "strict_eligible": bool(result["strict"][i]),
            "logit_good": float(result["logits"][i, 0]), "logit_defect": float(result["logits"][i, 1]),
            "p_good": float(p[0]), "p_defect": float(p[1]), "y_pred": int(p[1] >= threshold),
        })
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def train_run(run: dict, master: pd.DataFrame, data_root: Path, output_dir: Path, weights_path: Path | None,
              config: dict, allow_cpu: bool = False, max_epochs: int | None = None,
              limits: dict | None = None) -> dict:
    """Train one run and write model state, history, predictions and metrics into ``output_dir``.

    ``limits`` restricts the number of rows per partition (for example
    ``{"train": 16, "val": 8, "test": 8}``) and is intended for smoke tests.
    """
    tcfg = config["training"]
    seed = int(run["seed"])
    set_determinism(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA is not available; pass allow_cpu=True for a functional CPU run.")
    use_amp = device.type == "cuda" and bool(tcfg["amp_on_cuda"])
    max_epochs = int(max_epochs or tcfg["max_epochs"])
    batch_size = int(tcfg["batch_size"])
    patience = int(tcfg["early_stopping_patience"])
    min_delta = float(tcfg["early_stopping_min_delta"])
    threshold = float(tcfg["decision_threshold"])

    parts = partition_rows(master, run["protocol"], run["modality"], run["fold_id"])
    if limits:
        parts = {k: v.head(int(limits[k])) if k in limits else v for k, v in parts.items()}
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = make_loader(parts["train"], data_root, batch_size, True, generator, tcfg["num_workers"])
    val_loader = make_loader(parts["val"], data_root, batch_size, False, None, tcfg["num_workers"])
    test_loader = make_loader(parts["test"], data_root, batch_size, False, None, tcfg["num_workers"])

    architecture = run["architecture"]
    model = create_model(architecture, weights_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg["learning_rate"]),
                                  weight_decay=float(tcfg["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / "model_state.pt"
    best_nll, best_epoch, best_hash = float("inf"), None, None
    reference_nll, bad_epochs, stopped = float("inf"), 0, False
    history, steps = [], 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        for batch in train_loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels, reduction="mean")
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Non-finite training loss.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu().item()) * int(labels.shape[0])
            seen += int(labels.shape[0])
            steps += 1

        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.inference_mode():
            for batch in val_loader:
                images = batch["image"].to(device=device, dtype=torch.float32)
                labels = batch["label"].to(device=device, dtype=torch.long)
                total = nn.functional.cross_entropy(model(images), labels, reduction="sum")
                if not bool(torch.isfinite(total).item()):
                    raise RuntimeError("Non-finite validation NLL.")
                val_sum += float(total.detach().cpu().item())
                val_n += int(labels.shape[0])
        train_nll, val_nll = loss_sum / seen, val_sum / val_n

        updated = val_nll < best_nll
        if updated:
            best_nll, best_epoch = val_nll, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"run_id": run["run_id"], "epoch": int(epoch), "validation_nll": float(val_nll),
                        "model_state_dict": state}, state_file)
            shadow = create_model(architecture, weights_path)
            shadow.load_state_dict(state, strict=True)
            best_hash = model_state_sha256(shadow)
            del shadow
        significant = val_nll < reference_nll - min_delta
        if significant:
            reference_nll, bad_epochs = val_nll, 0
        else:
            bad_epochs += 1
        stop_now = bad_epochs >= patience
        history.append({"epoch": epoch, "train_nll": train_nll, "validation_nll": val_nll, "best_updated": updated,
                        "significant_improvement": significant, "bad_epochs": bad_epochs,
                        "early_stop_triggered": stop_now})
        print(f"  epoch {epoch:2d} train_nll={train_nll:.5f} val_nll={val_nll:.5f} bad={bad_epochs}")
        if stop_now:
            stopped = True
            break

    with open(output_dir / "training_history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)

    payload = torch.load(state_file, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device).eval()
    restored_hash = model_state_sha256(model)
    if restored_hash != best_hash or int(payload["epoch"]) != int(best_epoch):
        raise RuntimeError("Best model state was not restored exactly.")

    result = predict(model, test_loader, device)
    write_predictions(output_dir / "test_predictions.csv", run["run_id"], result, threshold)
    predictions = (result["probabilities"][:, 1] >= threshold).astype(np.int64)
    metrics = classification_metrics(result["labels"], result["probabilities"], predictions,
                                     tuple(config["metrics"]["nll_probability_clip"]))
    record = {
        **{k: run[k] for k in ("run_id", "architecture", "protocol", "modality", "fold_id", "seed")},
        "cohort": "PRIMARY_BINARY", "test_n": int(len(result["labels"])),
        "train_n": int(len(parts["train"])), "val_n": int(len(parts["val"])),
        "epochs_run": len(history), "best_epoch": int(best_epoch), "early_stop_triggered": stopped,
        "optimizer_steps": steps, "device": device.type, "amp": use_amp,
        "model_state_sha256": restored_hash, **metrics,
    }
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return record


def load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sample_id": str, "run_id": str})


def child_environment() -> dict:
    """Environment for a fresh training process with deterministic cuBLAS workspaces."""
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


GROUP_METRICS = ("macro_f1", "balanced_accuracy", "auroc", "auprc", "nll", "brier", "accuracy", "mcc")


def predictive_groups(records: list[dict], ddof: int = 1) -> pd.DataFrame:
    """Mean and sample SD over seeds of the per-run test metrics for each group."""
    frame = pd.DataFrame(records)
    rows = []
    for key, sub in frame.groupby(["architecture", "protocol", "modality", "fold_id"], sort=True):
        record = dict(zip(["architecture", "protocol", "modality", "fold"], key))
        record["test_n"] = int(sub["test_n"].iloc[0])
        record["n_seeds"] = int(len(sub))
        for metric in GROUP_METRICS:
            values = sub[metric].astype(float).to_numpy()
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_sd"] = float(np.std(values, ddof=ddof)) if values.size > ddof else float("nan")
        rows.append(record)
    return pd.DataFrame(rows)


def load_run_records(runs_dir: Path, runs: list[dict]) -> list[dict]:
    """Read test_metrics.json of every completed run in manifest order."""
    records = []
    for run in runs:
        path = Path(runs_dir) / run["run_id"] / "test_metrics.json"
        if path.is_file():
            with open(path, "r", encoding="utf-8") as handle:
                records.append({**json.load(handle), "manifest_index": run["manifest_index"]})
    return records
