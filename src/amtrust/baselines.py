"""Reference baselines on frozen ImageNet ResNet18 embeddings.

Embeddings: ImageNet ResNet18 (IMAGENET1K_V1) with ``fc`` replaced by
``nn.Identity``, all parameters frozen, evaluated on CPU one image at a time on
the standard preprocessed 512 x 512 input, giving one 512-dimensional float32
vector per primary-cohort image.  Rows are ordered OT before PB and then by the
lower-case source filename.

Baselines are fitted on the training partition of each task and evaluated on
its test partition:

* B0 majority: predicts Defect for every test image only if the training Defect
  prior exceeds 0.5; its score is the prior.
* B1 linear SVM: training-only ``StandardScaler`` and ``LinearSVC`` (C = 1,
  squared hinge, L2, primal solver), Defect when the decision value >= 0.
* B2 random forest: 500 trees, ``max_features="sqrt"``, seeds 20260916 to
  20260918, Defect when P(Defect) >= 0.5.

Macro-F1 and balanced accuracy use scikit-learn; AUROC and AUPRC use
``roc_auc_score`` and ``average_precision_score`` on the score.
"""

from __future__ import annotations

from pathlib import PurePosixPath

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, matthews_corrcoef, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .models import create_model
from .preprocessing import preprocess_image


def embedding_order(primary: pd.DataFrame) -> pd.DataFrame:
    """Primary cohort in embedding row order (OT first, then lower-case filename)."""
    frame = primary.copy()
    frame["_mod_rank"] = (frame["modality"] != "OT").astype(int)
    frame["_name"] = frame["original_member"].map(lambda m: PurePosixPath(m).name.lower())
    frame = frame.sort_values(["_mod_rank", "_name"], kind="mergesort").reset_index(drop=True)
    return frame.drop(columns=["_mod_rank", "_name"])


def embedding_model(weights_path) -> nn.Module:
    model = create_model("resnet18", weights_path)
    model.fc = nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def extract_embeddings(frame: pd.DataFrame, image_paths, weights_path, progress_every: int = 500) -> np.ndarray:
    """Return a float32 [N, 512] matrix of frozen ResNet18 embeddings on CPU."""
    model = embedding_model(weights_path)
    out = np.empty((len(frame), 512), dtype=np.float32)
    with torch.inference_mode():
        for i, (path, modality) in enumerate(zip(image_paths, frame["modality"])):
            tensor = preprocess_image(path, modality).unsqueeze(0)
            out[i] = model(tensor).cpu().numpy()[0].astype(np.float32)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  embeddings {i + 1}/{len(frame)}")
    if not np.isfinite(out).all():
        raise RuntimeError("Non-finite embedding values.")
    return out


def evaluate(y_true, y_pred, score) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    both = len(np.unique(y_true)) == 2
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, score)) if both else float("nan"),
        "auprc": float(average_precision_score(y_true, score)) if both else float("nan"),
        "single_class_prediction": bool(len(np.unique(y_pred)) == 1),
    }


def fit_task(X_train, y_train, X_test, y_test, config: dict) -> list[dict]:
    """Fit B0, B1 and the three B2 forests for one task; return metric records."""
    cfg = config["baselines"]
    records = []
    prior = float(np.mean(y_train))
    b0_pred = np.full(len(y_test), int(prior > 0.5), dtype=np.int8)
    b0_score = np.full(len(y_test), prior, dtype=np.float64)
    records.append({"baseline": "B0", "seed": "", **evaluate(y_test, b0_pred, b0_score),
                    "brier": float(brier_score_loss(y_test, b0_score))})

    svc_cfg = cfg["linear_svc"]
    scaler = StandardScaler(with_mean=True, with_std=True)
    svc = LinearSVC(C=svc_cfg["C"], penalty=svc_cfg["penalty"], loss=svc_cfg["loss"], dual=svc_cfg["dual"],
                    tol=svc_cfg["tol"], max_iter=svc_cfg["max_iter"], class_weight=None, fit_intercept=True,
                    intercept_scaling=1.0)
    svc.fit(scaler.fit_transform(X_train), y_train)
    b1_score = svc.decision_function(scaler.transform(X_test)).astype(np.float64)
    b1_pred = (b1_score >= svc_cfg["decision_threshold"]).astype(np.int8)
    records.append({"baseline": "B1", "seed": "", **evaluate(y_test, b1_pred, b1_score), "brier": float("nan")})

    rf_cfg = cfg["random_forest"]
    for seed in cfg["random_forest_seeds"]:
        forest = RandomForestClassifier(n_estimators=rf_cfg["n_estimators"], criterion=rf_cfg["criterion"],
                                        max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                        max_features=rf_cfg["max_features"], bootstrap=rf_cfg["bootstrap"],
                                        oob_score=False, class_weight=None, n_jobs=rf_cfg["n_jobs"],
                                        random_state=seed)
        forest.fit(X_train, y_train)
        probability = forest.predict_proba(X_test)[:, 1].astype(np.float64)
        pred = (probability >= rf_cfg["decision_threshold"]).astype(np.int8)
        records.append({"baseline": "B2", "seed": seed, **evaluate(y_test, pred, probability),
                        "brier": float(brier_score_loss(y_test, probability))})
    return records


def summarise(records: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    """Mean and SD over random-forest seeds; B0 and B1 are single fits."""
    keys = ["task", "protocol", "modality", "fold_id", "baseline"]
    rows = []
    for key, sub in records.groupby(keys, sort=False):
        record = dict(zip(keys, key))
        record["test_n"] = int(sub["test_n"].iloc[0]) if "test_n" in sub else None
        record["n_fits"] = int(len(sub))
        for metric in ("macro_f1", "balanced_accuracy", "accuracy", "mcc", "auroc", "auprc"):
            values = sub[metric].to_numpy(dtype=np.float64)
            record[metric] = float(np.mean(values))
            record[f"{metric}_sd"] = float(np.std(values, ddof=ddof)) if values.size > ddof else float("nan")
        record["single_class_fits"] = int(sub["single_class_prediction"].sum())
        rows.append(record)
    return pd.DataFrame(rows)
