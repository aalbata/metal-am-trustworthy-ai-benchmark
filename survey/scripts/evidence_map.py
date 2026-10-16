"""Code the extracted evidence into indicators and compute the evidence map.

Reads records/evidence_extraction.csv, keeps eligible studies, codes every indicator with
explicit keyword rules on the extracted text, and writes
  records/evidence_coded.csv   one row per eligible study with all indicators
  records/evidence_map.csv     proportions per stratum and stratum-weighted
  records/landscape.csv        task, input, model, and process counts (all eligible)
Evaluation-practice indicators are computed over studies whose full text was read.
"""
import glob
import re
from pathlib import Path

import pandas as pd

L = Path(__file__).resolve().parents[1]
X = L / "records"
order = pd.read_csv(X / "extraction_sample_order.csv")
SIZES = order.groupby("stratum").size().to_dict()

d = pd.read_csv(X / "evidence_extraction.csv", dtype=str, keep_default_na=False)
ov = pd.read_csv(X / "eligibility_overrides.csv", dtype=str)
for r in ov.itertuples():
    assert (d.study_id == r.study_id).sum() == 1, r.study_id
    d.loc[d.study_id == r.study_id, "eligible"] = r.eligible
d = d[d.eligible.str.strip() == "Y"].copy()
d["stratum"] = d.study_id.str[0]
d["full_text"] = ~d.access_level.str.lower().str.startswith("abstract")


def low(s):
    return str(s).lower()


def starts_none(s):
    s = low(s).strip()
    return s.startswith("none") or s.startswith("not reported") or s.startswith("not stated") or s == ""


def has(s, *pats):
    s = low(s)
    return any(re.search(p, s) for p in pats)


HELD = [r"held-out process", r"held-out geometry", r"held-out build", r"held-out machine",
        r"held-out material", r"external dataset", r"simulation-to-experiment"]

g = d.generalisation_setting
d["split_not_reported"] = g.map(lambda s: low(s).strip().startswith("not reported") and not has(s, *HELD, r"grouped", r"random"))
d["split_grouped"] = g.map(lambda s: has(s, r"grouped by part", r"grouped by layer", r"grouped by .*specimen"))
d["split_heldout_condition"] = g.map(lambda s: has(s, *HELD))
d["split_random_only"] = g.map(lambda s: has(s, r"random") and not has(s, *HELD, r"grouped"))
d["split_beyond_random"] = d.split_grouped | d.split_heldout_condition
d["heldout_machine"] = g.map(lambda s: has(s, r"held-out machine"))
d["heldout_material"] = g.map(lambda s: has(s, r"held-out material"))
d["heldout_build_or_experiment"] = (g.map(lambda s: has(s, r"held-out build", r"grouped by experiment"))
                                    | d.split_protocol.map(lambda s: has(s, r"leave-one-(build|experiment)")))
d["simulation_data_only"] = d.sensing_modality_or_input.map(
    lambda s: has(s, r"simulat", r"finite.element", r"\bfe\b", r"\bfea\b", r"\bfem\b") and not has(s, r"experiment", r"camera", r"measured", r"image"))
d["leakage_high"] = d.leakage_risk.map(lambda s: low(s).strip().startswith("high"))
d["leakage_unclear"] = d.leakage_risk.map(lambda s: low(s).strip().startswith("unclear"))
d["leakage_low"] = d.leakage_risk.map(lambda s: low(s).strip().startswith("low"))
d["calibration_any"] = d.calibration.map(lambda s: not starts_none(s))
d["uncertainty_any"] = d.uncertainty_method.map(lambda s: not starts_none(s))
d["uncertainty_validated"] = d.uncertainty_validated.map(lambda s: low(s).strip().startswith("y"))
d["robustness_any"] = d.robustness_evaluation.map(lambda s: not starts_none(s))
d["robustness_shift"] = d.robustness_evaluation.map(lambda s: not starts_none(s) and has(s, r"out-of-distribution", r"changed condition"))
d["robustness_perturbation"] = d.robustness_evaluation.map(
    lambda s: not starts_none(s) and has(s, r"noise", r"perturbation", r"corruption", r"adversarial", r"sensor fault"))
d["explanation_any"] = d.explanation_method.map(lambda s: not starts_none(s))
VALIDATION_TERMS = r"(quantitative|ground truth|physical validation|expert)"
d["explanation_validated"] = d.explanation_validation.map(
    lambda s: not starts_none(s) and has(re.sub(r"\bno\s+" + VALIDATION_TERMS, " ", low(s)), VALIDATION_TERMS))
d["explanation_qualitative_only"] = d.explanation_validation.map(lambda s: low(s).strip().startswith("qualitative"))
d["data_public"] = d.data_public.map(lambda s: low(s).strip().startswith("y") or low(s).strip().startswith("partial"))
d["code_public"] = d.code_public.map(lambda s: low(s).strip().startswith("y"))
d["repeats_reported"] = d.repeats_or_seeds.map(lambda s: not starts_none(s))
d["variability_reported"] = d.variability_reported.map(lambda s: not starts_none(s))
d["compute_any"] = d.computational_evidence.map(lambda s: not starts_none(s))
d["compute_inference"] = d.computational_evidence.map(lambda s: not starts_none(s) and has(s, r"inference", r"latency", r"throughput", r"fps"))
d["realtime_claim"] = d.real_time_claim.map(lambda s: low(s).strip().startswith("y"))
d["realtime_in_loop"] = d.real_time_measured_in_loop.map(lambda s: low(s).strip().startswith("y"))
d["closed_loop"] = d.closed_loop_control.map(lambda s: low(s).strip().startswith("y"))

IND = ["split_random_only", "split_grouped", "split_heldout_condition", "split_beyond_random", "split_not_reported", "heldout_machine", "heldout_material", "heldout_build_or_experiment", "simulation_data_only", "leakage_unclear",
       "leakage_high", "leakage_low", "calibration_any", "uncertainty_any", "uncertainty_validated", "robustness_any",
       "robustness_shift", "robustness_perturbation", "explanation_any", "explanation_validated",
       "explanation_qualitative_only", "data_public", "code_public", "repeats_reported", "variability_reported",
       "compute_any", "compute_inference", "realtime_claim", "realtime_in_loop", "closed_loop"]


# landscape over all eligible studies (full text or abstract)
def cat_task(s):
    head = low(s).split("(")[0]
    rules = [
        ("parameter optimisation or control", [r"parameter optimi", r"process parameter", r"process control"]),
        ("defect or anomaly detection", [r"defect", r"anomal", r"flaw", r"fault"]),
        ("melt-pool or thermal monitoring", [r"melt.?pool", r"thermal", r"temperature", r"keyhole"]),
        ("geometry or surface prediction", [r"geometr", r"bead", r"roughness", r"surface", r"distortion", r"deformation",
                                            r"dimension", r"layer height", r"height", r"segmentation of 3d"]),
        ("property or microstructure prediction", [r"propert", r"porosity", r"microstructure", r"strain", r"fatigue",
                                                   r"density", r"printability", r"damage"]),
    ]
    for k, pats in rules:
        if any(re.search(p, head) for p in pats):
            return k
    full = low(s)
    for k, pats in rules:
        if any(re.search(p, full) for p in pats):
            return k
    return "other"


def cat_input(s):
    s = low(s)
    rules = [
        ("images or video", [r"image", r"camera", r"video", r"tomograph", r"infrared", r"thermograph", r"vision",
                             r"\bct\b", r"x-ray", r"point cloud", r"photograph", r"micrograph"]),
        ("process parameters or features", [r"parameter", r"feature", r"composition", r"tabular", r"power", r"speed"]),
        ("acoustic, photodiode or other signals", [r"acoustic", r"photodiode", r"signal", r"spectr", r"current",
                                                   r"voltage", r"pyromet", r"sensor", r"telemetry"]),
        ("simulation data", [r"simulat", r"finite.element", r"\bfe\b", r"\bfem\b", r"\bfea\b", r"numerical"]),
    ]
    for k, pats in rules:
        if any(re.search(p, s) for p in pats):
            return k
    return "other or not reported"


def cat_model(s):
    s = low(s)
    rules = [
        ("physics-informed or hybrid", [r"physics-informed", r"physics informed", r"\bpinn", r"physics-guided",
                                        r"physics-based machine learning"]),
        ("deep neural network (CNN, transformer, recurrent)", [r"\bcnn", r"convolution", r"u-?net", r"resnet", r"yolo",
                                                              r"transformer", r"lstm", r"recurrent", r"deep",
                                                              r"\bvit\b", r"autoencoder", r"deeplab", r"segnet",
                                                              r"\bvae\b", r"gru\b"]),
        ("Gaussian process or Bayesian", [r"gaussian process", r"\bgpr\b", r"bayesian neural", r"bayesian regression",
                                          r"bayesian ridge", r"kriging"]),
        ("tree ensembles or kernel methods", [r"random forest", r"xgboost", r"boost", r"\bsvm", r"support vector",
                                              r"tree", r"\bsvr\b", r"k-nearest", r"knn"]),
        ("shallow neural network", [r"neural network", r"\bann\b", r"\bmlp\b", r"perceptron", r"feedforward", r"ffnn"]),
    ]
    for k, pats in rules:
        if any(re.search(p, s) for p in pats):
            return k
    return "other"


d["task_cat"] = d.task.map(cat_task)
d["input_cat"] = d.sensing_modality_or_input.map(cat_input)
d["model_cat"] = d.model_method.map(cat_model)
d["process_cat"] = d.process.map(lambda s: s.split(" ")[0] if s.split(" ")[0] in ("LPBF", "DED", "WAAM", "EB-PBF") else "other")

# corrections to the coded indicators (one row per study and indicator); the reason names the source:
# Verification (second check against the source), Rule (coding rules of Online Resource 1, Section S3),
# or Follows from (indicator that changes with another)
cpath = X / "coding_corrections.csv"
if cpath.exists():
    corr = pd.read_csv(cpath, dtype=str, keep_default_na=False)
    for r in corr.itertuples():
        hit = d.study_id == r.study_id
        assert hit.any(), "correction for a study that is not coded: %s" % r.study_id
        assert r.indicator in d.columns, r.indicator
        val = {"True": True, "False": False}.get(r.value, r.value)
        d.loc[hit, r.indicator] = val
    d["split_beyond_random"] = d.split_grouped | d.split_heldout_condition

ft = d[d.full_text]
rows = []
for ind in IND:
    r = {"indicator": ind}
    props = {}
    for s in ["F", "G"]:
        sub = ft[ft.stratum == s]
        r[f"{s}_n"] = len(sub)
        r[f"{s}_count"] = int(sub[ind].sum())
        props[s] = sub[ind].mean() if len(sub) else float("nan")
        r[f"{s}_prop"] = props[s]
    w = SIZES["F"] + SIZES["G"]
    r["weighted_prop"] = (SIZES["F"] * props["F"] + SIZES["G"] * props["G"]) / w
    r["all_count"] = int(ft[ind].sum())
    r["all_n"] = len(ft)
    rows.append(r)
emap = pd.DataFrame(rows)
land = []
for col in ["process_cat", "task_cat", "input_cat", "model_cat"]:
    for k, v in d[col].value_counts().items():
        land.append({"dimension": col, "category": k, "count": int(v), "n": len(d)})
pd.DataFrame(land).to_csv(X / "landscape.csv", index=False, lineterminator="\n")
d.to_csv(X / "evidence_coded.csv", index=False, lineterminator="\n")
emap.to_csv(X / "evidence_map.csv", index=False, lineterminator="\n")
print("eligible", len(d), "full text", len(ft), ft.groupby("stratum").size().to_dict())
print(emap[["indicator", "F_count", "F_n", "G_count", "G_n", "weighted_prop"]].to_string())
print(pd.DataFrame(land).to_string())
