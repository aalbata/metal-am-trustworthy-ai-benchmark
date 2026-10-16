"""Search stream definitions (process, method and trustworthiness blocks)."""

PROCESS_TERMS = [
    "metal additive manufacturing",
    "laser powder bed fusion",
    "selective laser melting",
    "electron beam melting",
    "powder bed fusion",
    "directed energy deposition",
    "laser metal deposition",
    "wire arc additive manufacturing",
    "binder jetting",
]

METHOD_TERMS = [
    "machine learning",
    "deep learning",
    "neural network",
    "convolutional",
    "transfer learning",
    "domain adaptation",
    "physics-informed",
    "Gaussian process",
    "random forest",
]

# Focused trustworthiness dimensions, in the order used in the survey.
DIMENSIONS = [
    ("F1", "generalisation and evaluation validity",
     ["generalization", "generalisation", "transferability", "cross-build", "domain shift",
      "out-of-distribution", "unseen"]),
    ("F2", "calibration and uncertainty",
     ["uncertainty", "uncertainty quantification", "Bayesian", "calibration", "conformal"]),
    ("F3", "robustness",
     ["robustness", "noise", "perturbation", "adversarial"]),
    ("F4", "explainability",
     ["explainable", "interpretable", "SHAP", "Grad-CAM", "saliency", "attention map"]),
    ("F5", "reproducibility",
     ["reproducibility", "open dataset", "benchmark dataset"]),
    ("F6", "computational efficiency",
     ["real-time", "latency", "edge", "FPGA", "inference time", "closed-loop control"]),
]

REVIEW_TERMS = ["review", "survey", "state of the art"]
REVIEW_METHOD_TERMS = ["machine learning", "deep learning", "artificial intelligence"]

GENERAL_CAP = 100
FOCUSED_CAP = 200
REVIEW_CAP = 200


def q(term):
    """Quote a term for the Semantic Scholar bulk boolean syntax."""
    return '"%s"' % term if (" " in term or "-" in term) else term


def or_group(terms):
    return "(" + " | ".join(q(t) for t in terms) + ")"


def general_streams():
    out = []
    for i, p in enumerate(PROCESS_TERMS, 1):
        for j, m in enumerate(METHOD_TERMS, 1):
            sid = "G%02d" % ((i - 1) * len(METHOD_TERMS) + j)
            out.append({"stream_id": sid, "kind": "general", "process": p, "method": m,
                        "endpoint": "bulk", "query": "%s + %s" % (q(p), q(m)), "cap": GENERAL_CAP})
    return out


def focused_streams():
    out = []
    for sid, name, terms in DIMENSIONS:
        query = "%s + %s + %s" % (or_group(PROCESS_TERMS), or_group(METHOD_TERMS), or_group(terms))
        out.append({"stream_id": sid, "kind": "focused", "dimension": name, "dim_terms": terms,
                    "endpoint": "bulk", "query": query, "cap": FOCUSED_CAP})
    return out


def review_stream():
    query = "%s + %s + %s" % (or_group(PROCESS_TERMS), or_group(REVIEW_TERMS), or_group(REVIEW_METHOD_TERMS))
    return [{"stream_id": "RV", "kind": "review", "endpoint": "bulk", "query": query,
             "dim_terms": REVIEW_TERMS, "cap": REVIEW_CAP}]


def all_streams():
    return general_streams() + focused_streams() + review_stream()
