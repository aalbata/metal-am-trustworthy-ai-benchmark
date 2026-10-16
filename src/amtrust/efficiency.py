"""Inference latency, throughput and peak memory of trained classifiers.

The reported protocol runs on CUDA only.  Each (architecture, precision)
context runs in a fresh Python process, requires zero allocated CUDA memory
before the model is created, and uses zero-valued float32 inputs of shape
[B, 3, 512, 512] preallocated on the device.  Settings: TF32 disabled for
matmul and cuDNN, ``cudnn.benchmark = True``, ``cudnn.deterministic = False``,
``torch.inference_mode()``; precision ``FP32_STRICT`` or ``AMP_FP16``
(``torch.autocast(device_type="cuda", dtype=torch.float16)``).

For each batch size the peak memory comes from one untimed probe forward pass
after ``reset_peak_memory_stats``; incremental peak = max allocated minus the
allocation before the probe (model and input included in the baseline).  Then
30 warm-up passes are followed by 100 timed passes with one CUDA event pair per
pass and a single synchronisation after the loop.  Batch-1 latency is the mean
(and median) of the per-pass times; batch-8 throughput is
8 x 100 x 1000 / (sum of the 100 batch times in ms).

The CPU option uses ``time.perf_counter`` around each pass on the CPU in FP32.
It is provided for functional checks only and is not comparable with the
reported CUDA measurements.
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time

import numpy as np
import torch

from .models import load_trained


def summarise(values) -> dict:
    x = np.asarray(values, dtype=np.float64)
    q1 = float(np.quantile(x, 0.25, method="linear"))
    q3 = float(np.quantile(x, 0.75, method="linear"))
    return {"n": int(x.size), "mean": float(np.mean(x)), "median": float(np.median(x)),
            "sd": float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
            "p25": q1, "p75": q3, "min": float(np.min(x)), "max": float(np.max(x))}


def _forward(model, tensor, precision: str, device_type: str):
    if precision == "FP32_STRICT":
        return model(tensor)
    if precision == "AMP_FP16":
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            return model(tensor)
    raise ValueError(f"Unsupported precision: {precision}")


def _cuda_timing(model, tensor, precision, warmups, repeats):
    for _ in range(warmups):
        with torch.inference_mode():
            _forward(model, tensor, precision, "cuda")
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        with torch.inference_mode():
            starts[i].record()
            _forward(model, tensor, precision, "cuda")
            ends[i].record()
    torch.cuda.synchronize()
    return [float(starts[i].elapsed_time(ends[i])) for i in range(repeats)]


def measure_cuda(architecture: str, state_path, precision: str, warmups: int = 30, repeats: int = 100) -> dict:
    """Measure one context on CUDA following the reported protocol."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the reported efficiency protocol.")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    pre_model = int(torch.cuda.memory_allocated())
    if pre_model != 0:
        raise RuntimeError(f"Process is not fresh: {pre_model} bytes already allocated.")
    model = load_trained(architecture, state_path).eval()
    model = model.to(device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    result = {"architecture": architecture, "precision": precision, "device": torch.cuda.get_device_name(device),
              "process_id": os.getpid(), "pre_model_allocated_bytes": pre_model,
              "model_only_allocated_bytes": int(torch.cuda.memory_allocated()) - pre_model,
              "warmup_iterations": warmups, "timed_iterations": repeats}
    for batch in (1, 8):
        tensor = torch.zeros((batch, 3, 512, 512), device=device, dtype=torch.float32)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = int(torch.cuda.memory_allocated())
        with torch.inference_mode():
            probe = _forward(model, tensor, precision, "cuda")
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated())
        times = _cuda_timing(model, tensor, precision, warmups, repeats)
        result[f"b{batch}_baseline_allocated_bytes"] = baseline
        result[f"b{batch}_peak_allocated_bytes"] = peak
        result[f"b{batch}_incremental_peak_allocated_bytes"] = peak - baseline
        result[f"b{batch}_time_ms"] = summarise(times)
        if batch == 8:
            result["b8_aggregate_samples_per_second"] = 8.0 * repeats * 1000.0 / float(math.fsum(times))
        del probe, tensor
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    del model
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    result["post_cleanup_allocated_bytes"] = int(torch.cuda.memory_allocated())
    return result


def measure_cpu(architecture: str, state_path, warmups: int = 30, repeats: int = 100) -> dict:
    """FP32 CPU timing for functional checks; not comparable with CUDA results."""
    model = load_trained(architecture, state_path).eval()
    result = {"architecture": architecture, "precision": "FP32_CPU", "device": "cpu",
              "comparable_with_reported": False, "warmup_iterations": warmups, "timed_iterations": repeats}
    for batch in (1, 8):
        tensor = torch.zeros((batch, 3, 512, 512), dtype=torch.float32)
        with torch.inference_mode():
            for _ in range(warmups):
                model(tensor)
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                model(tensor)
                times.append((time.perf_counter() - start) * 1000.0)
        result[f"b{batch}_time_ms"] = summarise(times)
        if batch == 8:
            result["b8_aggregate_samples_per_second"] = 8.0 * repeats * 1000.0 / float(math.fsum(times))
    return result


def flatten(result: dict) -> dict:
    """One CSV row per context with latency, throughput and memory in MiB."""
    mib = 2.0 ** 20
    row = {
        "architecture": result["architecture"],
        "precision": result["precision"],
        "device": result.get("device", ""),
        "b1_mean_latency_ms": result["b1_time_ms"]["mean"],
        "b1_median_latency_ms": result["b1_time_ms"]["median"],
        "b8_mean_batch_time_ms": result["b8_time_ms"]["mean"],
        "b8_aggregate_throughput_samples_per_s": result["b8_aggregate_samples_per_second"],
        "warmup_iterations_per_batch": result["warmup_iterations"],
        "timed_iterations_per_batch": result["timed_iterations"],
    }
    for batch in (1, 8):
        key = f"b{batch}_incremental_peak_allocated_bytes"
        if key in result:
            row[key] = result[key]
            row[f"b{batch}_incremental_peak_mib_derived"] = result[key] / mib
    for key in ("pre_model_allocated_bytes", "model_only_allocated_bytes", "post_cleanup_allocated_bytes"):
        if key in result:
            row[key] = result[key]
    return row


def environment() -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
