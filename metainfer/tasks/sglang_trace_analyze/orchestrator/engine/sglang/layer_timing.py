"""Per-layer decode timing analysis.

Uses the ``scheduler.run_batch`` and ``step[DECODE bs=N]`` annotations
from SGLang's profiler to identify decode steps and measure per-layer
execution time via anchor-kernel gaps.

Strategy:
1. Find ``step[DECODE bs=N]`` gpu_user_annotation events to determine
   batch_size and locate decode step boundaries.
2. Within a decode step, find all ``flash_fwd_splitkv_mla_fp8_sparse_kernel``
   GPU kernel events (one per transformer layer).
3. Sort by timestamp. The gap between consecutive anchor kernels
   approximates the time spent in each layer.
4. First and last layers are partial: first layer's start = step boundary,
   last layer's end = step boundary.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def extract_batch_size(trace_data: Dict[str, Any]) -> Optional[int]:
    """Extract decode batch_size from step annotations.

    SGLang profiler annotations look like ``step[DECODE bs=8]``.
    """
    events = trace_data.get("traceEvents", [])
    for e in events:
        if e.get("cat") == "gpu_user_annotation":
            name = e.get("name", "")
            m = re.search(r"step\[DECODE\s+bs=(\d+)\]", name)
            if m:
                return int(m.group(1))
    return None


def find_anchor_kernels(
    trace_data: Dict[str, Any],
    anchor_pattern: str = "flash_fwd_splitkv_mla",
) -> List[Dict[str, Any]]:
    """Find all anchor kernel events (one per layer per decode step).

    Returns list of {ts, dur, end, name} sorted by timestamp.
    """
    events = trace_data.get("traceEvents", [])
    anchors = []
    for e in events:
        if e.get("cat") != "kernel":
            continue
        name = e.get("name", "")
        if anchor_pattern in name:
            anchors.append({
                "ts": e.get("ts", 0),
                "dur": e.get("dur", 0),
                "end": e.get("ts", 0) + e.get("dur", 0),
                "name": name,
            })
    anchors.sort(key=lambda a: a["ts"])
    return anchors


def compute_layer_times(
    anchors: List[Dict[str, Any]],
    num_layers: int = 43,
) -> Dict[str, Any]:
    """Compute per-layer timing from anchor kernel timeline.

    Each anchor kernel corresponds to one transformer layer's attention
    computation. The gap between consecutive anchors approximates the
    total time spent in that layer (attention + MoE + communication).

    For a model with N layers and S decode steps, we have S*N anchors.
    We group them into S steps of N layers each, then average per layer.

    Returns:
        Dict with:
        - per_layer_us: list of (layer_idx, avg_time_us) sorted
        - total_per_step_us: average total decode step time
        - num_steps: number of decode steps identified
        - num_layers: layers per step
        - anchor_kernel: name of anchor kernel used
    """
    if len(anchors) < num_layers:
        # Not enough anchors for one full decode step
        return {
            "per_layer_us": [],
            "total_per_step_us": 0,
            "num_steps": 0,
            "num_layers": num_layers,
            "anchor_kernel": anchors[0]["name"] if anchors else "",
            "error": f"only {len(anchors)} anchors, need ≥{num_layers} for one step",
        }

    # Group anchors into steps of N layers each
    steps = []
    for i in range(0, len(anchors) - num_layers + 1, num_layers):
        step_anchors = anchors[i : i + num_layers]
        if len(step_anchors) == num_layers:
            steps.append(step_anchors)

    if not steps:
        return {
            "per_layer_us": [],
            "total_per_step_us": 0,
            "num_steps": 0,
            "num_layers": num_layers,
        }

    # For each layer position, compute average time across steps
    per_layer_times = []
    for layer_idx in range(num_layers - 1):  # last layer has no "next" anchor
        times = []
        for step_anchors in steps:
            current = step_anchors[layer_idx]
            next_anchor = step_anchors[layer_idx + 1]
            # Gap from end of current attention to start of next
            gap_us = next_anchor["ts"] - current["end"]
            if gap_us > 0:
                # Add current anchor's own duration
                layer_time = current["dur"] + gap_us
                times.append(layer_time)
        if times:
            avg_us = sum(times) / len(times)
            per_layer_times.append((layer_idx, avg_us, min(times), max(times), len(times)))

    # Total step time: from first anchor start to last anchor end
    step_times = []
    for step_anchors in steps:
        start = step_anchors[0]["ts"]
        end = step_anchors[-1]["end"]
        step_times.append(end - start)

    avg_step_us = sum(step_times) / len(step_times) if step_times else 0

    return {
        "per_layer_us": [(idx, round(avg, 1)) for idx, avg, *_ in per_layer_times],
        "per_layer_detail": [
            {"layer": idx, "avg_us": round(avg, 1), "min_us": round(mn, 1),
             "max_us": round(mx, 1), "samples": n}
            for idx, avg, mn, mx, n in per_layer_times
        ],
        "total_per_step_us": round(avg_step_us, 1),
        "num_steps": len(steps),
        "num_layers": num_layers,
        "steps_measured": len(per_layer_times),  # layers with timing data
        "anchor_kernel": anchors[0]["name"][:80] if anchors else "",
    }


def build_layer_timing_report(
    trace_data: Dict[str, Any],
    batch_size: int,
    stage: str,
    num_layers: int = 43,
    compress_ratios: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Produce the full layer_timing.json payload."""
    # Try to extract actual batch_size from annotations
    actual_bs = extract_batch_size(trace_data)
    if actual_bs is not None:
        batch_size = actual_bs

    anchors = find_anchor_kernels(trace_data)
    timing = compute_layer_times(anchors, num_layers)

    timing["batch_size"] = batch_size
    timing["stage"] = stage
    timing["batch_size_from_annotation"] = actual_bs is not None
    timing["compress_ratios"] = compress_ratios or []

    # Add layer type labels based on compress_ratios if available
    if compress_ratios:
        labels = []
        for i in range(len(timing.get("per_layer_detail", []))):
            if i < len(compress_ratios):
                cr = compress_ratios[i]
                if cr == 0:
                    labels.append("SWA")
                elif cr == 4:
                    labels.append("CSA")
                elif cr == 128:
                    labels.append("HCA")
                else:
                    labels.append(f"CR{cr}")
            else:
                labels.append("?")
        for i, detail in enumerate(timing.get("per_layer_detail", [])):
            if i < len(labels):
                detail["layer_type"] = labels[i]

    return timing
