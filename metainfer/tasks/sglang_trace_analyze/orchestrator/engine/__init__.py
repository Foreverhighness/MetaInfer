"""Inference engine layer — pluggable engine support.

Currently supports: sglang (default). Future: vllm.

To switch engines, set ``engine_name`` in requirements.json.
"""

from .base import BenchmarkRunner, FusePatternMatcher
from .sglang import (
    build_dir_name, run_benchmark, _setup_env, _K100_ENV,
    parse_trace, aggregate_kernels, aggregate_kernels_with_dims,
    detect_gaps, build_overlap_report,
    match_fuse_patterns, build_fuse_report, FUSE_PATTERNS,
)

# Default engine
DEFAULT_ENGINE = "sglang"

ENGINE_REGISTRY = {
    "sglang": {
        "benchmark": "engine.sglang.benchmark",
        "parser": "engine.sglang",
        "fuse": "engine.sglang",
    },
    # "vllm": {...},  # add when implemented
}
