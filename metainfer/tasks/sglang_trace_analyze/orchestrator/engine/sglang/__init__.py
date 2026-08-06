"""sglang inference engine — benchmark runner, trace parser, overlap, fuse."""

from .benchmark import build_dir_name, run_benchmark, _setup_env, _K100_ENV
from .parser import parse_trace, aggregate_kernels, aggregate_kernels_with_dims
from .overlap import detect_gaps, build_overlap_report
from .fuse import match_fuse_patterns, build_fuse_report, FUSE_PATTERNS
