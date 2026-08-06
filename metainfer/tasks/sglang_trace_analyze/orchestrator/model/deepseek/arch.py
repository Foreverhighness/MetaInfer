"""DeepSeek V4 model architecture knowledge.

**Authoritative source:** ``notebooks/models/deepseek-v4.md``
(loaded at module import time).

The markdown file is the single source of truth — edit it to update
architecture knowledge without touching Python code.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

# Path to the authoritative knowledge markdown
_NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "notebooks"
_ARCH_MD = _NOTEBOOKS_DIR / "models" / "deepseek-v4.md"


def _parse_table(lines: list) -> Dict[str, str]:
    """Parse a simple markdown table: key | value.
    Strips parenthetical comments like '58 (前三层 dense)' → '58'.
    """
    result = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("|--") or not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) >= 2:
            key = parts[0]
            val = re.sub(r"\s*\(.*\)", "", parts[1]).strip()
            result[key] = val
    return result


def _parse_kernel_map(lines: list) -> Dict[str, str]:
    """Parse kernel → component mapping table."""
    result = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("|--") or not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) >= 2:
            kernel = parts[0].replace("`", "").strip()
            component = parts[1].replace("`", "").strip()
            # Skip header row
            if kernel in ("GPU Kernel", "Kernel", "kernel"):
                continue
            # Strip parenthetical from key too: "Cijk_* (CK GEMM)" → "Cijk_*"
            kernel = re.sub(r"\s*\(.*\)", "", kernel).strip()
            if kernel and component:
                result[kernel] = component
    return result


def _load_arch_from_md(md_path: Path) -> Dict[str, Any]:
    """Parse architecture knowledge from markdown.

    Extracts the key parameters table and kernel→layer mapping table
    from structured markdown sections.
    """
    if not md_path.exists():
        return _DEFAULT_ARCH

    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    params = {}
    kernel_map = {}

    in_params = False
    in_kernel_map = False
    param_lines = []
    kernel_lines = []

    for line in lines:
        if "## 关键参数" in line:
            in_params = True
            in_kernel_map = False
            continue
        if "## Kernel → 模型层映射表" in line:
            in_params = False
            in_kernel_map = True
            continue
        if line.startswith("## ") and not line.startswith("###"):
            in_params = False
            in_kernel_map = False
            continue

        if in_params:
            param_lines.append(line)
        elif in_kernel_map:
            kernel_lines.append(line)

    params = _parse_table(param_lines)
    kernel_map = _parse_kernel_map(kernel_lines)

    def _int_or(key: str, default: int) -> int:
        try:
            return int(re.sub(r"[^0-9]", "", params.get(key, str(default))) or default)
        except (ValueError, TypeError):
            return default

    return {
        "family": "DeepSeek-V4",
        "num_layers": _int_or("num_layers", 43),
        "attention_type": params.get("attention_type", "hybrid"),
        "moe_layers": _int_or("moe_layers", 43),
        "shared_experts": _int_or("shared_experts", 1),
        "routed_experts_per_token": _int_or("top_k_experts", 6),
        "quantization": params.get("quantization", "int8_flash_channel"),
        "head_dim": _int_or("head_dim", 512),
        "q_lora_rank": _int_or("q_lora_rank", 1024),
        "o_lora_rank": _int_or("o_lora_rank", 1024),
        "hidden_size": _int_or("hidden_size", 4096),
        "intermediate_size": _int_or("intermediate_size", 2048),
        "mhc_enabled": True,
        "hc_mult": _int_or("hc_mult", 4),
        "page_size": _int_or("page_size", 256),
        "tp_size": _int_or("tp_size", 8),
        "num_experts": _int_or("num_experts", 256),
        "top_k_experts": _int_or("top_k_experts", 6),
        "kernel_map": kernel_map or _DEFAULT_ARCH["kernel_map"],
        "_source_md": str(md_path.relative_to(_NOTEBOOKS_DIR.parent)),
    }


# Fallback (if markdown is unavailable)
_DEFAULT_ARCH = {
    "family": "DeepSeek-V4",
    "num_layers": 43,
    "attention_type": "hybrid",        # CSA + HCA + SWA
    "moe_layers": 43,                  # all-MoE backbone
    "shared_experts": 1,
    "routed_experts_per_token": 6,     # top-6 from 256
    "quantization": "int8_flash_channel",
    "head_dim": 512,                   # query head dim
    "q_lora_rank": 1024,
    "o_lora_rank": 1024,
    "hidden_size": 4096,
    "intermediate_size": 2048,
    "mhc_enabled": True,
    "hc_mult": 4,
    "page_size": 256,
    "tp_size": 8,
    "num_experts": 256,
    "kernel_map": {
        "flash_fwd_splitkv_mla": "attention/mla",
        "fused_moe_kernel": "moe/experts",
        "w8a8_scaled_mm": "linear_int8",
        "Cijk_*": "linear_bf16",
        "cross_device_reduce_2stage": "tp_allreduce",
        "mhc_pre_*": "dsa/mhc_pre",
        "mhc_post_*": "dsa/mhc_post",
    },
}

# Module-level: loaded once from markdown at import time
DEEPSPEED_V4_ARCH = _load_arch_from_md(_ARCH_MD)
