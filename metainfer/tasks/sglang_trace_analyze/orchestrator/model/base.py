"""Model architecture abstraction layer.

Defines the interfaces that each model (DeepSeek, GLM, Qwen, ...)
must implement to plug into the trace analysis pipeline.

To add a new model:
1. Create ``model/<name>/arch.py`` — define ``MODEL_ARCH: dict``
2. Create ``model/<name>/mapper.py`` — implement ``KernelMapper``
3. Create ``model/<name>/flops.py`` — implement ``FlopsEstimator``
4. Register in ``model/__init__.py``'s ``MODEL_REGISTRY``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ModelArch(ABC):
    """Static architecture knowledge for one model family.

    Each model sub-package provides a ``MODEL_ARCH`` dict that
    describes layer structure, attention type, quantization method,
    and a kernel-name-to-layer-type mapping.
    """

    @property
    @abstractmethod
    def family(self) -> str:
        """Model family name, e.g. 'DeepSeek-V4', 'GLM-4', 'Qwen3'."""
        ...

    @property
    def num_layers(self) -> int:
        return 0

    @property
    def attention_type(self) -> str:
        """'mla', 'mha', 'gqa', 'flash_attn', etc."""
        return "mha"

    @property
    def kernel_map(self) -> Dict[str, str]:
        """Mapping from kernel name pattern → layer component.

        Example::
            {"flash_fwd_splitkv_mla": "attention/mla",
             "fused_moe_kernel":      "moe/experts"}
        """
        return {}

    def layer_for_kernel(self, kernel_name: str) -> Optional[str]:
        """Given a GPU kernel name, return the model-layer component it
        belongs to, or None."""
        import re
        name_lower = kernel_name.lower()
        for pattern, component in self.kernel_map.items():
            # patterns like "Cijk_*" match prefix; "*" is wildcard
            if pattern.endswith("*"):
                if name_lower.startswith(pattern[:-1]):
                    return component
            elif pattern in name_lower:
                return component
        return None


class KernelMapper:
    """Maps GPU kernel names → model structural elements.

    Uses :class:`ModelArch` for architecture-specific matching,
    plus call stack / CPU op hints from the trace.
    """

    def __init__(self, arch: ModelArch):
        self.arch = arch

    def map_kernel(
        self,
        kernel_name: str,
        call_stack: str = "",
        cpu_ops: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Map a single kernel to (op_type, category, model_layer, confidence).

        Subclasses may override this for model-specific heuristics.
        """
        op_type = self._infer_op_type(kernel_name, call_stack, cpu_ops)
        layer = self._infer_layer(kernel_name, call_stack, cpu_ops)
        confidence = self._confidence(kernel_name, call_stack, cpu_ops, op_type, layer)
        return {
            "kernel_name": kernel_name,
            "model_layer": layer,
            "op_type": op_type,
            "category": self._op_type_to_category(op_type),
            "call_stack": call_stack,
            "confidence": confidence,
        }

    # ── subclasses may override ──

    def _infer_layer(
        self, kernel_name: str, call_stack: str, cpu_ops: Optional[List[str]]
    ) -> Optional[str]:
        return self.arch.layer_for_kernel(kernel_name)

    def _infer_op_type(
        self, kernel_name: str, call_stack: str, cpu_ops: Optional[List[str]]
    ) -> str:
        return "Other"

    def _confidence(
        self, kernel_name: str, call_stack: str,
        cpu_ops: Optional[List[str]], op_type: str, layer: Optional[str],
    ) -> str:
        if call_stack:
            return "high" if layer else "medium"
        return "medium" if layer else "low"

    @staticmethod
    def _op_type_to_category(op_type: str) -> str:
        return op_type


class FlopsEstimator:
    """Estimates TFLOPS, bandwidth, and MFU for kernel entries.

    Model-specific because FLOPs formulas depend on the model's
    attention mechanism, MoE configuration, and quantization scheme.
    """

    def estimate(
        self,
        kernel: Dict[str, Any],
        gpu_spec,
        batch_size: int = 1,
        dtype: str = "bf16",
    ) -> Dict[str, Any]:
        """Augment a kernel entry with tflops_actual, mfu, bound, etc."""
        return kernel
