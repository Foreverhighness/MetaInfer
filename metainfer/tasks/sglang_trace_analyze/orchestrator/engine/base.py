"""Inference engine abstraction layer.

Defines the interfaces that each inference engine (sglang, vllm, ...)
must implement to plug into the trace analysis pipeline.

To add a new engine:
1. Create ``engine/<name>/benchmark.py`` — implement ``BenchmarkRunner``
2. Create ``engine/<name>/parser.py`` — implement engine-specific trace
   parsing (most logic is shared in ``engine/parser.py`` — see
   sglang/parser.py as an example)
3. Create ``engine/<name>/fuse.py`` — implement ``FusePatternMatcher``
4. Register in ``engine/__init__.py``'s ``ENGINE_REGISTRY``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class BenchmarkRunner(ABC):
    """Launches the engine's benchmark + profiling tool.

    Each engine has a different CLI, environment variables, and output
    directory convention.
    """

    @abstractmethod
    def run(
        self,
        model_path: str,
        batch_size: int,
        input_len: int,
        output_len: int,
        tp_size: int,
        pp_size: int,
        output_dir: str,
        disable_cuda_graph: bool = False,
        version: str = "default",
        **kwargs,
    ) -> bool:
        """Run one benchmark + profiling invocation. Returns True on success."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name, e.g. 'sglang', 'vllm'."""
        ...


class FusePatternMatcher:
    """Engine-specific fuse pattern rules.

    Each engine has different kernel naming conventions, so the
    fuse catalog must be engine-aware.
    """

    @property
    def patterns(self) -> List[Dict[str, Any]]:
        """Return the engine's fuse pattern catalog."""
        return []

    def match(self, kernel_names: List[str]) -> List[Dict[str, Any]]:
        """Scan a list of kernel names for known fuse patterns."""
        matches = []
        for pat in self.patterns:
            found = self._match_consecutive(kernel_names, pat.get("kernels", []))
            if found:
                matches.append({
                    "pattern": pat["pattern"],
                    "kernels": found,
                    "suggestion": pat.get("suggestion", ""),
                    "estimated_saving_us": pat.get("estimated_saving_us", 0),
                    "confidence": pat.get("confidence", "medium"),
                })
        return matches

    @staticmethod
    def _match_consecutive(
        names: List[str], pattern_kernels: List[str]
    ) -> List[str]:
        if len(pattern_kernels) > len(names):
            return []
        patterns_lower = [p.lower() for p in pattern_kernels]
        names_lower = [n.lower() for n in names]
        for i in range(len(names_lower) - len(patterns_lower) + 1):
            match = True
            for j, pat in enumerate(patterns_lower):
                if pat not in names_lower[i + j]:
                    match = False
                    break
            if match:
                return names[i : i + len(patterns_lower)]
        return []
