"""Qwen model family — to be implemented.

When adding Qwen support:
1. Implement arch.py — Qwen3 architecture (dense/MoE, GQA attention)
2. Implement mapper.py — Qwen-specific kernel→layer mapping
3. Implement flops.py — Qwen-specific FLOPs estimation
4. Register in model/__init__.py's MODEL_REGISTRY

Qwen3 supports both dense and MoE variants. The architecture knowledge
should handle both cases, detecting the variant from model config.
"""
