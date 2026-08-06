"""GLM model family — to be implemented.

When adding GLM support:
1. Implement arch.py — GLM-4 architecture (num_layers, attention_type, etc.)
2. Implement mapper.py — GLM-specific kernel→layer mapping heuristics
3. Implement flops.py — GLM-specific FLOPs estimation formulas
4. Register in model/__init__.py's MODEL_REGISTRY

GLM uses standard transformer with GQA attention (not MLA),
so the architecture knowledge can be simpler than DeepSeek.
"""
