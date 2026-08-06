"""vLLM inference engine — to be implemented.

When adding vLLM support:
1. Implement benchmark.py — wrap vllm bench or custom profiling script
2. Implement parser.py — handle vllm's torch profiler trace format
3. Implement fuse.py — vllm-specific fuse pattern catalog
4. Register in engine/__init__.py's ENGINE_REGISTRY

The Chrome trace format is the same across engines (torch.profiler),
so most parsing logic can be shared from engine/sglang/parser.py.
"""
