"""Model architecture layer — pluggable model support.

Currently supports: DeepSeek V4 (default). Future: GLM, Qwen.

To switch models, set ``model_family`` in requirements.json.
"""

from .base import ModelArch, KernelMapper, FlopsEstimator
from .deepseek import (
    DEEPSPEED_V4_ARCH,
    build_mapping, _map_one,
    calculate_mfu, extract_ck_tile_dims,
)

# Default model
DEFAULT_MODEL = "deepseek"

MODEL_REGISTRY = {
    "deepseek": {
        "arch": "model.deepseek.arch",
        "mapper": "model.deepseek.mapper",
        "flops": "model.deepseek.flops",
    },
    # "glm": {...},   # add when implemented
    # "qwen": {...},  # add when implemented
}
