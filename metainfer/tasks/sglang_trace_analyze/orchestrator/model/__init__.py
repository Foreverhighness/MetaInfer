"""Model architecture layer — DeepSeek-specific kernel mapping and FLOPs."""

from .mapper import build_mapping, _map_one
from .flops import calculate_mfu, extract_ck_tile_dims
from .arch import DEEPSPEED_V4_ARCH
