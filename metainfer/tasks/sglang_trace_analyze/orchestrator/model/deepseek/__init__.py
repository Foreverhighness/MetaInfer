"""DeepSeek V4 model — architecture knowledge, kernel mapping, FLOPs."""

from .arch import DEEPSPEED_V4_ARCH
from .mapper import build_mapping, _map_one
from .flops import calculate_mfu, extract_ck_tile_dims
