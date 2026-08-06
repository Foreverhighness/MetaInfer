"""DeepSeek V4 model architecture knowledge.

Sourced from MetaInfer's ``gen_infer_framework`` knowledge base at
``metainfer/tasks/gen_infer_framework/notebooks/02_model_specifics/02_deepseek_v3/``.

Used by :mod:`.mapper` to attribute GPU kernels to specific model layers.
"""

# DeepSeek V4 architecture reference
# --------------------------------
# Structure per decoder layer (61 layers total, ~58 with MoE):
#   RMSNorm → MLA Attention → residual
#   → RMSNorm → MoE MLP (1 shared expert + top-K routed experts) → residual
#
# Key components:
#   - MLA (Multi-head Latent Attention): compressed KV cache,
#     uses flash_fwd_splitkv_mla kernel on K100
#   - MoE: fine-grained, shared + routed experts, fused_moe_kernel
#   - MHC (Multi-Head Cache): mhc_pre/mhc_post kernels for
#     DSA (DeepSeek Sparse Attention, NSA successor)
#   - INT8 quantized weights (Flash Channel): w8a8_scaled_mm
#     for linear projections (QKV, O, gate, up, down)
#   - CK GEMM (Cijk_*): composable_kernel library for bf16 GEMM on K100
#   - cross_device_reduce_2stage_pcie: vllm-style TP allreduce
#   - RMSNorm: rms_norm before attention and MoE in each layer
#   - RoPE: rotary position embedding applied to Q and K
#   - TopK routing: MoE expert selection via radixFindKthValues/gatherTopK

DEEPSPEED_V4_ARCH = {
    "family": "DeepSeek-V4",
    "num_layers": 61,
    "attention_type": "mla",          # Multi-head Latent Attention
    "moe_layers": 58,                 # ~58 of 61 layers have MoE
    "shared_experts": 1,              # always computed
    "routed_experts_per_token": 8,    # top-K from total pool
    "quantization": "int8_flash_channel",
    "head_dim": 192,                  # MLA compressed head dim
    "kv_lora_rank": 512,              # MLA KV compression rank
    "hidden_size": 7168,              # model hidden dimension
    "intermediate_size": 2048,        # MoE expert intermediate
    "mhc_enabled": True,              # DSA sparse attention
    "mhc_pages": 256,
    "tp_size": 8,                     # tensor parallelism
    "kernel_map": {
        "flash_fwd_splitkv_mla":      "attention/mla",
        "fused_moe_kernel":           "moe/experts",
        "w8a8_scaled_mm":             "linear_int8",
        "Cijk_*":                     "linear_bf16",
        "cross_device_reduce_2stage": "tp_allreduce",
        "mhc_pre_*":                  "dsa/mhc_pre",
        "mhc_post_*":                 "dsa/mhc_post",
    },
}
