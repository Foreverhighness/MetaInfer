# DeepSeek V4 — Architecture Reference

用于 sglang trace analysis 中的 kernel → 模型结构映射。

## 结构示意

```
Input Token
    ↓
[Embedding]
    ↓
[DecoderLayer × 61]
    │
    ├── [RMSNorm]
    ├── [MLA Attention]          ← Multi-head Latent Attention
    │   ├── Q/KV compression     ← kv_lora_rank=512
    │   ├── RoPE on Q, K         ← rotary position embedding
    │   └── flash_fwd_splitkv_mla ← GPU kernel
    ├── [残差]
    ├── [RMSNorm]
    ├── [MoE MLP]
    │   ├── [共享专家 × 1]       ← always computed
    │   ├── [路由专家 × N 选 K]  ← top-K gating (K=8)
    │   │   ├── gate projection  ← CK GEMM / w8a8_scaled_mm
    │   │   ├── up projection    ← fused_moe_kernel
    │   │   └── down projection  ← CK GEMM / w8a8_scaled_mm
    │   └── [TopK routing]       ← mbtopk::radixFindKthValues + gatherTopK
    └── [残差]
    ↓
[RMSNorm]
    ↓
[LM Head]
```

## 关键参数

| 参数 | 值 |
|------|-----|
| num_layers | 61 |
| moe_layers | 58 (前三层 dense) |
| attention_type | MLA (Multi-head Latent Attention) |
| hidden_size | 7168 |
| head_dim | 192 |
| kv_lora_rank | 512 |
| num_experts | 256 |
| top_k_experts | 8 |
| shared_experts | 1 |
| intermediate_size | 2048 |
| quantization | INT8 Flash Channel |
| tp_size | 8 |
| swa_full_tokens_ratio | 0.1 |

## DSA (DeepSeek Sparse Attention)

- 替代 NSA，长上下文稀疏 attention
- MHC (Multi-Head Cache): mhc_pre / mhc_post kernel
- page_size=256, swa tokens per batch
- mhc_pages: 256

## Kernel → 模型层映射表

| GPU Kernel | 模型组件 |
|------------|----------|
| `flash_fwd_splitkv_mla` | attention/mla |
| `fused_moe_kernel` | moe/experts |
| `Cijk_*` (CK GEMM) | linear_bf16 |
| `w8a8_scaled_mm_*` | linear_int8 |
| `cross_device_reduce_2stage_pcie` | tp_allreduce |
| `mhc_pre_*` | dsa/mhc_pre |
| `mhc_post_*` | dsa/mhc_post |
| `mbtopk::radixFindKthValues` | moe/router |
| `sbtopk::gatherTopK` | moe/router |
| `rms_norm` | norm |

## TP 通信

- 8-way tensor parallelism
- `cross_device_reduce_2stage_pcie` (vllm-style two-stage allreduce)
- `ncclDevKernel_AllGather_Ring_Simple_Sum_i8` (NCCL allgather for MoE dispatch)
- 每层 MLA 和 MoE 后各有一次 allreduce

## 量化

- INT8 Flash Channel: 权重静态量化，activation 动态量化
- w8a8 scaled_mm: INT8 矩阵乘法 kernel
- CK GEMM: bf16 矩阵乘法 (composable_kernel 库, K100 GPU)

## 参考来源

- `metainfer/tasks/gen_infer_framework/notebooks/02_model_specifics/02_deepseek_v3/`
- sglang server_args.json (attention_backend='dsv4')
