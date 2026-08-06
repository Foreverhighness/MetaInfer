# DeepSeek V4 Flash — Architecture Reference

> 用于 sglang trace analysis 中的 kernel → 模型结构映射。
> 来源：DeepSeek-V4 Technical Report (arXiv 2606.19348, April 2026)、HuggingFace model card、SGLang 源码。

## 模型概览

DeepSeek V4 于 2026 年 4 月发布，是 DeepSeek 首个基于 **混合注意力 (Hybrid Attention)** 的模型家族。V2/V3 使用的 MLA (Multi-head Latent Attention) 在 V4 中被 CSA + HCA + Sliding Window 三合一注意力系统替代。

两个变体共享相同架构，仅规模不同：

| 参数 | V4-Flash | V4-Pro |
|------|----------|--------|
| 总参数量 | 284B | 1.6T |
| 每 token 激活参数 | 13B | 49B |
| 层数 | 43 | 61 |
| 专家数 | 256 routed + 1 shared | 384 routed |
| Top-K 路由 | 6 | 6 |
| hidden_size | 4096 | — |
| 上下文窗口 | 1M tokens | 1M tokens |
| 最大输出 | 384K tokens | 384K tokens |
| 量化 | FP8 Mixed (Base), FP4+FP8 (Chat) | 同 |
| 许可 | MIT | MIT |

> 我们 profile 的模型是 `DeepSeek-V4-Flash-Channel-INT8-w8a8`——V4-Flash 的 INT8 Flash Channel 量化版本。

## 核心架构创新

### 1. 混合注意力 (Hybrid Attention)

V4 **不使用 MLA**（那是 V2/V3）。V4 使用三层注意力交替：

```
Layer 0–1:     Sliding Window Attention (SWA) — 128 token local window
Layer 2–42:    CSA 与 HCA 交替

  CSA (Compressed Sparse Attention):
    - 每 4 个 token position 压缩为 1 个 entry (compress_ratio=4)
    - Indexer 从中选 top-512 个 entry 做稀疏 attention
    - query heads=128, head_dim=512
    - 对应 kernel: flash_fwd_splitkv_mla_fp8_sparse, indexer, c4_prefill

  HCA (Heavily Compressed Attention):
    - 每 128 个 token position 压缩为 1 个 entry (compress_ratio=128)
    - 对所有压缩 entry 做稠密 attention（因为历史已经足够短）
    - 对应 kernel: 同 CSA 但 compress_ratio=128, c128_prefill
```

### 2. mHC (Manifold-Constrained Hyper-Connections)

替代标准残差连接。每层的 residual stream 通过多条并行分支投影到 Birkhoff polytope（通过 Sinkhorn-Knopp 归一化），然后通过可学习的混合权重重新组合。

- `hc_mult=4` (4 条并行分支)
- `hc_split_sinkhorn` + `sigmoid` 归一化
- 最终 `softmax(dim=-1)` 混合
- 对应 kernel: `mhc_pre_*`, `mhc_post_*`

### 3. DeepSeekMoE

继承自 V3 的细粒度 MoE 框架：

- V4-Flash: 256 routed experts + 1 shared expert（所有层均为 MoE，无 dense MLP）
- Top-6 路由（每个 token 激活 6 个 routed expert + 1 shared = 7 expert）
- 共享 expert 始终计算
- 对应 kernel: `fused_moe_kernel`, expert routing (`mbtopk::radixFindKthValues`, `sbtopk::gatherTopK`)

### 4. MTP (Multi-Token Prediction)

内建投机解码：在主干网络后附加 MTP 头，一次预测多个未来 token。对应 kernel: MTP head GEMM。

## 结构示意

```
Input Token
    ↓
[Embedding]
    ↓
[RMSNorm]
    ↓
[mHC Residual]  ← 替代标准残差连接
    ↓
[Hybrid Attention]  ← CSA 或 HCA (非 MLA)
    │   ├── Q/K/V projection (linear_bf16 或 w8a8_scaled_mm)
    │   ├── RoPE on Q, K
    │   ├── CSA: compress → indexer → sparse attn
    │   └── HCA: compress → dense attn
    ↓
[mHC Residual]
    ↓
[RMSNorm]
    ↓
[MoE MLP]  ← 所有层均为 MoE
    │   ├── Gate projection → Top-6 routing
    │   ├── Shared Expert × 1 (always)
    │   ├── Routed Experts × 6 (from 256)
    │   │   ├── up projection (fused_moe_kernel)
    │   │   └── down projection (CK GEMM / w8a8_scaled_mm)
    │   └── Expert fusion & allreduce
    ↓
[mHC Residual]
    ↓
[RMSNorm] → [LM Head] → [MTP Heads (optional)]
```

## 各层类型分布

| 层范围 | attention_type | compress_ratio | 说明 |
|--------|---------------|----------------|------|
| 0–1 | SWA (full) | — | 前两层滑动窗口，完整 KV cache |
| 2, 4, 6, ... | CSA | 4 | 偶数层：压缩稀疏 attention |
| 3, 5, 7, ... | HCA | 128 | 奇数层：重度压缩稠密 attention |
| 末 N 层 | hash routing | — | paged MQA logits (num_hash_layers) |

## 关键参数 (V4-Flash)

| 参数 | 值 | 来源 |
|------|-----|------|
| num_layers | 43 | arxiv 2606.19348 §3.1 |
| attention_type | CSA + HCA + SWA (hybrid) | arxiv §2.1 |
| hidden_size | 4096 | arxiv §3.1 |
| q_lora_rank | 1024 | SGLang config |
| o_lora_rank | 1024 | SGLang config |
| o_groups | 8 | SGLang config |
| hc_mult | 4 | SGLang config |
| num_experts | 256 | SGLang config |
| top_k_experts | 6 | SGLang config |
| shared_experts | 1 | SGLang config |
| intermediate_size | — (MoE expert hidden) | |
| quantization | INT8 Flash Channel (w8a8) | model path |
| tp_size | 8 | benchmark config |
| page_size | 256 | SGLang config |
| swa_full_tokens_ratio | 0.1 | SGLang config |
| compress_ratios | [0, 0, 4, 128, 4, 128, ...] | SGLang config |
| num_hash_layers | (末 N 层) | SGLang config |

## Kernel → 模型层映射表

| GPU Kernel | 模型组件 | 说明 |
|------------|----------|------|
| `flash_fwd_splitkv_mla_fp8_sparse` | attention/fwd | CSA/HCA sparse attention (kernel 名保留 mla 历史遗留) |
| `fused_moe_kernel` | moe/experts | MoE expert 前向计算 |
| `Cijk_*` (CK GEMM) | linear_bf16 | bf16 矩阵乘法 (QKV/O/gate/up/down proj) |
| `w8a8_scaled_mm_*` | linear_int8 | INT8 量化矩阵乘法 |
| `cross_device_reduce_2stage_pcie` | tp_allreduce | TP 通信 (vllm-style 两阶段 allreduce) |
| `ncclDevKernel_AllGather_Ring_Simple_Sum_i8` | tp_communication | NCCL allgather (MoE dispatch) |
| `mhc_pre_*` | mhc/pre | mHC 残差连接预处理 |
| `mhc_post_*` | mhc/post | mHC 残差连接后处理 |
| `mbtopk::radixFindKthValues` | moe/router | MoE Top-K 专家选择 |
| `sbtopk::gatherTopK` | moe/router | MoE expert gather |
| `rms_norm` | norm | RMS 归一化 (每层 attention 和 MoE 前) |
| `indexer_*` | attention/indexer | CSA indexer (选择 sparse attention 的 KV entries) |
| `c4_prefill` | attention/csa | CSA prefill kernel |
| `c128_prefill` | attention/hca | HCA prefill kernel |
| `paged_mqa_logits` | attention/hash | Hash-table paged MQA (末 N 层) |

## TP 通信

- 8-way tensor parallelism
- 每层 attention 和 MoE 后各一次 allreduce
- `cross_device_reduce_2stage_pcie`: vllm 风格两阶段 allreduce
- `ncclDevKernel_AllGather_Ring_Simple_Sum_i8`: NCCL Ring AllGather (MoE expert dispatch)

## 参考来源

- DeepSeek-V4 Technical Report: [arXiv 2606.19348](https://arxiv.org/abs/2606.19348)
- HuggingFace: [deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- NVIDIA NeMo AutoModel: [deepseek-v4-flash](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/deepseek-v4-flash)
- SGLang v0.5.15 K100 fork: `server_args.json`, `environ.py`, `config.py`
- MetaInfer gen_infer_framework notebooks (V3 architecture reference)
