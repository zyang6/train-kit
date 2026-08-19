# sequence_packing

verl 中 **pack**（sequence packing / remove padding）特性的说明文档。整理自 verl 0.9.0 代码路径与训练实践中的常见问题。

## 是什么

在 verl 里，大家说的 **pack** 基本就是 **sequence packing**，开关是：

```bash
actor_rollout_ref.model.use_remove_padding=True
# 或
critic.model.use_remove_padding=True
```

官方文档表述：`use_remove_padding=True` for sequence packing（即 data packing + remove padding）。

核心动作：

1. **remove padding**：去掉 pad token
2. **sequence packing**：把 batch 内各样本的有效 token 首尾相接，拼成一条 packed 序列

## 和 `use_dynamic_bsz` 的区别

两者常一起开，但不是一回事。

| | `use_dynamic_bsz` | pack（`use_remove_padding`） |
|--|--|--|
| 解决什么 | 按 **token 总数** 切 micro-batch | 去掉 pad，把有效 token **拼成 packed 序列** 再算 |
| 改的是 | batch **切分策略** | 单次 forward 的 **张量布局 / attention 路径** |
| 关了会怎样 | 退回固定 `*_micro_batch_size_per_gpu` | 退回 `[B, S]` pad 布局，pad 也占算力 |
| 主要调参 | `*_max_token_len_per_gpu` | 开关本身；常配合 `pad_to_length` / SP / CP |

关系可以记成：

```
大 batch
  └─ use_dynamic_bsz: 按 token 数切成 micro-batches      ← 组 batch
        └─ use_remove_padding: 每个 micro-batch 去 pad 再 forward  ← 算内部
```

## 代码在哪里

### 配置开关

- `actor_rollout_ref.model.use_remove_padding`
- `critic.model.use_remove_padding`
- engine 侧：`verl/workers/config/engine.py`、`verl/workers/config/actor.py`

### 数据侧：去 pad → nested / packed

| 文件 | 作用 |
|------|------|
| `verl/workers/utils/padding.py` | `left_right_2_no_padding`、`unpad_input`、nested 转换 |
| `verl/utils/attention_utils.py` | CUDA/NPU 统一的 `unpad_input` / `pad_input` |

### FSDP 训练前向

| 文件 | 作用 |
|------|------|
| `verl/workers/engine/fsdp/transformer_impl.py` | `use_remove_padding=True` 时走 varlen；`attention_mask=None`；可选传 `cu_seqlens` |

### Megatron 训练前向

| 文件 | 作用 |
|------|------|
| `verl/models/mcore/util.py` | `preprocess_packed_seqs` / `postprocess_packed_seqs`、`PackedSeqParams` |
| `verl/workers/engine/megatron/transformer_impl.py` | `data_format = "thd" if use_remove_padding else "bshd"` |

### dynamic bsz（与 pack 独立）

| 文件 | 作用 |
|------|------|
| `verl/workers/engine/utils.py` | `prepare_micro_batches` |
| `verl/utils/seqlen_balancing.py` | `rearrange_micro_batches` |

### 文档

- `docs/perf/perf_tuning.rst` — Enable remove padding (sequence packing)

## 训练里 pack 会影响什么

1. **算力与显存**  
   pad token 基本不参与有效计算，吞吐通常更高；样本长度差异大时收益更明显。

2. **Attention 形态**  
   - 未 pack：`[B, S]` + `attention_mask`，标准 causal mask  
   - pack 后：`[1, total_nnz]` + `cu_seqlens`，走 FlashAttention varlen / Megatron THD

3. **log_prob / value / entropy 的布局**  
   先在 packed 一维上算，再按 `cu_seqlens` 嵌回 nested，最后可能 pad 到 `[B, max_response_len]` 做 PPO loss。

4. **与并行的绑定**  
   - Ulysses SP：要求 `use_remove_padding=True`  
   - Megatron CP / dynamic CP：通常要求 THD packing  
   - `pad_to_length`：给 packed 序列做桶对齐，需 pack 已开启

5. **不影响什么**  
   - 不改 PPO/GRPO 算法公式本身  
   - 不负责「一个 micro-batch 装多少条样本」（那是 `use_dynamic_bsz`）  
   - 一般不直接改 rollout 引擎；主要作用在 actor/ref/critic 的训练与 logprob forward

## 前向：未 pack vs pack

### 未 pack（BSHD）

```
input_ids:      [B, S]
attention_mask: [B, S]   # 0=pad, 1=有效
```

每条样本 pad 到同一长度 `S`。attention 在 batch 维天然隔离，样本之间不会互相看见。

### pack 后（THD / varlen）

```
input_ids:      [1, total_nnz]     # 只保留有效 token
cu_seqlens:     [0, L0, L0+L1, …]  # 累计边界
attention_mask: None               # 不再传二维 mask
```

模型只对有效 token 做计算；样本边界由 `cu_seqlens` 表达，而不是 pad mask。

## 为什么会「串台」

self-attention 默认把输入当成**一条连续序列**。对位置 `i`，causal 规则是：可以看自己和所有**前面的** token（`j ≤ i`），不是「只看本样本」。

若把两条样本拼成一条，又不告诉 kernel 在哪里断开：

```
真实含义:  [你好吗] [今天天气]
拼完:      你 好 吗 今 天 天 气
index:     0  1  2  3  4  5  6
```

算「今」(index=3) 时，causal 允许看 index 0,1,2,3，于是第二句会看到第一句——这就是**跨样本 attend（串台）**。

来源是 attention 核心计算：

\[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}+M\right)V
\]

对 token `i`，分数 \(s_{ij}=q_i\cdot k_j\) 决定它「看」哪些 token。causal mask 只挡**未来**（`j>i`），不挡**跨样本的前缀**。pack 后不同样本接在同一前缀里，就会串。

`cu_seqlens` / varlen attention 的作用：在样本交界处额外切断可见性，段内仍做 causal，段与段之间不建 attention 边。

## `cu_seqlens` 怎么用

### 是什么

一维累计长度表，长度 = `样本数 + 1`：

```
样本长度:  3, 2, 4
cu_seqlens = [0, 3, 5, 9]
```

第 `i` 条样本对应 token 区间：`tokens[cu_seqlens[i] : cu_seqlens[i+1]]`

### 怎么建

```python
# attention_mask: [B, S]，1=有效
seqlens = attention_mask.sum(dim=-1)
cu_seqlens = torch.zeros(B + 1, dtype=torch.int32)
cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
```

FSDP nested 路径：`cu_seqlens = input_ids.offsets()`，与上面等价。

Megatron：`preprocess_packed_seqs` 里从 `attention_mask.sum` 生成，再写入 `PackedSeqParams`。

### 前向传给谁

**FSDP / FlashAttention varlen：**

```python
model_inputs = {
    "input_ids": input_ids_rmpad,      # (1, total_nnz)
    "attention_mask": None,
    "position_ids": position_ids_rmpad,
    "cu_seqlens": cu_seqlens,           # 部分模型显式接收
}
```

**Megatron THD：**

```python
PackedSeqParams(
    qkv_format="thd",
    cu_seqlens_q=cu_seqlens_padded,
    cu_seqlens_kv=cu_seqlens_padded,
    max_seqlen_q=...,
    max_seqlen_kv=...,
)
```

kernel 对每个样本段 `[start:end)` 单独做段内 causal attention，不会跨段。

### 算完后「嵌回」

forward 输出是 `(total_nnz, …)` 一维结果。嵌回 = 用同一份 `cu_seqlens` 按样本边界切回 nested：

```python
log_probs = torch.nested.nested_tensor_from_jagged(log_probs_rmpad, cu_seqlens)
```

之后 PPO 可能再 pad 到 `[B, max_response_len]` 与 advantage 对齐。**嵌回不是把 pad token 填回去**，而是恢复「按样本组织」的结构。

## pack 后还用 `attention_mask` 吗

**会用到，但阶段不同。**

| 阶段 | 是否用 `[B,S]` attention_mask |
|------|-------------------------------|
| pack 前：数长度、抽有效 token | **用** |
| Megatron postprocess 填回 `[B,S]` | **用** |
| packed attention 计算本身 | **一般不用**，改传 `cu_seqlens` |
| `use_remove_padding=False` 的 fallback | 会从 nested **重建** mask |

packed 前向里常见：`attention_mask=None`，边界交给 `cu_seqlens`。

## 和相近名字的区别

| 名字 | 是不是 pack |
|------|-------------|
| `use_remove_padding` | **就是** pack 开关 |
| `use_dynamic_bsz` | 动态切 micro-batch，常和 pack 一起开 |
| Ray `STRICT_PACK` / `bin_pack` | 资源调度，与序列 packing 无关 |
| vLLM `packed_modules_mapping` | 权重/LoRA 模块融合（q/k/v→qkv），无关 |

## 常见配置示例

```bash
# 开 pack
actor_rollout_ref.model.use_remove_padding=True
critic.model.use_remove_padding=True

# 常配合 dynamic bsz
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=...

# Ulysses SP 必须开 pack
# actor_rollout_ref.actor.ulysses_sequence_parallel_size > 1 时
# use_remove_padding 必须为 True
```

## 限制与注意

- 只优化 **训练 / logprob forward** 路径，不直接改 rollout 引擎行为。
- 值按字符串/repr 比较时，`1` 与 `1.0` 可能判为不同（config 对比工具里需注意）。
- 部分模型/结构不支持 pack（如文档中 Qwen3.5 Gated Delta Net 曾注明保持 `use_remove_padding=False`）。
- NPU 路径依赖 `verl/utils/attention_utils.py` 中的 NPU flash attention 适配。

## 参考（verl 源码）

以 verl 0.9.0（`release/v0.9.0`）为准：

- `verl/workers/engine/fsdp/transformer_impl.py` — FSDP pack 前向
- `verl/models/mcore/util.py` — Megatron preprocess/postprocess packed seqs
- `verl/workers/utils/padding.py` — 数据 pack/unpack
- `docs/perf/perf_tuning.rst` — 性能调优中的 sequence packing 说明

## 测试：pack + FlashAttention varlen

`test_pack_flash_attn.py` 用最小例子演示：

1. 从 `attention_mask` 建 `cu_seqlens`
2. 把 `[B,S,D]` pack 成 `[1,total_nnz,D]`
3. 用 **varlen**（`cu_seqlens` 切段）做 causal attention
4. 对比 **错误** 的「整段一条序列」attention（会串台）

在仓库根目录执行：

```bash
# 自动：有 CUDA + flash-attn 走 FA，否则走 CPU 参考实现
python 02-features/01-verl/sequence_packing/test_pack_flash_attn.py

# 强制 CPU 参考实现（无需 flash-attn）
python 02-features/01-verl/sequence_packing/test_pack_flash_attn.py --backend ref

# 强制 FlashAttention varlen（需 CUDA + flash-attn）
pip install flash-attn
python 02-features/01-verl/sequence_packing/test_pack_flash_attn.py --backend flash
```

通过时会打印 `[PASS]`：varlen 结果与逐样本 padded attention 一致，而错误 whole-line pack 明显不同。
