#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal pack + FlashAttention varlen demo.

Shows how cu_seqlens prevents cross-sample attention when sequences are packed.

Usage (from repo root):
    python 02-features/01-verl/sequence_packing/test_pack_flash_attn.py
    python 02-features/01-verl/sequence_packing/test_pack_flash_attn.py --backend flash   # needs CUDA + flash-attn
    python 02-features/01-verl/sequence_packing/test_pack_flash_attn.py --backend ref     # pure PyTorch, CPU ok
"""
from __future__ import annotations

import argparse
import math
import sys

import torch


def build_cu_seqlens(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = torch.zeros(seqlens.numel() + 1, dtype=torch.int32, device=attention_mask.device)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    return cu_seqlens, seqlens


def pack_tokens(x: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = attention_mask.size(0)
    cu_seqlens, _ = build_cu_seqlens(attention_mask)
    chunks = [x[i, attention_mask[i].bool()] for i in range(bsz)]
    packed = torch.cat(chunks, dim=0).unsqueeze(0)
    return packed, cu_seqlens


def masked_qkv(
    attention_mask: torch.Tensor,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    bsz, seqlen = attention_mask.shape
    q = torch.randn(bsz, seqlen, dim, generator=gen, device=device, dtype=dtype)
    k = torch.randn(bsz, seqlen, dim, generator=gen, device=device, dtype=dtype)
    v = torch.randn(bsz, seqlen, dim, generator=gen, device=device, dtype=dtype)
    mask = attention_mask.unsqueeze(-1).to(dtype)
    return q * mask, k * mask, v * mask


def ref_padded_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    bsz, _, _ = q.shape
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(q.size(-1))
    for i in range(bsz):
        valid = attention_mask[i].bool()
        qi, ki, vi = q[i, valid], k[i, valid], v[i, valid]
        scores = torch.matmul(qi, ki.transpose(0, 1)) * scale
        causal = torch.triu(torch.ones(scores.shape, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out[i, valid] = torch.matmul(attn, vi)
    return out


def ref_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    q1, k1, v1 = q[0], k[0], v[0]
    out = torch.zeros_like(q1)
    scale = 1.0 / math.sqrt(q1.size(-1))
    cu = cu_seqlens.tolist()
    for start, end in zip(cu[:-1], cu[1:]):
        qi, ki, vi = q1[start:end], k1[start:end], v1[start:end]
        scores = torch.matmul(qi, ki.transpose(0, 1)) * scale
        causal = torch.triu(torch.ones(scores.shape, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out[start:end] = torch.matmul(attn, vi)
    return out.unsqueeze(0)


def wrong_whole_packed_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q1, k1, v1 = q[0], k[0], v[0]
    scale = 1.0 / math.sqrt(q1.size(-1))
    scores = torch.matmul(q1, k1.transpose(0, 1)) * scale
    causal = torch.triu(torch.ones(scores.shape, device=scores.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v1).unsqueeze(0)


def ref_varlen_attention_weights(q: torch.Tensor, k: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    q1, k1 = q[0], k[0]
    total = q1.size(0)
    weights = torch.zeros(total, total, device=q.device, dtype=q.dtype)
    scale = 1.0 / math.sqrt(q1.size(-1))
    cu = cu_seqlens.tolist()
    for start, end in zip(cu[:-1], cu[1:]):
        qi, ki = q1[start:end], k1[start:end]
        scores = torch.matmul(qi, ki.transpose(0, 1)) * scale
        causal = torch.triu(torch.ones(scores.shape, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        weights[start:end, start:end] = torch.softmax(scores, dim=-1)
    return weights


def whole_packed_attention_weights(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    q1, k1 = q[0], k[0]
    scale = 1.0 / math.sqrt(q1.size(-1))
    scores = torch.matmul(q1, k1.transpose(0, 1)) * scale
    causal = torch.triu(torch.ones(scores.shape, device=scores.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    return torch.softmax(scores, dim=-1)


def flash_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    from flash_attn import flash_attn_varlen_func

    q1, k1, v1 = q[0], k[0], v[0]
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    out = flash_attn_varlen_func(
        q1,
        k1,
        v1,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )
    return out.unsqueeze(0)


def make_demo_batch(device: torch.device):
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        device=device,
    )
    return attention_mask


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_demo(backend: str) -> int:
    if backend == "flash":
        if not torch.cuda.is_available():
            print("[FAIL] --backend flash requires CUDA")
            return 1
        try:
            import flash_attn
        except ImportError:
            print("[FAIL] flash-attn not installed. pip install flash-attn")
            return 1
        device = torch.device("cuda")
        dtype = torch.float16
        varlen_fn = flash_varlen_attention
        backend_name = f"flash-attn {flash_attn.__version__}"
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        varlen_fn = ref_varlen_attention
        backend_name = "reference PyTorch (CPU)"

    print(f"backend: {backend_name}")
    print(f"device:  {device}")

    attention_mask = make_demo_batch(device)
    cu_seqlens, seqlens = build_cu_seqlens(attention_mask)
    q, k, v = masked_qkv(attention_mask, dim=16, device=device, dtype=dtype, seed=42)
    q_packed, cu_packed = pack_tokens(q, attention_mask)
    k_packed, _ = pack_tokens(k, attention_mask)
    v_packed, _ = pack_tokens(v, attention_mask)

    print(f"batch seqlens: {seqlens.tolist()}")
    print(f"cu_seqlens:    {cu_packed.tolist()}")

    out_padded = ref_padded_attention(q, k, v, attention_mask)
    out_varlen = varlen_fn(q_packed, k_packed, v_packed, cu_packed.to(device))
    out_wrong = wrong_whole_packed_attention(q_packed, k_packed, v_packed)

    packed_ref = torch.cat([out_padded[i, attention_mask[i].bool()] for i in range(attention_mask.size(0))], dim=0)
    diff_ok = max_abs_diff(out_varlen[0], packed_ref)
    diff_wrong = max_abs_diff(out_varlen[0], out_wrong[0])

    # sample1 starts at packed index cu_seqlens[1] == 3
    start1 = int(cu_packed[1].item())
    cross_from_sample0 = max_abs_diff(out_varlen[0, start1], out_wrong[0, start1])

    print(f"varlen vs padded-ref max diff: {diff_ok:.6g}")
    print(f"varlen vs wrong-pack max diff: {diff_wrong:.6g}")
    print(f"sample1-first-token diff (varlen vs wrong): {cross_from_sample0:.6g}")

    if backend == "ref":
        w_varlen = ref_varlen_attention_weights(q_packed, k_packed, cu_packed.to(device))
        w_wrong = whole_packed_attention_weights(q_packed, k_packed)
        # Row start1 should not attend to columns [0, start1) under varlen.
        leak_varlen = w_varlen[start1, :start1].abs().max().item()
        leak_wrong = w_wrong[start1, :start1].abs().max().item()
        print(f"attention leak sample1->sample0 (varlen): {leak_varlen:.6g}")
        print(f"attention leak sample1->sample0 (wrong):  {leak_wrong:.6g}")
    else:
        leak_varlen = 0.0
        leak_wrong = cross_from_sample0

    ok = (
        diff_ok < 1e-2
        and diff_wrong > 1e-4
        and cross_from_sample0 > 1e-4
        and (backend != "ref" or (leak_varlen < 1e-6 and leak_wrong > 1e-4))
    )
    if ok:
        print("[PASS] cu_seqlens varlen matches per-sample attention; wrong whole-line pack leaks cross-sample.")
        return 0

    print("[FAIL] unexpected numeric result")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo pack + FlashAttention varlen with cu_seqlens")
    parser.add_argument(
        "--backend",
        choices=("auto", "flash", "ref"),
        default="auto",
        help="auto: flash-attn on CUDA if available, else reference CPU implementation",
    )
    args = parser.parse_args()

    if args.backend == "auto":
        if torch.cuda.is_available():
            try:
                import flash_attn  # noqa: F401

                backend = "flash"
            except ImportError:
                backend = "ref"
        else:
            backend = "ref"
    else:
        backend = args.backend

    return run_demo(backend)


if __name__ == "__main__":
    sys.exit(main())
