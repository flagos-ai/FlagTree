#!/usr/bin/env python3
"""Profile a small gpt-triton workload with FlagTree Profiler.

The upstream project is https://github.com/thevasudevgupta/gpt-triton.  This
runner intentionally uses a small random-weight GPT block so it can be used as a
multi-kernel profiler smoke test on Ascend without downloading HuggingFace
weights.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys

import triton.profiler as proton


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def _load_torch():
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is false")
    return torch


def _load_gpt_triton_kernels():
    default_source = _repo_root() / ".external" / "gpt-triton"
    source = pathlib.Path(os.environ.get("GPT_TRITON_SOURCE", default_source))
    if not (source / "kernels.py").exists():
        raise RuntimeError(f"gpt-triton source not found at {source}. Clone "
                           "https://github.com/thevasudevgupta/gpt-triton there or set "
                           "GPT_TRITON_SOURCE.")
    sys.path.insert(0, str(source))
    import kernels

    return kernels


def _make_weights(torch, device, *, vocab: int, seq_len: int, hidden: int, heads: int):
    del heads
    intermediate = hidden * 4
    dtype = torch.float16
    torch.manual_seed(1337)
    return {
        "wte": torch.randn((vocab, hidden), device=device, dtype=dtype),
        "wpe": torch.randn((seq_len, hidden), device=device, dtype=dtype),
        "ln1_w": torch.ones((hidden, ), device=device, dtype=dtype),
        "ln1_b": torch.zeros((hidden, ), device=device, dtype=dtype),
        "qkv_w": torch.randn((hidden, hidden * 3), device=device, dtype=dtype),
        "qkv_b": torch.randn((hidden * 3, ), device=device, dtype=dtype),
        "proj_w": torch.randn((hidden, hidden), device=device, dtype=dtype),
        "proj_b": torch.randn((hidden, ), device=device, dtype=dtype),
        "ln2_w": torch.ones((hidden, ), device=device, dtype=dtype),
        "ln2_b": torch.zeros((hidden, ), device=device, dtype=dtype),
        "ff1_w": torch.randn((hidden, intermediate), device=device, dtype=dtype),
        "ff1_b": torch.randn((intermediate, ), device=device, dtype=dtype),
        "ff2_w": torch.randn((intermediate, hidden), device=device, dtype=dtype),
        "ff2_b": torch.randn((hidden, ), device=device, dtype=dtype),
        "lnf_w": torch.ones((hidden, ), device=device, dtype=dtype),
        "lnf_b": torch.zeros((hidden, ), device=device, dtype=dtype),
    }


def _gpt_block(kernels, token_ids, weights, *, heads: int):
    with proton.scope("prefill"):
        with proton.scope("embeddings"):
            x = kernels.fused_embeddings(token_ids, weights["wte"], weights["wpe"])

        with proton.scope("self-attn"):
            residual = x
            with proton.scope("attn_ln"):
                x = kernels.fused_layer_norm(x, weights["ln1_w"], weights["ln1_b"])
            with proton.scope("qkv_proj"):
                q, k, v = kernels.matmul_and_split_qkv(x, weights["qkv_w"], weights["qkv_b"], heads)
            with proton.scope("attn"):
                x = kernels.flash_attention_v1(q, k, v)
            x = x.transpose(1, 2).contiguous().view(residual.shape)
            with proton.scope("o_proj"):
                x = kernels.fused_ffn(x, weights["proj_w"], bias=weights["proj_b"], residual=residual)

        with proton.scope("mlp"):
            residual = x
            with proton.scope("mlp_ln"):
                x = kernels.fused_layer_norm(x, weights["ln2_w"], weights["ln2_b"])
            with proton.scope("gate_up_proj"):
                x = kernels.fused_ffn(x, weights["ff1_w"], bias=weights["ff1_b"], add_gelu=True)
            with proton.scope("down_proj"):
                x = kernels.fused_ffn(x, weights["ff2_w"], bias=weights["ff2_b"], residual=residual)

        with proton.scope("final_ln"):
            return kernels.fused_layer_norm(x, weights["lnf_w"], weights["lnf_b"])


def main() -> int:
    torch = _load_torch()
    kernels = _load_gpt_triton_kernels()

    device_id = int(os.environ.get("FLAGTREE_PROFILER_DEVICE_ID", "1"))
    torch.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")

    batch = int(os.environ.get("GPT_TRITON_BATCH", "4"))
    seq_len = int(os.environ.get("GPT_TRITON_SEQ_LEN", "128"))
    hidden = int(os.environ.get("GPT_TRITON_HIDDEN", "256"))
    heads = int(os.environ.get("GPT_TRITON_HEADS", "8"))
    vocab = int(os.environ.get("GPT_TRITON_VOCAB", "1024"))

    out_dir = pathlib.Path(os.environ.get("GPT_TRITON_PROFILE_OUT", "/tmp/flagtree_profiler_gpt_triton"))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_base = out_dir / "profile"

    token_ids = torch.randint(0, vocab, (batch, seq_len), device=device, dtype=torch.long)
    weights = _make_weights(torch, device, vocab=vocab, seq_len=seq_len, hidden=hidden, heads=heads)

    _gpt_block(kernels, token_ids, weights, heads=heads)
    torch.npu.synchronize()

    sid = proton.start(
        name=str(profile_base),
        context="shadow",
        data="tree",
        backend="cann",
        hook="triton",
        mode=("runtime_base:"
              f"device_id={device_id}:"
              "vendor_metrics=aicore,bandwidth:"
              "mstx_enabled=true:"
              "mstx_domain=proton"),
    )
    try:
        _gpt_block(kernels, token_ids, weights, heads=heads)
        torch.npu.synchronize()
    finally:
        proton.finalize(sid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
