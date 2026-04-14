import importlib.util
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


def _load_topk_module():
    repo_root = Path(__file__).resolve().parents[4]
    mod_path = repo_root / "python" / "tutorials" / "tle" / "deepseek_v32" / "01-topk_selector.py"
    spec = importlib.util.spec_from_file_location("tle_topk_selector_tutorial", mod_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TOPK_MOD = _load_topk_module()


def _recall(pred: torch.Tensor, ref: torch.Tensor) -> float:
    pred_set = set(pred[0].cpu().tolist())
    ref_set = set(ref[0].cpu().tolist())
    return len(pred_set & ref_set) / ref.shape[1]


@triton.jit
def _fallback_only_kernel(
    x_ptr,
    out_ptr,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    seq_len,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_ptr = x_ptr + pid * stride_xm
    out_row = out_ptr + pid * stride_outm
    hist = tle.gpu.alloc([4096], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    write_cnt = tle.gpu.alloc([1], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    eq_cnt = tle.gpu.alloc([1], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    _TOPK_MOD._tle_topk_smem_overflow_fallback_fullscan(
        row_ptr,
        out_row,
        stride_xn,
        stride_outn,
        tl.zeros((), dtype=tl.int32),
        seq_len,
        seq_len,
        tle.gpu.local_ptr(hist, (0, )),
        tle.gpu.local_ptr(write_cnt, (0, )),
        tle.gpu.local_ptr(eq_cnt, (0, )),
        TOPK=TOPK,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tle_topk_smem_recall_seq262144():
    torch.manual_seed(1)
    seq_len = 262144
    topk = 2048
    x = torch.randn((1, seq_len), device=_TOPK_MOD.DEVICE, dtype=torch.float32)
    starts = torch.zeros((1, ), device=_TOPK_MOD.DEVICE, dtype=torch.int32)
    ends = torch.full((1, ), seq_len, device=_TOPK_MOD.DEVICE, dtype=torch.int32)
    ref = torch.topk(x, topk, dim=-1)[1]

    smem_out = _TOPK_MOD.tle_topk_selector_smem(
        x,
        starts,
        ends,
        topk,
        block_size=1024,
        assume_aligned=True,
    )
    assert _recall(smem_out, ref) == 1.0

    if _TOPK_MOD._supports_tle_cluster_remote():
        cluster_out = _TOPK_MOD.tle_topk_selector_smem_cluster(
            x,
            starts,
            ends,
            topk,
            block_size=1024,
            assume_aligned=True,
        )
        assert _recall(cluster_out, ref) == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("num_warps", [16, 32])
def test_tle_topk_fallback_fullscan_stable_high_warps(num_warps):
    torch.manual_seed(1)
    seq_len = 262144
    topk = 2048
    x = torch.randn((1, seq_len), device=_TOPK_MOD.DEVICE, dtype=torch.float32)
    ref = torch.topk(x, topk, dim=-1)[1]

    outputs = []
    for _ in range(3):
        out = torch.full((1, topk), -1, device=_TOPK_MOD.DEVICE, dtype=torch.int32)
        _fallback_only_kernel[(1, )](
            x,
            out,
            x.stride(0),
            x.stride(1),
            out.stride(0),
            out.stride(1),
            seq_len=seq_len,
            TOPK=topk,
            BLOCK_SIZE=1024,
            num_warps=num_warps,
            num_stages=1,
        )
        outputs.append(out.clone())
        assert _recall(out, ref) == 1.0

    out_sets = [set(o[0].cpu().tolist()) for o in outputs]
    assert out_sets[0] == out_sets[1]
    assert out_sets[1] == out_sets[2]
