import importlib.util
from pathlib import Path

import pytest
import torch
from triton import knobs
from triton._internal_testing import is_cuda, is_hip

SPARSE_MLA_PATH = (Path(__file__).resolve().parents[2] / "tutorials" / "tle" / "deepseek_v32" / "02-sparse-mla.py")
SERIALIZED_WGMMA_WARNING = "Potential Performance Loss: wgmma.mma_async instructions are serialized"
NO_SPILL_LINE = "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"


def _load_sparse_mla_module():
    spec = importlib.util.spec_from_file_location("sparse_mla_tutorial", SPARSE_MLA_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tle_sparse_mla_ptxas_has_no_spill_or_serialized_wgmma_warning(capfd, fresh_triton_cache):
    if is_hip():
        pytest.skip("CUDA-specific ptxas check")
    if not is_cuda() or not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("TLE sparse MLA WGMMA ptxas check requires sm90")

    module = _load_sparse_mla_module()
    q, kv, indices = module._build_sparse_mla_inputs(
        B=1,
        S=1,
        SKV=4096,
        H=128,
        HKV=1,
        DQK=576,
        topk=2048,
        dtype=torch.bfloat16,
        seed=1,
    )
    topk_length = module._compute_topk_length(indices, 2048)

    capfd.readouterr()
    with knobs.nvidia.scope():
        knobs.nvidia.dump_ptxas_log = True
        out, lse = module.tle_sparse_mla_fwd_interface(
            q,
            kv,
            indices,
            topk_length=topk_length,
            sm_scale=None,
            d_v=512,
            is_causal=True,
        )
        torch.cuda.synchronize()

    assert out.shape == (1, 1, 128, 512)
    assert lse.shape == (1, 1, 128)

    captured = capfd.readouterr()
    ptxas_log = captured.out + captured.err
    assert "Compiling entry function 'tle_sparse_mla_fwd'" in ptxas_log
    assert NO_SPILL_LINE in ptxas_log
    assert SERIALIZED_WGMMA_WARNING not in ptxas_log
