import importlib.util
from pathlib import Path

import pytest
import triton
from triton import knobs
from triton._internal_testing import is_ppu
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

SPARSE_MLA_PATH = (Path(__file__).resolve().parents[3] / "tutorials" / "tle" / "deepseek_v32" / "02-sparse-mla.py")
SM90_MAX_SHARED_MEMORY = 232448


def _cuda_backend_available():
    try:
        from triton.compiler.compiler import get_backend
        get_backend(GPUTarget("cuda", 90, 32))
        return True
    except Exception:
        return False


def _load_sparse_mla_module():
    spec = importlib.util.spec_from_file_location("sparse_mla_tutorial", SPARSE_MLA_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _compile_prefill_for_sm90(module):
    """Compile ``tle_flashmla_prefill_fwd`` for sm_90 the way the interface launches it (B=1, H=128, HKV=1,
    DQK=576, DV=512, topk=2048, BH=BK=64). The target is given, so no GPU is needed."""
    D, TD, K, H, VG, BH, BK = 512, 64, 2048, 128, 1, 64, 64
    DP = triton.next_power_of_2(D)
    TDP = triton.next_power_of_2(TD)
    G = H // VG
    DPH = DP // 2
    kernel = module.tle_flashmla_prefill_fwd
    signature = {
        "q_desc": f"tensordesc<bf16[{BH}, {DPH}]>",
        "tq_desc": f"tensordesc<bf16[{BH}, {TDP}]>",
        "output_desc": f"tensordesc<bf16[{BH}, {DPH}]>",
        "kv": "*bf16",
        "indices": "*i32",
        "topk_lengths": "*i32",
        "sm_scale": "constexpr",
        "lse": "*fp32",
        "B": "i32",
        "SQ": "i32",
        "SKV": "i32",
    }
    for name in ("K", "D", "TD", "DP", "TDP", "H", "G", "VG", "RH", "BK", "BH", "PAIR_BLOCKS"):
        signature[name] = "constexpr"
    constexprs = {
        "sm_scale": (D + TD)**-0.5,
        "K": K,
        "D": D,
        "TD": TD,
        "DP": DP,
        "TDP": TDP,
        "H": H,
        "G": G,
        "VG": VG,
        "RH": G // BH,
        "BK": BK,
        "BH": BH,
        "PAIR_BLOCKS": module.TLE_FLASHMLA_PREFILL_PAIR_BLOCKS,
    }
    names = list(kernel.arg_names)
    divisible = [["tt.divisibility", 16]]
    attrs = {(names.index(name), ): divisible for name in ("kv", "indices", "topk_lengths", "lse", "SQ", "SKV")}
    return triton.compile(ASTSource(kernel, signature, constexprs, attrs), target=GPUTarget("cuda", 90, 32),
                          options={"num_warps": 4, "num_stages": 1})


def test_tle_flashmla_prefill_fits_sm90_shared_memory_with_read_order_guard_on():
    """With the read-order guard on, a load of a pipe slot is held across the release of a pipe over the same
    buffer, and accelerate-matmul then stages it in a second 32 KiB buffer. The sK1_l and sK1_r pipes are over
    one allocation, so the first load of k1_r_blk in the second consumer would be held across the release of
    sK1_l and need 263920 bytes of shared memory against the 232448 of sm_90. The tutorial reads the slot again
    before the dot, which keeps the first load out of the held path: the kernel fits with the guard on.

    This catches the removal of that second read only where the sK copy is still an async copy. Where
    tritongpu-coalesce downgrades it to a load and a store (main since #1534), the held load is not staged and the
    kernel fits either way; the lit tests test_tle_keep_smem_read_order_{dot_staging,pipe_windows}.mlir pin the
    two cases at the IR level on every tree."""
    if not _cuda_backend_available():
        pytest.skip("requires cuda backend")
    if is_ppu():
        pytest.skip("sm_90 shared memory of the NVIDIA build")
    try:
        from triton.experimental.tle.compiler_knobs import compilation as tle_knobs
    except ImportError:
        pytest.skip("this build has no TLE compilation knobs")
    if not hasattr(triton._C.libtriton.passes.ttgpuir, "add_reorder_instructions_keep_smem_read_order"):
        pytest.skip("this build has no keep-smem-read-order option")

    module = _load_sparse_mla_module()
    with knobs.cache.scope(), knobs.runtime.scope(), tle_knobs.scope():
        tle_knobs.keep_smem_read_order = True
        compiled = _compile_prefill_for_sm90(module)

    assert compiled.metadata.shared <= SM90_MAX_SHARED_MEMORY, (
        f"tle_flashmla_prefill_fwd needs {compiled.metadata.shared} bytes of shared memory with "
        f"keep_smem_read_order on; sm_90 allows {SM90_MAX_SHARED_MEMORY}")
