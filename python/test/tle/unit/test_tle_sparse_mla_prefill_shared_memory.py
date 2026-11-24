import importlib.util
import re
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


# A load of a pipe slot that is held in registers and used as a dot operand is staged in a second shared memory
# buffer by accelerate-matmul: a tensor<64x256xbf16> local_alloc in the kernel.
STAGED_B_OPERAND = re.compile(r"ttg\.local_alloc %\S+ : \(tensor<64x256xbf16")


@pytest.mark.parametrize("keep_smem_read_order", [True, False])
def test_tle_flashmla_prefill_does_not_stage_the_k_operand(keep_smem_read_order):
    """The sparse-MLA prefill kernel fits the shared memory of sm_90 with the read-order guard on and off.

    The sK1_l and sK1_r pipes are declared over one buffer, but the writer of each fills one half of it. If the
    release of sK1_l were taken to give back the whole buffer, the guard would hold the load of the right half in
    registers across it, and accelerate-matmul would stage that value in a second 32 KiB buffer for the dot: 263920
    bytes of shared memory against the 232448 of sm_90. The staging buffer is what the test looks for, so that it
    fails on a tree where the sK copy has become a load and a store too (main since #1534), where the kernel fits
    for the wrong reason. The lit test test_tle_keep_smem_read_order_pipe_writers.mlir pins the decision on IR."""
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
        tle_knobs.keep_smem_read_order = keep_smem_read_order
        compiled = _compile_prefill_for_sm90(module)

    knob = f"keep_smem_read_order={keep_smem_read_order}"
    staged = STAGED_B_OPERAND.search(compiled.asm["ttgir"])
    assert not staged, f"the K operand of the second consumer is staged in shared memory with {knob}"
    assert compiled.metadata.shared <= SM90_MAX_SHARED_MEMORY, (
        f"tle_flashmla_prefill_fwd needs {compiled.metadata.shared} bytes of shared memory with {knob}; "
        f"sm_90 allows {SM90_MAX_SHARED_MEMORY}")
