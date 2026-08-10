import pathlib

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

_BASIC_IR = """
tt.func @kernel(%arg0: !tt.ptr<i32>) {
  %c42_i32 = arith.constant 42 : i32
  gpu.barrier
  ttg.warp_specialize(%arg0)
  default {
    tt.store %arg0, %c42_i32 : !tt.ptr<i32>
    gpu.barrier
    ttg.warp_yield
  }
  partition0(%arg1: !tt.ptr<i32>) num_warps(1) {
    %c5555_i32 = arith.constant 5555 : i32
    %c1_i32 = arith.constant 1 : i32
    gpu.barrier
    %ptr = tt.addptr %arg1, %c1_i32 : !tt.ptr<i32>, i32
    tt.store %ptr, %c5555_i32 : !tt.ptr<i32>
    ttg.warp_return
  } : (!tt.ptr<i32>) -> ()
  tt.return
}
"""


def _is_corex():
    try:
        target = triton.runtime.driver.active.get_current_target()
    except Exception:
        return False
    return target is not None and target.backend == "corex"


requires_corex = pytest.mark.skipif(not _is_corex(), reason="Requires an Iluvatar (corex) device")


@requires_corex
def test_warp_specialize_lowering(tmp_path: pathlib.Path):
    temp_file = tmp_path / "ws_basic.ttir"
    temp_file.write_text(_BASIC_IR)
    compiled = triton.compile(str(temp_file))

    llir = compiled.asm["llir"]
    code_lines = [ln for ln in llir.splitlines() if not ln.lstrip().startswith("!") and "DIFile" not in ln]
    assert "warp_specialize" not in "\n".join(code_lines), llir
    assert "__ws_namedbar_state" in llir, llir


@requires_corex
def test_warp_specialize_basic_e2e(tmp_path: pathlib.Path):
    """End-to-end: the default and worker partitions run concurrently and both
    write their results."""
    temp_file = tmp_path / "ws_basic_e2e.ttir"
    temp_file.write_text(_BASIC_IR)
    kernel = triton.compile(str(temp_file))

    out = torch.empty(2, dtype=torch.int32, device="cuda")
    kernel[(1, 1, 1)](out)
    assert out[0] == 42
    assert out[1] == 5555


# ===========================================================================
# `tle.gpu.warp_specialize` Python frontend tests
#
# Unlike the hand-written-IR tests above, these drive the full
# `tle.gpu.warp_specialize(...)` frontend (python/triton/experimental/tle):
# JIT default/worker partition functions -> `ttg.warp_specialize` op ->
# Iluvatar lowering -> execution. This validates the Iluvatar TLE Python
# bindings (create_warp_* builders + WarpSpecializeOp accessors added in
# third_party/iluvatar/tle/triton_iluvatar_tle.cc) together with the
# ivcore11 software-barrier lowering.
#
# NOTE: warp-specialized kernels require num_warps to be a multiple of 4, and
# worker partitions that use block-level tensors must run with the same warp
# count as the default group so layout inference stays consistent.
# ===========================================================================


@triton.jit
def _ws_fe_default_store(out_ptr):
    tl.store(out_ptr, 42)


@triton.jit
def _ws_fe_worker_store(out_ptr):
    tl.store(out_ptr + 1, 5555)


@triton.jit
def _ws_fe_basic_kernel(out_ptr):
    tle.gpu.warp_specialize(
        [
            (_ws_fe_default_store, (out_ptr, )),
            (_ws_fe_worker_store, (out_ptr, )),
        ],
        worker_num_warps=[1],
        worker_num_regs=[80],
    )


@requires_corex
def test_tle_gpu_warp_specialize_frontend_basic_e2e():
    """The `tle.gpu.warp_specialize` frontend must emit `ttg.warp_specialize`,
    lower it away on Iluvatar, and run both partitions to completion."""
    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    compiled = _ws_fe_basic_kernel[(1, )](out, num_warps=4)

    # The frontend must have produced a real warp-specialized region...
    assert "ttg.warp_specialize" in compiled.asm["ttgir"], compiled.asm["ttgir"]
    # ...that is fully lowered away by the Iluvatar WS pass (ignore debug-info
    # metadata lines whose paths may embed the string).
    llir = compiled.asm["llir"]
    code_lines = [ln for ln in llir.splitlines() if not ln.lstrip().startswith("!") and "DIFile" not in ln]
    assert "warp_specialize" not in "\n".join(code_lines), llir

    assert out[0] == 42
    assert out[1] == 5555


@triton.jit
def _ws_fe_double(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs) * 2.0)


@triton.jit
def _ws_fe_negate(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, -tl.load(x_ptr + offs))


@triton.jit
def _ws_fe_compute_kernel(x_ptr, o0_ptr, o1_ptr, BLOCK: tl.constexpr):
    tle.gpu.warp_specialize(
        [
            (_ws_fe_double, (x_ptr, o0_ptr, BLOCK)),
            (_ws_fe_negate, (x_ptr, o1_ptr, BLOCK)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[80],
    )


@requires_corex
def test_tle_gpu_warp_specialize_frontend_compute_e2e():
    """Two partitions do real block-tensor compute concurrently on independent
    outputs; both results must be numerically correct."""
    BLOCK = 64
    torch.manual_seed(0)
    x = torch.randn(BLOCK, dtype=torch.float32, device="cuda")
    o0 = torch.zeros(BLOCK, dtype=torch.float32, device="cuda")
    o1 = torch.zeros(BLOCK, dtype=torch.float32, device="cuda")

    _ws_fe_compute_kernel[(1, )](x, o0, o1, BLOCK=BLOCK, num_warps=4)

    torch.testing.assert_close(o0, x * 2.0, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(o1, -x, atol=1e-5, rtol=1e-5)


@triton.jit
def _ws_fe_default_ret(x_ptr):
    return tl.load(x_ptr) * 3


@triton.jit
def _ws_fe_worker_side(out_ptr):
    tl.store(out_ptr + 1, 7777)


@triton.jit
def _ws_fe_ret_kernel(x_ptr, out_ptr):
    r = tle.gpu.warp_specialize(
        [
            (_ws_fe_default_ret, (x_ptr, )),
            (_ws_fe_worker_side, (out_ptr, )),
        ],
        worker_num_warps=[1],
        worker_num_regs=[80],
    )
    tl.store(out_ptr, r)


@requires_corex
def test_tle_gpu_warp_specialize_frontend_return_e2e():
    """The default partition returns a value (via `ttg.warp_yield`); the region
    result must be usable after the warp-specialized region."""
    x = torch.tensor([11], dtype=torch.int32, device="cuda")
    out = torch.zeros(2, dtype=torch.int32, device="cuda")

    _ws_fe_ret_kernel[(1, )](x, out, num_warps=4)

    assert out[0] == 33  # default partition: 11 * 3
    assert out[1] == 7777  # worker partition side effect
