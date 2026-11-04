import pathlib
import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler.compiler import ASTSource, GPUTarget

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

_MIXED_SLOT_IR = """
tt.func @kernel() {
  gpu.barrier
  ttg.warp_specialize()
  default {
    gpu.barrier
    ttg.warp_yield
  }
  partition0() num_warps(2) {
    gpu.barrier
    ttg.warp_return
  } : () -> ()
  ttg.warp_specialize()
  default {
    gpu.barrier
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    gpu.barrier
    ttg.warp_return
  } : () -> ()
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
def test_warp_specialize_mixed_slot_uses_generation_counter(tmp_path: pathlib.Path):
    """Different participant counts reusing one slot must not round tickets
    against the current participant count; the packed generation protocol
    advances only after the final leader arrives.
    """
    temp_file = tmp_path / "ws_mixed_slot.ttir"
    temp_file.write_text(_MIXED_SLOT_IR)
    compiled = triton.compile(
        str(temp_file),
        options={"num_warps": 8},
    )

    llir = compiled.asm["llir"]
    assert llir.count("cmpxchg ptr addrspace(3)") >= 2, llir
    assert re.search(r"and i32 %\d+, -256", llir), llir
    assert re.search(r"and i32 %\d+, 255", llir), llir
    assert re.search(r"add i32 %\d+, 256", llir), llir


def test_warp_specialize_arrival_is_not_repeated_by_poll(tmp_path: pathlib.Path):
    """A leader must increment one software-barrier phase exactly once.

    The acquire polling block may branch to itself, but the block containing
    the counter RMW must not have a self-edge.  A self-edge would turn a slow
    wait into repeated arrivals and corrupt the phase counter.
    """
    temp_file = tmp_path / "ws_arrival_once.ttir"
    temp_file.write_text(_MIXED_SLOT_IR)
    compiled = triton.compile(
        str(temp_file),
        target=GPUTarget("corex", 71, 64),
        options={"num_warps": 8},
    )

    llir = compiled.asm["llir"]
    lines = llir.splitlines()
    blocks = []
    current_label = None
    current_body = []
    block_label = re.compile(r"^([A-Za-z0-9_.]+):(?:\s+;.*)?$")
    for line in lines:
        match = block_label.match(line)
        if match:
            if current_label is not None:
                blocks.append((current_label, "\n".join(current_body)))
            current_label = match.group(1)
            current_body = []
        elif current_label is not None:
            current_body.append(line)
    if current_label is not None:
        blocks.append((current_label, "\n".join(current_body)))

    arrival_blocks = [
        (label, body)
        for label, body in blocks
        if "atomicrmw add ptr addrspace(3)" in body
        and "@__ws_namedbar_state" in body
    ]
    assert arrival_blocks, llir
    for label, body in arrival_blocks:
        assert f"label %{label}" not in body, (label, body)


def _assert_named_barrier_completion_vote(llir, warp_size):
    arrivals = re.findall(r"atomicrmw add ptr addrspace\(3\)[^\n]*@__ws_namedbar_state", llir)
    assert arrivals, llir
    votes = re.findall(r"call i32 @llvm.bi.vote.any\(i32 (%[-\w.]+)\)", llir)
    if warp_size == 64:
        assert len(votes) == len(arrivals), llir
        # The dispatch header still normalizes its integer warp ID by shuffle.
        assert len(re.findall(r"call i32 @llvm.bi.slb.shfl.idx.b32", llir)) == 1, llir
        functions = re.findall(r"(?ms)^define [^\n]+\{\n(.*?)^\}", llir)
        for function in functions:
            for value in re.findall(r"call i32 @llvm.bi.vote.any\(i32 (%[-\w.]+)\)", function):
                definition = re.search(rf"(?m)^\s*{re.escape(value)} = phi i32 ([^\n]+)", function)
                if definition:
                    assert re.search(r"\[ 1, %[-\w.]+ \]", definition.group(1)), definition.group(0)
                    assert re.search(r"\[ 0, %[-\w.]+ \]", definition.group(1)), definition.group(0)
                else:
                    # LLVM may replace the constant merge phi by the lane-0
                    # predicate, but the convergent vote must remain after poll.
                    extension = re.search(rf"{re.escape(value)} = zext i1 (%[-\w.]+) to i32", function)
                    assert extension, function
                    predicate = re.search(
                        rf"{re.escape(extension.group(1))} = icmp eq i32 (%[-\w.]+), 0", function)
                    assert predicate, function
                    assert re.search(rf"{re.escape(predicate.group(1))} = .*call i32 @llvm.bi.lane.id", function)
            block_parts = re.split(r"(?m)^([-\w.]+):[^\n]*\n", function)
            blocks = list(zip(block_parts[1::2], block_parts[2::2]))
            vote_blocks = {label for label, body in blocks if "call i32 @llvm.bi.vote.any" in body}
            for label, body in blocks:
                if "load atomic" not in body or "@__ws_namedbar_state" not in body:
                    continue
                successors = set(re.findall(r"\blabel %([-\w.]+)", body))
                assert label in successors, body
                assert len(successors & vote_blocks) == 1, body
    else:
        assert not votes, llir
        assert "llvm.bi.slb.shfl.idx.b32" in llir, llir
    assert 'syncscope("workgroup") release' in llir
    assert 'syncscope("workgroup") acquire' in llir


@pytest.mark.parametrize("source_ir", [_BASIC_IR, _MIXED_SLOT_IR], ids=["fixed", "mixed"])
@pytest.mark.parametrize("warp_size", [32, 64])
def test_warp_specialize_completion_codegen(tmp_path, source_ir, warp_size):
    source = tmp_path / "ws_completion.ttir"
    source.write_text(source_ir)
    compiled = triton.compile(
        str(source), target=GPUTarget("corex", 71, 64),
        options={"num_warps": 8, "warp_size": warp_size},
    )
    _assert_named_barrier_completion_vote(compiled.asm["llir"], warp_size)
    if source_ir == _MIXED_SLOT_IR:
        llir = compiled.asm["llir"]
        assert llir.count("cmpxchg ptr addrspace(3)") >= 2
        assert re.search(r"and i32 %[-\w.]+, -256", llir), llir
        assert re.search(r"and i32 %[-\w.]+, 255", llir), llir
        assert re.search(r"add i32 %[-\w.]+, 256", llir), llir


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


@triton.jit(noinline=True)
def _ws_fe_noinline_helper_barrier(out_ptr):
    # This helper is called only by the default warp group.  Its barrier must
    # therefore be lowered to the default group's software barrier, not CTA-wide.
    tl.debug_barrier()
    tl.store(out_ptr, 4242)


@triton.jit
def _ws_fe_noinline_helper_barrier_kernel(out_ptr):
    tle.gpu.warp_specialize(
        [
            (_ws_fe_noinline_helper_barrier, (out_ptr, )),
            (_ws_fe_worker_store, (out_ptr, )),
        ],
        worker_num_warps=[1],
        worker_num_regs=[80],
    )


@requires_corex
def test_tle_gpu_warp_specialize_noinline_helper_barrier_e2e():
    """A helper CTA barrier must respect its caller's warp-group scope."""
    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    compiled = _ws_fe_noinline_helper_barrier_kernel[(1,)](out, num_warps=4)

    assert "__tle_ws_barrier" in compiled.asm["llir"], compiled.asm["llir"]
    assert out[0] == 4242
    assert out[1] == 5555


def test_warp_specialize_helper_barrier_completion_codegen():
    compiled = triton.compile(
        ASTSource(_ws_fe_noinline_helper_barrier_kernel, signature={"out_ptr": "*i32"}),
        target=GPUTarget("corex", 71, 64), options={"num_warps": 4},
    )
    llir = compiled.asm["llir"]
    assert "__tle_ws_barrier" in llir
    _assert_named_barrier_completion_vote(llir, 64)


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
