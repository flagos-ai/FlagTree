"""Standard storage facts do not block a proved queue schedule (CPU only)."""
from types import SimpleNamespace
import re

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)


@triton.jit
def pipelined_mm(A, B, C):
    row = tl.program_id(0) * 256
    col = tl.program_id(1) * 256
    a_ptr = tl.make_block_ptr(A, (768, 1536), (1536, 1), (row, 0), (256, 16), (1, 0))
    b_ptr = tl.make_block_ptr(B, (1536, 512), (512, 1), (0, col), (16, 256), (1, 0))
    acc = tl.zeros((256, 256), tl.float32)
    for _ in tl.range(0, 1536, 16, loop_unroll_factor=4):
        a = tle.load(a_ptr, boundary_check=(0, 1), padding_option="zero", is_async=True)
        b = tle.load(b_ptr, boundary_check=(0, 1), padding_option="zero", is_async=True)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptr = tl.advance(a_ptr, (0, 16))
        b_ptr = tl.advance(b_ptr, (16, 0))
    offsets = (row + tl.arange(0, 256))[:, None] * 512 + (col + tl.arange(0, 256))[None, :]
    tl.store(C + offsets, acc.to(C.dtype.element_ty))


@pytest.fixture(autouse=True)
def cpu_driver(monkeypatch):
    target = GPUTarget("hip", "gfx936", 64)
    monkeypatch.setattr(triton.runtime.driver, "_active", SimpleNamespace(
        get_current_device=lambda: 0, get_current_stream=lambda _: 0,
        get_current_target=lambda: target))
    monkeypatch.setenv("FLAGTREE_AABS", "0")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")


def compile_mm(dtype, post=False, phase=False):
    source = ASTSource(pipelined_mm, {name: "*" + dtype for name in ["A", "B", "C"]}, {},
                       attrs={(i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)] for i in range(3)})
    return triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options=dict(
        num_warps=8, num_stages=5, loop_unroll_stage="post-pipeline",
        schedule_hint="local-prefetch", lds_dma_alias_workaround=True,
        prefer_wide_async_copy=True, kpack=2, mmac_warps_m=4,
        mmac_transposed=True, optimize_epilogue=True,
        coalesce_lgkm_waits=True, mma_phase_schedule=phase, post_mma_wait=post))


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_storage_assumptions_preserved_and_unknown_effects_rejected(dtype):
    src = compile_mm(dtype).asm["llir"]
    line = next(line for line in src.splitlines(keepends=True)
                if "call void @llvm.assume(i1 true)" in line)
    header = next(line for line in src.splitlines() if line.startswith("define "))
    pointers = re.findall(r"ptr addrspace\(1\)[^,%]*\s(%[\w.]+)", header)
    assert len(pointers) >= 3
    conditional = (f"  %conditional.assume.test = icmp ne ptr addrspace(1) {pointers[0]}, null\n"
                   "  call void @llvm.assume(i1 %conditional.assume.test)\n")
    # LLVM requires true conditions on assumptions carrying operand bundles.
    # The conditional/false negatives are valid ordinary assumptions without bundles.
    cases = [("true_storage_facts", src, 1),
             ("conditional", src.replace(line, conditional, 1), 0),
             ("false", src.replace(line, "  call void @llvm.assume(i1 false)\n", 1), 0),
             ("unknown_call", src.replace(line, line + "  call void @unknown_effect()\n", 1)
              + "\ndeclare void @unknown_effect()\n", 0),
             ("volatile_store", src.replace(line, line +
              f"  store volatile i32 0, ptr addrspace(1) {pointers[2]}\n", 1), 0)]
    for name, ir, expected in cases:
        changed, loops, reason = libtriton.llvm.prepare_hcu_post_mma_wait(ir, "gfx936", 512, 65536)
        assert loops == expected, (dtype, name, reason)
        if loops:
            assert changed.count("call void @llvm.assume(i1 true)") == ir.count("call void @llvm.assume(i1 true)")
            assert "CROSS_WAVE_FIXED_POINT" in reason
        else:
            assert changed == ir
        print("MODEL_CASE", dtype, name, loops, reason)


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("phase", [False, True])
def test_scheduled_native_completion(dtype, phase):
    compiled = compile_mm(dtype, post=True, phase=phase)
    metadata = compiled.metadata._asdict()
    assert metadata["hcu_post_mma_wait_loops"] == 1
    assert metadata["hcu_schedule_native_readiness"]["accepted"]
    assert "CROSS_WAVE_FIXED_POINT" in metadata["hcu_post_mma_wait_preparation_result"]
