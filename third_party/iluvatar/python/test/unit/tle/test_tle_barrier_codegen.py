"""Software pipe completion broadcasts retain leader and phase contracts."""

import re

import pytest
import triton
from triton.compiler.compiler import GPUTarget


@pytest.mark.parametrize("arrival_attrs", [
    "",
    "participant_arrive, drain_arrive",
    "participant_arrive, sme_async_payload",
    "participant_arrive, warp_aggregated_reader_release",
    "participant_arrive, sme_async_payload, cta_aggregated_writer_commit, "
    "cta_aggregated_writer_leader_tid = 0 : i32",
    "participant_arrive, warp_aggregated_reader_release, cta_aggregated_reader_release",
])
@pytest.mark.parametrize("warp_size", [32, 64])
@pytest.mark.parametrize("static_count", [True, False])
def test_pipe_barrier_completion_codegen(tmp_path, arrival_attrs, warp_size, static_count):
    count = 4 * warp_size
    wait_attrs = f"static_arrive_count = {count} : i32" if static_count else ""
    source = tmp_path / "pipe_barrier.ttgir"
    source.write_text("""
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:71", "ttg.threads-per-warp" = WARP_SIZE : i32} {
  tt.func public @kernel(%out: !tt.ptr<i32>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    iluvatar_tle.init_barrier %bar, ARRIVE_COUNT : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    gpu.barrier
    iluvatar_tle.arrive_barrier %bar, ARRIVE_COUNT {ARRIVAL_ATTRS} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    iluvatar_tle.wait_barrier %bar, %zero {WAIT_ATTRS} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.store %out, %one : !tt.ptr<i32>
    tt.return
  }
}
""".replace("ARRIVAL_ATTRS", arrival_attrs).replace("WAIT_ATTRS", wait_attrs).replace(
        "ARRIVE_COUNT", str(count)).replace("WARP_SIZE", str(warp_size)))
    compiled = triton.compile(str(source), target=GPUTarget("corex", 71, 64))
    llir = compiled.asm["llir"]
    assert re.search(rf"and i32 %[-\w.]+, {warp_size - 1}\b", llir), llir
    assert 'syncscope("workgroup") acquire' in llir
    assert 'syncscope("workgroup") release' in llir or not arrival_attrs
    # Initialization and CTA aggregation still elect one partition thread,
    # not one lane per warp. They must retain the CTA thread-ID read.
    assert "@llvm.nvvm.read.ptx.sreg.tid.x()" in llir
    block_parts = re.split(r"(?m)^([-\w.]+):[^\n]*\n", llir)
    blocks = list(zip(block_parts[1::2], block_parts[2::2]))
    if "cta_aggregated" in arrival_attrs or not arrival_attrs:
        assert re.search(rf"atomicrmw add .*i32 {count} .*\b(?:release|acq_rel)\b", llir), llir
        init_pred = re.search(r"br i1 (%[-\w.]+), label", llir).group(1)
        arrival_block = next(label for label, body in blocks if "atomicrmw add" in body)
        assert f"br i1 {init_pred}, label %{arrival_block}," in llir, llir
    else:
        assert re.search(rf"atomicrmw add .*i32 {warp_size} .*release", llir), llir

    poll_blocks = [(label, body) for label, body in blocks if "load atomic" in body]
    assert len(poll_blocks) == 1, llir
    label, body = poll_blocks[0]
    successors = re.findall(r"\blabel %([-\w.]+)", body)
    assert label not in successors, body
    assert len(successors) == 1, body
    assert "llvm.bi.slb.shfl" not in body, body
    assert "llvm.bi.vote.any" not in body, body
    assert "atomicrmw" not in body, body
    completion_intrinsic = "llvm.bi.vote.any" if warp_size == 64 else "llvm.bi.slb.shfl.idx.b32"
    merge_blocks = [(label, body) for label, body in blocks if f"call i32 @{completion_intrinsic}" in body]
    assert len(merge_blocks) == 1, llir
    merge_label, merge_body = merge_blocks[0]
    assert successors == [merge_label], body
    assert re.search(r"phi i32 .*\[ 0, %[-\w.]+ \]", merge_body), merge_body
    if warp_size == 64:
        assert "@llvm.bi.vote.any(i32" in llir, llir
        assert "llvm.bi.slb.shfl.idx.b32" not in llir, llir
    else:
        assert "llvm.bi.vote.any" not in llir, llir
        assert "llvm.bi.slb.shfl.idx.b32" in llir, llir
