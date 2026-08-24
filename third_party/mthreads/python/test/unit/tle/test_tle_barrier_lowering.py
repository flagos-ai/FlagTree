"""Lowering and runtime coverage for mthreads TLE hardware barriers."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C import libtriton
from triton._C.libtriton import ir, passes
from triton.compiler import ASTSource
from triton.tools.tensor_descriptor import TensorDescriptor

from test_tle_utils import mthreads_backend, musa_target, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _completion_barrier_phase_reuse_pipeline(desc, out, STAGES: tl.constexpr):
    block: tl.constexpr = 128
    # Exercise two complete phase-parity cycles for every physical slot.
    iterations: tl.constexpr = 4 * STAGES
    smem = tle.gpu.alloc(
        (STAGES, block),
        dtype=tl.float16,
        nv_mma_shared_layout=False,
    )
    full = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=1,
        init=tle.gpu.PENDING,
        expect_bytes=block * 2,
    )
    empty = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=4,
        init=tle.gpu.READY,
    )
    for iteration in tl.static_range(0, iterations):
        tle.gpu.barrier_wait(empty[iteration % STAGES], phaseIdx=iteration // STAGES)
        tle.gpu.copy(
            desc,
            smem.slot(iteration % STAGES),
            (block, ),
            (iteration * block, ),
            barrier=full[iteration % STAGES],
        )
        tle.gpu.barrier_wait(full[iteration % STAGES], phaseIdx=iteration // STAGES)
        value = tl.load(tle.gpu.local_ptr(smem.slot(iteration % STAGES), (0, )))
        tl.store(out + iteration, value)
        tle.gpu.barrier_arrive(empty[iteration % STAGES], phaseIdx=iteration // STAGES)


@triton.jit
def _non_ws_single_slot_pipeline(desc, out, STAGES: tl.constexpr, SLOT: tl.constexpr):
    block: tl.constexpr = 128
    smem = tle.gpu.alloc(
        (STAGES, block),
        dtype=tl.float16,
        nv_mma_shared_layout=False,
    )
    full = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=1,
        init=tle.gpu.PENDING,
        expect_bytes=block * 2,
    )
    empty = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=4,
        init=tle.gpu.READY,
    )
    tle.gpu.barrier_wait(empty[SLOT], phaseIdx=0)
    tle.gpu.copy(
        desc,
        smem.slot(SLOT),
        (block, ),
        (SLOT * block, ),
        barrier=full[SLOT],
    )
    tle.gpu.barrier_wait(full[SLOT], phaseIdx=0)
    value = tl.load(tle.gpu.local_ptr(smem.slot(SLOT), (0, )))
    tl.store(out, value)
    tle.gpu.barrier_arrive(empty[SLOT], phaseIdx=0)


@triton.jit
def _ready_barrier_probe(out):
    ready = tle.gpu.alloc_barriers(1, arrive_count=4, init=tle.gpu.READY)
    tle.gpu.barrier_wait(ready[0], phaseIdx=0)
    tl.store(out, 1)
    tle.gpu.barrier_arrive(ready[0], phaseIdx=0)


@triton.jit(do_not_specialize=["slot", "first_phase"])
def _ready_barrier_array_phase_reuse(out, slot, first_phase):
    ready = tle.gpu.alloc_barriers(2, arrive_count=4, init=tle.gpu.READY)
    for iteration in tl.static_range(0, 2):
        phase = first_phase + iteration
        tle.gpu.barrier_wait(ready[slot], phaseIdx=phase)
        tl.store(out + iteration, slot * 10 + iteration)
        tle.gpu.barrier_arrive(ready[slot], phaseIdx=phase)


def _parse_fixture(tmp_path, body, name="barrier_operations"):
    fixture = f"""module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
    ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32}} {{
  tt.func public @{name}() attributes {{musa.max_bar_id = 3 : i32, noinline = false}} {{
{body}    tt.return
  }}
}}
"""
    fixture_path = tmp_path / f"{name}.ttgir"
    fixture_path.write_text(fixture)
    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    return context, ir.parse_mlir_module(str(fixture_path), context)


def _lower_barrier_operations(module, context):
    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_tle_lower_barrier_operations(pm)
    pm.run(module, "lower_tle_barrier_operations")
    return module.str_nodebug()


def _compile_phase_reuse(stages):
    source = ASTSource(
        fn=_completion_barrier_phase_reuse_pipeline,
        signature={"desc": "tensordesc<fp16[128]>", "out": "*fp16", "STAGES": "constexpr"},
        constexprs={"STAGES": stages},
        attrs={(0, ): [["musa.tme_tail_divisibility", 4]]},
    )
    return triton.compile(
        source,
        target=musa_target(),
        options={"num_warps": 4, "num_stages": 1},
    )


def test_mthreads_tle_barrier_operations_lower_and_are_idempotent(tmp_path):
    body = """    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    musa_tle.barrier.wait %c1, %c0
    musa_tle.barrier.wait %c2, %c1
    musa_tle.barrier.arrive %c3, %c1 {arrive_count = 1 : i32}
"""
    context, module = _parse_fixture(tmp_path, body)
    lowered = _lower_barrier_operations(module, context)

    assert "musa_tle.barrier.wait" not in lowered, lowered
    assert "musa_tle.barrier.arrive" not in lowered, lowered
    assert lowered.count("ttmg.wait_barrier") == 2, lowered
    assert lowered.count("ttmg.warp_arrive_barrier") == 1, lowered
    assert "ttmg.arrive_barrier_noret" not in lowered, lowered
    constants = {
        name: int(value)
        for name, value in re.findall(
            r"(%[-\w.]+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*i32",
            lowered,
        )
    }
    waits = re.findall(r"ttmg\.wait_barrier\s+(%[-\w.]+),\s*(%[-\w.]+)", lowered)
    assert [(constants[barrier], constants[phase]) for barrier, phase in waits] == [(1, 0), (2, 1)]
    arrive = re.search(r"ttmg\.warp_arrive_barrier\s+(%[-\w.]+)", lowered).group(1)
    assert constants[arrive] == 3

    rerun = _lower_barrier_operations(module, context)
    assert rerun == lowered


def test_mthreads_tle_consumer_arrival_is_warp_convergent_in_llvm(tmp_path):
    body = """    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    ttmg.bar_record %c1 : i32
    musa_tle.barrier.wait %c1, %c0
    musa_tle.barrier.arrive %c1, %c0 {arrive_count = 1 : i32}
"""
    context, module = _parse_fixture(tmp_path, body, "barrier_operations_llvm")
    _lower_barrier_operations(module, context)

    pm = ir.pass_manager(context)
    passes.convert.add_scf_to_cf(pm)
    passes.convert.add_index_to_llvmir(pm)
    libtriton.mthreads.passes.ttgpuir.add_allocate_shared_memory(pm, 31)
    libtriton.mthreads.passes.ttgpuir.add_mtgpu_to_llvm(pm, 31)
    libtriton.mthreads.passes.ttgpuir.add_to_llvmir(pm, 31)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.convert.add_cf_to_llvmir(pm)
    passes.convert.add_arith_to_llvmir(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    pm.run(module, "lower_tle_barrier_operations_to_llvm")
    llir = module.str_nodebug()

    assert llir.count('llvm.call_intrinsic "llvm.musa.async.wait"') == 1, llir
    assert llir.count('llvm.call_intrinsic "llvm.musa.async.arrive.none.phaseid"') == 1, llir
    assert "gpu.thread_id" not in llir, llir
    assert "llvm.musa.read.ptx.sreg.tid.x" not in llir, llir


@pytest.mark.parametrize("stages", [1, 2])
def test_mthreads_tle_completion_copy_phase_reuse_compiles_to_mubin(stages):
    compiled = _compile_phase_reuse(stages)
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]

    assert "musa_tle.barrier." not in ttgir, ttgir
    assert ttgir.count("ttmg.wait_barrier") == 8 * stages, ttgir
    assert ttgir.count("ttmg.warp_arrive_barrier") == 4 * stages, ttgir
    assert ttgir.count("ttmg.arrive_barrier_noret") == 4 * stages, ttgir
    assert "llvm.musa.async.wait" in llir, llir
    assert "llvm.musa.async.arrive.none.phaseid" in llir, llir
    assert "llvm.musa.tme.ld.tile.1d" in llir, llir
    assert re.search(r"\bphi\s+i64\b", llir) is None, llir
    assert compiled.asm["mubin"], compiled.asm.keys()


@pytest.mark.parametrize("stages,slot", [(1, 0), (2, 0), (2, 1)])
def test_mthreads_tle_completion_copy_single_slot_compiles_to_mubin(stages, slot):
    source = ASTSource(
        fn=_non_ws_single_slot_pipeline,
        signature={
            "desc": "tensordesc<fp16[128]>",
            "out": "*fp16",
            "STAGES": "constexpr",
            "SLOT": "constexpr",
        },
        constexprs={"STAGES": stages, "SLOT": slot},
        attrs={(0, ): [["musa.tme_tail_divisibility", 4]]},
    )
    compiled = triton.compile(source, target=musa_target(), options={"num_warps": 4, "num_stages": 1})
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]

    assert "musa_tle.barrier." not in ttgir, ttgir
    assert ttgir.count("ttmg.wait_barrier") == 2, ttgir
    assert ttgir.count("ttmg.warp_arrive_barrier") == 1, ttgir
    assert ttgir.count("ttmg.arrive_barrier_noret") == 1, ttgir
    assert "llvm.musa.async.wait" in llir, llir
    assert "llvm.musa.async.arrive" in llir, llir
    assert "llvm.musa.async.arrive.none.phaseid" in llir, llir
    assert "llvm.musa.tme.ld.tile.1d" in llir, llir
    assert compiled.asm["mubin"], compiled.asm.keys()


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_mthreads_tle_ready_barrier_reinitializes_across_launches():
    for _ in range(2):
        out = torch.zeros((1, ), device="musa", dtype=torch.int32)
        _ready_barrier_probe[(1, )](out, num_warps=4, num_stages=1)
        torch.musa.synchronize()
        assert out.cpu().item() == 1


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
@pytest.mark.parametrize("slot", [0, 1])
def test_mthreads_tle_ready_barrier_array_dynamic_slot_and_phase_runtime(slot):
    for _ in range(2):
        out = torch.empty((2, ), device="musa", dtype=torch.int32)
        _ready_barrier_array_phase_reuse[(1, )](
            out,
            slot,
            0,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()

        expected = torch.tensor([slot * 10, slot * 10 + 1], dtype=torch.int32)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
@pytest.mark.parametrize("stages", [1, 2])
def test_mthreads_tle_completion_copy_phase_reuse_runtime(stages):
    block = 128
    iterations = 4 * stages
    for launch in range(2):
        start = launch * iterations * block
        src = torch.arange(
            start,
            start + iterations * block,
            device="musa",
            dtype=torch.float16,
        )
        out = torch.empty((iterations, ), device="musa", dtype=torch.float16)
        desc = TensorDescriptor.from_tensor(src, [block])

        _completion_barrier_phase_reuse_pipeline[(1, )](
            desc,
            out,
            STAGES=stages,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()

        torch.testing.assert_close(out.cpu(), src[::block].cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
@pytest.mark.parametrize("stages,slot", [(1, 0), (2, 0), (2, 1)])
def test_mthreads_tle_completion_copy_single_slot_runtime(stages, slot):
    block = 128
    src = torch.arange(
        stages * block,
        device="musa",
        dtype=torch.float16,
    )
    out = torch.empty((1, ), device="musa", dtype=torch.float16)
    desc = TensorDescriptor.from_tensor(src, [block])

    _non_ws_single_slot_pipeline[(1, )](
        desc,
        out,
        STAGES=stages,
        SLOT=slot,
        num_warps=4,
        num_stages=1,
    )
    torch.musa.synchronize()

    expected = src[slot * block:slot * block + 1]
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=0, atol=0)
