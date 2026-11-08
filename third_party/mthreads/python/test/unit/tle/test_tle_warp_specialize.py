"""Compile and runtime coverage for the MUSA TLE warp-specialize container."""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.experimental.tle.language.gpu.mthreads import (
    warp_specialize as mthreads_warp_specialize, )
from triton._C import libtriton
from triton._C.libtriton import ir
from triton.backends.compiler import Language
from triton.compiler import ASTSource
from triton.compiler.errors import CompilationError

from test_tle_utils import compile_musa, mthreads_backend, musa_target, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _ws_default(out, value):
    tl.store(out, value + 1)


@triton.jit
def _ws_worker(out, value, smem, duplicate_out, BIAS: tl.constexpr):
    local = tle.gpu.local_ptr(smem, (0, ))
    tl.store(local, value)
    loaded = tl.load(local)
    tl.store(out + 1, loaded + BIAS)
    tl.store(duplicate_out + 2, loaded + BIAS + 1)


@triton.jit
def _ws_container_kernel(out, value, BIAS: tl.constexpr):
    smem = tle.gpu.alloc(
        (1, ),
        dtype=tl.int32,
        nv_mma_shared_layout=False,
    )
    tle.gpu.warp_specialize(
        [
            (_ws_default, (out, value)),
            (_ws_worker, (out, value, smem, out, BIAS)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_simple_default(out):
    tl.store(out, 1)


@triton.jit
def _ws_simple_worker(out):
    tl.store(out + 1, 2)


@triton.jit
def _ws_multi_default(out):
    smem = tle.gpu.alloc((1, ), dtype=tl.int32, nv_mma_shared_layout=False)
    local = tle.gpu.local_ptr(smem, (0, ))
    tl.store(local, 11)
    tl.store(out, tl.load(local))


@triton.jit
def _ws_multi_worker0(out):
    smem = tle.gpu.alloc((1, ), dtype=tl.int32, nv_mma_shared_layout=False)
    local = tle.gpu.local_ptr(smem, (0, ))
    tl.store(local, 22)
    tl.store(out + 1, tl.load(local))


@triton.jit
def _ws_multi_worker1(out):
    smem = tle.gpu.alloc((1, ), dtype=tl.int32, nv_mma_shared_layout=False)
    local = tle.gpu.local_ptr(smem, (0, ))
    tl.store(local, 33)
    tl.store(out + 2, tl.load(local))


@triton.jit
def _ws_multi_worker2(out):
    smem = tle.gpu.alloc((1, ), dtype=tl.int32, nv_mma_shared_layout=False)
    local = tle.gpu.local_ptr(smem, (0, ))
    tl.store(local, 44)
    tl.store(out + 3, tl.load(local))


@triton.jit
def _ws_multi_worker_kernel(out):
    tle.gpu.warp_specialize(
        [
            (_ws_multi_default, (out, )),
            (_ws_multi_worker0, (out, )),
            (_ws_multi_worker1, (out, )),
            (_ws_multi_worker2, (out, )),
        ],
        worker_num_warps=[1, 2, 4],
        worker_num_regs=[24, 32, 40],
    )


@triton.jit
def _ws_barrier_default(out):
    tl.store(out, 11)


@triton.jit
def _ws_barrier_worker0(out, barrier):
    tle.gpu.barrier_arrive(barrier, phaseIdx=0)
    tle.gpu.barrier_wait(barrier, phaseIdx=0)
    tl.store(out + 1, 22)


@triton.jit
def _ws_barrier_worker1(out, barrier):
    tle.gpu.barrier_arrive(barrier, phaseIdx=0)
    tle.gpu.barrier_wait(barrier, phaseIdx=0)
    tl.store(out + 2, 33)


@triton.jit
def _ws_barrier_worker2(out, barrier):
    tle.gpu.barrier_arrive(barrier, phaseIdx=0)
    tle.gpu.barrier_wait(barrier, phaseIdx=0)
    tl.store(out + 3, 44)


@triton.jit
def _ws_multi_worker_barrier_kernel(out):
    barrier0 = tle.gpu.alloc_barrier(arrive_count=1)
    barrier1 = tle.gpu.alloc_barrier(arrive_count=2)
    barrier2 = tle.gpu.alloc_barrier(arrive_count=4)
    tle.gpu.warp_specialize(
        [
            (_ws_barrier_default, (out, )),
            (_ws_barrier_worker0, (out, barrier0)),
            (_ws_barrier_worker1, (out, barrier1)),
            (_ws_barrier_worker2, (out, barrier2)),
        ],
        worker_num_warps=[1, 2, 4],
        worker_num_regs=[24, 32, 40],
    )


@triton.jit
def _ws_dynamic_warps_kernel(out, worker_warps):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[worker_warps],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_dynamic_regs_kernel(out, worker_regs):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[4],
        worker_num_regs=[worker_regs],
    )


@triton.jit
def _ws_non_sequence_warps_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=4,
        worker_num_regs=[24],
    )


@triton.jit
def _ws_non_sequence_regs_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[4],
        worker_num_regs=24,
    )


@triton.jit
def _ws_zero_warps_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[0],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_non_power_of_two_warps_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[3],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_non_terminal_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )
    tl.store(out + 2, 3)


@triton.jit
def _ws_warp_count_mismatch_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_register_count_mismatch_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[4],
        worker_num_regs=[],
    )


@triton.jit
def _ws_empty_functions_kernel(out):
    tle.gpu.warp_specialize([], worker_num_warps=[], worker_num_regs=[])


@triton.jit
def _ws_no_workers_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, ))],
        worker_num_warps=[],
        worker_num_regs=[],
    )


@triton.jit
def _ws_invalid_entry_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), _ws_simple_worker],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_invalid_args_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, out)],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_constexpr_tuple_config_kernel(out, WORKER_WARPS: tl.constexpr, WORKER_REGS: tl.constexpr):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=(WORKER_WARPS, ),
        worker_num_regs=(WORKER_REGS, ),
    )


def _compile_ws_ir(
    fn=_ws_container_kernel,
    signature=None,
    constexprs=None,
    consumer_warps=16,
    include_late=False,
    reserved_ids=None,
    lowering_target=None,
):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": consumer_warps, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(
        fn=fn,
        signature=signature or {"out": "*i32", "value": "i32"},
        constexprs={"BIAS": 7} if constexprs is None else constexprs,
    )
    module = src.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    stages = {}
    backend.add_stages(stages, options, Language.TRITON)
    metadata = {}
    module = stages["ttir"](module, metadata)
    ttir = module.str_nodebug()
    module = stages["ttgir"](module, metadata)
    ttgir = module.str_nodebug()
    if include_late:
        builder = ir.builder(context)
        if reserved_ids is not None:
            function = module.get_function(module.get_entry_func_name())
            for name in ("musa.next_bar_id", "musa.max_bar_id"):
                function.set_attr(name, builder.get_int32_attr(reserved_ids))
        pm = ir.pass_manager(context)
        libtriton.passes.convert.add_scf_to_cf(pm)
        libtriton.passes.convert.add_index_to_llvmir(pm)
        libtriton.mthreads.passes.ttgpuir.add_allocate_shared_memory(pm, 31)
        libtriton.mthreads.passes.ttgpuir.add_mtgpu_to_llvm(pm, 31)
        libtriton.mthreads.passes.ttgpuir.add_to_llvmir(pm, 31)
        pm.run(module, "lower_static_ws_to_llvm_cfg")
        if lowering_target is not None:
            module.set_attr("ttg.target", builder.get_string_attr(lowering_target))
        pm = ir.pass_manager(context)
        libtriton.mthreads.passes.ttgpuir.add_tle_lower_warp_specialize(pm)
        pm.run(module, "lower_partition_barriers")
        return ttir, ttgir, module.str_nodebug()
    return ttir, ttgir


def test_tle_warp_specialize_static_multi_worker_runtime():
    out = torch.full((4, ), -1, dtype=torch.int32, device="musa")
    compiled = _ws_multi_worker_kernel.warmup(
        out,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    assert compiled.metadata.num_warps == 11
    assert "llvm.musa.barrier0" not in compiled.asm["llir"]
    # Concurrent partition-local buffers must have distinct backing addresses.
    # Check disjointness, not allocator-chosen offsets or dispatch instructions.
    shared_offsets = {0}
    shared_offsets.update(
        int(offset) for offset in re.findall(
            r"getelementptr \(i8, ptr addrspace\(3\) @global_smem, i32 (\d+)\)",
            compiled.asm["llir"],
        ))
    assert len(shared_offsets) == 4
    for _ in range(4):
        out.fill_(-1)
        _ws_multi_worker_kernel[(1, )](out, num_warps=4, num_stages=1)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu(), torch.tensor([11, 22, 33, 44], dtype=torch.int32))


def test_tle_warp_specialize_multi_worker_barrier_runtime():
    out = torch.full((4, ), -1, dtype=torch.int32, device="musa")
    compiled = _ws_multi_worker_barrier_kernel.warmup(
        out,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    assert compiled.metadata.num_warps == 11
    assert "llvm.musa.barrier0" not in compiled.asm["llir"]
    for _ in range(2):
        out.fill_(-1)
        _ws_multi_worker_barrier_kernel[(1, )](out, num_warps=4, num_stages=1)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu(), torch.tensor([11, 22, 33, 44], dtype=torch.int32))


def test_tle_warp_specialize_musa_container_contract():
    builder = libtriton.ir.builder
    expected_capabilities = {
        "mthreads_tle_pipe_sqmma_version": 2,
        "mthreads_tle_multifield_pipe_version": 3,
        "mthreads_tle_one_shot_pipe_version": 1,
        "mthreads_tle_spmc_pipe_version": 2,
        "mthreads_tle_dynamic_loops_version": 1,
        "mthreads_tle_split_m_sqmma_version": 1,
        "mthreads_tle_split128_sqmma_version": 1,
        "mthreads_tle_local_barrier_version": 1,
        "mthreads_tle_persistent_ordered_sqmma_version": 1,
        "mthreads_tle_dynamic_partition_sync_version": 1,
        "mthreads_tle_grouped_completion_version": 1,
        "mthreads_tle_16_warp_persistent_version": 1,
        "mthreads_tle_fused_pipe_consumer_version": 1,
    }
    for capability, minimum in expected_capabilities.items():
        assert getattr(builder, capability, 0) >= minimum
    for method in (
            "create_mthreads_local_barrier",
            "create_warp_specialize",
            "create_warp_specialize_partitions",
            "create_warp_yield",
            "create_warp_return",
    ):
        assert hasattr(builder, method)
    assert hasattr(libtriton.mthreads.ir, "WarpSpecializeOp")

    out = torch.empty((3, ), dtype=torch.int32, device="musa")
    compiled = _ws_container_kernel.warmup(out, 5, BIAS=7, grid=(1, ), num_warps=16, num_stages=1)
    assert compiled.metadata.num_warps == 20
    assert "llvm.musa.barrier0" not in compiled.asm["llir"]
    # Exercise dynamic scalar and shared-memory captures, duplicate pointers,
    # and the constexpr argument through their observable results.
    for value in (5, 11):
        out.fill_(-1)
        compiled[(1, 1, 1)](out, value, 7)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu(), torch.tensor([value + 1, value + 7, value + 8], dtype=torch.int32))


def test_tle_warp_specialize_keeps_old_builder_call_compatible():

    class OldBuilder:

        def create_warp_specialize(self, result_types, worker_num_warps):
            return result_types, worker_num_warps

    builder = OldBuilder()
    assert mthreads_warp_specialize.create_op(builder, ["result"], [4]) == (
        ["result"],
        [4],
    )


@pytest.mark.parametrize(
    "consumer_warps,producer_warps",
    [(4, 4), (16, 1), (16, 2), (16, 4), (16, 8)],
    ids=["equal_4_4", "producer_1", "producer_2", "producer_4", "producer_8"],
)
def test_tle_warp_specialize_static_two_branch_runtime(consumer_warps, producer_warps):
    out = torch.zeros((2, ), dtype=torch.int32, device="musa")
    compiled = _ws_constexpr_tuple_config_kernel.warmup(
        out,
        WORKER_WARPS=producer_warps,
        WORKER_REGS=24,
        grid=(1, ),
        num_warps=consumer_warps,
        num_stages=1,
    )
    assert compiled.metadata.num_warps == consumer_warps + producer_warps
    assert "llvm.musa.barrier0" not in compiled.asm["llir"]

    for _ in range(2):
        out.zero_()
        _ws_constexpr_tuple_config_kernel[(1, )](
            out,
            WORKER_WARPS=producer_warps,
            WORKER_REGS=24,
            num_warps=consumer_warps,
            num_stages=1,
        )
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu(), torch.tensor([1, 2], dtype=torch.int32))


@pytest.mark.parametrize(
    "kernel,diagnostic",
    [
        (
            _ws_non_terminal_kernel,
            "MUSA TLE static warp_specialize must be the final operation before tt.return",
        ),
    ],
)
def test_tle_warp_specialize_lowering_rejects_unsupported_contract(capfd, kernel, diagnostic):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_ws_ir(kernel, {"out": "*i32"}, constexprs={})
    assert diagnostic in capfd.readouterr().err


# Explicit TME/SQMMA pipeline integration is covered by
# test_tle_warp_specialize_integration.py; keep only container-specific tests here.


@pytest.mark.parametrize(
    "kernel,signature,diagnostic",
    [
        (
            _ws_dynamic_warps_kernel,
            {"out": "*i32", "worker_warps": "i32"},
            r"MUSA TLE warp_specialize worker_num_warps\[0\] must be a compile-time integer",
        ),
        (
            _ws_dynamic_regs_kernel,
            {"out": "*i32", "worker_regs": "i32"},
            r"MUSA TLE warp_specialize worker_num_regs\[0\] must be a compile-time integer",
        ),
        (
            _ws_non_sequence_warps_kernel,
            {"out": "*i32"},
            "MUSA TLE warp_specialize worker_num_warps must be a static sequence",
        ),
        (
            _ws_non_sequence_regs_kernel,
            {"out": "*i32"},
            "MUSA TLE warp_specialize worker_num_regs must be a static sequence",
        ),
        (
            _ws_zero_warps_kernel,
            {"out": "*i32"},
            r"MUSA TLE warp_specialize worker_num_warps\[0\] must be positive",
        ),
        (
            _ws_warp_count_mismatch_kernel,
            {"out": "*i32"},
            "warp_specialize got 1 worker functions but 0 warp counts",
        ),
        (
            _ws_register_count_mismatch_kernel,
            {"out": "*i32"},
            "warp_specialize got 1 worker functions but 0 register counts",
        ),
        (
            _ws_empty_functions_kernel,
            {"out": "*i32"},
            "warp_specialize requires at least a default partition function",
        ),
        (
            _ws_invalid_entry_kernel,
            {"out": "*i32"},
            "warp_specialize entry 1 must be a tuple",
        ),
        (
            _ws_invalid_args_kernel,
            {"out": "*i32"},
            "warp_specialize entry 1 args must be a tuple",
        ),
    ],
)
def test_tle_warp_specialize_frontend_rejects_invalid_static_configuration(kernel, signature, diagnostic):
    with pytest.raises(CompilationError, match=diagnostic):
        _compile_ws_ir(kernel, signature, constexprs={})


def test_tle_warp_specialize_rejects_default_only_configuration(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_ws_ir(_ws_no_workers_kernel, {"out": "*i32"}, constexprs={})
    assert "MUSA TLE static warp_specialize requires at least one worker partition" in capfd.readouterr().err


def test_tle_warp_specialize_dialect_rejects_non_power_of_two_warps(capfd):
    with pytest.raises(RuntimeError, match="error encountered during parsing"):
        _compile_ws_ir(_ws_non_power_of_two_warps_kernel, {"out": "*i32"}, constexprs={})
    diagnostic = "'ttg.warp_specialize' op partition #0 number of warps (3) must be a power of 2"
    assert diagnostic in capfd.readouterr().err


_WS_INVALID_DIALECT_FIXTURES = [
    (
        """
module attributes {"ttg.num-warps" = 16 : i32} {
  tt.func @bad_partition_count() {
    "ttg.warp_specialize"() ({
      "ttg.warp_yield"() : () -> ()
    }, {
      "ttg.warp_specialize.partitions"() : () -> ()
    }) {partitionNumWarps = array<i32: 4>} : () -> ()
    tt.return
  }
}
""",
        "'ttg.warp_specialize' op has 0 partitions but `partitionNumWarps` has 1 elements",
    ),
    (
        """
module attributes {"ttg.num-warps" = 16 : i32} {
  tt.func @bad_capture_count(%arg0: i32) {
    ttg.warp_specialize(%arg0)
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : (i32) -> ()
    tt.return
  }
}
""",
        "'ttg.warp_specialize.partitions' op partition region #0 has 0 arguments but expected 1",
    ),
    (
        """
module attributes {"ttg.num-warps" = 16 : i32} {
  tt.func @bad_capture_type(%arg0: i32) {
    ttg.warp_specialize(%arg0)
    default {
      ttg.warp_yield
    }
    partition0(%arg1: i64) num_warps(4) {
      ttg.warp_return
    } : (i32) -> ()
    tt.return
  }
}
""",
        "'ttg.warp_specialize.partitions' op partition region #0 argument #0 has type 'i64' "
        "but corresponding capture has type 'i32'",
    ),
]


@pytest.mark.parametrize("fixture,diagnostic", _WS_INVALID_DIALECT_FIXTURES)
def test_tle_warp_specialize_dialect_rejects_invalid_structure(tmp_path, capfd, fixture, diagnostic):
    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    fixture_path = tmp_path / "invalid_ws_capture.mlir"
    fixture_path.write_text(fixture)

    with pytest.raises(RuntimeError):
        ir.parse_mlir_module(str(fixture_path), context)
    assert diagnostic in capfd.readouterr().err


@triton.jit
def _ws_cta_barrier_idle_role():
    pass


@triton.jit
def _ws_implicit_tme_role(desc):
    smem = tle.gpu.alloc((16, 64), dtype=tl.float16, nv_mma_shared_layout=False)
    tle.gpu.copy(desc, smem, (16, 64), (0, 0))


@triton.jit
def _ws_cta_barrier_probe(desc, MODE: tl.constexpr):
    if MODE == 0:
        tle.gpu.warp_specialize([(_ws_implicit_tme_role, (desc, )), (_ws_cta_barrier_idle_role, ())], [4], [24])
    elif MODE == 1:
        tle.gpu.warp_specialize([(_ws_cta_barrier_idle_role, ()), (_ws_implicit_tme_role, (desc, ))], [4], [24])
    else:
        _ws_implicit_tme_role(desc)
        tle.gpu.warp_specialize([(_ws_cta_barrier_idle_role, ()), (_ws_cta_barrier_idle_role, ())], [4], [24])


@pytest.mark.parametrize("mode", [0, 1])
def test_unknown_cta_barrier_in_partition_rejected(mode, capfd):
    # Compile only: an unmatched CTA barrier must never be launched.
    with pytest.raises(RuntimeError):
        compile_musa(_ws_cta_barrier_probe, {"desc": "tensordesc<fp16[16, 64]>"}, {"MODE": mode})
    assert "CTA barrier inside MUSA TLE static warp_specialize partition" in capfd.readouterr().err


def test_cta_barrier_before_dispatch_preserved():
    compiled = compile_musa(_ws_cta_barrier_probe, {"desc": "tensordesc<fp16[16, 64]>"}, {"MODE": 2})
    assert "call void @llvm.musa.barrier0()" in compiled.asm["llir"]


# Partition synchronization must retain local-memory completion semantics.
# Every partition exchanges data across its own warps at the same time; with
# a named arrival alone, PH1 lets the next store overtake pending shared loads.


@triton.jit
def _ws_lma_exchange(x, out, shared, SLOT: tl.constexpr, WARPS: tl.constexpr, ITERS: tl.constexpr):
    layout: tl.constexpr = tle.gpu.BlockEncoding([4], [32], [WARPS], [0])
    index = tle.gpu.set_layout(tl.arange(0, 2048), layout)
    base = (tl.program_id(0) * 3 + SLOT) * 2048
    values = tl.load(x + base + index)
    total = tl.full((2048, ), 0, tl.int32)
    for step in range(ITERS):
        tl.store(tle.gpu.set_layout(tle.gpu.local_ptr(shared), layout), values + step)
        # Exchange across warps; the compiler must synchronize shared stores.
        value = tl.load(tle.gpu.local_ptr(shared, (index ^ 128, )))
        total += value
    tl.store(out + base + index, total)


@triton.jit
def _ws_lma_kernel(x, out, ITERS: tl.constexpr):
    s0 = tle.gpu.alloc((2048, ), tl.int32, nv_mma_shared_layout=False)
    s1 = tle.gpu.alloc((2048, ), tl.int32, nv_mma_shared_layout=False)
    s2 = tle.gpu.alloc((2048, ), tl.int32, nv_mma_shared_layout=False)
    tle.gpu.warp_specialize([
        (_ws_lma_exchange, (x, out, s0, 0, 8, ITERS)),
        (_ws_lma_exchange, (x, out, s1, 1, 8, ITERS)),
        (_ws_lma_exchange, (x, out, s2, 2, 4, ITERS)),
    ], worker_num_warps=[8, 4], worker_num_regs=[64, 64])


def test_ws_shared_completion_compile():
    source = ASTSource(_ws_lma_kernel, {'x': '*i32', 'out': '*i32', 'ITERS': 'constexpr'}, constexprs={'ITERS': 17})
    compiled = triton.compile(source, target=musa_target(), options={'num_warps': 8})
    llir = compiled.asm['llir']
    arrivals = re.findall(r'^.*\bcall\b.*@llvm\.musa\.async\.arrive(?:\.none\.phaseid)?\(.*$', llir, re.MULTILINE)
    assert arrivals, 'test must exercise compiler-generated partition barriers'
    fenced = re.findall(r'call void @llvm\.musa\.lma\.wait\(\)', llir)
    assert len(fenced) == len(arrivals), 'partition barrier lost local-memory completion'
    assert not re.search(r'call i32 @llvm\.musa\.async\.arrive\(', llir), 'phase must persist explicitly'
    assert 'alloca ' not in llir, 'phase state must be promoted to registers'


@pytest.mark.skipif(not torch.musa.is_available(), reason='MUSA device required')
def test_ws_shared_completion_runtime():
    torch.manual_seed(1041)
    cpu = torch.randint(-100, 100, (3, 3, 2048), dtype=torch.int32)
    x = cpu.to('musa')
    out = torch.empty_like(x)
    iters = 257
    expected = cpu[:, :, torch.arange(2048) ^ 128] * iters + iters * (iters - 1) // 2
    for _ in range(5):
        out.fill_(-1)
        _ws_lma_kernel[(3, )](x, out, iters, num_warps=8)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@triton.jit
def _ws_barrier_idle():
    return


@triton.jit
def _ws_barrier_sequence(out, trips):
    value = tl.load(out + tl.arange(0, 128))
    for iteration in range(trips):
        result = 0
        for inner in range(iteration % 3 + 1):
            if (iteration + inner) % 2:
                result += tl.sum(value + inner, 0)
            else:
                result += tl.max(value + inner, 0)
        tl.store(out + 128 + iteration, result)


@triton.jit
def _ws_barrier_sequence_kernel(out, trips):
    tle.gpu.warp_specialize([(_ws_barrier_sequence, (out, trips)), (_ws_barrier_idle, ())], worker_num_warps=[1],
                            worker_num_regs=[24])


@pytest.mark.parametrize("reserved_ids, target", [
    (61, "musa:31"),
    (61, "musa:ph1"),
    (62, "musa:22"),
])
def test_ws_barrier_reservation_boundary(reserved_ids, target):
    _compile_ws_ir(_ws_barrier_sequence_kernel, signature={"out": "*i32", "trips": "i32"}, constexprs={},
                   consumer_warps=4, include_late=True, reserved_ids=reserved_ids, lowering_target=target)


@pytest.mark.parametrize("target", ["musa:31", "musa:ph1"])
def test_ws_paired_barrier_exhaustion(target, capfd):
    with pytest.raises(RuntimeError):
        _compile_ws_ir(_ws_barrier_sequence_kernel, signature={"out": "*i32", "trips": "i32"}, constexprs={},
                       consumer_warps=4, include_late=True, reserved_ids=62, lowering_target=target)
    diagnostic = capfd.readouterr().err
    assert "requires 2 barrier slots (2 per synchronization group)" in diagnostic
    assert "available hardware barrier slots: 1" in diagnostic


@triton.jit
def _ws_late_wait_part(out):
    value = tl.load(out + tl.arange(0, 128))
    tl.store(out + 128, tl.sum(value, 0))


@triton.jit
def _ws_late_wait_kernel(out):
    tle.gpu.warp_specialize([(_ws_late_wait_part, (out, )), (_ws_barrier_idle, ())], worker_num_warps=[1],
                            worker_num_regs=[24])


def _run_late_wait_probe():
    from triton import knobs

    def hook(*args):
        if not args:
            return ("ws-late-wait-regression", "v1")
        _, stages, _, _, _ = args
        original = stages["llir"]

        def instrument(source, metadata):
            llir = original(source, metadata)
            sites = 0

            def replace(match):
                nonlocal sites
                sites += 1
                if sites == 1:
                    return (match[1] + "  call void @__ws_delay(ptr addrspace(1) %0)\n" + match[2] +
                            "\n  call void @__ws_delay(ptr addrspace(1) %0)")
                return (match[0] + f"\n  call void @__ws_check(i32 {match[3]}, i32 {match[4]}, "
                        "ptr addrspace(1) %0)")

            llir = re.sub(
                r"((?:tail )?call void @llvm\.musa\.async\.arrive\.none\.phaseid\(i32 [^)]+\)[^\n]*\n)"
                r"\s*((?:tail )?call void @llvm\.musa\.async\.wait\(i32 ([^,]+), i32 ([^)]+)\)[^\n]*)", replace, llir)
            assert sites == 2
            # Only delay and observe the production lowering. Do not rewrite
            # its IDs, phases, arrivals, waits, or initialization.
            llir += """
declare <2 x i32> @llvm.musa.async.read.bar(i32) convergent
define internal void @__ws_delay(ptr addrspace(1) %out) alwaysinline convergent {
entry:
  %tid = call i32 @llvm.musa.read.ptx.sreg.tid.x()
  %warp = lshr i32 %tid, 5
  %slow = icmp eq i32 %warp, 1
  %index = add i32 %tid, 256
  %scratch = getelementptr i32, ptr addrspace(1) %out, i32 %index
  br i1 %slow, label %loop, label %done
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  store volatile i32 %i, ptr addrspace(1) %scratch
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, 256
  br i1 %more, label %loop, label %done
done:
  ret void
}
define internal void @__ws_check(i32 %bar, i32 %old, ptr addrspace(1) %out) alwaysinline convergent {
entry:
  %state = call <2 x i32> @llvm.musa.async.read.bar(i32 %bar)
  %low = extractelement <2 x i32> %state, i32 0
  %phase = and i32 %low, 1
  %incomplete = icmp eq i32 %phase, %old
  %flag = zext i1 %incomplete to i32
  %tid = call i32 @llvm.musa.read.ptx.sreg.tid.x()
  %ptr = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store i32 %flag, ptr addrspace(1) %ptr
  ret void
}
"""
            return llir

        stages["llir"] = instrument

    out = torch.zeros((512, ), dtype=torch.int32, device="musa")
    with knobs.runtime.scope():
        knobs.runtime.add_stages_inspection_hook = hook
        for _ in range(20):
            out.zero_()
            _ws_late_wait_kernel[(1, )](out, num_warps=4, num_stages=1, llc_options="-enable-misched=false")
            assert not out[:128].cpu().any(), "WAIT returned before its phase completed"
    # The production state must follow dynamic branches and nested loop edges.
    values = torch.arange(128, dtype=torch.int32)
    out[:128] = values.to("musa")
    _ws_barrier_sequence_kernel[(1, )](out, 5, num_warps=4, num_stages=1)
    expected = []
    for iteration in range(5):
        result = 0
        for inner in range(iteration % 3 + 1):
            current = values + inner
            result += int(current.sum() if (iteration + inner) % 2 else current.max())
        expected.append(result)
    assert out[128:133].cpu().tolist() == expected


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_ws_late_wait_and_nested_loops_runtime(tmp_path):
    import os

    directory = str(Path(__file__).resolve().parent)
    command = (f"import sys; sys.path.insert(0, {directory!r}); "
               "from test_tle_warp_specialize import _run_late_wait_probe; _run_late_wait_probe()")
    environment = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / "cache"))
    result = subprocess.run([sys.executable, "-c", command], env=environment, capture_output=True, text=True,
                            timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
