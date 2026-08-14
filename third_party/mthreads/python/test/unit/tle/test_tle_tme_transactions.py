"""Compile-only coverage for mthreads TLE TME completion transactions."""

import re

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C import libtriton
from triton._C.libtriton import ir, passes
from triton.backends.compiler import Language
from triton.compiler import ASTSource

from test_tle_utils import mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _completion_transaction_kernel(
    desc,
    dynamic_k,
    STAGES: tl.constexpr,
    SLOT: tl.constexpr,
):
    smem = tle.gpu.alloc(
        (STAGES, 256, 64),
        dtype=tl.float16,
        nv_mma_shared_layout=False,
    )
    full = tle.gpu.alloc_barriers(STAGES, expect_bytes=32768)
    tle.gpu.copy(
        desc,
        smem.slot(SLOT),
        (256, 64),
        (0, dynamic_k),
        barrier=full[SLOT],
    )


def _compile_transaction_module(stages=1, slot=0):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 4, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(
        fn=_completion_transaction_kernel,
        signature={
            "desc": "tensordesc<fp16[256, 64]>",
            "dynamic_k": "i32",
            "STAGES": "constexpr",
            "SLOT": "constexpr",
        },
        constexprs={"STAGES": stages, "SLOT": slot},
    )
    module = src.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    compiler_stages = {}
    backend.add_stages(compiler_stages, options, Language.TRITON)
    metadata = {}
    module = compiler_stages["ttir"](module, metadata)
    module = compiler_stages["ttgir"](module, metadata)
    return backend, context, module


def _lower_to_llvm_dialect(module, context):
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
    pm.run(module, "lower_tle_tme_transaction_to_llvm")
    return module.str_nodebug()


@pytest.mark.parametrize("stages,slot", [(1, 0), (2, 0), (2, 1)])
def test_mthreads_tle_completion_transaction_is_single_and_idempotent(stages, slot):
    _, context, module = _compile_transaction_module(stages, slot)
    ttgir = module.str_nodebug()

    assert ttgir.count("ttmg.barrier_add_trans") == 1, ttgir
    assert ttgir.count("ttmg.async_tme_copy_global_to_local") == 1, ttgir
    assert ttgir.count("ttmg.arrive_barrier_noret") == 1, ttgir
    assert ttgir.count("musa.tme.explicit_completion") == 3, ttgir
    assert ttgir.count("musa.tme.issue_thread = 0 : i32") == 3, ttgir
    assert "musa_tle.expect_bytes" not in ttgir, ttgir
    assert "ttmg.wait_barrier" not in ttgir, ttgir

    add_pos = ttgir.index("ttmg.barrier_add_trans")
    copy_pos = ttgir.index("ttmg.async_tme_copy_global_to_local")
    arrive_pos = ttgir.index("ttmg.arrive_barrier_noret")
    assert add_pos < copy_pos < arrive_pos, ttgir
    add_barrier, byte_value = re.search(r"ttmg\.barrier_add_trans\s+(%[-\w.]+),\s*(%[-\w.]+)", ttgir).groups()
    copy_barrier = re.search(r"ttmg\.async_tme_copy_global_to_local\s+[^\n]*\],\s*(%[-\w.]+)", ttgir).group(1)
    arrive_barrier = re.search(r"ttmg\.arrive_barrier_noret\s+(%[-\w.]+)", ttgir).group(1)
    constants = {
        name: int(value)
        for name, value in re.findall(r"(%[-\w.]+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*i32", ttgir)
    }
    assert add_barrier == copy_barrier == arrive_barrier, ttgir
    assert constants[byte_value] == 32768, ttgir

    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_tle_lower_tme_transactions(pm)
    pm.run(module, "rerun_tle_tme_transaction_lowering")
    rerun = module.str_nodebug()
    assert rerun.count("ttmg.barrier_add_trans") == 1, rerun
    assert rerun.count("ttmg.arrive_barrier_noret") == 1, rerun


def test_mthreads_tle_explicit_completion_uses_raw_issue_thread_without_barrier0(tmp_path):
    backend, _, module = _compile_transaction_module()
    fixture = module.str_nodebug().replace(
        "musa.tme.issue_thread = 0 : i32",
        "musa.tme.issue_thread = 512 : i32",
    )
    fixture_path = tmp_path / "raw_issue_thread_512.ttgir"
    fixture_path.write_text(fixture)

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    reparsed = ir.parse_mlir_module(str(fixture_path), context)
    llir = _lower_to_llvm_dialect(reparsed, context)

    assert "llvm.musa.barrier0" not in llir, llir
    assert llir.count('llvm.call_intrinsic "llvm.musa.async.add.trans"') == 1, llir
    assert llir.count('llvm.call_intrinsic "llvm.musa.tme.ld.tile.2d"') == 1, llir
    assert llir.count('llvm.call_intrinsic "llvm.musa.async.arrive.none.phaseid"') == 1, llir
    issue_constant = re.search(r"(%\d+) = llvm\.mlir\.constant\(512 : i32\) : i32", llir).group(1)
    assert len(re.findall(rf'llvm\.icmp "eq" %\d+, {re.escape(issue_constant)} : i32', llir)) == 3, llir
    assert (llir.index("llvm.musa.async.add.trans") < llir.index("llvm.musa.tme.ld.tile.2d") <
            llir.index("llvm.musa.async.arrive.none.phaseid")), llir


def test_mthreads_tle_completion_copy_rejects_consumer_partition(tmp_path, capfd):
    fixture = """#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32,
    ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @copy_in_consumer(%desc: !tt.tensordesc<tensor<256x64xf16, #shared>>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %smem = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    ttg.warp_specialize() attributes {requestedRegisters = array<i32: 24>}
    default {
      ttmg.async_tme_copy_global_to_local %desc[%c0, %c0], %c1, %smem, %true {
        blockShape = array<i32: 256, 64>, cachePolicy = 0 : i32,
        innerPersistence = 2 : i32, musa_tle.expect_bytes = 32768 : i32,
        outerPersistence = 2 : i32, prefetchSize = 0 : i32,
        swizzleGranularity = 0 : i32, swizzleLine = 1 : i32,
        swizzleStride = 3 : i32
      } : !tt.tensordesc<tensor<256x64xf16, #shared>>, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
"""
    fixture_path = tmp_path / "completion_copy_in_consumer.ttgir"
    fixture_path.write_text(fixture)

    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(fixture_path), context)
    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_tle_lower_tme_transactions(pm)
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        pm.run(module, "reject_completion_copy_in_consumer")
    assert ("mthreads TLE completion TME copy must be in the producer partition" in capfd.readouterr().err)
