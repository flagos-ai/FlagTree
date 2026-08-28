"""Compile and runtime coverage for the MUSA TLE warp-specialize container."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C import libtriton
from triton._C.libtriton import ir
from triton.backends.compiler import Language
from triton.compiler import ASTSource
from triton.compiler.errors import CompilationError

from test_tle_utils import mthreads_backend, require_mthreads_libtriton

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
        pm = ir.pass_manager(context)
        libtriton.passes.convert.add_scf_to_cf(pm)
        libtriton.passes.convert.add_index_to_llvmir(pm)
        libtriton.mthreads.passes.ttgpuir.add_allocate_shared_memory(pm, 31)
        libtriton.mthreads.passes.ttgpuir.add_mtgpu_to_llvm(pm, 31)
        libtriton.mthreads.passes.ttgpuir.add_to_llvmir(pm, 31)
        libtriton.mthreads.passes.ttgpuir.add_tle_lower_warp_specialize(pm)
        pm.run(module, "lower_static_ws_to_llvm_cfg")
        return ttir, ttgir, module.str_nodebug()
    return ttir, ttgir


def _split_top_level(text):
    values = []
    start = 0
    depth = 0
    pairs = {"<": ">", "[": "]", "{": "}", "(": ")"}
    closing = set(pairs.values())
    for index, char in enumerate(text):
        if char in pairs:
            depth += 1
        elif char in closing:
            depth -= 1
        elif char == "," and depth == 0:
            values.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        values.append(tail)
    return values


def _assert_ws_container(ir_text, producer_warps=4):
    assert ir_text.count("ttg.warp_specialize") == 1, ir_text
    assert ir_text.count("ttg.warp_yield") == 1, ir_text
    assert ir_text.count("ttg.warp_return") == 1, ir_text
    assert "partition0" in ir_text, ir_text
    assert "partition1" not in ir_text, ir_text
    assert f"num_warps({producer_warps})" in ir_text, ir_text
    assert re.search(r"requestedRegisters\s*=\s*array<i32:\s*24>", ir_text), ir_text
    assert "actualRegisters" not in ir_text, ir_text
    assert "ttg.maxnreg" not in ir_text, ir_text
    assert "tle.wgmma_pipeline_mode" not in ir_text, ir_text
    assert "tt.call" not in ir_text, ir_text

    ws_match = re.search(
        r"ttg\.warp_specialize\((?P<captures>[^)]*)\).*?"
        rf"partition0\((?P<args>[^)]*)\)\s*num_warps\({producer_warps}\).*?"
        r":\s*\((?P<types>[^)]*)\)\s*->\s*\(\)",
        ir_text,
        re.DOTALL,
    )
    assert ws_match, ir_text
    captures = _split_top_level(ws_match.group("captures"))
    args = _split_top_level(ws_match.group("args"))
    types = _split_top_level(ws_match.group("types"))

    # Five Python worker arguments flatten to three unique SSA captures:
    # pointer, dynamic i32, and shared memdesc.  The repeated pointer is
    # remapped to the first block argument and constexpr BIAS consumes no SSA.
    assert len(captures) == 3, ir_text
    assert len(args) == 3, ir_text
    assert len(types) == 3, ir_text
    assert sum("ptr" in ty for ty in types) == 1, ir_text
    assert sum(ty == "i32" for ty in types) == 1, ir_text
    assert sum("memdesc" in ty for ty in types) == 1, ir_text
    assert any(re.search(r"\bi32\b", arg) for arg in args), ir_text
    assert any("memdesc" in arg for arg in args), ir_text

    assert re.search(r"(?:arith\.)?constant\s+7\s*:\s*i32", ir_text), ir_text
    assert re.search(r"ttg\.warp_yield\s*(?:\n|\})", ir_text), ir_text
    assert re.search(r"ttg\.warp_return\s*(?:\n|\})", ir_text), ir_text
    assert re.search(r"\)\s*->\s*\(\)\s*\n\s*tt\.return", ir_text), ir_text


def _assert_prepared_ws_ttgir(ir_text, consumer_warps=16, producer_warps=4):
    total_warps = consumer_warps + producer_warps
    assert f'"ttg.total-num-warps" = {total_warps} : i32' in ir_text, ir_text
    assert f"warpGroupStartIds = array<i32: {consumer_warps}>" in ir_text, ir_text
    assert ir_text.count("ttg.warp_specialize") == 1, ir_text
    assert ir_text.count("ttg.warp_yield") == 1, ir_text
    assert ir_text.count("ttg.warp_return") == 1, ir_text
    assert "musa_tle.static_warp_specialize" in ir_text, ir_text
    assert "gpu.thread_id" not in ir_text, ir_text
    assert "musa_tle.static_ws.split" not in ir_text, ir_text
    assert "musa_tle.static_ws.role" not in ir_text, ir_text
    assert "musa_tle.static_ws.num_warps" not in ir_text, ir_text
    assert "musa_tle.static_ws.thread_offset" not in ir_text, ir_text
    assert "musa_tle.static_ws.split_candidate" not in ir_text, ir_text


def _assert_static_ws_late_ir(ir_text, consumer_warps=16, producer_warps=4):
    producer_begin = consumer_warps * 32
    assert "ttg.warp_specialize" not in ir_text, ir_text
    assert "ttg.warp_yield" not in ir_text, ir_text
    assert "ttg.warp_return" not in ir_text, ir_text
    assert "musa_tle.static_warp_specialize" not in ir_text, ir_text
    dispatch = re.search(
        rf'(?P<tid>%[-\w.]+) = llvm\.call_intrinsic '
        rf'"llvm\.musa\.read\.ptx\.sreg\.tid\.x"\(\).*?'
        rf'arith\.constant {producer_begin} : i32.*?'
        rf'arith\.cmpi uge, (?P=tid),',
        ir_text,
        re.DOTALL,
    )
    assert dispatch, ir_text
    assert re.search(rf"arith\.cmpi ult, {re.escape(dispatch.group('tid'))},", ir_text), ir_text
    assert ir_text.count("arith.cmpi uge") == 1, ir_text
    assert ir_text.count("arith.cmpi ult") == 1, ir_text
    assert ir_text.count("cf.cond_br") == 2, ir_text
    assert re.search(rf"arith\.constant {producer_begin} : i32", ir_text), ir_text
    assert ir_text.index("arith.cmpi uge") < ir_text.index("arith.cmpi ult"), ir_text
    assert "llvm.switch" not in ir_text, ir_text
    assert "musa_tle.static_ws." not in ir_text, ir_text
    assert "builtin.unrealized_conversion_cast" not in ir_text, ir_text


def _assert_multi_worker_ws_ttgir(ir_text):
    assert ir_text.count("ttg.warp_specialize") == 1, ir_text
    assert ir_text.count("ttg.warp_return") == 3, ir_text
    assert '"ttg.total-num-warps" = 11 : i32' in ir_text, ir_text
    assert "partition0" in ir_text and "num_warps(1)" in ir_text, ir_text
    assert "partition1" in ir_text and "num_warps(2)" in ir_text, ir_text
    assert "partition2" in ir_text and "num_warps(4)" in ir_text, ir_text
    assert "warpGroupStartIds = array<i32: 4, 5, 7>" in ir_text, ir_text
    assert "requestedRegisters = array<i32: 24, 32, 40>" in ir_text, ir_text
    assert ir_text.count("ttg.local_alloc") == 4, ir_text
    for group in range(4):
        assert f"musa_tle.barrier_group = {group} : i64" in ir_text, ir_text


def _assert_multi_worker_late_ir(ir_text, expected_shared_offsets=None):
    assert "ttg.warp_specialize" not in ir_text, ir_text
    assert "ttg.warp_return" not in ir_text, ir_text
    assert "ttg.warp_yield" not in ir_text, ir_text
    assert "musa_tle.static_warp_specialize" not in ir_text, ir_text
    assert "cf.switch" not in ir_text, ir_text
    assert ir_text.count("icmp ult") >= 3, ir_text
    assert ir_text.count("br i1") >= 4, ir_text
    for boundary in (128, 160, 224):
        assert re.search(rf"icmp ult i32 %[-\w.]+, {boundary},", ir_text), ir_text
        assert re.search(rf"icmp eq i32 %[-\w.]+, {boundary}", ir_text), ir_text
    assert '"maxntidx", i32 352' in ir_text, ir_text

    if expected_shared_offsets is not None:
        # Each partition-local allocation must have a distinct backing
        # address; this catches an allocator treating concurrently live
        # worker regions as mutually exclusive. The default allocation is the
        # base address (offset zero), while the three workers use distinct
        # aligned offsets.
        shared_offsets = {0}
        shared_offsets.update(
            int(offset) for offset in re.findall(
                r"getelementptr \(i8, ptr addrspace\(3\) @global_smem, i32 (\d+)\)",
                ir_text,
            ))
        assert shared_offsets == expected_shared_offsets, ir_text


def test_tle_warp_specialize_static_multi_worker_runtime():
    out = torch.full((4, ), -1, dtype=torch.int32, device="musa")
    compiled = _ws_multi_worker_kernel.warmup(
        out,
        grid=(1, ),
        num_warps=4,
        num_stages=1,
    )
    assert compiled.metadata.num_warps == 11
    _assert_multi_worker_ws_ttgir(compiled.asm["ttgir"])
    _assert_multi_worker_late_ir(compiled.asm["llir"], expected_shared_offsets={0, 16, 32, 48})
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
    ttgir = compiled.asm["ttgir"]
    assert ttgir.count("ttmg.init_arrival") == 3, ttgir
    assert "ttmg.init_arrival %barrier0_2, %barrier0_0" in ttgir, ttgir
    assert "ttmg.init_arrival %barrier1_5, %barrier1_3" in ttgir, ttgir
    assert "ttmg.init_arrival %barrier2_8, %barrier2_6" in ttgir, ttgir
    assert 'musa.max_bar_id = 3 : i32' in ttgir, ttgir
    _assert_multi_worker_late_ir(compiled.asm["llir"])
    for _ in range(2):
        out.fill_(-1)
        _ws_multi_worker_barrier_kernel[(1, )](out, num_warps=4, num_stages=1)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu(), torch.tensor([11, 22, 33, 44], dtype=torch.int32))


def test_tle_warp_specialize_musa_container_contract():
    ttir, ttgir, late_ir = _compile_ws_ir(include_late=True)
    _assert_ws_container(ttir)
    assert "musa_tle.static_warp_specialize" in ttir, ttir
    _assert_ws_container(ttgir)
    _assert_prepared_ws_ttgir(ttgir)
    _assert_static_ws_late_ir(late_ir)


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
    _assert_prepared_ws_ttgir(compiled.asm["ttgir"], consumer_warps, producer_warps)
    assert "ttg.warp_specialize" not in compiled.asm["llir"]
    assert f'!"maxntidx", i32 {(consumer_warps + producer_warps) * 32}' in compiled.asm["llir"]

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


def test_tle_warp_specialize_builder_bindings_are_available():
    builder = libtriton.ir.builder
    for method in (
            "create_warp_specialize",
            "create_warp_specialize_partitions",
            "create_warp_yield",
            "create_warp_return",
    ):
        assert hasattr(builder, method)
    assert hasattr(libtriton.mthreads.ir, "WarpSpecializeOp")


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


@triton.jit
def _ws_consumer(
    a_smem,
    b_smem,
    full_a,
    full_b,
    empty_a,
    empty_b,
    out,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    BIAS: tl.constexpr,
):
    PERIOD: tl.constexpr = 2 * STAGES
    FULL_PERIODS: tl.constexpr = K_TILES // PERIOD
    TAIL_TILES: tl.constexpr = K_TILES % PERIOD

    for period_idx in tl.range(0, FULL_PERIODS, num_stages=1):
        for u in tl.static_range(0, PERIOD):
            k_iter = period_idx * PERIOD + u
            tle.gpu.barrier_wait(full_a[u % STAGES], phaseIdx=u // STAGES)
            tle.gpu.barrier_wait(full_b[u % STAGES], phaseIdx=u // STAGES)

            a = tl.load(tle.gpu.local_ptr(a_smem.slot(u % STAGES), (0, 0)))
            b = tl.load(tle.gpu.local_ptr(b_smem.slot(u % STAGES), (0, 0)))
            tl.store(out + 2 * k_iter, a + BIAS)
            tl.store(out + 2 * k_iter + 1, b + BIAS)

            tle.gpu.barrier_arrive(empty_a[u % STAGES], phaseIdx=u // STAGES)
            tle.gpu.barrier_arrive(empty_b[u % STAGES], phaseIdx=u // STAGES)

    for u in tl.static_range(0, TAIL_TILES):
        k_iter = FULL_PERIODS * PERIOD + u
        tle.gpu.barrier_wait(full_a[u % STAGES], phaseIdx=u // STAGES)
        tle.gpu.barrier_wait(full_b[u % STAGES], phaseIdx=u // STAGES)

        a = tl.load(tle.gpu.local_ptr(a_smem.slot(u % STAGES), (0, 0)))
        b = tl.load(tle.gpu.local_ptr(b_smem.slot(u % STAGES), (0, 0)))
        tl.store(out + 2 * k_iter, a + BIAS)
        tl.store(out + 2 * k_iter + 1, b + BIAS)

        tle.gpu.barrier_arrive(empty_a[u % STAGES], phaseIdx=u // STAGES)
        tle.gpu.barrier_arrive(empty_b[u % STAGES], phaseIdx=u // STAGES)


@triton.jit
def _ws_producer(
    a_desc,
    b_desc,
    a_smem,
    b_smem,
    full_a,
    full_b,
    empty_a,
    empty_b,
    out,
    duplicate_out,
    dynamic_k,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    BIAS: tl.constexpr,
):
    # out, duplicate_out and BIAS intentionally exercise capture reconstruction.
    tl.store(duplicate_out, tl.load(out))
    PERIOD: tl.constexpr = 2 * STAGES
    FULL_PERIODS: tl.constexpr = K_TILES // PERIOD
    TAIL_TILES: tl.constexpr = K_TILES % PERIOD

    for period_idx in tl.range(0, FULL_PERIODS, num_stages=1):
        period_base = period_idx * PERIOD
        for u in tl.static_range(0, PERIOD):
            k_offset = dynamic_k + period_base + u
            tle.gpu.barrier_wait(empty_a[u % STAGES], phaseIdx=u // STAGES)
            tle.gpu.barrier_wait(empty_b[u % STAGES], phaseIdx=u // STAGES)
            tle.gpu.copy(
                a_desc,
                a_smem.slot(u % STAGES),
                (256, 64),
                (0, k_offset),
                barrier=full_a[u % STAGES],
            )
            tle.gpu.copy(
                b_desc,
                b_smem.slot(u % STAGES),
                (64, 256),
                (k_offset, 0),
                barrier=full_b[u % STAGES],
            )

    for u in tl.static_range(0, TAIL_TILES):
        k_offset = dynamic_k + FULL_PERIODS * PERIOD + u
        tle.gpu.barrier_wait(empty_a[u % STAGES], phaseIdx=u // STAGES)
        tle.gpu.barrier_wait(empty_b[u % STAGES], phaseIdx=u // STAGES)
        tle.gpu.copy(
            a_desc,
            a_smem.slot(u % STAGES),
            (256, 64),
            (0, k_offset),
            barrier=full_a[u % STAGES],
        )
        tle.gpu.copy(
            b_desc,
            b_smem.slot(u % STAGES),
            (64, 256),
            (k_offset, 0),
            barrier=full_b[u % STAGES],
        )


@triton.jit
def _ws_dot_consumer(
    a_smem,
    b_smem,
    full_a,
    full_b,
    empty_a,
    empty_b,
    out,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
):
    PERIOD: tl.constexpr = 2 * STAGES
    FULL_PERIODS: tl.constexpr = K_TILES // PERIOD
    TAIL_TILES: tl.constexpr = K_TILES % PERIOD

    acc = tl.zeros((256, 256), dtype=tl.float32)
    for _period_idx in tl.range(0, FULL_PERIODS, num_stages=1):
        for u in tl.static_range(0, PERIOD):
            tle.gpu.barrier_wait(full_a[u % STAGES], phaseIdx=u // STAGES)
            tle.gpu.barrier_wait(full_b[u % STAGES], phaseIdx=u // STAGES)

            a = tl.load(tle.gpu.local_ptr(a_smem.slot(u % STAGES)))
            b = tl.load(tle.gpu.local_ptr(b_smem.slot(u % STAGES)))
            acc = tl.dot(a, b, acc=acc)

            tle.gpu.barrier_arrive(empty_a[u % STAGES], phaseIdx=u // STAGES)
            tle.gpu.barrier_arrive(empty_b[u % STAGES], phaseIdx=u // STAGES)

    for u in tl.static_range(0, TAIL_TILES):
        tle.gpu.barrier_wait(full_a[u % STAGES], phaseIdx=u // STAGES)
        tle.gpu.barrier_wait(full_b[u % STAGES], phaseIdx=u // STAGES)

        a = tl.load(tle.gpu.local_ptr(a_smem.slot(u % STAGES)))
        b = tl.load(tle.gpu.local_ptr(b_smem.slot(u % STAGES)))
        acc = tl.dot(a, b, acc=acc)

        tle.gpu.barrier_arrive(empty_a[u % STAGES], phaseIdx=u // STAGES)
        tle.gpu.barrier_arrive(empty_b[u % STAGES], phaseIdx=u // STAGES)

    offsets = tl.arange(0, 256)[:, None] * 256 + tl.arange(0, 256)[None, :]
    tl.store(out + offsets, acc.to(tl.float16))


@triton.jit
def _ws_integration_kernel(
    a_desc,
    b_desc,
    out,
    dynamic_k,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    BIAS: tl.constexpr,
):
    a_smem = tle.gpu.alloc(
        (STAGES, 256, 64),
        dtype=tl.float16,
        nv_mma_shared_layout=False,
    )
    b_smem = tle.gpu.alloc(
        (STAGES, 64, 256),
        dtype=tl.float16,
        nv_mma_shared_layout=False,
    )
    full_a = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=1,
        init=tle.gpu.PENDING,
        expect_bytes=32768,
    )
    full_b = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=1,
        init=tle.gpu.PENDING,
        expect_bytes=32768,
    )
    empty_a = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=16,
        init=tle.gpu.READY,
    )
    empty_b = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=16,
        init=tle.gpu.READY,
    )

    tle.gpu.warp_specialize(
        [
            (
                _ws_consumer,
                (a_smem, b_smem, full_a, full_b, empty_a, empty_b, out, K_TILES, STAGES, BIAS),
            ),
            (
                _ws_producer,
                (
                    a_desc,
                    b_desc,
                    a_smem,
                    b_smem,
                    full_a,
                    full_b,
                    empty_a,
                    empty_b,
                    out,
                    out,
                    dynamic_k,
                    K_TILES,
                    STAGES,
                    BIAS,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_dot_integration_kernel(
    a_desc,
    b_desc,
    out,
    dynamic_k,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
):
    a_smem = tle.gpu.alloc(
        (STAGES, 256, 64),
        dtype=tl.float16,
        nv_mma_shared_layout=False,
    )
    b_smem = tle.gpu.alloc(
        (STAGES, 64, 256),
        dtype=tl.float16,
        nv_mma_shared_layout=False,
    )
    full_a = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=1,
        init=tle.gpu.PENDING,
        expect_bytes=32768,
    )
    full_b = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=1,
        init=tle.gpu.PENDING,
        expect_bytes=32768,
    )
    empty_a = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=16,
        init=tle.gpu.READY,
    )
    empty_b = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=16,
        init=tle.gpu.READY,
    )

    tle.gpu.warp_specialize(
        [
            (
                _ws_dot_consumer,
                (a_smem, b_smem, full_a, full_b, empty_a, empty_b, out, K_TILES, STAGES),
            ),
            (
                _ws_producer,
                (
                    a_desc,
                    b_desc,
                    a_smem,
                    b_smem,
                    full_a,
                    full_b,
                    empty_a,
                    empty_b,
                    out,
                    out,
                    dynamic_k,
                    K_TILES,
                    STAGES,
                    1,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


def _compile_ws_integration(stages, k_tiles):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 16, "num_stages": 1})
    assert options.num_warps == 16
    assert options.num_stages == 1
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(
        fn=_ws_integration_kernel,
        signature={
            "a_desc": "tensordesc<fp16[256, 64]>",
            "b_desc": "tensordesc<fp16[64, 256]>",
            "out": "*fp16",
            "dynamic_k": "i32",
            "K_TILES": "constexpr",
            "STAGES": "constexpr",
            "BIAS": "constexpr",
        },
        constexprs={"K_TILES": k_tiles, "STAGES": stages, "BIAS": 1},
        attrs={(0, ): [["musa.tme_tail_divisibility", 4]], (1, ): [["musa.tme_tail_divisibility", 4]]},
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
    ttir = module.str_nodebug()
    module = compiler_stages["ttgir"](module, metadata)
    ttgir = module.str_nodebug()

    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_allocate_shared_memory(pm, 31)
    pm.run(module, "allocate_ws_integration_shared_memory")
    return ttir, ttgir, module.str_nodebug()


def _compile_ws_dot_integration(stages):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 16, "num_stages": 1})
    assert options.num_warps == 16
    assert options.num_stages == 1
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(
        fn=_ws_dot_integration_kernel,
        signature={
            "a_desc": "tensordesc<fp16[256, 64]>",
            "b_desc": "tensordesc<fp16[64, 256]>",
            "out": "*fp16",
            "dynamic_k": "i32",
            "K_TILES": "constexpr",
            "STAGES": "constexpr",
        },
        constexprs={"K_TILES": 16, "STAGES": stages},
        attrs={(0, ): [["musa.tme_tail_divisibility", 4]], (1, ): [["musa.tme_tail_divisibility", 4]]},
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
    ttir = module.str_nodebug()
    module = compiler_stages["ttgir"](module, metadata)
    ttgir = module.str_nodebug()

    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_allocate_shared_memory(pm, 31)
    pm.run(module, "allocate_ws_dot_integration_shared_memory")
    allocated = module.str_nodebug()

    pm = ir.pass_manager(context)
    libtriton.passes.convert.add_scf_to_cf(pm)
    libtriton.passes.convert.add_index_to_llvmir(pm)
    libtriton.mthreads.passes.ttgpuir.add_mtgpu_to_llvm(pm, 31)
    libtriton.mthreads.passes.ttgpuir.add_to_llvmir(pm, 31)
    libtriton.mthreads.passes.ttgpuir.add_tle_lower_warp_specialize(pm)
    libtriton.passes.convert.add_scf_to_cf(pm)
    pm.run(module, "lower_ws_dot_integration_to_llvm_cfg")
    return ttir, ttgir, allocated, module.str_nodebug()


def _constants(region):
    return {
        name: int(value)
        for name, value in re.findall(
            r"(%[-\w.]+)\s*=\s*(?:arith\.)?constant"
            r"(?:\s+\{[^}\n]*\})?\s+(-?\d+)\s*:\s*i32",
            region,
        )
    }


def _ssa_definitions(region):
    definitions = {}
    for line in region.splitlines():
        match = re.match(r"\s*(%[-\w.]+)\s*=\s*(.*)", line)
        if match:
            definitions[match.group(1)] = set(re.findall(r"%[-\w.]+", match.group(2)))
    return definitions


def _depends_on(value, root, definitions, seen=None):
    if value == root:
        return True
    seen = set() if seen is None else seen
    if value in seen:
        return False
    return any(_depends_on(operand, root, definitions, seen | {value}) for operand in definitions.get(value, ()))


def _extract_ws_regions(ir_text):
    ws_match = re.search(
        r"ttg\.warp_specialize\((?P<captures>[^)]*)\)\s+"
        r"attributes\s+\{(?P<attrs>[^}]*)\}",
        ir_text,
    )
    default_match = re.search(
        r"\bdefault\s*\{(?P<body>.*?)\n\s*\}\s*\n\s*partition0\(",
        ir_text,
        re.DOTALL,
    )
    partition_match = re.search(
        r"partition0\((?P<args>.*?)\)\s*num_warps\(4\)\s*\{"
        r"(?P<body>.*?)\n\s*ttg\.warp_return",
        ir_text,
        re.DOTALL,
    )
    assert ws_match and default_match and partition_match, ir_text
    return ws_match, default_match.group("body"), partition_match


def _index_defs(region, op_name, enclosing_constants=None):
    constants = dict(enclosing_constants or {})
    constants.update(_constants(region))
    definitions = {}
    for result, source, index in re.findall(
            rf"(%[-\w.]+)\s*=\s*{re.escape(op_name)}\s+(%[-\w.]+)\[(%[-\w.]+)\]",
            region,
    ):
        assert index in constants, (op_name, index, region)
        definitions[result] = (source, constants[index])
    return definitions


def _barrier_phases(region, op_name, barrier_defs, selected_sources, enclosing_constants=None):
    constants = dict(enclosing_constants or {})
    constants.update(_constants(region))
    phases = []
    for slot, phase in re.findall(
            rf"{re.escape(op_name)}\s+(%[-\w.]+),\s*(%[-\w.]+)",
            region,
    ):
        source, _ = barrier_defs[slot]
        if source in selected_sources:
            assert phase in constants, (op_name, phase, region)
            phases.append(constants[phase])
    return phases


def _physical_barrier_defs(region, base_args):
    constants = _constants(region)
    definitions = {}
    for result, base, slot in re.findall(
            r"(%[-\w.]+)\s*=\s*arith\.addi\s+(%[-\w.]+),\s*(%[-\w.]+)"
            r"\s*(?:\{[^}\n]*\}\s*)?:\s*i32",
            region,
    ):
        if base in base_args and slot in constants:
            definitions[result] = (base, constants[slot])
    return definitions


def _physical_barrier_uses(region, op_name, enclosing_constants):
    constants = dict(enclosing_constants)
    constants.update(_constants(region))
    uses = []
    for bar_id, phase in re.findall(
            rf"{re.escape(op_name)}\s+(%[-\w.]+),\s*(%[-\w.]+)",
            region,
    ):
        uses.append((constants[bar_id], constants[phase]))
    return uses


def _emitted_static_tiles(stages, k_tiles):
    period = 2 * stages
    full_periods = k_tiles // period
    tail_tiles = k_tiles % period
    return ([*range(period)] if full_periods else []) + [*range(tail_tiles)]


def _assert_single_compiler_stage(ir_text):
    assert re.findall(r"tt\.num_stages = (\d+) : i32", ir_text) == ["1", "1"], ir_text


def _assert_common_ws_ir(ir_text, stages, k_tiles, barriers_lowered=False):
    emitted_tiles = _emitted_static_tiles(stages, k_tiles)
    _assert_single_compiler_stage(ir_text)
    if barriers_lowered:
        assert ir_text.count("ttg.warp_specialize") == 1, ir_text
        assert ir_text.count("ttg.warp_yield") == 1, ir_text
        assert ir_text.count("ttg.warp_return") == 1, ir_text
        assert "musa_tle.static_warp_specialize" in ir_text, ir_text
        assert "warpGroupStartIds = array<i32: 16>" in ir_text, ir_text
        assert '"ttg.total-num-warps" = 20 : i32' in ir_text, ir_text
        assert "gpu.thread_id" not in ir_text, ir_text
        assert "musa_tle.static_ws.split" not in ir_text, ir_text
        assert "musa_tle.static_ws.role" not in ir_text, ir_text
        assert "musa_tle.static_ws.num_warps" not in ir_text, ir_text
        assert "musa_tle.static_ws.thread_offset" not in ir_text, ir_text
        assert "musa_tle.static_ws.split_candidate" not in ir_text, ir_text
        assert "builtin.unrealized_conversion_cast" not in ir_text, ir_text
    else:
        assert ir_text.count("ttg.warp_specialize") == 1, ir_text
        assert ir_text.count("ttg.warp_yield") == 1, ir_text
        assert ir_text.count("ttg.warp_return") == 1, ir_text
        assert "musa_tle.static_warp_specialize" in ir_text, ir_text
    assert ir_text.count("scf.for") == 2, ir_text
    assert ir_text.count("musa_tle.barrier.alloc") == (0 if barriers_lowered else 4), ir_text
    wait_op = "ttmg.wait_barrier" if barriers_lowered else "musa_tle.barrier.wait"
    arrive_op = "ttmg.warp_arrive_barrier" if barriers_lowered else "musa_tle.barrier.arrive"
    assert ir_text.count(wait_op) == 4 * len(emitted_tiles), ir_text
    assert ir_text.count(arrive_op) == 2 * len(emitted_tiles), ir_text
    if barriers_lowered:
        assert "musa_tle.barrier.wait" not in ir_text, ir_text
        assert "musa_tle.barrier.arrive" not in ir_text, ir_text
    assert ir_text.count("ttg.memdesc_index") >= 4 * stages, ir_text
    assert ir_text.count("ttg.local_alloc") == 2, ir_text
    assert "tle.wgmma_pipeline_mode" not in ir_text, ir_text
    assert "#ttg.nvmma_shared" not in ir_text, ir_text
    assert "ttg.maxnreg" not in ir_text, ir_text
    assert "actualRegisters" not in ir_text, ir_text
    assert not re.search(r"(?<!musa_)tle\.barrier\.", ir_text), ir_text

    alloc_lines = [line for line in ir_text.splitlines() if "musa_tle.barrier.alloc" in line]
    if barriers_lowered:
        assert "musa_tle.barrier.index" not in ir_text, ir_text
        assert ir_text.count("ttmg.init_arrival") == 4 * stages, ir_text
        assert f"musa.max_bar_id = {4 * stages}" in ir_text, ir_text
        assert "musa.next_bar_id" not in ir_text, ir_text
        assert "ttmg.bar_record" in ir_text, ir_text
        assert ir_text.count("ttg.barrier local") == 1, ir_text
        init_positions = [match.start() for match in re.finditer("ttmg.init_arrival", ir_text)]
        rendezvous = ir_text.index("ttg.barrier local")
        warp_specialize = ir_text.index("ttg.warp_specialize")
        assert ir_text.index("ttmg.bar_record") < min(init_positions), ir_text
        assert max(init_positions) < rendezvous < warp_specialize, ir_text
    else:
        assert sum("arrive_count = 1" in line and "expect_bytes = 32768" in line and "init_polarity = 0" in line
                   for line in alloc_lines) == 2, ir_text
        assert sum("arrive_count = 16" in line and "expect_bytes" not in line and "init_polarity = 1" in line
                   for line in alloc_lines) == 2, ir_text
        assert all(f"num_barriers = {stages}" in line for line in alloc_lines), ir_text
        assert all("memdesc" not in line and "allocation.offset" not in line for line in alloc_lines), ir_text

    assert f"!ttg.memdesc<{stages}x256x64xf16" in ir_text, ir_text
    assert f"!ttg.memdesc<{stages}x64x256xf16" in ir_text, ir_text
    assert "!ttg.memdesc<256x64xf16" in ir_text, ir_text
    assert "!ttg.memdesc<64x256xf16" in ir_text, ir_text
    assert "#smem, mutable>" in ir_text, ir_text

    ws_match, default_region, partition_match = _extract_ws_regions(ir_text)
    producer_region = partition_match.group("body")
    if barriers_lowered:
        partition_args = []
    else:
        captures = _split_top_level(ws_match.group("captures"))
        partition_args = _split_top_level(partition_match.group("args"))
        assert len(captures) == 10, (captures, ir_text)
        assert len(set(captures)) == 10, (captures, ir_text)
        assert len(partition_args) == 10, (partition_args, ir_text)
        assert sum("tensordesc" in arg for arg in partition_args) == 2, partition_args
        assert sum("memdesc" in arg for arg in partition_args) == 2, partition_args
        assert sum(": i32" in arg for arg in partition_args) == 5, partition_args
        assert sum("ptr<f16>" in arg for arg in partition_args) == 1, partition_args
        assert "requestedRegisters = array<i32: 24>" in ws_match.group("attrs"), ir_text
        assert re.search(r"\)\s*->\s*\(\)\s*\n\s*tt\.return", ir_text), ir_text
    assert "ttg.barrier local" not in default_region, default_region
    assert "ttg.barrier local" not in producer_region, producer_region

    enclosing_constants = _constants(ir_text)
    assert default_region.count("scf.for") == 1, default_region
    assert not re.search(r"arith\.(?:rem|div)", default_region), default_region
    default_memdescs = _index_defs(default_region, "ttg.memdesc_index", enclosing_constants)
    expected_logical = [u // stages for u in emitted_tiles for _ in range(2)]
    expected_ready = [phase ^ 1 for phase in expected_logical]
    if barriers_lowered:
        expected_slots = [u % stages for u in emitted_tiles]
        expected_waits = [
            item for slot, phase in zip(expected_slots, expected_logical[::2])
            for item in ((1 + slot, phase), (1 + stages + slot, phase))
        ]
        expected_arrives = [
            item for slot, phase in zip(expected_slots, expected_ready[::2])
            for item in ((1 + 2 * stages + slot, phase), (1 + 3 * stages + slot, phase))
        ]
        assert _physical_barrier_uses(default_region, "ttmg.wait_barrier",
                                      enclosing_constants) == expected_waits, default_region
        assert _physical_barrier_uses(default_region, "ttmg.warp_arrive_barrier",
                                      enclosing_constants) == expected_arrives, default_region
    else:
        default_barriers = _index_defs(default_region, "musa_tle.barrier.index", enclosing_constants)
        allocation_results = re.findall(
            r"(%[-\w.]+)\s*=\s*musa_tle\.barrier\.alloc",
            ir_text,
        )
        assert len(allocation_results) == 4, ir_text
        full_sources = set(allocation_results[:2])
        empty_sources = set(allocation_results[2:])
        assert _barrier_phases(
            default_region,
            "musa_tle.barrier.wait",
            default_barriers,
            full_sources,
            enclosing_constants,
        ) == expected_logical, default_region
        assert _barrier_phases(
            default_region,
            "musa_tle.barrier.arrive",
            default_barriers,
            empty_sources,
            enclosing_constants,
        ) == expected_ready, default_region

    assert producer_region.count("scf.for") == 1, producer_region
    assert not re.search(r"arith\.(?:rem|div)", producer_region), producer_region
    producer_arg_names = [arg.split(":", 1)[0].strip() for arg in partition_args]
    if barriers_lowered:
        # Hardware barrier bases are constants, rematerialized directly in the
        # isolated producer. They must not consume WS capture-mailbox SMEM.
        producer_constants = _constants(producer_region)
        producer_full = {name for name, value in producer_constants.items() if value in {1, 1 + stages}}
        producer_empty = {
            name
            for name, value in producer_constants.items()
            if value in {1 + 2 * stages, 1 + 3 * stages}
        }
        producer_barriers = _physical_barrier_defs(producer_region, producer_full | producer_empty)
    else:
        producer_full = set(producer_arg_names[4:6])
        producer_empty = set(producer_arg_names[6:8])
        producer_barriers = _index_defs(producer_region, "musa_tle.barrier.index")
    assert _barrier_phases(
        producer_region,
        "ttmg.wait_barrier" if barriers_lowered else "musa_tle.barrier.wait",
        producer_barriers,
        producer_empty,
    ) == expected_ready, producer_region

    slot_values = {value for _, value in default_memdescs.values()}
    if not barriers_lowered:
        slot_values.update(value for _, value in default_barriers.values())
    slot_values.update(value for _, value in producer_barriers.values())
    memdesc_defs = _index_defs(producer_region, "ttg.memdesc_index")
    slot_values.update(value for _, value in memdesc_defs.values())
    assert slot_values == set(range(stages)), producer_region
    return producer_region, producer_barriers, producer_full, memdesc_defs


def _assert_ttir_copy_association(ttir, stages, k_tiles):
    emitted_tiles = _emitted_static_tiles(stages, k_tiles)
    producer_region, barrier_defs, full_sources, memdesc_defs = _assert_common_ws_ir(ttir, stages, k_tiles)
    copies = re.findall(
        r"ttg\.tma_copy\s+(%[-\w.]+),\s*(%[-\w.]+),\s*"
        r"\[(?P<offsets>[^]]+)\]\s+barrier\s+(%[-\w.]+)",
        producer_region,
    )
    assert len(copies) == 2 * len(emitted_tiles), producer_region
    dynamic_offsets = []
    for copy_index, (_, destination, offsets, barrier) in enumerate(copies):
        barrier_source, barrier_slot = barrier_defs[barrier]
        _, destination_slot = memdesc_defs[destination]
        assert barrier_source in full_sources, (barrier, producer_region)
        assert barrier_slot == destination_slot, (barrier, destination, producer_region)
        offset_values = [value.strip() for value in offsets.split(",")]
        assert len(offset_values) == 2, offsets
        constants = _constants(producer_region)
        if copy_index % 2 == 0:
            assert constants[offset_values[0]] == 0, offsets
            dynamic_offsets.append(offset_values[1])
        else:
            assert constants[offset_values[1]] == 0, offsets
            dynamic_offsets.append(offset_values[0])
    assert all(dynamic_offsets[index] == dynamic_offsets[index + 1]
               for index in range(0, len(dynamic_offsets), 2)), copies
    assert len(set(dynamic_offsets[::2])) == len(emitted_tiles), copies
    loop_induction = re.search(r"scf\.for\s+(%[-\w.]+)\s*=", producer_region).group(1)
    definitions = _ssa_definitions(producer_region)
    period_offset_count = 4 * stages
    assert all(_depends_on(value, loop_induction, definitions) for value in dynamic_offsets[:period_offset_count])
    assert all(not _depends_on(value, loop_induction, definitions) for value in dynamic_offsets[period_offset_count:])
    assert ttir.count("ttg.tma_copy") == 2 * len(emitted_tiles), ttir
    assert ttir.count("expect_bytes = 32768 : i32") == 2 + 2 * len(emitted_tiles), ttir


def _assert_ttgir_copy_association(ttgir, stages, k_tiles):
    emitted_tiles = _emitted_static_tiles(stages, k_tiles)
    producer_region, barrier_defs, full_sources, memdesc_defs = _assert_common_ws_ir(
        ttgir, stages, k_tiles, barriers_lowered=True)
    copies = re.findall(
        r"ttmg\.async_tme_copy_global_to_local\s+(%[-\w.]+)"
        r"\[(?P<offsets>[^]]+)\],\s*(%[-\w.]+),\s*(%[-\w.]+),",
        producer_region,
    )
    assert len(copies) == 2 * len(emitted_tiles), producer_region
    dynamic_offsets = []
    constants = _constants(producer_region)
    for copy_index, (_, offsets, barrier, destination) in enumerate(copies):
        barrier_source, barrier_slot = barrier_defs[barrier]
        _, destination_slot = memdesc_defs[destination]
        assert barrier_source in full_sources, (barrier, producer_region)
        assert barrier_slot == destination_slot, (barrier, destination, producer_region)
        offset_values = [value.strip() for value in offsets.split(",")]
        assert len(offset_values) == 2, offsets
        if copy_index % 2 == 0:
            assert constants[offset_values[0]] == 0, offsets
            dynamic_offsets.append(offset_values[1])
        else:
            assert constants[offset_values[1]] == 0, offsets
            dynamic_offsets.append(offset_values[0])
    assert all(dynamic_offsets[index] == dynamic_offsets[index + 1]
               for index in range(0, len(dynamic_offsets), 2)), copies
    assert len(set(dynamic_offsets[::2])) == len(emitted_tiles), copies
    loop_induction = re.search(r"scf\.for\s+(%[-\w.]+)\s*=", producer_region).group(1)
    definitions = _ssa_definitions(producer_region)
    period_offset_count = 4 * stages
    assert all(_depends_on(value, loop_induction, definitions) for value in dynamic_offsets[:period_offset_count])
    assert all(not _depends_on(value, loop_induction, definitions) for value in dynamic_offsets[period_offset_count:])
    assert ttgir.count("ttmg.async_tme_copy_global_to_local") == 2 * len(emitted_tiles), ttgir
    assert ttgir.count("musa.tme.explicit_completion") == 3 * 2 * len(emitted_tiles), ttgir
    assert ttgir.count("musa.tme.issue_thread = 512 : i32") == 3 * 2 * len(emitted_tiles), ttgir
    assert ttgir.count("blockShape = array<i32: 256, 64>") == len(emitted_tiles), ttgir
    assert ttgir.count("blockShape = array<i32: 64, 256>") == len(emitted_tiles), ttgir
    assert ttgir.count("ttmg.init_arrival") == 4 * stages, ttgir
    assert ttgir.count("ttmg.barrier_add_trans") == 2 * len(emitted_tiles), ttgir
    assert ttgir.count("ttmg.arrive_barrier_noret") == 2 * len(emitted_tiles), ttgir


def _assert_explicit_shared_allocations(allocated, stages):
    explicit_bytes = stages * 65536
    local_allocs = [line for line in allocated.splitlines() if "ttg.local_alloc" in line]
    assert len(local_allocs) == 2, allocated
    assert "musa_tle.barrier.alloc" not in allocated, allocated
    assert "musa_tle.barrier.index" not in allocated, allocated
    assert allocated.count("ttmg.init_arrival") == 4 * stages, allocated
    assert f"musa.max_bar_id = {4 * stages}" in allocated, allocated
    assert f"!ttg.memdesc<{stages}x256x64xf16" in local_allocs[0], local_allocs
    assert f"!ttg.memdesc<{stages}x64x256xf16" in local_allocs[1], local_allocs
    assert "allocation.offset = 0 : i32" in local_allocs[0], local_allocs
    assert f"allocation.offset = {stages * 32768} : i32" in local_allocs[1], local_allocs
    assert allocated.count("ttg.warp_specialize") == 1, allocated
    ws_line = next(line for line in allocated.splitlines() if "ttg.warp_specialize" in line)
    assert "allocation.offset" not in ws_line, ws_line
    shared_bytes = int(re.search(r"ttg\.shared = (\d+) : i32", allocated).group(1))
    assert shared_bytes == explicit_bytes, allocated


def _assert_dot_pipeline_resources(ttir, ttgir, allocated, stages):
    _assert_single_compiler_stage(ttir)
    _assert_single_compiler_stage(ttgir)
    assert "#ttg.swizzled_shared" in ttir, ttir
    assert "#ttg.swizzled_shared" in ttgir, ttgir
    assert "#ttg.nvmma_shared" not in ttir, ttir
    assert "#ttg.nvmma_shared" not in ttgir, ttgir
    assert "musa_tle.barrier.wait" not in ttgir, ttgir
    assert "musa_tle.barrier.arrive" not in ttgir, ttgir

    _, default_region, _ = _extract_ws_regions(ttgir)
    wait_segments = default_region.split("ttmg.squad_dot_wait")[1:]
    assert len(wait_segments) == 2 * stages, default_region
    for segment in wait_segments:
        before_next_dot = segment.split("ttmg.squad_dot", 1)[0]
        assert before_next_dot.count("ttmg.warp_arrive_barrier") == 2, before_next_dot

    assert ttir.count("ttg.local_alloc") == 2, ttir
    assert f"!ttg.memdesc<{stages}x256x64xf16" in ttir, ttir
    assert f"!ttg.memdesc<{stages}x64x256xf16" in ttir, ttir

    ttgir_allocs = [line for line in ttgir.splitlines() if "ttg.local_alloc" in line]
    explicit_ttgir_allocs = [line for line in ttgir_allocs if "sqmma.op_idx" not in line]
    sqmma_ttgir_allocs = [line for line in ttgir_allocs if "sqmma.op_idx" in line]
    assert len(explicit_ttgir_allocs) == 2, ttgir
    assert len(sqmma_ttgir_allocs) == 4 * stages, ttgir
    assert f"!ttg.memdesc<{stages}x256x64xf16" in explicit_ttgir_allocs[0], explicit_ttgir_allocs
    assert f"!ttg.memdesc<{stages}x64x256xf16" in explicit_ttgir_allocs[1], explicit_ttgir_allocs
    assert sum("sqmma.op_idx = 0" in line for line in sqmma_ttgir_allocs) == 2 * stages, sqmma_ttgir_allocs
    assert sum("sqmma.op_idx = 1" in line for line in sqmma_ttgir_allocs) == 2 * stages, sqmma_ttgir_allocs
    assert all("sqmma.elem_bytes = 2" in line for line in sqmma_ttgir_allocs), sqmma_ttgir_allocs
    assert sum("!ttg.memdesc<256x64xf16" in line for line in sqmma_ttgir_allocs) == 2 * stages
    assert sum("!ttg.memdesc<64x256xf16" in line for line in sqmma_ttgir_allocs) == 2 * stages

    allocated_lines = allocated.splitlines()
    allocated_local = [line for line in allocated_lines if "ttg.local_alloc" in line]
    explicit_allocs = [line for line in allocated_local if "sqmma.op_idx" not in line]
    sqmma_allocs = [line for line in allocated_local if "sqmma.op_idx" in line]
    assert len(explicit_allocs) == 2, allocated
    assert len(sqmma_allocs) == 4 * stages, allocated
    assert "musa_tle.barrier.alloc" not in allocated, allocated
    assert "musa_tle.barrier.index" not in allocated, allocated
    assert "musa_tle.barrier.wait" not in allocated, allocated
    assert "musa_tle.barrier.arrive" not in allocated, allocated
    assert allocated.count("ttmg.init_arrival") == 4 * stages, allocated
    assert f"musa.max_bar_id = {4 * stages}" in allocated, allocated

    explicit_offsets = [int(re.search(r"allocation\.offset = (\d+)", line).group(1)) for line in explicit_allocs]
    assert explicit_offsets == [131072, 131072 + stages * 32768], explicit_allocs
    sqmma_offsets = [int(re.search(r"allocation\.offset = (\d+)", line).group(1)) for line in sqmma_allocs]
    explicit_end = 131072 + stages * 65536
    expected_sqmma_offsets = [index * 32768 for index in range(4)]
    expected_sqmma_offsets += [explicit_end + index * 32768 for index in range(4 * (stages - 1))]
    assert sqmma_offsets == expected_sqmma_offsets, sqmma_allocs

    output_conversion = next(
        line for line in allocated_lines
        if "ttg.convert_layout" in line and "tensor<256x256xf16" in line and "allocation.offset" in line)
    assert "allocation.offset = 0 : i32" in output_conversion, output_conversion
    assert allocated.count("ttg.warp_specialize") == 1, allocated
    ws_line = next(line for line in allocated.splitlines() if "ttg.warp_specialize" in line)
    assert "allocation.offset" not in ws_line, ws_line

    func_line = next(line for line in allocated_lines if "tt.func public @_ws_dot_integration_kernel" in line)
    assert "allocation.offset" not in func_line, func_line
    shared_bytes = int(re.search(r"ttg\.shared = (\d+) : i32", allocated).group(1))
    expected_shared = max(
        explicit_end,
        max(offset + 32768 for offset in expected_sqmma_offsets),
        131072,
    )
    assert shared_bytes == expected_shared, allocated


def _assert_late_ws_dot_cfg(late, stages):
    assert "ttg.warp_specialize" not in late, late
    assert "ttg.warp_yield" not in late, late
    assert "ttg.warp_return" not in late, late
    assert "musa_tle.static_warp_specialize" not in late, late
    assert "musa_tle.static_ws." not in late, late
    assert "cf.switch" not in late, late

    comparisons = re.findall(
        r"(%[-\w.]+)\s*=\s*arith\.cmpi\s+(uge|ult),\s*"
        r"(%[-\w.]+),\s*(%[-\w.]+)",
        late,
    )
    producer = [comparison for comparison in comparisons if comparison[1] == "uge"]
    consumer = [comparison for comparison in comparisons if comparison[1] == "ult"]
    assert len(producer) == 1, late
    assert len(consumer) == 1, late
    assert producer[0][2:] == consumer[0][2:], (producer, consumer, late)
    constants = _constants(late)
    assert constants[producer[0][3]] == 16 * 32, (producer, constants, late)
    assert late.index(producer[0][0]) < late.index(consumer[0][0]), late
    assert re.search(rf"cf\.cond_br\s+{re.escape(producer[0][0])},", late), late
    assert re.search(rf"cf\.cond_br\s+{re.escape(consumer[0][0])},", late), late
    # Two partition dispatches plus the thread-0-only barrier initializer.
    assert late.count("cf.cond_br") == 3, late

    # The pre-existing post-initialization CTA rendezvous is reused.  The
    # static dispatch and partition barriers must not allocate control SMEM.
    assert late.count('"llvm.musa.syncthreads.lm"') == 1, late
    initialized_barriers = late.count('"llvm.musa.async.init.arrival"')
    record = re.search(
        r'llvm\.call_intrinsic\s+"llvm\.musa\.async\.bar\.record"'
        r'\((%[-\w.]+)\)',
        late,
    )
    assert record, late
    recorded_barriers = constants[record.group(1)]
    assert recorded_barriers == initialized_barriers, (record.group(1), constants, late)
    assert recorded_barriers > 4 * stages, (recorded_barriers, stages, late)


@pytest.mark.parametrize("stages,k_tiles", [(1, 16), (2, 16), (2, 10)])
def test_mthreads_tle_warp_specialize_integration_contract(stages, k_tiles):
    ttir, ttgir, allocated = _compile_ws_integration(stages, k_tiles)
    _assert_ttir_copy_association(ttir, stages, k_tiles)
    _assert_ttgir_copy_association(ttgir, stages, k_tiles)
    _assert_explicit_shared_allocations(allocated, stages)


@pytest.mark.parametrize("stages", [1, 2])
def test_mthreads_tle_warp_specialize_dot_pipeline_resources(stages):
    ttir, ttgir, allocated, late = _compile_ws_dot_integration(stages)
    _assert_dot_pipeline_resources(ttir, ttgir, allocated, stages)
    _assert_late_ws_dot_cfg(late, stages)


def test_mthreads_tle_warp_specialize_stage_three_remains_deferred():
    with pytest.raises(CompilationError, match="Shape element 0 must be a power of 2"):
        _compile_ws_integration(3, 16)


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
