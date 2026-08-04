"""Compile-only integration coverage for mthreads explicit TLE warp specialization."""

import re

import pytest
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
    return ttir, ttgir, module.str_nodebug()


def _split_top_level(text):
    values = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "<[{(":
            depth += 1
        elif char in ">]})":
            depth -= 1
        elif char == "," and depth == 0:
            values.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        values.append(tail)
    return values


def _constants(region):
    return {
        name: int(value)
        for name, value in re.findall(
            r"(%[-\w.]+)\s*=\s*(?:arith\.)?constant\s+(-?\d+)\s*:\s*i32",
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
            r"(%[-\w.]+)\s*=\s*arith\.addi\s+(%[-\w.]+),\s*(%[-\w.]+)\s*:\s*i32",
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
    assert ir_text.count("ttg.warp_specialize") == 1, ir_text
    assert ir_text.count("ttg.warp_yield") == 1, ir_text
    assert ir_text.count("ttg.warp_return") == 1, ir_text
    assert ir_text.count("scf.for") == 2, ir_text
    assert ir_text.count("musa_tle.barrier.alloc") == (0 if barriers_lowered else 4), ir_text
    assert ir_text.count("musa_tle.barrier.wait") == 4 * len(emitted_tiles), ir_text
    assert ir_text.count("musa_tle.barrier.arrive") == 2 * len(emitted_tiles), ir_text
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
        warp_specialize = ir_text.index("ttg.warp_specialize(")
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
    assert "ttg.barrier local" not in default_region, default_region
    assert "ttg.barrier local" not in partition_match.group("body"), partition_match.group("body")
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
        assert _physical_barrier_uses(default_region, "musa_tle.barrier.wait",
                                      enclosing_constants) == expected_waits, default_region
        assert _physical_barrier_uses(default_region, "musa_tle.barrier.arrive",
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

    producer_region = partition_match.group("body")
    assert producer_region.count("scf.for") == 1, producer_region
    assert not re.search(r"arith\.(?:rem|div)", producer_region), producer_region
    producer_arg_names = [arg.split(":", 1)[0].strip() for arg in partition_args]
    producer_full = set(producer_arg_names[4:6])
    producer_empty = set(producer_arg_names[6:8])
    producer_barriers = (_physical_barrier_defs(producer_region, producer_full | producer_empty)
                         if barriers_lowered else _index_defs(producer_region, "musa_tle.barrier.index"))
    assert _barrier_phases(
        producer_region,
        "musa_tle.barrier.wait",
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
    assert "ttmg.wait_barrier" not in ttgir, ttgir


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
    ws_line = next(line for line in allocated.splitlines() if "ttg.warp_specialize(" in line)
    assert f"allocation.offset = {explicit_bytes} : i32" in ws_line, ws_line
    shared_bytes = int(re.search(r"ttg\.shared = (\d+) : i32", allocated).group(1))
    # The generic ttg.warp_specialize allocation currently serializes this
    # probe's ten explicit captures into an 88-byte scratch area.  Keep this
    # temporary equality strict so no other implicit shared allocation can be
    # introduced unnoticed.  Static mthreads WS lowering must replace it with
    # shared_bytes == explicit_bytes once captures are remapped directly.
    assert shared_bytes == explicit_bytes + 88, allocated


def _assert_dot_pipeline_resources(ttir, ttgir, allocated, stages):
    _assert_single_compiler_stage(ttir)
    _assert_single_compiler_stage(ttgir)
    assert "#ttg.swizzled_shared" in ttir, ttir
    assert "#ttg.swizzled_shared" in ttgir, ttgir
    assert "#ttg.nvmma_shared" not in ttir, ttir
    assert "#ttg.nvmma_shared" not in ttgir, ttgir

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
    assert allocated.count("ttmg.init_arrival") == 4 * stages, allocated
    assert f"musa.max_bar_id = {4 * stages}" in allocated, allocated

    explicit_offsets = [int(re.search(r"allocation\.offset = (\d+)", line).group(1)) for line in explicit_allocs]
    assert explicit_offsets == [131072, 131072 + stages * 32768], explicit_allocs
    sqmma_offsets = [int(re.search(r"allocation\.offset = (\d+)", line).group(1)) for line in sqmma_allocs]
    sqmma_base = 131072 + stages * 65536
    assert sqmma_offsets == [sqmma_base + index * 32768 for index in range(4 * stages)], sqmma_allocs

    output_conversion = next(
        line for line in allocated_lines
        if "ttg.convert_layout" in line and "tensor<256x256xf16" in line and "allocation.offset" in line)
    assert "allocation.offset = 0 : i32" in output_conversion, output_conversion
    ws_line = next(line for line in allocated_lines if "ttg.warp_specialize(" in line)
    assert "allocation.offset = 0 : i32" in ws_line, ws_line

    expected_scratch_end = sqmma_base + 4 * stages * 32768
    func_line = next(line for line in allocated_lines if "tt.func public @_ws_dot_integration_kernel" in line)
    assert f"allocation.offset = {expected_scratch_end} : i32" in func_line, func_line
    shared_bytes = int(re.search(r"ttg\.shared = (\d+) : i32", allocated).group(1))
    assert shared_bytes == expected_scratch_end + 4, allocated


@pytest.mark.parametrize("stages,k_tiles", [(1, 16), (2, 16), (2, 10)])
def test_mthreads_tle_warp_specialize_integration_contract(stages, k_tiles):
    ttir, ttgir, allocated = _compile_ws_integration(stages, k_tiles)
    _assert_ttir_copy_association(ttir, stages, k_tiles)
    _assert_ttgir_copy_association(ttgir, stages, k_tiles)
    _assert_explicit_shared_allocations(allocated, stages)


@pytest.mark.parametrize("stages", [1, 2])
def test_mthreads_tle_warp_specialize_dot_pipeline_resources(stages):
    ttir, ttgir, allocated = _compile_ws_dot_integration(stages)
    _assert_dot_pipeline_resources(ttir, ttgir, allocated, stages)


def test_mthreads_tle_warp_specialize_stage_three_remains_deferred():
    with pytest.raises(CompilationError, match="Shape element 0 must be a power of 2"):
        _compile_ws_integration(3, 16)
