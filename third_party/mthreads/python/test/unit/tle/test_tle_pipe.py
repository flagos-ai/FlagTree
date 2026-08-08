"""Compile and runtime coverage for the mthreads single-field TLE pipe contract."""

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
from triton.tools.tensor_descriptor import TensorDescriptor

from test_tle_utils import mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _pipe_consumer(reader, out, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        wait = reader.wait(iteration)
        tl.store(out + iteration, iteration + tl.where(wait.is_closed, 1000, 0))
        reader.release(iteration)


@triton.jit
def _pipe_producer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)


@triton.jit
def _dual_pipe_consumer(first, second, out, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        first_wait = first.wait(iteration)
        second_wait = second.wait(iteration)
        tl.store(out + iteration,
                 iteration + tl.where(first_wait.is_closed, 1000, 0) + tl.where(second_wait.is_closed, 2000, 0))
        first.release(iteration)
        second.release(iteration)


@triton.jit
def _dual_pipe_producer(first, second, first_desc, second_desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        first_slot = first.acquire(iteration)
        second_slot = second.acquire(iteration)
        tle.gpu.copy(first_desc, first_slot.data, (BLOCK, ), (iteration * BLOCK, ))
        tle.gpu.copy(second_desc, second_slot.data, (BLOCK, ), (iteration * BLOCK, ))
        first.commit(iteration)
        second.commit(iteration)


@triton.jit
def _dual_pipe_kernel(
    first_desc,
    second_desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    first_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    second_smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    first = tle.pipe(capacity=STAGES, name="first", data=first_smem)
    second = tle.pipe(capacity=STAGES, name="second", data=second_smem)
    tle.gpu.warp_specialize(
        [
            (_dual_pipe_consumer, (first.reader(), second.reader(), out, ITERATIONS)),
            (
                _dual_pipe_producer,
                (first.writer(), second.writer(), first_desc, second_desc, BLOCK, ITERATIONS),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _pipe_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="data_pipe", data=smem)
    tle.gpu.warp_specialize(
        [
            (_pipe_consumer, (pipe.reader(), out, ITERATIONS)),
            (_pipe_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _non_ws_pipe_mm_kernel(
    a_desc,
    b_desc,
    out,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    a_smem = tle.gpu.alloc(
        (STAGES, BLOCK_M, BLOCK_K),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=False,
    )
    b_smem = tle.gpu.alloc(
        (STAGES, BLOCK_K, BLOCK_N),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=False,
    )
    a_pipe = tle.pipe(capacity=STAGES, name="runtime_a", a=a_smem)
    b_pipe = tle.pipe(capacity=STAGES, name="runtime_b", b=b_smem)
    a_writer = a_pipe.writer()
    b_writer = b_pipe.writer()
    a_reader = a_pipe.reader()
    b_reader = b_pipe.reader()

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in tl.static_range(0, K_TILES):
        a_slot = a_writer.acquire(k_iter)
        b_slot = b_writer.acquire(k_iter)
        k_offset = k_iter * BLOCK_K
        tle.gpu.copy(a_desc, a_slot.a, (BLOCK_M, BLOCK_K), (0, k_offset))
        tle.gpu.copy(b_desc, b_slot.b, (BLOCK_K, BLOCK_N), (k_offset, 0))
        a_writer.commit(k_iter)
        b_writer.commit(k_iter)

        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        acc = tle.gpu.wgmma(a_wait.slot.a, b_wait.slot.b, acc)
        acc = tle.gpu.wgmma_wait(0, acc)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    offsets = tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out + offsets, acc.to(tl.float16))


@triton.jit
def _ws_pipe_mm_consumer(
    a_reader,
    b_reader,
    out,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in tl.range(0, K_TILES, num_stages=1):
        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        acc = tle.gpu.wgmma(a_wait.slot.a, b_wait.slot.b, acc)
        acc = tle.gpu.wgmma_wait(0, acc)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    offsets = tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out + offsets, acc.to(tl.float16))


@triton.jit
def _ws_pipe_mm_producer(
    a_writer,
    b_writer,
    a_desc,
    b_desc,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    for k_iter in tl.range(0, K_TILES, num_stages=1):
        a_slot = a_writer.acquire(k_iter)
        b_slot = b_writer.acquire(k_iter)
        k_offset = k_iter * BLOCK_K
        tle.gpu.copy(a_desc, a_slot.a, (BLOCK_M, BLOCK_K), (0, k_offset))
        tle.gpu.copy(b_desc, b_slot.b, (BLOCK_K, BLOCK_N), (k_offset, 0))
        a_writer.commit(k_iter)
        b_writer.commit(k_iter)


@triton.jit
def _ws_pipe_mm_kernel(
    a_desc,
    b_desc,
    out,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    a_smem = tle.gpu.alloc(
        (STAGES, BLOCK_M, BLOCK_K),
        dtype=tl.float16,
        nv_mma_shared_layout=False,
    )
    b_smem = tle.gpu.alloc(
        (STAGES, BLOCK_K, BLOCK_N),
        dtype=tl.float16,
        nv_mma_shared_layout=False,
    )
    a_pipe = tle.pipe(capacity=STAGES, name="ws_runtime_a", a=a_smem)
    b_pipe = tle.pipe(capacity=STAGES, name="ws_runtime_b", b=b_smem)
    tle.gpu.warp_specialize(
        [
            (
                _ws_pipe_mm_consumer,
                (a_pipe.reader(), b_pipe.reader(), out, K_TILES, BLOCK_M, BLOCK_N),
            ),
            (
                _ws_pipe_mm_producer,
                (
                    a_pipe.writer(),
                    b_pipe.writer(),
                    a_desc,
                    b_desc,
                    K_TILES,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _baseline_consumer(out, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        tl.store(out + iteration, iteration)


@triton.jit
def _baseline_producer(smem, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr, STAGES: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        tle.gpu.copy(desc, smem.slot(iteration % STAGES), (BLOCK, ), (iteration * BLOCK, ))


@triton.jit
def _baseline_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    tle.gpu.warp_specialize(
        [
            (_baseline_consumer, (out, ITERATIONS)),
            (_baseline_producer, (smem, desc, BLOCK, ITERATIONS, STAGES)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _local_store_producer(writer, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tl.store(tle.gpu.local_ptr(slot.data, (0, )), 0.0)
        writer.commit(iteration)


@triton.jit
def _double_tme_producer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)


@triton.jit
def _close_producer(writer, desc, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    for iteration in tl.static_range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        tle.gpu.copy(desc, slot.data, (BLOCK, ), (iteration * BLOCK, ))
        writer.commit(iteration)
        writer.close(iteration)


@triton.jit
def _invalid_pipe_kernel(
    desc,
    out,
    STAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    ITERATIONS: tl.constexpr,
    KIND: tl.constexpr,
):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="invalid_pipe", data=smem)
    if KIND == 0:
        tle.gpu.warp_specialize(
            [
                (_pipe_consumer, (pipe.reader(), out, ITERATIONS)),
                (_local_store_producer, (pipe.writer(), BLOCK, ITERATIONS)),
            ],
            worker_num_warps=[4],
            worker_num_regs=[24],
        )
    elif KIND == 1:
        tle.gpu.warp_specialize(
            [
                (_pipe_consumer, (pipe.reader(), out, ITERATIONS)),
                (_double_tme_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
            ],
            worker_num_warps=[4],
            worker_num_regs=[24],
        )
    else:
        tle.gpu.warp_specialize(
            [
                (_pipe_consumer, (pipe.reader(), out, ITERATIONS)),
                (_close_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
            ],
            worker_num_warps=[4],
            worker_num_regs=[24],
        )


@triton.jit
def _misplaced_pipe_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK),
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="misplaced", data=smem)
    tle.gpu.warp_specialize(
        [
            (_pipe_producer, (pipe.writer(), desc, BLOCK, ITERATIONS)),
            (_pipe_consumer, (pipe.reader(), out, ITERATIONS)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _multi_field_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    first = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    second = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    tle.pipe(capacity=STAGES, first=first, second=second)


@triton.jit
def _named_reader_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    tle.pipe(capacity=STAGES, readers=("consumer", ), data=smem)


@triton.jit
def _one_shot_kernel(desc, out, STAGES: tl.constexpr, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
    smem = tle.gpu.alloc((STAGES, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    tle.pipe(capacity=STAGES, one_shot=True, data=smem)


def _compile_pipeline(fn, stages):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 16, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(
        fn=fn,
        signature={
            "desc": "tensordesc<fp16[128]>",
            "out": "*i32",
            "STAGES": "constexpr",
            "BLOCK": "constexpr",
            "ITERATIONS": "constexpr",
        },
        constexprs={"STAGES": stages, "BLOCK": 128, "ITERATIONS": 2 * stages},
    )
    module = source.make_ir(
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
    pm.run(module, "allocate_pipe_shared_memory")
    return ttir, ttgir, module.str_nodebug()


def _compile_invalid_pipeline(fn, kind=None):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 16, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    signature = {
        "desc": "tensordesc<fp16[128]>",
        "out": "*i32",
        "STAGES": "constexpr",
        "BLOCK": "constexpr",
        "ITERATIONS": "constexpr",
    }
    constexprs = {"STAGES": 2, "BLOCK": 128, "ITERATIONS": 2}
    if kind is not None:
        signature["KIND"] = "constexpr"
        constexprs["KIND"] = kind
    source = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    module = source.make_ir(
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
    return compiler_stages["ttgir"](module, metadata)


def _compile_dual_pipeline(stages):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 16, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(
        fn=_dual_pipe_kernel,
        signature={
            "first_desc": "tensordesc<fp16[128]>",
            "second_desc": "tensordesc<fp16[128]>",
            "out": "*i32",
            "STAGES": "constexpr",
            "BLOCK": "constexpr",
            "ITERATIONS": "constexpr",
        },
        constexprs={"STAGES": stages, "BLOCK": 128, "ITERATIONS": 2 * stages},
    )
    module = source.make_ir(
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
    return module.str_nodebug()


def _shared_bytes(allocated):
    return int(re.search(r"ttg\.shared = (\d+) : i32", allocated).group(1))


def _i32_constants(ir_text):
    return {
        name: int(value)
        for name, value in re.findall(r"(%[-\w.]+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*i32", ir_text)
    }


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_mthreads_single_field_pipe_lowers_to_hardware_barriers(stages):
    ttir, ttgir, allocated = _compile_pipeline(_pipe_kernel, stages)

    assert ttir.count("tle.pipe.create") == 1, ttir
    assert ttir.count("tle.pipe.writer_acquire") == 2 * stages, ttir
    assert ttir.count("tle.pipe.writer_commit") == 2 * stages, ttir
    assert ttir.count("tle.pipe.reader_wait") == 2 * stages, ttir
    assert ttir.count("tle.pipe.reader_release") == 2 * stages, ttir
    assert "tle.pipe." not in ttgir, ttgir
    assert ttgir.count("ttmg.init_arrival") == 2 * stages, ttgir
    assert f"musa.max_bar_id = {2 * stages}" in ttgir, ttgir
    assert "ttmg.barrier_add_trans" in ttgir, ttgir
    assert "ttmg.wait_barrier" in ttgir, ttgir
    assert "ttmg.warp_arrive_barrier" in ttgir, ttgir
    assert "ttmg.async_tme_copy_global_to_local" in ttgir, ttgir
    assert "tle.pipe." not in allocated, allocated
    assert "1000" not in ttgir, ttgir

    constants = _i32_constants(ttgir)
    arrivals = re.findall(r"ttmg\.init_arrival\s+(%[-\w.]+),\s+(%[-\w.]+),\s+(%[-\w.]+)", ttgir)
    assert [constants[count] for _, count, _ in arrivals] == [1] * stages + [16] * stages
    assert [constants[phase] for _, _, phase in arrivals] == [0] * (2 * stages)
    assert ttgir.count("ttmg.barrier_add_trans") == 2 * stages
    assert all("%c256" in line for line in ttgir.splitlines() if "ttmg.barrier_add_trans" in line)


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_mthreads_pipe_barrier_ids_add_no_shared_memory(stages):
    _, _, pipe_allocated = _compile_pipeline(_pipe_kernel, stages)
    _, _, baseline_allocated = _compile_pipeline(_baseline_kernel, stages)

    assert _shared_bytes(pipe_allocated) == _shared_bytes(baseline_allocated)
    assert pipe_allocated.count("ttg.warp_specialize") == 1, pipe_allocated
    ws_line = next(line for line in pipe_allocated.splitlines() if "ttg.warp_specialize" in line)
    assert "allocation.offset" not in ws_line, ws_line
    assert "musa_tle.static_ws." not in pipe_allocated, pipe_allocated


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_mthreads_two_single_field_pipes_keep_independent_barrier_rings(stages):
    ttgir = _compile_dual_pipeline(stages)
    assert "tle.pipe." not in ttgir, ttgir
    assert ttgir.count("ttmg.init_arrival") == 4 * stages, ttgir
    assert ttgir.count("ttmg.barrier_add_trans") == 4 * stages, ttgir
    assert ttgir.count("ttmg.async_tme_copy_global_to_local") == 4 * stages, ttgir
    assert f"musa.max_bar_id = {4 * stages}" in ttgir, ttgir


@pytest.mark.parametrize(
    "kernel,diagnostic",
    [
        (_multi_field_kernel, "requires exactly one payload field"),
        (_named_reader_kernel, "supports only the default SPSC reader"),
        (_one_shot_kernel, "does not support one_shot=True"),
    ],
)
def test_mthreads_pipe_rejects_unsupported_frontend_options(kernel, diagnostic):
    with pytest.raises(CompilationError, match=diagnostic):
        _compile_invalid_pipeline(kernel)


@pytest.mark.parametrize(
    "kind,diagnostic",
    [
        (0, "requires exactly one TME copy between acquire and commit; found 0"),
        (1, "requires exactly one TME copy between acquire and commit; found 2"),
        (2, "does not support writer.close"),
    ],
)
def test_mthreads_pipe_rejects_unsupported_producer_protocol(capfd, kind, diagnostic):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(_invalid_pipe_kernel, kind)
    assert diagnostic in capfd.readouterr().err


def test_mthreads_pipe_rejects_writer_in_default_partition(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_invalid_pipeline(_misplaced_pipe_kernel)
    assert "requires writer operations either outside warp_specialize or in the worker partition 0" in capfd.readouterr(
    ).err


@pytest.mark.parametrize(
    "block_m,block_n,k_tiles,stages",
    [
        pytest.param(128, 128, 4, 1, id="m128-n128-k256-stage1"),
        pytest.param(128, 128, 4, 2, id="m128-n128-k256-stage2"),
        pytest.param(128, 128, 7, 3, id="m128-n128-k448-stage3"),
        pytest.param(256, 256, 1, 1, id="m256-n256-k64-stage1"),
    ],
)
def test_mthreads_non_ws_pipe_mm_runtime(block_m, block_n, k_tiles, stages):
    torch.manual_seed(42)
    block_k = 64
    k = k_tiles * block_k
    a = torch.randn((block_m, k), dtype=torch.float16, device="musa")
    b = torch.randn((k, block_n), dtype=torch.float16, device="musa")
    out = torch.empty((block_m, block_n), dtype=torch.float16, device="musa")
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n])
    reference = torch.matmul(a.to(torch.float32), b.to(torch.float32))

    for _ in range(2):
        out.fill_(float("nan"))
        _non_ws_pipe_mm_kernel[(1, )](
            a_desc,
            b_desc,
            out,
            K_TILES=k_tiles,
            STAGES=stages,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=1,
        )
        torch.musa.synchronize()
        torch.testing.assert_close(out.to(torch.float32), reference, rtol=1.25e-1, atol=1.25e-1)


@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_mthreads_ws_pipe_mm_runtime(stages):
    torch.manual_seed(42)
    block_m = block_n = 256
    block_k = 64
    # Seven tiles exercise a full two-phase cycle and then reuse slot zero for
    # the third generation when capacity is three.
    k_tiles = 7
    k = k_tiles * block_k
    a = torch.randn((block_m, k), dtype=torch.float16, device="musa")
    b = torch.randn((k, block_n), dtype=torch.float16, device="musa")
    out = torch.empty((block_m, block_n), dtype=torch.float16, device="musa")
    a_desc = TensorDescriptor.from_tensor(a, [block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(b, [block_k, block_n])
    reference = torch.matmul(a.to(torch.float32), b.to(torch.float32))

    compiled = _ws_pipe_mm_kernel.warmup(
        a_desc,
        b_desc,
        out,
        K_TILES=k_tiles,
        STAGES=stages,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        grid=(1, ),
        num_warps=16,
        num_stages=1,
    )
    assert compiled.metadata.num_warps == 20
    assert compiled.metadata.shared == stages * 65536
    assert '"ttg.total-num-warps" = 20 : i32' in compiled.asm["ttgir"]
    assert compiled.asm["ttgir"].count("ttg.warp_specialize") == 1
    assert "musa_tle.static_ws." not in compiled.asm["ttgir"]
    assert "ttg.warp_specialize" not in compiled.asm["llir"]
    assert "ttg.convert_layout" not in compiled.asm["ttgir"]
    assert "swizzleGranularity = 1 : i32" in compiled.asm["ttgir"]
    assert "swizzleGranularity = 2 : i32" in compiled.asm["ttgir"]
    assert "builtin.unrealized_conversion_cast" not in compiled.asm["llir"]
    assert compiled.asm["llir"].count("call void @llvm.musa.syncthreads.lm()") == 1
    assert f"llvm.musa.async.bar.record(i32 {4 * stages})" in compiled.asm["llir"]

    for _ in range(2):
        out.fill_(float("nan"))
        _ws_pipe_mm_kernel[(1, )](
            a_desc,
            b_desc,
            out,
            K_TILES=k_tiles,
            STAGES=stages,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=16,
            num_stages=1,
        )
        torch.musa.synchronize()
        torch.testing.assert_close(out.to(torch.float32), reference, rtol=1.25e-1, atol=1.25e-1)


def test_mthreads_pipe_bindings_are_optional_and_backend_local():
    assert hasattr(libtriton.ir.builder, "create_pipe_create")
    assert hasattr(libtriton.ir.builder, "create_pipe_writer_acquire")
    assert hasattr(libtriton.ir.builder, "create_pipe_writer_commit")
    assert hasattr(libtriton.ir.builder, "create_pipe_writer_close")
    assert hasattr(libtriton.ir.builder, "create_pipe_reader_wait")
    assert hasattr(libtriton.ir.builder, "create_pipe_reader_release")
    assert hasattr(libtriton.mthreads.passes.ttgpuir, "add_tle_lower_pipe")
