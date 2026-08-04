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

from test_tle_utils import compile_musa, mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


def _i32_constants(ir_text):
    return {
        name: int(value)
        for name, value in re.findall(
            r"(%[-\w.]+)\s*=\s*(?:arith\.)?constant\s+(-?\d+)\s*:\s*i32",
            ir_text,
        )
    }


@triton.jit
def _normal_copy_roundtrip_kernel(src, dst, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.float32, nv_mma_shared_layout=False)
    tle.gpu.copy(src + offsets, smem, (BLOCK, ))
    tle.gpu.copy(smem, dst + offsets, (BLOCK, ))


@triton.jit
def _normal_copy_shape_mismatch_kernel(src, dst, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.float32, nv_mma_shared_layout=False)
    tle.gpu.copy(src + offsets, smem, (BLOCK // 2, ))
    tle.gpu.copy(smem, dst + offsets, (BLOCK, ))


@triton.jit
def _tma_copy_desc_to_smem_kernel(desc, dst, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.float16, nv_mma_shared_layout=False)
    tle.gpu.copy(desc, smem, (BLOCK, ), (0, ))
    values = tl.load(tle.gpu.local_ptr(smem))
    tl.store(dst + offsets, values)


@triton.jit
def _tma_copy_smem_to_desc_kernel(desc, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    values = (rows * 10 + cols).to(tl.float16)
    smem = tle.gpu.alloc((BLOCK_M, BLOCK_N), dtype=tl.float16, nv_mma_shared_layout=False)
    tl.store(tle.gpu.local_ptr(smem), values)
    tle.gpu.copy(smem, desc, (BLOCK_M, BLOCK_N), (0, 0))


@triton.jit
def _tma_copy_missing_offsets_kernel(desc, BLOCK: tl.constexpr):
    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.float16, nv_mma_shared_layout=False)
    tle.gpu.copy(desc, smem, (BLOCK, ))


@triton.jit
def _tma_copy_wrong_offset_rank_kernel(desc, BLOCK: tl.constexpr):
    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.float16, nv_mma_shared_layout=False)
    tle.gpu.copy(desc, smem, (BLOCK, ), (0, 0))


@triton.jit
def _tma_completion_copy_kernel(
    desc,
    dynamic_k,
    STAGES: tl.constexpr,
    SLOT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TRANSPOSE_OFFSETS: tl.constexpr,
):
    smem = tle.gpu.alloc(
        (STAGES, BLOCK_M, BLOCK_N),
        dtype=tl.float16,
        nv_mma_shared_layout=False,
    )
    full = tle.gpu.alloc_barriers(
        STAGES,
        arrive_count=1,
        init=tle.gpu.PENDING,
        expect_bytes=32768,
    )
    offsets = (dynamic_k, 0) if TRANSPOSE_OFFSETS else (0, dynamic_k)
    tle.gpu.copy(
        desc,
        smem.slot(SLOT),
        (BLOCK_M, BLOCK_N),
        offsets,
        barrier=full[SLOT],
    )


@triton.jit
def _tma_implicit_completion_copy_kernel(desc):
    smem = tle.gpu.alloc((256, 64), dtype=tl.float16, nv_mma_shared_layout=False)
    tle.gpu.copy(desc, smem, (256, 64), (0, 0))


@triton.jit
def _tma_completion_unindexed_barrier_kernel(desc):
    smem = tle.gpu.alloc((256, 64), dtype=tl.float16, nv_mma_shared_layout=False)
    full = tle.gpu.alloc_barriers(2, expect_bytes=32768)
    tle.gpu.copy(desc, smem, (256, 64), (0, 0), barrier=full)


@triton.jit
def _tma_completion_missing_bytes_kernel(desc):
    smem = tle.gpu.alloc((256, 64), dtype=tl.float16, nv_mma_shared_layout=False)
    full = tle.gpu.alloc_barrier()
    tle.gpu.copy(desc, smem, (256, 64), (0, 0), barrier=full)


@triton.jit
def _tma_completion_wrong_direction_kernel(desc):
    smem = tle.gpu.alloc((256, 64), dtype=tl.float16, nv_mma_shared_layout=False)
    full = tle.gpu.alloc_barrier(expect_bytes=32768)
    tle.gpu.copy(smem, desc, (256, 64), (0, 0), barrier=full)


@triton.jit
def _tma_completion_dynamic_bytes_kernel(desc, expect_bytes):
    smem = tle.gpu.alloc((256, 64), dtype=tl.float16, nv_mma_shared_layout=False)
    full = tle.gpu.alloc_barrier(expect_bytes=expect_bytes)
    tle.gpu.copy(desc, smem, (256, 64), (0, 0), barrier=full)


@triton.jit
def _tma_completion_wrong_barrier_type_kernel(desc):
    smem = tle.gpu.alloc((256, 64), dtype=tl.float16, nv_mma_shared_layout=False)
    tle.gpu.copy(desc, smem, (256, 64), (0, 0), barrier=0)


def _compile_tma_completion_ir(fn, signature, constexprs=None):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs or {})
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

    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_allocate_shared_memory(pm, 31)
    pm.run(module, "allocate_tma_completion_shared_memory")
    return ttir, ttgir, module.str_nodebug()


@pytest.mark.parametrize(
    "stages,slot,block_m,block_n,transpose_offsets",
    [
        (1, 0, 256, 64, False),
        (1, 0, 64, 256, True),
        (2, 0, 256, 64, False),
        (2, 1, 256, 64, False),
        (2, 0, 64, 256, True),
        (2, 1, 64, 256, True),
    ],
)
def test_tle_tma_completion_barrier_preserves_mthreads_contract(
    stages,
    slot,
    block_m,
    block_n,
    transpose_offsets,
):
    ttir, ttgir, allocated = _compile_tma_completion_ir(
        _tma_completion_copy_kernel,
        signature={
            "desc": f"tensordesc<fp16[{block_m}, {block_n}]>",
            "dynamic_k": "i32",
            "STAGES": "constexpr",
            "SLOT": "constexpr",
            "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr",
            "TRANSPOSE_OFFSETS": "constexpr",
        },
        constexprs={
            "STAGES": stages,
            "SLOT": slot,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "TRANSPOSE_OFFSETS": transpose_offsets,
        },
    )

    assert ttir.count("ttg.tma_copy") == 1, ttir
    assert ttir.count("musa_tle.barrier.alloc") == 1, ttir
    assert ttir.count("musa_tle.barrier.index") == 1, ttir
    assert ttir.count("ttg.memdesc_index") == 1, ttir
    assert "barrier %" in ttir, ttir
    assert "expect_bytes = 32768 : i32" in ttir, ttir
    assert f"tensor<{block_m}x{block_n}xf16>" in ttir, ttir
    assert re.search(
        rf"!tt\.tensordesc<tensor<{block_m}x{block_n}xf16",
        ttir,
    ), ttir
    assert "tt.make_tensor_descriptor" not in ttir, ttir
    if transpose_offsets:
        assert re.search(r"\[%arg\d+, %c0_i32\] barrier", ttir), ttir
    else:
        assert re.search(r"\[%c0_i32, %arg\d+\] barrier", ttir), ttir

    assert "musa_tle.barrier.alloc" not in ttgir, ttgir
    assert "musa_tle.barrier.index" not in ttgir, ttgir
    async_line = next(line for line in ttgir.splitlines() if "ttmg.async_tme_copy_global_to_local" in line)
    barrier_name = re.search(r"\],\s*(%[-\w.]+),\s*%", async_line).group(1)
    assert _i32_constants(ttgir)[barrier_name] == slot + 1, ttgir
    assert f"blockShape = array<i32: {block_m}, {block_n}>" in async_line, ttgir
    assert "musa_tle.expect_bytes = 32768 : i32" in async_line, ttgir
    init_lines = [line for line in ttgir.splitlines() if "ttmg.init_arrival" in line]
    assert len(init_lines) == stages, ttgir
    constants = _i32_constants(ttgir)
    init_ids = [constants[re.search(r"ttmg\.init_arrival\s+(%[-\w.]+)", line).group(1)] for line in init_lines]
    assert init_ids == list(range(1, stages + 1)), ttgir
    assert "ttmg.barrier_add_trans" not in ttgir, ttgir
    assert "ttmg.arrive_barrier" not in ttgir, ttgir
    assert "ttmg.wait_barrier" not in ttgir, ttgir
    assert f"musa.max_bar_id = {stages}" in ttgir, ttgir

    assert f"ttg.shared = {stages * 32768} : i32" in allocated, allocated


def test_tle_tma_copy_without_completion_barrier_keeps_implicit_sync():
    ttir, ttgir, allocated = _compile_tma_completion_ir(
        _tma_implicit_completion_copy_kernel,
        signature={"desc": "tensordesc<fp16[256, 64]>"},
    )

    assert "ttg.tma_copy" in ttir, ttir
    assert "!tt.tensordesc<tensor<256x64xf16" in ttir, ttir
    assert "tt.make_tensor_descriptor" not in ttir, ttir
    assert " barrier " not in next(line for line in ttir.splitlines() if "ttg.tma_copy" in line), ttir
    assert "expect_bytes" not in next(line for line in ttir.splitlines() if "ttg.tma_copy" in line), ttir
    assert "ttmg.init_arrival" in ttgir, ttgir
    assert "ttmg.barrier_add_trans" in ttgir, ttgir
    assert "ttmg.async_tme_copy_global_to_local" in ttgir, ttgir
    assert "ttmg.arrive_barrier_noret" in ttgir, ttgir
    assert "ttmg.wait_barrier" in ttgir, ttgir
    assert "musa.max_bar_id = 1" in ttgir, ttgir
    assert "ttg.shared = 32768 : i32" in allocated, allocated


@pytest.mark.parametrize(
    "fn,signature,message",
    [
        (
            _tma_completion_unindexed_barrier_kernel,
            {"desc": "tensordesc<fp16[256, 64]>"},
            "TMA copy barrier arrays must be indexed",
        ),
        (
            _tma_completion_missing_bytes_kernel,
            {"desc": "tensordesc<fp16[256, 64]>"},
            "TMA copy barrier must be allocated with expect_bytes",
        ),
        (
            _tma_completion_wrong_direction_kernel,
            {"desc": "tensordesc<fp16[256, 64]>"},
            "TMA copy barrier is only supported for global-to-shared TMA copy",
        ),
        (
            _tma_completion_dynamic_bytes_kernel,
            {"desc": "tensordesc<fp16[256, 64]>", "expect_bytes": "i32"},
            "expect_bytes must be a compile-time integer or None",
        ),
        (
            _tma_completion_wrong_barrier_type_kernel,
            {"desc": "tensordesc<fp16[256, 64]>"},
            "TMA copy barrier expects tle.gpu barrier",
        ),
    ],
)
def test_tle_tma_completion_barrier_rejects_invalid_inputs(fn, signature, message):
    with pytest.raises(CompilationError, match=message):
        _compile_tma_completion_ir(fn, signature)


def test_tle_copy_normal_gmem_to_smem_lowers_to_async_copy():
    compiled = compile_musa(
        _normal_copy_roundtrip_kernel,
        signature={"src": "*fp32", "dst": "*fp32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 64},
    )

    ttgir = compiled.asm["ttgir"]
    assert "ttg.async_copy_global_to_local" in ttgir, ttgir
    assert "musa_tle.local_ptr_async_store" in ttgir, ttgir
    assert "ttg.local_load" in ttgir, ttgir
    assert "musa_tle.local_pointers" not in compiled.asm["llir"], compiled.asm["llir"]


def test_tle_copy_normal_rejects_shape_mismatch():
    from triton.compiler.errors import CompilationError

    with pytest.raises(CompilationError, match="copy shape .* must match"):
        compile_musa(
            _normal_copy_shape_mismatch_kernel,
            signature={"src": "*fp32", "dst": "*fp32", "BLOCK": "constexpr"},
            constexprs={"BLOCK": 64},
        )


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_copy_normal_roundtrip_runtime():
    block = 64
    src = torch.arange(0, block, device="musa", dtype=torch.float32)
    dst = torch.empty((block, ), device="musa", dtype=torch.float32)

    _normal_copy_roundtrip_kernel[(1, )](src, dst, BLOCK=block, num_warps=1)

    torch.testing.assert_close(dst.cpu(), src.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_copy_tma_desc_to_smem_lowers_to_musa_tme():
    block = 128
    src = torch.arange(0, block, device="musa", dtype=torch.float16)
    dst = torch.empty((block, ), device="musa", dtype=torch.float16)
    desc = TensorDescriptor.from_tensor(src, [block])

    compiled = _tma_copy_desc_to_smem_kernel.warmup(desc, dst, BLOCK=block, grid=(1, ), num_warps=4)
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]

    assert "ttg.tma_copy" not in ttgir, ttgir
    assert "ttmg.async_tme_copy_global_to_local" in ttgir, ttgir
    assert "ttmg.wait_barrier" in ttgir, ttgir
    assert "llvm.musa.tme.ld.tile.1d" in llir, llir


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_copy_tma_rejects_missing_offsets():
    from triton.compiler.errors import CompilationError

    block = 128
    src = torch.empty((block, ), device="musa", dtype=torch.float16)
    desc = TensorDescriptor.from_tensor(src, [block])

    with pytest.raises(CompilationError, match="descriptor-based tle.gpu.copy requires offsets"):
        _tma_copy_missing_offsets_kernel.warmup(desc, BLOCK=block, grid=(1, ), num_warps=4)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_copy_tma_rejects_wrong_offset_rank():
    from triton.compiler.errors import CompilationError

    block = 128
    src = torch.empty((block, ), device="musa", dtype=torch.float16)
    desc = TensorDescriptor.from_tensor(src, [block])

    with pytest.raises(CompilationError, match="offsets must provide 1 values, got 2"):
        _tma_copy_wrong_offset_rank_kernel.warmup(desc, BLOCK=block, grid=(1, ), num_warps=4)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_copy_tma_desc_to_smem_runtime():
    block = 128
    src = torch.arange(0, block, device="musa", dtype=torch.float16)
    dst = torch.empty((block, ), device="musa", dtype=torch.float16)
    desc = TensorDescriptor.from_tensor(src, [block])

    _tma_copy_desc_to_smem_kernel[(1, )](desc, dst, BLOCK=block, num_warps=4)

    torch.testing.assert_close(dst.cpu(), src.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_copy_tma_smem_to_desc_lowers_to_musa_tme():
    block_m = 16
    block_n = 32
    dst = torch.empty((block_m, block_n), device="musa", dtype=torch.float16)
    desc = TensorDescriptor.from_tensor(dst, [block_m, block_n])

    compiled = _tma_copy_smem_to_desc_kernel.warmup(
        desc,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        grid=(1, ),
        num_warps=4,
    )
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]

    assert "ttg.tma_copy" not in ttgir, ttgir
    assert "ttmg.async_tme_copy_local_to_global" in ttgir, ttgir
    assert "ttmg.tme_store_commit" in ttgir, ttgir
    assert "llvm.musa.tme.st.2d" in llir, llir


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_copy_tma_smem_to_desc_runtime():
    block_m = 16
    block_n = 32
    dst = torch.empty((block_m, block_n), device="musa", dtype=torch.float16)
    desc = TensorDescriptor.from_tensor(dst, [block_m, block_n])

    _tma_copy_smem_to_desc_kernel[(1, )](desc, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=4)

    rows = torch.arange(0, block_m, dtype=torch.float16)[:, None]
    cols = torch.arange(0, block_n, dtype=torch.float16)[None, :]
    ref = rows * 10 + cols
    torch.testing.assert_close(dst.cpu(), ref, rtol=0, atol=0)
