# flagtree tle
"""Code-generation tests for loop-carried typed-pipe cursors."""

import pytest
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend

_HOPPER_TARGET = GPUTarget("cuda", 90, 32)


@triton.jit
def _pipe_cursor_loop_kernel(out_ptr, start_iter, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc([4, BLOCK], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=4, scope="cta", name="cursor_test", data=smem)
    writer = pipe.writer()
    cursor = writer.cursor(start_iter)
    for iteration in range(0, 8):
        slot = writer.acquire(cursor)
        ptrs = tle.gpu.local_ptr(slot.data, (offsets, ))
        tl.store(ptrs, offsets + iteration)
        writer.commit(cursor)
        cursor = cursor.advance()
    tl.store(out_ptr, cursor.stage)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_pipe_cursor_is_carried_as_stage_and_phase_without_loop_division():
    backend = make_backend(_HOPPER_TARGET)
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(
        fn=_pipe_cursor_loop_kernel,
        signature={"out_ptr": "*i32", "start_iter": "i32"},
        constexprs={"BLOCK": 64},
    )
    ttir = str(
        source.make_ir(
            _HOPPER_TARGET,
            options,
            backend.get_codegen_implementation(options),
            backend.get_module_map(),
            context,
        ))

    loop_line = next(line for line in ttir.splitlines() if "scf.for" in line)
    loop_body = ttir[ttir.index(loop_line):ttir.index("scf.yield", ttir.index(loop_line))]
    assert "iter_args" in loop_line
    assert "i32, i1" in loop_line
    assert "tle.pipe.writer_acquire" in ttir
    assert "tle.pipe.writer_commit" in ttir
    assert "arith.remsi" not in loop_body
    assert "arith.divsi" not in loop_body
