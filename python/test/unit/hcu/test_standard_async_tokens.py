"""Expose existing TTGIR completion dependencies without target layout tokens."""
import re

import pytest
import triton
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource
from triton.experimental.gluon.language.amd.cdna4.async_copy import (
    buffer_load_to_shared, commit_group, wait_group,
)

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler bindings are required", allow_module_level=True)


@g.jit
def _copy(A, O, WITH_TOKEN: gl.constexpr, BAD_TOKEN: gl.constexpr = False):
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [4], [0])
    i = gl.arange(0, 2048, layout=layout)
    mem = gl.allocate_shared_memory(A.dtype.element_ty, [2048], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    buffer_load_to_shared(mem, A, i, i < 1536, other=0)
    commit_group()
    ready = wait_group(0, return_token=WITH_TOKEN)
    if BAD_TOKEN:
        result = mem.load(layout, token=42)
    elif WITH_TOKEN:
        result = mem.load(layout, token=ready)
    else:
        result = mem.load(layout)
    gl.store(O + i, result)


def compile_copy(dtype, with_token, bad_token=False):
    source = GluonASTSource(_copy,
        {"A": "*" + dtype, "O": "*" + dtype,
         "WITH_TOKEN": "constexpr", "BAD_TOKEN": "constexpr"},
        {"WITH_TOKEN": with_token, "BAD_TOKEN": bad_token},
        attrs={(0,): [("tt.divisibility", 16)], (1,): [("tt.divisibility", 16)]})
    return triton.compile(source, target=GPUTarget("hip", "gfx936", 64),
                          options=dict(num_warps=4, num_stages=1))


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("with_token", [False, True])
def test_standard_wait_result_reaches_local_load(dtype, with_token):
    # Explicit target compilation does not query or initialize a GPU driver.
    kernel = compile_copy(dtype, with_token)
    loads = re.findall(r"ttg.local_load[^\n]+", kernel.asm["ttgir"])
    assert loads and any(" token " in line for line in loads) == with_token
    assert "ttg.async_wait" in kernel.asm["ttgir"]
    assert "mls_shared" not in kernel.asm["ttgir"]
    assert "hcu.packet" not in kernel.asm["ttgir"]


def test_shared_load_rejects_non_token():
    with pytest.raises(triton.CompilationError, match="expected an async wait token"):
        compile_copy("bf16", True, bad_token=True)


def test_wait_token_preserves_read_before_overwrite(tmp_path):
    from test_physical_shared_regions import module_source, lower_source
    text = module_source(select_load=True)
    text = text.replace("    ttg.local_barrier", "    %ready = ttg.async_wait {num = 0 : i32}\n    ttg.local_barrier")
    text = text.replace("ttg.local_load %choice :", "ttg.local_load %choice token %ready :")
    # Completion of a previous write does not complete a later shared read.
    # Writing B must still wait for the read which could select B.
    assert lower_source(tmp_path, text, update_async_wait=True).count("rocdl.s.barrier") == 2


from triton.experimental.gluon.language.hcu import sched_barrier


@g.jit
def _schedule_boundary(MASK: gl.constexpr):
    sched_barrier(MASK)


@pytest.mark.parametrize("mask",[0,1,-1,1<<31])
def test_target_schedule_boundary_has_no_packet_or_thread_sync(mask):
    source=GluonASTSource(_schedule_boundary,{"MASK":"constexpr"},{"MASK":mask})
    if mask<0 or mask>0x7fffffff:
        with pytest.raises(triton.CompilationError):
            triton.compile(source,target=GPUTarget("hip","gfx936",64),options=dict(num_warps=4))
        return
    kernel=triton.compile(source,target=GPUTarget("hip","gfx936",64),options=dict(num_warps=4))
    assert f"@llvm.amdgcn.sched.barrier(i32 {mask})" in kernel.asm["llir"]
    assert "rocdl.s.barrier" not in kernel.asm["ttgir"]
    assert "whole_packet" not in kernel.asm["ttgir"]
