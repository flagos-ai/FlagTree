import re

import pytest
import triton
import triton.experimental.tle.language as tle
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, get_backend, make_backend

_HOPPER_TARGET = GPUTarget("cuda", 90, 32)


def _cuda_backend_available():
    try:
        get_backend(_HOPPER_TARGET)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _cuda_backend_available(), reason="requires cuda backend")


def _make_ttir(kernel):
    backend = make_backend(_HOPPER_TARGET)
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(fn=kernel, signature={}, constexprs={})
    module = source.make_ir(
        _HOPPER_TARGET,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    return str(module)


@triton.jit
def _wait_on_named_barrier(bar):
    tle.gpu.barrier_wait(bar)


@triton.jit
def _two_barriers_passed_to_one_helper():
    first = tle.gpu.alloc_barrier(arrive_count=64)
    second = tle.gpu.alloc_barrier(arrive_count=64)
    _wait_on_named_barrier(first)
    _wait_on_named_barrier(second)


def test_allocated_named_barrier_identity_participates_in_helper_mangling():
    ttir = _make_ttir(_two_barriers_passed_to_one_helper)

    helper_defs = [line for line in ttir.splitlines() if "tt.func private" in line and "_wait_on_named_barrier" in line]
    assert len(helper_defs) == 2
    assert any(re.search(r"barrier_.*_16(?:__|\W|$)", line) for line in helper_defs)
    assert any(re.search(r"barrier_.*_17(?:__|\W|$)", line) for line in helper_defs)
    assert ttir.count("named_id = 16") == 1
    assert ttir.count("named_id = 17") == 1


@triton.jit
def _first_helper_local_barrier():
    bar = tle.gpu.alloc_barrier(arrive_count=64)
    tle.gpu.barrier_wait(bar)


@triton.jit
def _second_helper_local_barrier():
    bar = tle.gpu.alloc_barrier(arrive_count=64)
    tle.gpu.barrier_wait(bar)


@triton.jit
def _barriers_allocated_in_distinct_helpers():
    _first_helper_local_barrier()
    _second_helper_local_barrier()


def test_named_barrier_virtual_ids_are_unique_across_codegen_module():
    ttir = _make_ttir(_barriers_allocated_in_distinct_helpers)

    assert ttir.count("named_id = 16") == 1
    assert ttir.count("named_id = 17") == 1
