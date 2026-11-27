from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.raw as raw
from triton._flagtree_backend import FLAGTREE_BACKEND
from triton.experimental.tle.raw import dialect

pytestmark = pytest.mark.skipif(
    FLAGTREE_BACKEND not in ("", "nvidia") or torch.version.hip is not None,
    reason="requires the NVIDIA CUDA raw backend",
)

INPUTS = Path(__file__).parent / "inputs"
ENTRIES = ("fnx", "fn", "symbols::fnx", "symbols::fn")


def bind_entry(source, name, *, deferred=False):

    @dialect(name="cuda", file=INPUTS / f"symbolset_{source}.cu", extern_func_name=name, deferred=deferred)
    def entry(*args):
        ...

    return entry


@triton.jit
def call_pair(state, FIRST: tl.constexpr, SECOND: tl.constexpr):
    raw.call(FIRST, (state, ))
    raw.call(SECOND, (state, ))


@pytest.mark.parametrize("first_source,second_source", [("a", "b"), ("b", "a")], ids=["a_then_b", "b_then_a"])
@pytest.mark.parametrize("first_name", ENTRIES)
@pytest.mark.parametrize("second_name", ENTRIES)
def test_eager_symbol_import(first_source, second_source, first_name, second_name):
    first = bind_entry(first_source, first_name)
    second = bind_entry(second_source, second_name)
    state = torch.zeros(2, dtype=torch.int32, device="cuda")
    compiled = call_pair[(1, )](state, first, second, num_warps=1)

    # A owns the even bits and B the odd bits. The second word records
    # the mask before the last call so the test also checks call order.
    first_bit = 2 * ENTRIES.index(first_name) + int(first_source == "b")
    second_bit = 2 * ENTRIES.index(second_name) + int(second_source == "b")
    assert state.tolist() == [(1 << first_bit) | (1 << second_bit), 1 << first_bit]
    assert "__tle_raw_remove_after_ir_parse__" not in compiled.asm["ttir"]


@triton.jit
def deferred_sequence(state, FIRST: tl.constexpr, SECOND: tl.constexpr):
    raw.call(FIRST, (state, ), output_indices=[])
    raw.call(SECOND, (state, ), output_indices=[])
    raw.call(FIRST, (state, ), output_indices=[])


@pytest.mark.parametrize("first,second", [("a", "b"), ("b", "a")])
def test_deferred_helper_identity(first, second):
    state = torch.zeros(3, dtype=torch.int32, device="cuda")
    deferred_sequence[(1, )](state, bind_entry(first, "accumulate", deferred=True),
                             bind_entry(second, "accumulate", deferred=True), num_warps=1)
    first_step, second_step = (1, 3) if first == "a" else (3, 1)
    assert state.tolist() == [2 * first_step + second_step, first_step + second_step, first_step]
