"""HCU scalar buffer stores combine only when both halfwords can be written."""

import re

import pytest
import triton
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource
from triton.experimental.gluon.language.amd.cdna3 import buffer_store

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)


@g.jit
def _store(C, BOUND: gl.constexpr, START: gl.constexpr, STRIDE: gl.constexpr):
    layout: gl.constexpr = gl.DistributedLinearLayout(
        [[2], [1]], [[4], [8], [16], [32], [64], [128]], [], [], [256]
    )
    i = gl.arange(0, 256, layout=layout)
    offset = START + i * STRIDE
    buffer_store((i + 19).to(gl.int16), C, offset, i < BOUND)


@pytest.mark.parametrize(
    "bound,start,stride,alignment,expected",
    [
        (254, 0, 1, 16, "dword"),
        (253, 0, 1, 16, "short"),
        (254, 1, 1, 16, "short"),
        (254, 0, 2, 16, "short"),
        (254, 0, 1, 2, "short"),
        # The last scalar offset is in range but a pair there would cross the
        # end of the gfx936 raw descriptor.
        (256, 0x3FFFFF00, 1, 16, "mixed"),
    ],
)
def test_buffer_store_pairs_preserve_mask_alignment_and_bounds(
    bound, start, stride, alignment, expected
):
    constants = {"BOUND": bound, "START": start, "STRIDE": stride}
    source = GluonASTSource(
        _store,
        {"C": "*i16", **dict.fromkeys(constants, "constexpr")},
        constants,
        attrs={(0,): [("tt.divisibility", alignment)]},
    )
    kernel = triton.compile(
        source, target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 1}
    )
    stores = re.findall(r"^\s*buffer_store_(\w+)\s", kernel.asm["amdgcn"], re.M)
    expected_stores = {
        "dword": ["dword"] * 2,
        "short": ["short"] * 4,
        "mixed": ["dword", "short", "short"],
    }
    assert sorted(stores) == expected_stores[expected]


@g.jit
def _store_from_origin(C):
    layout: gl.constexpr = gl.DistributedLinearLayout(
        [[2], [1]], [[4], [8], [16], [32], [64], [128]], [], [], [256]
    )
    i = gl.arange(0, 256, layout=layout)
    origin = gl.program_id(0) * 256
    buffer_store(i.to(gl.int16), C, origin + i, i < 254)


def test_buffer_store_pairs_skip_possible_descriptor_boundary():
    source = GluonASTSource(
        _store_from_origin,
        {"C": "*i16"},
        attrs={(0,): [("tt.divisibility", 16)]},
    )
    kernel = triton.compile(
        source, target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 1}
    )
    stores = re.findall(r"^\s*buffer_store_(\w+)\s", kernel.asm["amdgcn"], re.M)
    assert sorted(stores) == ["dword", "short", "short"]
