import pytest

import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._flagtree_backend import FLAGTREE_BACKEND
from triton.experimental.tle import get_supported_primitives

pytestmark = pytest.mark.skipif(FLAGTREE_BACKEND != "hcu", reason="requires the FlagTree HCU backend")


@triton.jit
def _set_layout_kernel(src, dst, BLOCK: tl.constexpr, LAYOUT: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tle.gpu.set_layout(tl.load(src + offsets), LAYOUT)
    tl.store(tle.gpu.set_layout(dst + offsets, LAYOUT), values)


def test_set_layout_lowers_with_hard_layout():
    target = triton.runtime.driver.active.get_current_target()
    block = target.warp_size * 4
    layout = tle.gpu.BlockEncoding([2], [target.warp_size], [2], [0])
    source = triton.compiler.ASTSource(
        fn=_set_layout_kernel,
        signature={
            "src": "*fp32",
            "dst": "*fp32",
            "BLOCK": "constexpr",
            "LAYOUT": "constexpr",
        },
        constexprs={"BLOCK": block, "LAYOUT": layout},
    )

    compiled = triton.compile(source, target=target, options={"num_warps": 2})
    assert "gpu.set_layout" in compiled.asm["ttir"]
    assert "gpu.set_layout" not in compiled.asm["ttgir"]
    assert "tle.explicit" in compiled.asm["ttgir"]
    assert "gpu.set_layout" not in compiled.asm["llir"]
    assert "gpu.set_layout" in get_supported_primitives()
