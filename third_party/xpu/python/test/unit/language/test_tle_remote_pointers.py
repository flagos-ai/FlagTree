"""Compile coverage for XPU TLE remote device pointers."""

import re
import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from triton.backends.xpu.tle_supported import TLE_SUPPORTED_PRIMITIVES

pytestmark = pytest.mark.skipif(
    "remote" not in TLE_SUPPORTED_PRIMITIVES,
    reason="XPU TLE remote pointer lowering has not been enabled",
)


@triton.jit
def _remote_read_kernel(out_ptr, dmem, dtype: tl.constexpr):
    pid = tl.program_id(0)
    remote_ptr = tle.remote(dmem, space="device", dtype=dtype, shard_id=1, offset=pid)
    tl.store(out_ptr + pid, tl.load(remote_ptr))


@triton.jit
def _remote_without_offset_kernel(out_ptr, dmem):
    remote_ptr = tle.remote(dmem, space="device", dtype=tl.float32, shard_id=1)
    tl.store(out_ptr, tl.load(remote_ptr))


@pytest.mark.parametrize("dtype,scale", [(tl.float16, 1), (tl.float32, 2)])
def test_remote_device_read_compiles_through_llvm(dtype, scale):
    source = triton.compiler.ASTSource(
        fn=_remote_read_kernel,
        signature={"out_ptr": "*fp32", "dmem": "i64", "dtype": "constexpr"},
        constexprs={"dtype": dtype},
    )
    compiled = triton.compile(source)
    assert compiled.metadata.xpu_uses_flagcx
    if compiled.asm["ttir"]:
        assert "triton_xpu.remote_pointer" in compiled.asm["ttir"]
    if compiled.asm["llir"]:
        assert "triton_xpu.remote_pointer" not in compiled.asm["llir"]
        assert "flagcxGetIntraPointerC" in compiled.asm["llir"]
        assert re.search(rf"\b(?:shl|mul)\b[^\n]*, (?:{scale}|{1 << scale})\b", compiled.asm["llir"])


def test_device_remote_requires_offset():
    source = triton.compiler.ASTSource(
        fn=_remote_without_offset_kernel,
        signature={"out_ptr": "*fp32", "dmem": "i64"},
        constexprs={},
    )
    with pytest.raises(Exception, match="offset"):
        triton.compile(source)
