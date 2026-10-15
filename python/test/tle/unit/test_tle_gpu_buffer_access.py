"""Buffer load/store use the same API with CommonIR enabled or disabled."""
import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._common_ir import ENABLED

pytestmark = pytest.mark.require_tle("gpu.alloc", "gpu.buffered_tensor.load", "gpu.buffered_tensor.store",
                                     "gpu.buffered_tensor.slot")


@triton.jit
def _buffer_access(src, dst, ROWS: tl.constexpr, COLS: tl.constexpr, USE_SLOT: tl.constexpr):
    idx = tl.arange(0, ROWS)[:, None] * COLS + tl.arange(0, COLS)[None, :]
    value = tl.load(src + idx)
    if USE_SLOT:
        allocation = tle.gpu.alloc([2, ROWS, COLS], value.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        buf = allocation.slot(1)
    else:
        buf = tle.gpu.alloc([ROWS, COLS], value.dtype, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    buf.store(value)
    value = buf.load()
    buf.store(value + 1)
    tl.store(dst + idx, buf.load(writable=False))


@pytest.mark.parametrize("shape", [(1, 64), (8, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("use_slot", [False, True])
def test_buffer_access_roundtrip(shape, dtype, use_slot):
    rows, cols = shape
    src = torch.arange(rows * cols, device="cuda", dtype=dtype).reshape(shape)
    dst = torch.empty_like(src)
    kernel = _buffer_access[(1, )](src, dst, rows, cols, use_slot)
    torch.testing.assert_close(dst, src + 1, atol=0, rtol=0)
    assert "tile." not in kernel.asm["ttgir"]
    assert "unrealized_conversion_cast" not in kernel.asm["ttgir"]


@pytest.mark.parametrize("use_slot", [False, True])
def test_buffer_access_frontend_ir(use_slot):
    from triton._C.libtriton import ir
    from triton.compiler.compiler import ASTSource, make_backend

    target = triton.runtime.driver.active.get_current_target()
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(_buffer_access, {"src": "*fp32", "dst": "*fp32"}, {"ROWS": 8, "COLS": 16, "USE_SLOT": use_slot})
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                            context)
    assert module.verify()
    operations = []
    module.walk(operations.append)
    names = {op.get_name() for op in operations}
    if ENABLED:
        assert {"tile.alloc", "tile.to_tensor", "tile.store_tensor"} <= names
        for op in operations:
            if op.get_name() in ("tile.alloc", "tile.subview"):
                assert str(op.get_result(0).get_type()).startswith("!tile.buf<")
        if use_slot:
            assert "tile.subview" in names
    else:
        assert not any(name.startswith("tile.") for name in names)
        assert {"ttg.local_alloc", "tle.local_pointers"} <= names
