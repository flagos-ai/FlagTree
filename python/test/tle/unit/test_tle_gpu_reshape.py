import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from triton._internal_testing import is_hopper_or_newer
from triton.experimental.tle.language.gpu.types import _shared_linear_layout

_ROWS = 32
_OLD_COLS = 512
_GROUPS = 4
_NEW_COLS = 128
_NUMEL = _ROWS * _OLD_COLS

_K_ROWS = tl.constexpr(32)
_K_OLD_COLS = tl.constexpr(512)
_K_GROUPS = tl.constexpr(4)
_K_NEW_COLS = tl.constexpr(128)
_K_NUMEL = tl.constexpr(16384)


class _FakeBuilder:

    def __init__(self, inferred_layout):
        self.inferred_layout = inferred_layout
        self.reshape_args = None
        self.shared_linear_args = None

    def create_memdesc_reshape(self, src, shape):
        self.reshape_args = (src, list(shape))
        return "reshaped_handle"

    def get_tle_shared_layout_from_memdesc(self, handle):
        assert handle == "reshaped_handle"
        return self.inferred_layout

    def make_shared_linear_encoding_attr(self, offset_bases, block_bases, alignment, rank):
        self.shared_linear_args = (offset_bases, block_bases, alignment, rank)
        return "shared_linear_encoding"


class _FakeSemantic:

    def __init__(self, inferred_layout):
        self.builder = _FakeBuilder(inferred_layout)


def test_buffered_tensor_reshape_uses_inferred_shared_linear_and_preserves_alloc_prefix():
    inferred_layout = {
        "kind": "shared_linear",
        "offset_bases": [[0, 0, 1], [0, 0, 2], [0, 1, 0], [1, 0, 0]],
        "block_bases": [],
        "alignment": 16,
        "rank": 3,
    }
    semantic = _FakeSemantic(inferred_layout)
    source_layout = tle.gpu.nv_mma_shared_layout.make_default([_ROWS, _OLD_COLS], tl.float16)
    source = tle.gpu.buffered_tensor(
        "source_handle",
        tl.float16,
        [_ROWS, _OLD_COLS],
        tle.gpu.smem,
        source_layout,
        semantic,
        alloc_shape=[3, _ROWS, _OLD_COLS],
    )

    reshaped = source.reshape([_ROWS, _GROUPS, _NEW_COLS], _semantic=semantic)

    assert semantic.builder.reshape_args == ("source_handle", [_ROWS, _GROUPS, _NEW_COLS])
    assert reshaped.handle == "reshaped_handle"
    assert reshaped.shape == [_ROWS, _GROUPS, _NEW_COLS]
    assert reshaped.type.alloc_shape == [3, _ROWS, _GROUPS, _NEW_COLS]
    assert isinstance(reshaped.type.layout, _shared_linear_layout)
    assert reshaped.type.layout.to_ir(semantic.builder) == "shared_linear_encoding"
    assert semantic.builder.shared_linear_args == (
        inferred_layout["offset_bases"],
        inferred_layout["block_bases"],
        inferred_layout["alignment"],
        inferred_layout["rank"],
    )


def test_buffered_tensor_reshape_keeps_inferred_nv_mma_encoding():
    inferred_layout = {
        "kind": "nv_mma",
        "transposed": False,
        "fp4_padded": False,
        "swizzled": True,
        "ctas_per_cga": [1, 1],
        "cta_split_num": [1, 1],
        "cta_order": [1, 0],
    }
    semantic = _FakeSemantic(inferred_layout)
    source_shape = [_ROWS, _GROUPS, _NEW_COLS]
    source_layout = tle.gpu.nv_mma_shared_layout.make_default(source_shape, tl.float16)
    source = tle.gpu.buffered_tensor(
        "source_handle",
        tl.float16,
        source_shape,
        tle.gpu.smem,
        source_layout,
        semantic,
    )

    reshaped = source.reshape([_ROWS * _GROUPS, _NEW_COLS], _semantic=semantic)

    assert isinstance(reshaped.type.layout, tle.gpu.nv_mma_shared_layout)
    assert reshaped.type.layout.shape == [_ROWS * _GROUPS, _NEW_COLS]
    assert reshaped.type.layout.order == [1, 0]
    assert reshaped.type.alloc_shape == [_ROWS * _GROUPS, _NEW_COLS]


@triton.jit
def _reshape_fallback_kernel(src, dst):
    flat_offsets = tl.arange(0, _K_NUMEL)
    src_ptrs = tl.reshape(src + flat_offsets, (_K_ROWS, _K_OLD_COLS))
    smem = tle.gpu.alloc(
        [_K_ROWS, _K_OLD_COLS],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    tle.gpu.copy(src_ptrs, smem, [_K_ROWS, _K_OLD_COLS])

    reshaped = smem.reshape([_K_ROWS, _K_GROUPS, _K_NEW_COLS])
    rows = tl.broadcast_to(tl.arange(0, _K_ROWS)[:, None, None], (_K_ROWS, _K_GROUPS, _K_NEW_COLS))
    groups = tl.broadcast_to(tl.arange(0, _K_GROUPS)[None, :, None], (_K_ROWS, _K_GROUPS, _K_NEW_COLS))
    cols = tl.broadcast_to(tl.arange(0, _K_NEW_COLS)[None, None, :], (_K_ROWS, _K_GROUPS, _K_NEW_COLS))
    values = tl.load(tle.gpu.local_ptr(reshaped, (rows, groups, cols)))
    dst_offsets = rows * _K_OLD_COLS + groups * _K_NEW_COLS + cols
    tl.store(dst + dst_offsets, values)


@triton.jit
def _reshape_nv_mma_preserving_kernel(dst):
    smem = tle.gpu.alloc(
        [_K_ROWS, _K_GROUPS, _K_NEW_COLS],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    reshaped = smem.reshape([_K_ROWS * _K_GROUPS, _K_NEW_COLS])
    ptr = tle.gpu.local_ptr(reshaped, (0, 0))
    tl.store(ptr, 1.0)
    tl.store(dst, tl.load(ptr))


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires NVIDIA Hopper or newer")
@pytest.mark.require_tle("gpu.alloc", "gpu.buffered_tensor.reshape", "gpu.copy", "gpu.local_ptr")
def test_buffered_tensor_reshape_shared_linear_fallback_executes_without_copying():
    src = torch.arange(_NUMEL, device="cuda", dtype=torch.float32).to(torch.float16)
    dst = torch.empty_like(src)

    compiled = _reshape_fallback_kernel.warmup(src, dst, grid=(1, ), num_warps=8)
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_reshape" in ttgir
    assert "#ttg.shared_linear" in ttgir

    _reshape_fallback_kernel[(1, )](src, dst, num_warps=8)
    torch.testing.assert_close(dst, src, atol=0, rtol=0)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="requires NVIDIA Hopper or newer")
@pytest.mark.require_tle("gpu.alloc", "gpu.buffered_tensor.reshape", "gpu.local_ptr")
def test_buffered_tensor_reshape_preserves_nv_mma_when_exactly_representable():
    dst = torch.empty(1, device="cuda", dtype=torch.float16)

    compiled = _reshape_nv_mma_preserving_kernel.warmup(dst, grid=(1, ), num_warps=4)
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_reshape" in ttgir
    assert "#ttg.nvmma_shared" in ttgir
    assert "#ttg.shared_linear" not in ttgir
