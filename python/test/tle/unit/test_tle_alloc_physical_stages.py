"""Automatic and explicit shared layouts must agree on physical stage extents."""
from types import SimpleNamespace

import pytest
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.experimental.tle.language.gpu import core as gpu_core


class _Builder:
    def get_bf16_ty(self):
        return "bf16"

    def make_swizzled_shared_encoding_attr(self, *args):
        return ("swizzled", *args)

    def make_nv_mma_shared_encoding_attr(self, *args):
        return ("nvmma", *args)

    def create_local_alloc(self, shape, dtype, layout):
        self.operation = ("alloc", shape, dtype, layout)
        return "buffer"

    def get_memdesc_type(self, shape, dtype, layout, storage):
        return (shape, dtype, layout, storage)

    def create_memdesc_alias(self, result_type, source, offset):
        self.operation = ("alias", result_type, source, offset)
        return "buffer"


@pytest.fixture
def semantic(monkeypatch):
    monkeypatch.setattr(gpu_core.tle_semantic, "COMMON_IR_ENABLED", False)
    monkeypatch.setattr(gpu_core.mthreads_common, "enabled", lambda: False)
    monkeypatch.setattr(gpu_core.iluvatar_copy, "enabled", lambda: False)
    return SimpleNamespace(builder=_Builder())


@pytest.mark.parametrize("shape", [(3, 16, 2, 32), (3, 64, 2, 2, 64), (5, 16, 2, 32), (2, 64, 2, 2, 64)])
@pytest.mark.parametrize("mma", [False, True])
@pytest.mark.parametrize("aliased", [False, True])
def test_automatic_physical_stage_matches_explicit_layout(semantic, shape, mma, aliased):
    layout = (tle.gpu.nv_mma_shared_layout.make_default(list(shape), tl.bfloat16) if mma
              else tle.gpu.swizzled_shared_layout.make_default(len(shape)))
    arena = tle.gpu.buffered_tensor("arena", tl.bfloat16, [262144], tle.gpu.smem,
                                   tle.gpu.swizzled_shared_layout.make_default(1), semantic)
    kwargs = {"alias": arena, "alias_offset_bytes": 1024} if aliased else {}
    explicit = tle.gpu.alloc(shape, tl.bfloat16, layout=layout, nv_mma_shared_layout=mma,
                             _semantic=semantic, **kwargs)
    expected_operation = semantic.builder.operation
    automatic = tle.gpu.alloc(shape, tl.bfloat16, nv_mma_shared_layout=mma,
                              _semantic=semantic, **kwargs)
    assert semantic.builder.operation == expected_operation
    assert automatic.shape == explicit.shape == list(shape)
    assert automatic.type.alloc_shape == explicit.type.alloc_shape == list(shape)
    assert automatic.type.layout == explicit.type.layout


@pytest.mark.parametrize("shape", [(3, 64), (3, 3, 64), (3, 64, 2, 3, 64), (2, 64, 2, 2, 96)])
def test_physical_stage_support_does_not_relax_logical_or_trailing_dimensions(semantic, shape):
    with pytest.raises(RuntimeError, match="non-power-of-two|rank-2 payload"):
        tle.gpu.alloc(shape, tl.bfloat16, _semantic=semantic)
