import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler.errors import CompilationError
from triton.experimental.tle.language.gpu.iluvatar import layout as iluvatar_layout

from utils import compile_iluvatar


@triton.jit(noinline=True)
def _consume_alloc(buf, out_ptr):
    tl.store(out_ptr, 0.0)


@triton.jit
def _alloc_tcu_f16_kernel(out_ptr):
    buf = tle.gpu.alloc((32, 32), dtype=tl.float16, nv_mma_shared_layout=True)
    _consume_alloc(buf, out_ptr)


@triton.jit
def _alloc_default_tcu_f16_kernel(out_ptr):
    buf = tle.gpu.alloc((32, 32), dtype=tl.float16)
    _consume_alloc(buf, out_ptr)


@triton.jit
def _alloc_tcu_f32_kernel(out_ptr):
    buf = tle.gpu.alloc((32, 16), dtype=tl.float32, nv_mma_shared_layout=True)
    _consume_alloc(buf, out_ptr)


@triton.jit
def _alloc_tcu_slot_f16_kernel(out_ptr):
    buf = tle.gpu.alloc((2, 32, 32), dtype=tl.float16, nv_mma_shared_layout=True)
    _consume_alloc(buf.slot(0), out_ptr)


@triton.jit
def _alloc_true_1d_kernel(out_ptr):
    buf = tle.gpu.alloc((16, ), dtype=tl.float32, nv_mma_shared_layout=True)
    _consume_alloc(buf, out_ptr)


@triton.jit
def _alloc_true_int32_kernel(out_ptr):
    buf = tle.gpu.alloc((32, 16), dtype=tl.int32, nv_mma_shared_layout=True)
    _consume_alloc(buf, out_ptr)


@triton.jit
def _alloc_true_small_contig_kernel(out_ptr):
    buf = tle.gpu.alloc((32, 8), dtype=tl.float32, nv_mma_shared_layout=True)
    _consume_alloc(buf, out_ptr)


@triton.jit
def _alloc_false_2d_kernel(out_ptr):
    buf = tle.gpu.alloc((32, 32), dtype=tl.float16, nv_mma_shared_layout=False)
    _consume_alloc(buf, out_ptr)


@triton.jit
def _alloc_explicit_layout_kernel(out_ptr, LAYOUT: tl.constexpr):
    buf = tle.gpu.alloc((32, 32), dtype=tl.float16, layout=LAYOUT)
    _consume_alloc(buf, out_ptr)


def _assert_swizzled(ttgir, *, use_tcu, vec=None):
    assert "ttg.local_alloc" in ttgir, ttgir
    assert "#ttg.swizzled_shared" in ttgir, ttgir
    assert "#ttg.nvmma_shared" not in ttgir, ttgir
    if use_tcu:
        assert "useTcu = true" in ttgir, ttgir
        if vec is not None:
            assert f"vec = {vec}" in ttgir or f"vec={vec}" in ttgir, ttgir
    else:
        assert "useTcu = true" not in ttgir, ttgir


def test_is_tcu_eligible_gates():
    assert iluvatar_layout.is_tcu_eligible((32, 32), tl.float16)
    assert iluvatar_layout.is_tcu_eligible((32, 16), tl.float32)
    assert iluvatar_layout.is_tcu_eligible((2, 32, 32), tl.float16)
    assert iluvatar_layout.is_tcu_eligible((64, 64), tl.int8)
    assert not iluvatar_layout.is_tcu_eligible((16, ), tl.float32)
    assert not iluvatar_layout.is_tcu_eligible((32, 16), tl.int32)
    assert not iluvatar_layout.is_tcu_eligible((32, 8), tl.float32)
    assert not iluvatar_layout.is_tcu_eligible((32, 16), tl.float16)


def test_col_major_tcu_layout():
    assert iluvatar_layout.is_tcu_eligible((16, 32), tl.float32, col_major=True)
    assert not iluvatar_layout.is_tcu_eligible((8, 32), tl.float32, col_major=True)

    row = iluvatar_layout.make_tcu_swizzled_layout((32, 32), tl.float16)
    col = iluvatar_layout.make_tcu_swizzled_layout((32, 32), tl.float16, col_major=True)
    assert list(row.order) == [1, 0] and list(col.order) == [0, 1]
    assert col.use_tcu and col.vectorSize == row.vectorSize
    assert list(iluvatar_layout.make_tcu_swizzled_layout((2, 32, 32), tl.float16, col_major=True).order) == [1, 2, 0]


def test_trunk_swizzled_layout_stays_backend_agnostic():
    import inspect
    from triton.experimental.tle.language.gpu import types as tle_types
    trunk = tle_types.swizzled_shared_layout
    assert "use_tcu" not in inspect.signature(trunk.__init__).parameters
    assert not getattr(trunk.make_default(2), "use_tcu", False)


def test_slot_and_permute_preserve_use_tcu():
    from triton.experimental.tle.language.gpu import types as tle_types
    tcu = iluvatar_layout.IluvatarTcuSwizzledSharedLayout
    layout = iluvatar_layout.make_tcu_swizzled_layout((2, 32, 16), tl.float32)
    assert layout.use_tcu and layout.vectorSize == 16

    slotted = tle_types._make_slot_layout(layout, [32, 16])
    assert isinstance(slotted, tcu) and slotted.use_tcu

    permuted = layout.make_permute((0, 2, 1))
    assert isinstance(permuted, tcu) and permuted.use_tcu

    generic = tle_types.swizzled_shared_layout.make_default(3)
    assert not getattr(tle_types._make_slot_layout(generic, [32, 16]), "use_tcu", False)


def test_slot_of_tcu_buffer_keeps_usetcu_in_ttgir():
    compiled = compile_iluvatar(_alloc_tcu_slot_f16_kernel, signature={"out_ptr": "*fp32"})
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_index" in ttgir, ttgir
    # Both the buffer and the sliced view must carry the TCU encoding.
    assert ttgir.count("useTcu = true") >= 2, ttgir


def test_alloc_true_eligible_f16_emits_usetcu():
    compiled = compile_iluvatar(_alloc_tcu_f16_kernel, signature={"out_ptr": "*fp32"})
    _assert_swizzled(compiled.asm["ttgir"], use_tcu=True, vec=32)


def test_alloc_default_eligible_f16_emits_usetcu():
    compiled = compile_iluvatar(_alloc_default_tcu_f16_kernel, signature={"out_ptr": "*fp32"})
    _assert_swizzled(compiled.asm["ttgir"], use_tcu=True, vec=32)


def test_alloc_true_eligible_f32_emits_usetcu():
    compiled = compile_iluvatar(_alloc_tcu_f32_kernel, signature={"out_ptr": "*fp32"})
    _assert_swizzled(compiled.asm["ttgir"], use_tcu=True, vec=16)


def test_alloc_true_1d_falls_back_to_generic_swizzled():
    compiled = compile_iluvatar(_alloc_true_1d_kernel, signature={"out_ptr": "*fp32"})
    _assert_swizzled(compiled.asm["ttgir"], use_tcu=False)


def test_alloc_true_int32_falls_back_to_generic_swizzled():
    compiled = compile_iluvatar(_alloc_true_int32_kernel, signature={"out_ptr": "*fp32"})
    _assert_swizzled(compiled.asm["ttgir"], use_tcu=False)


def test_alloc_true_small_contig_falls_back_to_generic_swizzled():
    compiled = compile_iluvatar(_alloc_true_small_contig_kernel, signature={"out_ptr": "*fp32"})
    _assert_swizzled(compiled.asm["ttgir"], use_tcu=False)


def test_alloc_false_stays_generic_swizzled():
    compiled = compile_iluvatar(_alloc_false_2d_kernel, signature={"out_ptr": "*fp32"})
    _assert_swizzled(compiled.asm["ttgir"], use_tcu=False)


def test_alloc_explicit_nv_mma_shared_layout_raises():
    layout = tle.gpu.nv_mma_shared_layout.make_default((32, 32), tl.float16)
    with pytest.raises(CompilationError, match="iluvatar TLE alloc does not support nv_mma_shared_layout=True"):
        compile_iluvatar(_alloc_explicit_layout_kernel, signature={"out_ptr": "*fp32", "LAYOUT": "constexpr"},
                         constexprs={"LAYOUT": layout})
