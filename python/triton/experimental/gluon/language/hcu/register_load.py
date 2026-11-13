"""Explicit bounded structured-register loads. No shared-memory descriptor."""
from triton._C.libtriton import hcu
from triton.experimental.gluon.language import _core as core
from triton.experimental.gluon.language._core import builtin,_unwrap_if_constexpr
def _builder(semantic):
    return semantic.builder.get_op_builder(), semantic.builder.get_loc()


def _integer(semantic, value, dtype):
    value = semantic.to_tensor(_unwrap_if_constexpr(value))
    assert value.type.scalar.is_int()
    assert not value.type.is_block(), "extent/stride/origin must be scalar"
    return semantic.cast(value, dtype).handle


@builtin
def structured_register_load(base,shape,strides,indices,tile_shape,layout,_semantic=None):
    """Load a rank-two contiguous-row f16/bf16 window directly into registers.

    shape=(runtime_rows,constant_K), strides=(K,1), indices=(row_origin,
    col_origin,uniform_enable). K must be positive and2*K<0x4000.
    Negative/partial rows zero-fill; invalid complete K window or false enable
    returns zeros. The compiler independently verifies uniformity and every
    coordinate/vector group. Unsupported requests fail compilation.
    """
    base=_semantic.to_tensor(base)
    assert base.type.is_ptr() and base.type.scalar.element_ty in (core.float16,core.bfloat16)
    assert len(shape)==len(strides)==2 and len(indices)==3
    tile=tuple(_unwrap_if_constexpr(x)for x in _unwrap_if_constexpr(tile_shape))
    layout=_unwrap_if_constexpr(layout)
    assert len(tile)==2 and all(type(x)is int and x>0 for x in tile)
    ty=core.distributed_type(base.type.scalar.element_ty,list(tile),layout)
    builder,loc=_builder(_semantic)
    v=hcu.create_structured_register_load(builder,loc,base.handle,ty.to_ir(_semantic.builder),
        [_integer(_semantic,x,core.int32)for x in shape],
        [_integer(_semantic,x,core.int64)for x in strides],
        [_integer(_semantic,x,core.int32)for x in indices])
    return core.tensor(v,ty)
