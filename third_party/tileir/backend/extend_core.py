import re

from triton.language import core as tl

MLIR_DYNAMIC = -9223372036854775808


def _safe_mangle(prefix, text):
    return prefix + re.sub(r"[^0-9A-Za-z_]+", "_", str(text))


class mem_token_type(tl.dtype):

    def __init__(self):
        self.name = "mem_token"

    def to_ir(self, builder):
        return builder.get_tileir_mem_token_ty()

    def __str__(self):
        return self.name

    __repr__ = __str__

    def mangle(self):
        return "MTmem_token"

    def is_ptr(self):
        return False

    def __eq__(self, other):
        return isinstance(other, mem_token_type)


class tensor_view_type(tl.dtype):

    def __init__(self, element_ty, shape, stride):
        if not isinstance(element_ty, tl.dtype):
            raise TypeError(f"element_ty is a {type(element_ty).__name__}.")
        if len(shape) != len(stride):
            raise ValueError("tensor_view_type expects shape and stride to have the same rank")
        self.element_ty = element_ty
        self.shape = shape
        self.stride = stride
        shape_text = [x if x != MLIR_DYNAMIC else "?" for x in shape]
        stride_text = [x if x != MLIR_DYNAMIC else "?" for x in stride]
        if shape_text:
            self.name = f"tensor_view<{'x'.join(map(str, shape_text))}x{element_ty}, {stride_text}>"
        else:
            self.name = f"tensor_view<{element_ty}>"

    def to_ir(self, builder):
        return builder.get_tensor_view_ty(
            self.element_ty.to_ir(builder),
            self.shape,
            self.stride,
        )

    def __str__(self):
        return self.name

    __repr__ = __str__

    def mangle(self):
        return _safe_mangle("TV", self.name)

    def is_ptr(self):
        return True

    def __eq__(self, other):
        return (isinstance(other, tensor_view_type) and self.element_ty == other.element_ty
                and self.shape == other.shape and self.stride == other.stride)


class partition_view_type(tl.dtype):

    def __init__(self, tensor_view, element_ty, tile_shape, tile_dim_map):
        self.ir_type = None
        self.tensor_view = tensor_view
        self.element_ty = element_ty
        self.tile_shape = tile_shape
        self.tile_dim_map = tile_dim_map
        self.name = "partition_view"

    def to_ir(self, builder):
        if self.ir_type is None:
            raise RuntimeError("partition_view_type IR type is only available after make_partition_view")
        return self.ir_type

    def __str__(self):
        return self.name

    __repr__ = __str__

    def mangle(self):
        return _safe_mangle("PV", f"{self.tile_shape}_{self.tile_dim_map}_{self.element_ty}")

    def is_ptr(self):
        return False

    def __eq__(self, other):
        return (isinstance(other, partition_view_type) and self.tensor_view == other.tensor_view
                and self.element_ty == other.element_ty and self.tile_shape == other.tile_shape
                and self.tile_dim_map == other.tile_dim_map)


def _require_tileir_builder(_semantic, op_name):
    semantic_op_name = "create_mem_token" if op_name == "mem_token" else op_name
    if _semantic is None or not hasattr(_semantic, semantic_op_name):
        raise NotImplementedError(f"tl.ext.{op_name} is only supported by FlagTree's TileIR path; "
                                  "set FLAGTREE_USE_TILEIR=1 for kernels that only use TileIR TKO view ops")


def _unwrap_list(values):
    return [tl._unwrap_if_constexpr(x) for x in values]


@tl.builtin
def extract_slice(input, offsets, tiles, _semantic=None):
    """Compatibility shim for nv-Triton tl.ext.extract_slice.

    This currently supports the Ocean FFT pattern that slices one element from
    the last dimension of a tensor whose last dimension is size 2. It maps to
    Triton's existing split operation, which removes that last dimension.
    """
    offsets = [tl._unwrap_if_constexpr(x) for x in offsets]
    tiles = [tl._unwrap_if_constexpr(x) for x in tiles]
    input_shape = [tl._unwrap_if_constexpr(x) for x in input.shape]
    if len(offsets) != len(input_shape) or len(tiles) != len(input_shape):
        raise ValueError("tl.ext.extract_slice expects rank-matched offsets and tiles")
    if offsets[:-1] != [0] * (len(offsets) - 1):
        raise NotImplementedError("tl.ext.extract_slice only supports leading zero offsets")
    if tiles[:-1] != input_shape[:-1] or tiles[-1] != 1:
        raise NotImplementedError("tl.ext.extract_slice only supports full leading tiles and last tile 1")
    if input_shape[-1] != 2:
        raise NotImplementedError("tl.ext.extract_slice only supports last dimension size 2")
    if offsets[-1] == 0:
        return tl.split(input, _semantic=_semantic)[0]
    if offsets[-1] == 1:
        return tl.split(input, _semantic=_semantic)[1]
    raise NotImplementedError("tl.ext.extract_slice only supports last offset 0 or 1")


@tl.builtin
def cat(input, other, can_reorder=False, dim=-1, _semantic=None):
    """Compatibility shim for nv-Triton tl.ext.cat on the last dimension."""
    dim = tl._unwrap_if_constexpr(dim)
    if dim not in (-1, len(input.shape)):
        raise NotImplementedError("tl.ext.cat currently only supports appending a last dimension")
    return tl.join(input, other, _semantic=_semantic)


def _unsupported(name):

    def fn(*args, **kwargs):
        raise NotImplementedError(f"tl.ext.{name} is not supported by FlagTree's TileIR frontend yet")

    return fn


@tl.builtin
def make_tensor_view(base, shapes, strides, _semantic=None):
    _require_tileir_builder(_semantic, "make_tensor_view")
    if not base.dtype.is_ptr():
        raise TypeError("tl.ext.make_tensor_view base must be a pointer")

    shapes_ty = [x.value if isinstance(x, tl.constexpr) else MLIR_DYNAMIC for x in shapes]
    strides_ty = [x.value if isinstance(x, tl.constexpr) else MLIR_DYNAMIC for x in strides]
    shapes_val = [x for x in shapes if not isinstance(x, tl.constexpr)]
    strides_val = [x for x in strides if not isinstance(x, tl.constexpr)]
    res_type = tensor_view_type(base.dtype.element_ty, shapes_ty, strides_ty)
    return _semantic.make_tensor_view(res_type, base, shapes_val, strides_val)


@tl.builtin
def make_partition_view(src_tensor_view, tile_shape, tile_dim_map, pad_val="zero", _semantic=None):
    _require_tileir_builder(_semantic, "make_partition_view")
    pad_val = tl._unwrap_if_constexpr(pad_val)
    tile_shape = _unwrap_list(tile_shape)
    tile_dim_map = _unwrap_list(tile_dim_map)
    res_type = partition_view_type(
        src_tensor_view.handle,
        src_tensor_view.dtype.element_ty,
        tile_shape,
        tile_dim_map,
    )
    return _semantic.make_partition_view(src_tensor_view, tile_shape, tile_dim_map, res_type, pad_val)


@tl.builtin
def make_view(base, shapes, strides, tile_shape, tile_dim_map, traversal_strides=None, pad_val="zero", _semantic=None):
    if traversal_strides is not None:
        raise NotImplementedError("tl.ext.make_view traversal_strides is outside FlagTree's TKO-load MVP")
    if len(tile_shape) != len(tile_dim_map):
        raise ValueError("tl.ext.make_view expects tile_shape and tile_dim_map to have the same rank")
    if len(tile_dim_map) != len(shapes):
        raise ValueError("tl.ext.make_view expects every source dimension to appear in tile_dim_map")
    tile_dim_map = _unwrap_list(tile_dim_map)
    if len(set(tile_dim_map)) != len(shapes):
        raise ValueError("tl.ext.make_view expects every source dimension to appear exactly once")
    tensor_view = make_tensor_view(base, shapes, strides, _semantic=_semantic)
    return make_partition_view(tensor_view, tile_shape, tile_dim_map, pad_val, _semantic=_semantic)


@tl.builtin
def load_view_tko(
    src,
    coords,
    memToken=None,
    semantics="weak",
    scope="",
    has_result_token=False,
    cost=-1,
    _semantic=None,
):
    _require_tileir_builder(_semantic, "load_view_tko")
    has_result_token = bool(tl._unwrap_if_constexpr(has_result_token))
    return _semantic.load_view_tko(
        src,
        coords,
        memToken,
        semantics,
        scope,
        mem_token_type(),
        has_result_token,
        int(tl._unwrap_if_constexpr(cost)),
    )


@tl.builtin
def store_view_tko(
    dst,
    value,
    coords,
    memToken=None,
    semantics="weak",
    scope="",
    has_result_token=False,
    cost=-1,
    allow_tma=None,
    _semantic=None,
):
    _require_tileir_builder(_semantic, "store_view_tko")
    has_result_token = bool(tl._unwrap_if_constexpr(has_result_token))
    if allow_tma is not None:
        raise NotImplementedError("tl.ext.store_view_tko allow_tma is outside FlagTree's TKO-load MVP")
    value = _semantic.to_tensor(value)
    return _semantic.store_view_tko(
        dst,
        value,
        coords,
        memToken,
        semantics,
        scope,
        mem_token_type(),
        has_result_token,
        int(tl._unwrap_if_constexpr(cost)),
    )


@tl.builtin
def dim(src, index, _semantic=None):
    _require_tileir_builder(_semantic, "dim")
    index = tl._unwrap_if_constexpr(index)
    return _semantic.dim(src, index)


@tl.builtin
def create_mem_token(_semantic=None):
    _require_tileir_builder(_semantic, "mem_token")
    return tl.tensor(_semantic.create_mem_token(), mem_token_type())


@tl.builtin
def join_mem_tokens(*tokens, _semantic=None):
    _require_tileir_builder(_semantic, "join_mem_tokens")
    if len(tokens) == 0:
        return create_mem_token(_semantic=_semantic)
    result = tokens[0]
    if not isinstance(result.type, mem_token_type):
        raise TypeError(f"{result} is not a mem_token_type")
    for token in tokens[1:]:
        if not isinstance(token.type, mem_token_type):
            raise TypeError(f"{token} is not a mem_token_type")
        result = tl.tensor(
            _semantic.join_mem_tokens(result, token),
            mem_token_type(),
        )
    return result


load = _unsupported("load")
store = _unsupported("store")
atomic_cas = _unsupported("atomic_cas")
atomic_xchg = _unsupported("atomic_xchg")
tileir_tiled_atomic_rmw = _unsupported("tileir_tiled_atomic_rmw")
