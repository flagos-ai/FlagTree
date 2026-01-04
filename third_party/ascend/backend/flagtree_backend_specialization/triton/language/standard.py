from math import pi as math_pi
from triton.language import core, math
from triton.language.core import float32, int1, int32
from triton.runtime.jit import jit


@core._tensor_member_fn
@jit
@math._add_math_1arg_docstr("sigmoid")
def sigmoid(x):
    _is_int8_type: core.constexpr = x.dtype.is_int8()
    core.static_assert(not _is_int8_type, f"Expected dtype fp16/fp32/bf16, but got int8 or int1")
    _is_floating_type: core.constexpr = x.dtype.is_floating()
    core.static_assert(_is_floating_type == True, f"Expected dtype fp16/fp32/bf16, but got {core.constexpr(x.dtype)}")
    return (1 / (1 + math.exp(-x.to(core.float32)))).to(x.dtype)

@core._tensor_member_fn
@jit
@math._add_math_1arg_docstr("softmax")
def softmax(x, ieee_rounding=False):
    _is_int8_type: core.constexpr = x.dtype.is_int8()
    core.static_assert(not _is_int8_type, f"Expected dtype fp16/fp32/bf16, but got int8 or int1")
    _is_floating_type: core.constexpr = x.dtype.is_floating()
    core.static_assert(_is_floating_type == True, f"Expected dtype fp16/fp32/bf16, but got {core.constexpr(x.dtype)}")
    z = x.to(core.float32) - max(x, 0)
    num = math.exp(z)
    den = sum(num, 0)
    return math.fdiv(num, den, ieee_rounding).to(x.dtype)

@core._tensor_member_fn
@jit
@math._add_math_1arg_docstr("isfinited")
def isfinited(x):
    _is_int8_type: core.constexpr = x.dtype.is_int8()
    core.static_assert(not _is_int8_type, f"Expected dtype fp16/fp32/bf16, but got int8 or int1")
    _is_floating_type: core.constexpr = x.dtype.is_floating()
    core.static_assert(_is_floating_type == True, f"Expected dtype fp16/fp32/bf16, but got {core.constexpr(x.dtype)}")
    nan_mask = math.isnan(x)
    inf_mask = math.isinf(x)
    return (~nan_mask & ~inf_mask).to(int1)

@core._tensor_member_fn
@jit
@math._add_math_1arg_docstr("finitef")
def finitef(x):
    _is_int8_type: core.constexpr = x.dtype.is_int8()
    core.static_assert(not _is_int8_type, f"finitef only supports float32, but got int8 or int1")
    core.static_assert(x.dtype == float32, f"finitef only supports float32, but got {core.constexpr(x.dtype)}")
    nan_mask = math.isnan(x)
    inf_mask = math.isinf(x)
    return (~nan_mask & ~inf_mask).to(int1)

@core._tensor_member_fn
@jit
@math._add_math_1arg_docstr("rint")
def rint(x):
    _is_int8_type: core.constexpr = x.dtype.is_int8()
    core.static_assert(not _is_int8_type, f"Expected dtype fp16/fp32/bf16, but got int8 or int1")
    _is_floating_type: core.constexpr = x.dtype.is_floating()
    core.static_assert(_is_floating_type == True, f"Expected dtype fp16/fp32/bf16, but got {core.constexpr(x.dtype)}")
    # Calculate integer part and fractional part
    floor_x = math.floor(x)
    fractional = x - floor_x
    # Check if fractional part is close to 0.5
    is_half = math.abs(fractional - 0.5) < 1e-8
    # Check if integer part is even
    floor_int = floor_x.to(int32)
    is_even = (floor_int % 2) == 0
    # Apply bankers rounding rules:
    # - If fractional part is 0.5: keep integer part if even, add 1 if odd
    # - Otherwise: round to the nearest integer directly
    return core.where(
        is_half,
        core.where(is_even, floor_x, floor_x + 1.0),
        core.where(x >= 0, math.floor(x + 0.5), math.ceil(x - 0.5))
    )

pi: core.constexpr = math_pi

@core._tensor_member_fn
@jit
@math._add_math_2arg_docstr("atan2")
def atan2(y, x):
    _is_int8_type_x: core.constexpr = x.dtype.is_int8()
    core.static_assert(not _is_int8_type_x, f"Expected dtype fp16/fp32/bf16, but got int8 or int1")
    _is_int8_type_y: core.constexpr = y.dtype.is_int8()
    core.static_assert(not _is_int8_type_y, f"Expected dtype fp16/fp32/bf16, but got int8 or int1")
    _is_floating_type_x: core.constexpr = x.dtype.is_floating()
    core.static_assert(_is_floating_type_x == True, f"Expected dtype fp16/fp32/bf16, but got {core.constexpr(x.dtype)}")
    _is_floating_type_y: core.constexpr = y.dtype.is_floating()
    core.static_assert(_is_floating_type_y == True, f"Expected dtype fp16/fp32/bf16, but got {core.constexpr(y.dtype)}")
    half_pi: core.constexpr = 0.5 * pi
    base = core.where(x == 0, 0.0, math.atan(y.to(float32) / x.to(float32)))
    base = core.where((x == 0) & (y > 0), half_pi, base)
    base = core.where((x == 0) & (y < 0), -half_pi, base)

    add_pi = core.where((x < 0) & (y >= 0), pi, 0.0)
    sub_pi = core.where((x < 0) & (y < 0), -pi, 0.0)
    return (base + add_pi + sub_pi).to(x.dtype)

@jit
def _argmax_combine(value1, index1, value2, index2, tie_break_left):
    if tie_break_left:
        tie = value1 == value2 and index1 < index2
    else:
        tie = value1 == value2 and index1 > index2
    gt = value1 > value2 or tie
    v_ret = core.where(gt, value1, value2)
    i_ret = core.where(gt, index1, index2)
    return v_ret, i_ret


@jit
def _argmax_combine_tie_break_left(value1, index1, value2, index2):
    return _argmax_combine(value1, index1, value2, index2, True)


@jit
def _argmax_combine_tie_break_fast(value1, index1, value2, index2):
    return _argmax_combine(value1, index1, value2, index2, False)


@jit
def _elementwise_max_default(a, b):
    return core.maximum(a, b)


@jit
def _elementwise_max_propagate_nan(a, b):
    return core.maximum(a, b, propagate_nan = core.PropagateNan.ALL)


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("maximum", return_indices_arg="return_indices",
                            tie_break_arg="return_indices_tie_break_left")
def max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False, propagate_nan = False):
    input = core._promote_bfloat16_to_float32(input)
    if return_indices:
        if return_indices_tie_break_left:
            return core._reduce_with_indices(input, axis, _argmax_combine_tie_break_left, keep_dims=keep_dims)
        else:
            return core._reduce_with_indices(input, axis, _argmax_combine_tie_break_fast, keep_dims=keep_dims)
    else:
        if core.constexpr(input.dtype.primitive_bitwidth) < core.constexpr(32):
            if core.constexpr(input.dtype.is_floating()):
                input = input.to(core.float32)
            else:
                assert input.dtype.is_int(), "Expecting input to be integer type"
        if not propagate_nan:
            return core.reduce(input, axis, _elementwise_max_default, keep_dims=keep_dims)
        else:
            return core.reduce(input, axis, _elementwise_max_propagate_nan, keep_dims=keep_dims)


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("maximum index", tie_break_arg="tie_break_left")
def argmax(input, axis, tie_break_left=True, keep_dims=False):
    (_, ret) = max(input, axis, return_indices=True, return_indices_tie_break_left=tie_break_left, keep_dims=keep_dims)
    return ret

@jit
def _argmin_combine(value1, index1, value2, index2, tie_break_left):
    if tie_break_left:
        tie = value1 == value2 and index1 < index2
    else:
        tie = value1 == value2 and index1 > index2
    lt = value1 < value2 or tie
    value_ret = core.where(lt, value1, value2)
    index_ret = core.where(lt, index1, index2)
    return value_ret, index_ret


@jit
def _argmin_combine_tie_break_left(value1, index1, value2, index2):
    return _argmin_combine(value1, index1, value2, index2, True)


@jit
def _argmin_combine_tie_break_fast(value1, index1, value2, index2):
    return _argmin_combine(value1, index1, value2, index2, False)


@jit
def _elementwise_min(a, b):
    return core.minimum(a, b)


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("minimum", return_indices_arg="return_indices",
                            tie_break_arg="return_indices_tie_break_left")
def min(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False):
    input = core._promote_bfloat16_to_float32(input)
    if return_indices:
        if return_indices_tie_break_left:
            return core._reduce_with_indices(input, axis, _argmin_combine_tie_break_left, keep_dims=keep_dims)
        else:
            return core._reduce_with_indices(input, axis, _argmin_combine_tie_break_fast, keep_dims=keep_dims)
    else:
        if core.constexpr(input.dtype.primitive_bitwidth) < 32:
            if core.constexpr(input.dtype.is_floating()):
                input = input.to(core.float32)
            else:
                assert input.dtype.is_int(), "Expecting input to be integer type"
        return core.reduce(input, axis, _elementwise_min, keep_dims=keep_dims)


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("minimum index", tie_break_arg="tie_break_left")
def argmin(input, axis, tie_break_left=True, keep_dims=False):
    _, ret = min(input, axis, return_indices=True, return_indices_tie_break_left=tie_break_left, keep_dims=keep_dims)
    return ret


@jit
def _xor_combine(a, b):
    return a ^ b

# xor sum

@core._tensor_member_fn
@jit
@core._add_reduction_docstr("xor sum")
def xor_sum(x, axis=None, keep_dims=False):
    core.static_assert(x.type.scalar.is_int(), "xor_sum only supported for integers")
    return core.reduce(x, axis, _xor_combine, keep_dims=keep_dims)

# sort

@jit
def _indicator(n_dims: core.constexpr, j: core.constexpr):
    ar = core.arange(0, 2)
    ar = core.reshape(ar, [1] * (n_dims - j - 1) + [2] + [1] * j)
    return ar


@jit
def _compare_and_swap(x, flip_dim, i: core.constexpr):
    # compare-and-swap on the ith *innermost* dimension
    n_dims: core.constexpr = _log2(x.numel)

    # flip along middle dimension (the bitwise XORs will be optimised away):
    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ix = x.to(idtype, bitcast=True)
    iy = ix ^ xor_sum(ix, n_dims - 1 - i, True)
    y = iy.to(x.dtype, bitcast=True)

    # determines whether we are in the right (rather than left) position along the axis:
    is_right = _indicator(n_dims, i)

    # conditional swap:
    ret = core.where((x > y) != (flip_dim ^ is_right), y, x)
    return ret


@jit
def _bitonic_merge_hypercube(x, stage: core.constexpr, order: core.constexpr):
    '''
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    '''
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        flip_dim = _indicator(_log2(x.numel), stage)
    else:
        flip_dim = order
    # perform `stage` rounds of `compare-and-swap`
    for i in core.static_range(stage):
        x = _compare_and_swap(x, flip_dim, stage - 1 - i)
    return x


@jit
def sort_impl(x, k: core.constexpr = None, dim: core.constexpr = None, descending: core.constexpr = core.CONSTEXPR_0):
    """
    Sorts a tensor along a specified dimension.

    :param x: The input tensor to be sorted.
    :type x: Tensor
    :param dim: The dimension along which to sort the tensor. If None, the tensor is sorted along the last dimension. Currently, only sorting along the last dimension is supported.
    :type dim: int, optional
    :param k: the number of top elements to select. If none, assume k = x.shape[dim]
    :type k: int, optional
    :param descending: If set to True, the tensor is sorted in descending order. If set to False, the tensor is sorted in ascending order.
    :type descending: bool, optional
    """
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = len(x.shape) - 1 if dim is None else dim
    core.static_assert(_dim == len(x.shape) - 1, "only minor dimension is currently supported")

    log_n: core.constexpr = _log2(x.shape[_dim])
    log_k: core.constexpr = log_n if k is None else _log2(k)

    n_dims: core.constexpr = _log2(x.numel)

    # reshape to hypercube:
    h = core.reshape(x, [2] * n_dims)

    # run first log_k bitonic sort iterations:
    for i in core.static_range(1, log_k + 1):
        h = _bitonic_merge_hypercube(h, i, 2 if i < log_n else descending)

    # select top k elements using bitonic top-k
    # https://www.doc.ic.ac.uk/~hlgr/pdfs/MassivelyParallelTopK.pdf
    for i in core.static_range(log_k + 1, log_n + 1):
        h = real_max(h, axis=(_log2(h.numel) - 1 - log_k)) if descending else min(h, axis=(_log2(h.numel) - 1 - log_k))
        h = _bitonic_merge_hypercube(h, log_k, 2 if i < log_n else descending)

    # reshape back:
    x = core.reshape(h, x.shape[:-1] + [2**log_k])
    return x


@jit
def topk(x, k: core.constexpr, dim: core.constexpr = None):
    return sort_impl(x, k=k, dim=dim, descending=True)

standard_ext_spec_func_list = ["sigmoid", "softmax", "isfinited", "finitef", "rint", "atan2", "argmax", "argmin",
                               "topk", "max"]
