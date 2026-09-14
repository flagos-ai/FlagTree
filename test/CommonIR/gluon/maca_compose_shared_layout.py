"""Python model of MACAMmaEncodingAttr::composeSharedLayoutForOperand.

This helper mirrors the C500 implementation in
lib/Dialect/TritonGPU/IR/Dialect.cpp.  It is a constexpr function so a Gluon
kernel can derive a SwizzledSharedLayout from the complete MMA component
instead of keeping a hand-written swizzle that becomes stale when
elements_mnk changes.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.constexpr_function
def compose_maca_shared_layout_for_operand(
    elements_mnk,
    operand_index,
    operand_shape,
    shared_order,
    type_width_in_bits,
    need_trans=False,
    is_a_trans=False,
    is_b_trans=False,
    elements_stride=(1, 1),
    disable_swizzle=False,
):
    """Return the C500 shared layout composed for one MACA dot operand.

    The arguments correspond to the fields consumed by the C++
    ``MACAMmaEncodingAttr::composeSharedLayoutForOperand`` implementation.
    CTA layout is omitted because the tutorial kernels use ``num_ctas=1``.
    """

    # Gluon can represent a local constexpr sequence with its own aggregate
    # type, so normalize each element rather than relying on list/tuple type
    # checks. ``constexpr.__index__`` makes ``int`` valid for both wrapped and
    # ordinary compile-time integers.
    elements_mnk = [int(value) for value in elements_mnk]
    operand_index = int(operand_index)
    operand_shape = [int(value) for value in operand_shape]
    shared_order = [int(value) for value in shared_order]
    type_width_in_bits = int(type_width_in_bits)
    elements_stride = [int(value) for value in elements_stride]

    assert operand_index == 0 or operand_index == 1
    assert len(elements_mnk) == 3
    assert len(operand_shape) == 2
    assert len(shared_order) == 2
    assert len(elements_stride) == 2
    assert type_width_in_bits > 0 and type_width_in_bits % 8 == 0

    if disable_swizzle:
        return gl.SwizzledSharedLayout(
            vec=1,
            per_phase=1,
            max_phase=1,
            order=shared_order,
        )

    elt_byte = type_width_in_bits // 8
    per_phase = 128 // (operand_shape[shared_order[0]] * elt_byte)
    per_phase = max(per_phase, 1)

    tm, tn, tk = elements_mnk
    enable_lds_trans = is_a_trans if operand_index == 0 else is_b_trans
    mma_thread_shape = (4, 4, 16) if enable_lds_trans else (16, 16, 4)

    if operand_index == 0:
        if is_a_trans:
            stride = elements_stride[0]
            assert tk >= stride and tk % stride == 0
            tm = tm * stride
            tk = tk // stride

        shape_m = operand_shape[1] if need_trans else operand_shape[0]
        shape_k = operand_shape[0] if need_trans else operand_shape[1]
        assert shape_m >= mma_thread_shape[0] * tm
        assert shape_k >= mma_thread_shape[2] * tk
        assert shape_m % (mma_thread_shape[0] * tm) == 0
        assert shape_k % (mma_thread_shape[2] * tk) == 0

        is_at = (
            (not need_trans and shared_order[0] == 1)
            or (need_trans and shared_order[0] == 0)
        )
        is_an = (
            (not need_trans and shared_order[0] == 0)
            or (need_trans and shared_order[0] == 1)
        )

        if is_at:
            assert not is_a_trans
            per_phase = max(tm, per_phase)
            mul = max(per_phase // tm, 1)
            vec = max(128 // (16 * elt_byte), tk)
            max_phase = max(128 // (vec * elt_byte) // mul, 8 // mul)
            max_phase = min(max_phase, shape_k // vec)
            if max_phase == 1:
                vec = tk
            return gl.SwizzledSharedLayout(
                vec=vec,
                per_phase=per_phase,
                max_phase=max_phase,
                order=shared_order,
            )

        assert is_an
        per_phase = max(tk, per_phase)
        mul = max(per_phase // tk, 1)
        vec = mma_thread_shape[0] * tm
        max_phase = max(128 // (vec * elt_byte) // mul, 1)
        max_phase = min(max_phase, shape_m // vec)
        if max_phase == 1:
            vec = tm
        return gl.SwizzledSharedLayout(
            vec=vec,
            per_phase=per_phase,
            max_phase=max_phase,
            order=shared_order,
        )

    if is_b_trans:
        stride = elements_stride[1]
        assert tk >= stride and tk % stride == 0
        tk = tk // stride
        tn = tn * stride

    shape_k = operand_shape[1] if need_trans else operand_shape[0]
    shape_n = operand_shape[0] if need_trans else operand_shape[1]
    assert shape_k >= mma_thread_shape[2] * tk
    assert shape_n >= mma_thread_shape[1] * tn
    assert shape_k % (mma_thread_shape[2] * tk) == 0
    assert shape_n % (mma_thread_shape[1] * tn) == 0

    is_bt = (
        (not need_trans and shared_order[0] == 1)
        or (need_trans and shared_order[0] == 0)
    )
    is_bn = (
        (not need_trans and shared_order[0] == 0)
        or (need_trans and shared_order[0] == 1)
    )

    if is_bn:
        assert not is_b_trans
        per_phase = max(tn, per_phase)
        mul = max(per_phase // tn, 1)
        vec = max(128 // (16 * elt_byte), tk)
        max_phase = max(128 // (vec * elt_byte) // mul, 8 // mul)
        max_phase = min(max_phase, shape_k // vec)
        if max_phase == 1:
            vec = tk
        return gl.SwizzledSharedLayout(
            vec=vec,
            per_phase=per_phase,
            max_phase=max_phase,
            order=shared_order,
        )

    assert is_bt
    per_phase = max(tk, per_phase)
    mul = max(per_phase // tk, 1)
    vec = mma_thread_shape[1] * tn
    max_phase = max(128 // (vec * elt_byte) // mul, 1)
    max_phase = min(max_phase, shape_n // vec)
    if max_phase == 1:
        vec = tn
    return gl.SwizzledSharedLayout(
        vec=vec,
        per_phase=per_phase,
        max_phase=max_phase,
        order=shared_order,
    )
