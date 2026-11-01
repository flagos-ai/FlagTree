"""Declare ordered FP32 K partials followed by one BF16 publication."""

from triton.flagmega.ir import DType


def ordered_projection_parts(lhs, rhs, emit, *, name, parts):
    """RHS is [N, K]; each partial uses the original BF16 operands."""
    extent = lhs.type.shape[-1].fixed_value
    if parts < 1 or extent % parts:
        raise ValueError("Projection partitions must divide K")
    width = extent // parts
    partials = []
    for index in range(parts):
        begin, end = index * width, (index + 1) * width
        operands = tuple(emit("tensors.slice", f"{name}.part{index}.{label}", (value,), {
            "starts": (begin,), "ends": (end,), "axes": (1,), "steps": (1,),
        }) for label, value in (("input", lhs), ("weight", rhs)))
        partials.append(emit("math.matmul", f"{name}.part{index}.accumulator", operands, {
            "transpose_a": False, "transpose_b": True, "output_data_type": "float32",
            "reduction_order": "mma_k16",
        }))
    combined = partials[0]
    for index, partial in enumerate(partials[1:], 1):
        combined = emit("math.add", f"{name}.combine{index}", (combined, partial))
    return emit("tensors.cast", name, (combined,), {
        "dtype": DType.BFLOAT16.value, "rounding_contract": "preserve",
    })


def qwen35_native_projection_parts(lhs, rhs):
    # Pinned TP1 CUDA GEMM contract: the small-M 4096 -> 2048 GDN
    # output projection publishes two FP32 K=2048 partials before BF16.
    # The measured M=128/1024/2048 kernels accumulate the complete K.
    shape = tuple(d.fixed_value if d.is_fixed else None for d in lhs.type.shape)
    weight_shape = tuple(d.fixed_value if d.is_fixed else None for d in rhs.type.shape)
    return 2 if (lhs.type.dtype == rhs.type.dtype == DType.BFLOAT16
                 and shape in ((1, 4096), (32, 4096)) and weight_shape == (2048, 4096)) else 1
