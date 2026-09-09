# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Expose the native GDN prefill numerical stages as ordinary editable IR."""

from triton.flagmega.ir import DType, tensor_type
from triton.flagmega.ir.ops.nn.gdn_recurrent_core import GatedDeltaNetRecurrentCore as Recurrent


def emit_gdn_prefill(source, inputs, emit):
    """Replace a recurrent frontend node; emit is a typed node constructor."""
    prefix, attrs = source.id, source.attrs
    qkv, z, projected, state = (parameter.read(inputs) for parameter in (Recurrent.qkv, Recurrent.z,
                                                                         Recurrent.projection_input, Recurrent.state))
    tokens = qkv.type.shape[0].fixed_value
    key_heads, value_heads = attrs["num_key_heads"], attrs["num_value_heads"]
    key_dim, value_dim = attrs["key_head_dim"], attrs["value_head_dim"]

    def make(op, suffix, operands=(), attributes=None):
        return emit(op, prefix + "." + suffix, operands, attributes or {})

    def cast(value, dtype, suffix):
        return value if value.type.dtype == dtype else make("tensors.cast", suffix, (value, ), {"dtype": dtype.value})

    def reshape(value, shape, suffix):
        return make("tensors.reshape", suffix, (value, ), {"shape": shape})

    projections = tuple(
        make("math.matmul", name, (projected, parameter.read(inputs)), {"transpose_b": True})
        for name, parameter in (("a", Recurrent.a_weight), ("b", Recurrent.b_weight)))
    gates = make("nn.delta_rule_gates", "gates",
                 (*projections, Recurrent.a_log.read(inputs), Recurrent.dt_bias.read(inputs)),
                 {"alpha_exp_mode": "accurate", "softplus_threshold": 20.})
    alpha, beta = tuple(
        make("builtin.get_item", name, (gates, ), {"index": index}) for index, name in enumerate(("alpha", "beta")))
    qkv_parts = []
    start = 0
    for name, heads, dimension in (("query", key_heads, key_dim), ("key", key_heads, key_dim), ("value", value_heads,
                                                                                                value_dim)):
        end = start + heads * dimension
        part = make("tensors.slice", name + "_slice", (qkv, ), {"starts": (start, ), "ends": (end, ), "axes": (1, )})
        part = reshape(part, (tokens, heads, dimension), name + "_heads")
        if name != "value":
            part = make(
                "nn.l2_normalization", name, (part, ),
                {"axes": (-1, ), "epsilon": 1e-6, "epsilon_mode": "add", "division_mode": "reciprocal_multiply"})
        qkv_parts.append(part)
        start = end
    query, key, value = qkv_parts
    coefficients = make("nn.delta_rule_coefficients", "coefficients", (key, beta), {"block_size": 64})
    prefix_log = make("nn.delta_rule_log_prefix", "log_prefix", (alpha, ),
                      {"block_size": 64, "scan_group_size": 32, "epsilon": 1e-10, "log2_mode": "fast"})
    update = make(
        "nn.delta_rule_block_update", "block_update", (query, key, value, coefficients, prefix_log, state), {
            "scale": key_dim**-.5, "state_field": "recurrent", "state_layout":
            ("layer", "head", "value", "key"), "state_vector_axes": ("key", )
        })
    core, updated = tuple(
        make("builtin.get_item", name, (update, ), {"index": index}) for index, name in enumerate(("core", "state")))
    # Native stores the core and runtime norm weight as BF16, but performs
    # RMS normalization and the following SiLU(z) product in FP32.
    weight = cast(Recurrent.norm_weight.read(inputs), DType.BFLOAT16, "norm_weight_bf16")
    weight = cast(weight, DType.FLOAT32, "norm_weight_wide")
    bias = make("builtin.splat_const", "norm_zero",
                attributes={"result_type": tensor_type("float32", (value_dim, )), "value": 0.})
    stats = make("nn.norm_stats", "norm_stats", (core, ), {"axis": -1, "use_mean": False})
    normalized = make("nn.norm_apply", "norm_apply", (core, stats, weight, bias), {
        "axis": -1, "epsilon": attrs["epsilon"], "use_mean": False, "round_before_scale": False, "output_dtype":
        "float32"
    })
    z = reshape(cast(z, DType.FLOAT32, "z_wide"), (tokens, value_heads, value_dim), "z_heads")
    gate = make("math.silu", "z_gate", (z, ))
    output = cast(make("math.mul", "gated_wide", (normalized, gate)), DType.BFLOAT16, "gated")
    output = reshape(output, (tokens, value_heads * value_dim), "output")
    return emit("builtin.tuple", prefix, (output, updated), {})
