# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Statically declared functional patterns for FlagMega operations."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from triton.flagmega.ir.model import DType, Effect, IRType, Node
from triton.flagmega.ir.ops.builtin.call import Call as BuiltinCall
from triton.flagmega.ir.ops.builtin.get_item import GetItem
from triton.flagmega.ir.ops.builtin.none import NoneValue
from triton.flagmega.ir.ops.builtin.tuple import TupleValue
from triton.flagmega.ir.ops.builtin.const_asset import ConstAsset
from triton.flagmega.ir.ops.builtin.scalar_const import ScalarConst as BuiltinScalarConst
from triton.flagmega.ir.ops.builtin.splat_const import SplatConst
from triton.flagmega.ir.ops.distributed.boxing import Boxing
from triton.flagmega.ir.ops.distributed.force_boxing import ForceBoxing
from triton.flagmega.ir.ops.distributed.materialize_local_shards import (
    MaterializeLocalShards,
)
from triton.flagmega.ir.ops.distributed.sharded_view import ShardedView
from triton.flagmega.ir.ops.math.add import Add
from triton.flagmega.ir.ops.math.div import Div
from triton.flagmega.ir.ops.math.sigmoid import Sigmoid
from triton.flagmega.ir.ops.math.reduce_sum import ReduceSum
from triton.flagmega.ir.ops.nn.softmax import Softmax
from triton.flagmega.ir.ops.tensors.broadcast_to import BroadcastTo
from triton.flagmega.ir.ops.tensors.slice import Slice
from triton.flagmega.ir.ops.tensors.top_k import TopK
from triton.flagmega.ir.ops.math.block_scaled_matmul import BlockScaledMatMul
from triton.flagmega.ir.ops.math.matmul import MatMul
from triton.flagmega.ir.ops.math.mul import Mul
from triton.flagmega.ir.ops.math.packed_block_scaled_matmul import PackedBlockScaledMatMul
from triton.flagmega.ir.ops.math.packed_dense_matmul import PackedDenseMatMul
from triton.flagmega.ir.ops.math.silu import Silu
from triton.flagmega.ir.ops.math.vectorized_binary import VectorizedBinary
from triton.flagmega.ir.ops.math.vectorized_matmul import VectorizedMatMul
from triton.flagmega.ir.ops.math.vectorized_unary import VectorizedUnary
from triton.flagmega.ir.ops.nn.embedding import Embedding
from triton.flagmega.ir.ops.nn.greedy_sample import GreedySample
from triton.flagmega.ir.ops.nn.sparse_experts import SparseExperts
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.ir.ops.nn.dense_matmul_glu import DenseMatMulGlu
from triton.flagmega.ir.ops.nn.gated_delta_net import GatedDeltaNet
from triton.flagmega.ir.ops.nn.gdn_convolution import GatedDeltaNetConvolution
from triton.flagmega.ir.ops.nn.gdn_recurrent_core import GatedDeltaNetRecurrentCore
from triton.flagmega.ir.ops.nn.delta_rule_coefficients import DeltaRuleCoefficients
from triton.flagmega.ir.ops.nn.delta_rule_log_prefix import DeltaRuleLogPrefix
from triton.flagmega.ir.ops.nn.delta_rule_block_update import DeltaRuleBlockUpdate
from triton.flagmega.ir.ops.nn.delta_rule_gates import DeltaRuleGates
from triton.flagmega.ir.ops.nn.l2_normalization import L2Normalization
from triton.flagmega.ir.ops.nn.gdn_state_slice import GatedDeltaNetStateSlice
from triton.flagmega.ir.ops.nn.matmul_glu import MatMulGlu
from triton.flagmega.ir.ops.nn.packed_matmul_glu import PackedMatMulGlu
from triton.flagmega.ir.ops.nn.packed_dense_matmul_glu import PackedDenseMatMulGlu
from triton.flagmega.ir.ops.nn.packed_qwen3_paged_attention import PackedQwen3PagedAttention
from triton.flagmega.ir.ops.nn.rms_norm import RMSNorm
from triton.flagmega.ir.ops.nn.qkv_parallel_linear import QKVParallelLinear
from triton.flagmega.ir.ops.nn.qwen3_paged_attention import Qwen3PagedAttention
from triton.flagmega.ir.ops.nn.qkv_rope_with_cache import QKVRoPEWithCache
from triton.flagmega.ir.ops.nn.vectorized_rms_norm import VectorizedRMSNorm
from triton.flagmega.ir.ops.nn.bind_norm_stats import BindNormStats
from triton.flagmega.ir.ops.nn.layer_norm import LayerNorm
from triton.flagmega.ir.ops.nn.norm_apply import NormApply
from triton.flagmega.ir.ops.nn.norm_stats import NormStats
from triton.flagmega.ir.ops.nn.paged_attention import PagedAttention
from triton.flagmega.ir.ops.nn.rope import RoPE
from triton.flagmega.ir.ops.nn.rotary_embedding import RotaryEmbedding
from triton.flagmega.ir.ops.nn.update_paged_attention_kv_cache import (
    UpdatePagedAttentionKVCache,
)
from triton.flagmega.ir.ops.ntt.vectorized_cast import VectorizedCast
from triton.flagmega.ir.ops.ntt.vectorized_rope import VectorizedRoPE
from triton.flagmega.ir.ops.ntt.matmul_norm_stats_combine import MatMulNormStatsCombine
from triton.flagmega.ir.ops.ntt.gather_reduce_add_norm_apply import (
    GatherReduceAddNormApply,
)
from triton.flagmega.ir.ops.ntt.gather_reduce_norm_apply import (
    GatherReduceNormApply,
)
from triton.flagmega.ir.ops.ntt.gather_reduce_qkv_rope_with_cache import (
    GatherReduceQKVRoPEWithCache,
)
from triton.flagmega.ir.ops.ntt.matmul_norm_stats import MatMulNormStats
from triton.flagmega.ir.ops.ntt.packed_matmul import PackedMatMul
from triton.flagmega.ir.ops.ntt.packed_qkv_parallel_linear import PackedQKVParallelLinear
from triton.flagmega.ir.ops.ntt.packed_qkv_parallel_linear_combine import (
    PackedQKVParallelLinearCombine,
)
from triton.flagmega.ir.ops.ntt.paged_attention_combine import PagedAttentionCombine
from triton.flagmega.ir.ops.ntt.paged_attention_partial import PagedAttentionPartial
from triton.flagmega.ir.ops.tensors.bitcast import Bitcast
from triton.flagmega.ir.ops.tensors.cast import Cast
from triton.flagmega.ir.ops.tensors.concat import Concat
from triton.flagmega.ir.ops.tensors.pack import Pack
from triton.flagmega.ir.ops.tensors.pad import Pad
from triton.flagmega.ir.ops.tensors.permute import Permute
from triton.flagmega.ir.ops.tensors.reshape import Reshape
from triton.flagmega.ir.ops.tensors.slice_to_shape import SliceToShape
from triton.flagmega.ir.ops.tensors.unpack import Unpack
from triton.flagmega.ir.ops.tir.barrier import Barrier
from triton.flagmega.ir.ops.tir.buffer import Buffer
from triton.flagmega.ir.ops.tir.buffer_view import BufferView
from triton.flagmega.ir.ops.tir.ref_slice import RefSlice
from triton.flagmega.ir.ops.tir.kernel import Kernel
from triton.flagmega.ir.ops.tir.call import Call as TIRCall
from triton.flagmega.ir.ops.tir.scalar_const import ScalarConst as TIRScalarConst
from triton.flagmega.pattern_match.call_pattern import CallPattern
from triton.flagmega.pattern_match.op_pattern import OpPattern
from triton.flagmega.pattern_match.or_pattern import is_alt
from triton.flagmega.pattern_match.pattern import ExprPattern, Pattern, wildcard
from triton.flagmega.pattern_match.vargs_pattern import VArgsPattern, is_vargs_repeat


PatternInput = Pattern | Node | None
Condition = Callable[[Node], bool] | None


def _sparse_dtype_pattern(dtype):
    return None if dtype is None else SparseExperts.normalize_attrs({"output_dtype": dtype})["output_dtype"]


def _as_pattern(value: PatternInput, parameter) -> Pattern:
    if value is None:
        return wildcard(type_pattern=parameter.type_pattern)
    if isinstance(value, Pattern):
        return value
    if isinstance(value, Node):
        return ExprPattern(
            lambda candidate: candidate.id == value.id,
            type_pattern=parameter.type_pattern,
        )
    raise TypeError(f"Pattern parameter {parameter.name!r} must be a Pattern, Node, or None.")


def _call_pattern(
    definition,
    operands: tuple[PatternInput, ...],
    *,
    attributes: Mapping[str, object] | None = None,
    target_name: str | None = None,
    call_name: str | None = None,
    condition: Condition = None,
) -> CallPattern:
    """Shared mechanics; public pattern signatures remain handwritten."""

    parameters = definition.input_parameters
    variadic = parameters[-1] if parameters and parameters[-1].variadic else None
    fixed = parameters[:-1] if variadic is not None else parameters
    if len(operands) < len(fixed) or (variadic is None and len(operands) != len(fixed)):
        raise TypeError(f"{definition.op_name} pattern received an invalid operand count.")
    fields = [
        _as_pattern(value, parameter)
        for value, parameter in zip(operands[:len(fixed)], fixed)
    ]
    if variadic is not None:
        fields.extend(_as_pattern(value, variadic) for value in operands[len(fixed):])
    constraints = {
        name: value
        for name, value in (attributes or {}).items()
        if value is not None
    }
    target = OpPattern(
        definition.op_name,
        condition,
        target_name,
        attributes=constraints,
    )
    if variadic is not None and len(operands) == len(fixed):
        arguments = is_vargs_repeat(lambda: wildcard(type_pattern=variadic.type_pattern))
    else:
        arguments = VArgsPattern(fields)
    return CallPattern(target, arguments, call_name)


def _op_pattern_function(definition):
    def decorate(function):
        function.__flagmega_op_definition__ = definition
        return function

    return decorate


class _builtin:
    @staticmethod
    @_op_pattern_function(NoneValue)
    def is_none(
        *,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            NoneValue,
            (),
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(TupleValue)
    def is_tuple(
        *fields: PatternInput,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            TupleValue,
            fields,
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(BuiltinCall)
    def is_call(
        *arguments: PatternInput,
        result_type: IRType | None = None,
        callee: str | None = None,
        effect: Effect | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            BuiltinCall,
            arguments,
            attributes={"callee": callee},
            target_name=target_name,
            call_name=call_name,
            condition=_with_node_constraints(
                condition, result_type=result_type, effect=effect
            ),
        )

    @staticmethod
    @_op_pattern_function(BuiltinScalarConst)
    def is_scalar_const(
        result_type: IRType | None = None,
        value: bool | int | float | None = None,
        *,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            BuiltinScalarConst,
            (),
            attributes={"value": value},
            target_name=target_name,
            call_name=call_name,
            condition=_with_node_constraints(condition, result_type=result_type),
        )

    @staticmethod
    @_op_pattern_function(SplatConst)
    def is_splat_const(
        result_type: IRType | None = None,
        value: bool | int | float | None = None,
        *,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            SplatConst,
            (),
            attributes={"value": value},
            target_name=target_name,
            call_name=call_name,
            condition=_with_node_constraints(condition, result_type=result_type),
        )

    @staticmethod
    @_op_pattern_function(ConstAsset)
    def is_const_asset(
        *,
        result_type: IRType | None = None,
        recipe: str | None = None,
        output: str | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            ConstAsset,
            (),
            attributes={"recipe": recipe, "output": output},
            target_name=target_name,
            call_name=call_name,
            condition=_with_node_constraints(condition, result_type=result_type),
        )


class _distributed:
    @staticmethod
    @_op_pattern_function(MaterializeLocalShards)
    def is_materialize_local_shards(
        value: PatternInput = None,
        *,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            MaterializeLocalShards,
            (value,),
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(Boxing)
    def is_boxing(
        value: PatternInput = None,
        new_type: IRType | None = None,
        *,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            Boxing, (value,), attributes={"new_type": new_type},
            target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(ShardedView)
    def is_sharded_view(
        value: PatternInput = None,
        new_type: IRType | None = None,
        *,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            ShardedView, (value,), attributes={"new_type": new_type},
            target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(ForceBoxing)
    def is_force_boxing(
        value: PatternInput = None,
        new_type: IRType | None = None,
        *,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            ForceBoxing, (value,), attributes={"new_type": new_type},
            target_name=target_name, call_name=call_name, condition=condition)


class _math:

    @staticmethod
    @_op_pattern_function(Div)
    def is_div(lhs: PatternInput = None, rhs: PatternInput = None, *, target_name: str | None = None,
               call_name: str | None = None, condition: Condition = None) -> CallPattern:
        return _call_pattern(Div, (lhs, rhs), target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(Sigmoid)
    def is_sigmoid(value: PatternInput = None, *, target_name: str | None = None, call_name: str | None = None,
                   condition: Condition = None) -> CallPattern:
        return _call_pattern(Sigmoid, (value, ), target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(ReduceSum)
    def is_reduce_sum(value: PatternInput = None, *, axes: tuple[int, ...] | None = None, keep_dims: bool | None = None,
                      target_name: str | None = None, call_name: str | None = None,
                      condition: Condition = None) -> CallPattern:
        return _call_pattern(ReduceSum, (value, ), attributes={"axes": axes, "keep_dims": keep_dims},
                             target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(Add)
    def is_add(
        lhs: PatternInput = None,
        rhs: PatternInput = None,
        *,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            Add, (lhs, rhs), target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    def is_add_commutative(
        lhs: PatternInput = None,
        rhs: PatternInput = None,
        *,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> Pattern:
        """Match add in either operand order, preferring the declared order."""

        return is_alt(
            _math.is_add(
                lhs, rhs, target_name=target_name, call_name=call_name, condition=condition),
            _math.is_add(
                rhs, lhs, target_name=target_name, call_name=call_name, condition=condition),
        )

    @staticmethod
    @_op_pattern_function(Mul)
    def is_mul(
        lhs: PatternInput = None,
        rhs: PatternInput = None,
        *,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            Mul, (lhs, rhs), target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    def is_mul_commutative(
        lhs: PatternInput = None,
        rhs: PatternInput = None,
        *,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> Pattern:
        """Match multiply in either operand order, preferring the declared order."""

        return is_alt(
            _math.is_mul(
                lhs, rhs, target_name=target_name, call_name=call_name, condition=condition),
            _math.is_mul(
                rhs, lhs, target_name=target_name, call_name=call_name, condition=condition),
        )

    @staticmethod
    @_op_pattern_function(Silu)
    def is_silu(
        value: PatternInput = None,
        *,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            Silu, (value,), target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    def is_silu_mul(
        value: PatternInput = None,
        multiplier: PatternInput = None,
        *,
        silu_name: str | None = None,
        mul_name: str | None = None,
    ) -> CallPattern:
        """Match ``mul(silu(value), multiplier)``."""

        return _math.is_mul(
            _math.is_silu(value, call_name=silu_name),
            multiplier,
            call_name=mul_name,
        )

    @staticmethod
    @_op_pattern_function(MatMul)
    def is_matmul(
        lhs: PatternInput = None,
        rhs: PatternInput = None,
        *,
        transpose_a: bool | None = None,
        transpose_b: bool | None = None,
        output_data_type: DType | str | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            MatMul,
            (lhs, rhs),
            attributes={"transpose_a": transpose_a, "transpose_b": transpose_b,
                        "output_data_type": output_data_type},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(PackedDenseMatMul)
    def is_packed_dense_matmul(
        lhs: PatternInput = None,
        weight: PatternInput = None,
        *,
        packed_layout: str | None = None,
        logical_n: int | None = None,
        output_data_type: DType | str | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            PackedDenseMatMul,
            (lhs, weight),
            attributes={"packed_layout": packed_layout, "logical_n": logical_n,
                        "output_data_type": output_data_type},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(VectorizedBinary)
    def is_vectorized_binary(
        lhs: PatternInput = None,
        rhs: PatternInput = None,
        *,
        binary_op: str | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            VectorizedBinary, (lhs, rhs), attributes={"binary_op": binary_op},
            target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(VectorizedUnary)
    def is_vectorized_unary(
        value: PatternInput = None,
        *,
        unary_op: str | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            VectorizedUnary, (value,), attributes={"unary_op": unary_op},
            target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(VectorizedMatMul)
    def is_vectorized_matmul(
        lhs: PatternInput = None,
        rhs: PatternInput = None,
        *,
        lhs_axes: tuple[int, ...] | None = None,
        rhs_axes: tuple[int, ...] | None = None,
        output_axes: tuple[int, ...] | None = None,
        output_lanes: tuple[int, ...] | None = None,
        transpose_a: bool | None = None,
        transpose_b: bool | None = None,
        output_data_type: DType | str | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            VectorizedMatMul,
            (lhs, rhs),
            attributes={
                "lhs_axes": lhs_axes, "rhs_axes": rhs_axes, "output_axes": output_axes,
                "output_lanes": output_lanes, "transpose_a": transpose_a, "transpose_b": transpose_b,
                "output_data_type": output_data_type,
            },
            target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(BlockScaledMatMul)
    def is_block_scaled_matmul(
        value: PatternInput = None,
        weight: PatternInput = None,
        weight_scale: PatternInput = None,
        *,
        weight_block_n: int | None = None,
        weight_block_k: int | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            BlockScaledMatMul,
            (value, weight, weight_scale),
            attributes={"weight_block_n": weight_block_n, "weight_block_k": weight_block_k},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(PackedBlockScaledMatMul)
    def is_packed_block_scaled_matmul(
        value: PatternInput = None,
        weight: PatternInput = None,
        weight_scale: PatternInput = None,
        *,
        weight_block_n: int | None = None,
        weight_block_k: int | None = None,
        k_pack: int | None = None,
        k_vector: int | None = None,
        packed_layout: str | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            PackedBlockScaledMatMul,
            (value, weight, weight_scale),
            attributes={
                "weight_block_n": weight_block_n,
                "weight_block_k": weight_block_k,
                "k_pack": k_pack,
                "k_vector": k_vector,
                "packed_layout": packed_layout,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )


class _nn:

    @staticmethod
    @_op_pattern_function(L2Normalization)
    def is_l2_normalization(value: PatternInput = None, *, axes: tuple[int, ...] | None = None,
                            epsilon: float | None = None, epsilon_mode: str | None = None,
                            division_mode: str | None = None, target_name: str | None = None,
                            call_name: str | None = None, condition: Condition = None) -> CallPattern:
        return _call_pattern(L2Normalization, (value,), attributes={"axes": axes, "epsilon": epsilon,
                             "epsilon_mode": epsilon_mode, "division_mode": division_mode},
                             target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(DeltaRuleGates)
    def is_delta_rule_gates(a: PatternInput = None, b: PatternInput = None, a_log: PatternInput = None,
                            dt_bias: PatternInput = None, *, softplus_threshold: float | None = None,
                            alpha_exp_mode: str | None = None, target_name: str | None = None,
                            call_name: str | None = None, condition: Condition = None) -> CallPattern:
        return _call_pattern(DeltaRuleGates, (a, b, a_log, dt_bias),
                             attributes={"softplus_threshold": softplus_threshold, "alpha_exp_mode": alpha_exp_mode},
                             target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(DeltaRuleBlockUpdate)
    def is_delta_rule_block_update(query: PatternInput = None, key: PatternInput = None, value: PatternInput = None,
                                    coefficients: PatternInput = None, log_prefix: PatternInput = None,
                                    state: PatternInput = None, *, scale: float | None = None,
                                    state_field: str | None = None, state_layout: tuple[str, ...] | None = None,
                                    state_vector_axes: tuple[str, ...] | None = None, target_name: str | None = None,
                                    call_name: str | None = None, condition: Condition = None) -> CallPattern:
        return _call_pattern(DeltaRuleBlockUpdate, (query, key, value, coefficients, log_prefix, state),
                             attributes={"scale": scale, "state_field": state_field, "state_layout": state_layout,
                                         "state_vector_axes": state_vector_axes}, target_name=target_name,
                             call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(DeltaRuleLogPrefix)
    def is_delta_rule_log_prefix(alpha: PatternInput = None, *, block_size: int | None = None,
                                 scan_group_size: int | None = None, epsilon: float | None = None,
                                 log2_mode: str | None = None, target_name: str | None = None,
                                 call_name: str | None = None, condition: Condition = None) -> CallPattern:
        return _call_pattern(DeltaRuleLogPrefix, (alpha,), attributes={"block_size": block_size,
                             "scan_group_size": scan_group_size, "epsilon": epsilon, "log2_mode": log2_mode},
                             target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(DeltaRuleCoefficients)
    def is_delta_rule_coefficients(key: PatternInput = None, beta: PatternInput = None, *, block_size: int | None = None,
                                   target_name: str | None = None, call_name: str | None = None,
                                   condition: Condition = None) -> CallPattern:
        return _call_pattern(DeltaRuleCoefficients, (key, beta), attributes={"block_size": block_size},
                             target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(GatedDeltaNetStateSlice)
    def is_gated_delta_net_state_slice(state: PatternInput = None, layer_id: PatternInput = None, *,
                                       target_name: str | None = None, call_name: str | None = None,
                                       condition: Condition = None) -> CallPattern:
        return _call_pattern(GatedDeltaNetStateSlice, (state, layer_id), target_name=target_name, call_name=call_name,
                             condition=condition)

    @staticmethod
    @_op_pattern_function(Softmax)
    def is_softmax(value: PatternInput = None, *, axis: int | None = None, target_name: str | None = None,
                   call_name: str | None = None, condition: Condition = None) -> CallPattern:
        return _call_pattern(Softmax, (value, ), attributes={"axis": axis}, target_name=target_name,
                             call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(SparseExperts)
    def is_sparse_experts(
        q: PatternInput = None,
        router_expert_ids: PatternInput = None,
        router_expert_weights: PatternInput = None,
        gate_input_scale: PatternInput = None,
        gate_weight: PatternInput = None,
        gate_proj_scale: PatternInput = None,
        down_input_scale: PatternInput = None,
        down_weight: PatternInput = None,
        down_proj_scale: PatternInput = None,
        up_input_scale: PatternInput = None,
        up_weight: PatternInput = None,
        up_proj_scale: PatternInput = None,
        *,
        output_dtype=None,
        intermediate_dtype=None,
        round_projections: bool | None = None,
        round_activation: bool | None = None,
        round_down_projection: bool | None = None,
        round_weighted_output: bool | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            SparseExperts,
            (q, router_expert_ids, router_expert_weights, gate_input_scale, gate_weight, gate_proj_scale,
             down_input_scale, down_weight, down_proj_scale, up_input_scale, up_weight, up_proj_scale),
            attributes={
                "output_dtype": _sparse_dtype_pattern(output_dtype), "intermediate_dtype":
                _sparse_dtype_pattern(intermediate_dtype), "round_projections": round_projections, "round_activation":
                round_activation, "round_down_projection": round_down_projection, "round_weighted_output":
                round_weighted_output
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(SparseExpertsGateUp)
    def is_sparse_experts_gate_up(
        q: PatternInput = None,
        router_expert_ids: PatternInput = None,
        gate_input_scale: PatternInput = None,
        gate_weight: PatternInput = None,
        gate_proj_scale: PatternInput = None,
        up_input_scale: PatternInput = None,
        up_weight: PatternInput = None,
        up_proj_scale: PatternInput = None,
        *,
        output_dtype=None,
        round_projections: bool | None = None,
        round_activation: bool | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            SparseExpertsGateUp,
            (q, router_expert_ids, gate_input_scale, gate_weight, gate_proj_scale, up_input_scale, up_weight,
             up_proj_scale),
            attributes={
                "output_dtype": _sparse_dtype_pattern(output_dtype), "round_projections": round_projections,
                "round_activation": round_activation
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(SparseExpertsDown)
    def is_sparse_experts_down(
        activations: PatternInput = None,
        router_expert_ids: PatternInput = None,
        router_expert_weights: PatternInput = None,
        down_input_scale: PatternInput = None,
        down_weight: PatternInput = None,
        down_proj_scale: PatternInput = None,
        *,
        output_dtype=None,
        round_projection: bool | None = None,
        round_weighted_output: bool | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            SparseExpertsDown,
            (activations, router_expert_ids, router_expert_weights, down_input_scale, down_weight, down_proj_scale),
            attributes={
                "output_dtype": _sparse_dtype_pattern(output_dtype), "round_projection": round_projection,
                "round_weighted_output": round_weighted_output
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(QKVParallelLinear)
    def is_qkv_parallel_linear(
        input: PatternInput = None,
        q_weight: PatternInput = None,
        k_weight: PatternInput = None,
        v_weight: PatternInput = None,
        q_bias: PatternInput = None,
        k_bias: PatternInput = None,
        v_bias: PatternInput = None,
        q_input_scale: PatternInput = None,
        k_input_scale: PatternInput = None,
        v_input_scale: PatternInput = None,
        q_weight_scale: PatternInput = None,
        k_weight_scale: PatternInput = None,
        v_weight_scale: PatternInput = None,
        *,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
        output_data_type: str | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            QKVParallelLinear,
            (
                input,
                q_weight,
                k_weight,
                v_weight,
                q_bias,
                k_bias,
                v_bias,
                q_input_scale,
                k_input_scale,
                v_input_scale,
                q_weight_scale,
                k_weight_scale,
                v_weight_scale,
            ),
            attributes={
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "output_data_type": output_data_type,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(NormStats)
    def is_norm_stats(
        input: PatternInput = None,
        *,
        axis: int | None = None,
        use_mean: bool | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            NormStats,
            (input,),
            attributes={"axis": axis, "use_mean": use_mean},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(BindNormStats)
    def is_bind_norm_stats(
        input: PatternInput = None,
        stats: PatternInput = None,
        *,
        axis: int | None = None,
        use_mean: bool | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            BindNormStats,
            (input, stats),
            attributes={"axis": axis, "use_mean": use_mean},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(NormApply)
    def is_norm_apply(
        input: PatternInput = None,
        stats: PatternInput = None,
        scale: PatternInput = None,
        bias: PatternInput = None,
        *,
        axis: int | None = None,
        epsilon: float | None = None,
        use_mean: bool | None = None,
        round_before_scale: bool | None = None,
        output_dtype: DType | str | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            NormApply,
            (input, stats, scale, bias),
            attributes={"axis": axis, "epsilon": epsilon, "use_mean": use_mean,
                        "round_before_scale": round_before_scale, "output_dtype": output_dtype},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(LayerNorm)
    def is_layer_norm(
        input: PatternInput = None,
        scale: PatternInput = None,
        bias: PatternInput = None,
        *,
        axis: int | None = None,
        epsilon: float | None = None,
        use_mean: bool | None = None,
        round_before_scale: bool | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            LayerNorm,
            (input, scale, bias),
            attributes={"axis": axis, "epsilon": epsilon, "use_mean": use_mean,
                        "round_before_scale": round_before_scale},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(GreedySample)
    def is_greedy_sample(
        logits: PatternInput = None,
        *,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            GreedySample,
            (logits,),
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(DenseMatMulGlu)
    def is_dense_matmul_glu(
        value: PatternInput = None,
        gate_weight: PatternInput = None,
        up_weight: PatternInput = None,
        *,
        activation: str | None = None,
        round_activation: bool | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            DenseMatMulGlu,
            (value, gate_weight, up_weight),
            attributes={"activation": activation, "round_activation": round_activation},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(PackedDenseMatMulGlu)
    def is_packed_dense_matmul_glu(
        value: PatternInput = None,
        gate_weight: PatternInput = None,
        up_weight: PatternInput = None,
        *,
        activation: str | None = None,
        round_activation: bool | None = None,
        packed_layout: str | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            PackedDenseMatMulGlu,
            (value, gate_weight, up_weight),
            attributes={
                "activation": activation,
                "round_activation": round_activation,
                "packed_layout": packed_layout,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(Embedding)
    def is_embedding(
        indices: PatternInput = None,
        weight: PatternInput = None,
        *,
        padding_idx: int | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            Embedding,
            (indices, weight),
            attributes={"padding_idx": padding_idx},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(VectorizedRMSNorm)
    def is_vectorized_rms_norm(
        value: PatternInput = None,
        weight: PatternInput = None,
        *,
        value_axes: tuple[int, ...] | None = None,
        weight_axes: tuple[int, ...] | None = None,
        logical_extent: int | None = None,
        epsilon: float | None = None,
        weight_bias: float | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            VectorizedRMSNorm,
            (value, weight),
            attributes={
                "value_axes": value_axes, "weight_axes": weight_axes,
                "logical_extent": logical_extent, "epsilon": epsilon, "weight_bias": weight_bias,
            },
            target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(RMSNorm)
    def is_rms_norm(
        value: PatternInput = None,
        weight: PatternInput = None,
        *,
        epsilon: float | None = None,
        weight_bias: float | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            RMSNorm,
            (value, weight),
            attributes={"epsilon": epsilon, "weight_bias": weight_bias},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(MatMulGlu)
    def is_matmul_glu(
        value: PatternInput = None,
        gate_weight: PatternInput = None,
        up_weight: PatternInput = None,
        gate_scale: PatternInput = None,
        up_scale: PatternInput = None,
        *,
        activation: str | None = None,
        weight_block_n: int | None = None,
        weight_block_k: int | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            MatMulGlu,
            (value, gate_weight, up_weight, gate_scale, up_scale),
            attributes={
                "activation": activation,
                "weight_block_n": weight_block_n,
                "weight_block_k": weight_block_k,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(PackedMatMulGlu)
    def is_packed_matmul_glu(
        value: PatternInput = None,
        gate_weight: PatternInput = None,
        up_weight: PatternInput = None,
        gate_scale: PatternInput = None,
        up_scale: PatternInput = None,
        *,
        activation: str | None = None,
        weight_block_n: int | None = None,
        weight_block_k: int | None = None,
        k_pack: int | None = None,
        k_vector: int | None = None,
        packed_layout: str | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            PackedMatMulGlu,
            (value, gate_weight, up_weight, gate_scale, up_scale),
            attributes={
                "activation": activation,
                "weight_block_n": weight_block_n,
                "weight_block_k": weight_block_k,
                "k_pack": k_pack,
                "k_vector": k_vector,
                "packed_layout": packed_layout,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(GatedDeltaNet)
    def is_gated_delta_net(
        value: PatternInput = None,
        state: PatternInput = None,
        qkv_weight: PatternInput = None,
        qkv_scale: PatternInput = None,
        z_weight: PatternInput = None,
        z_scale: PatternInput = None,
        b_weight: PatternInput = None,
        a_weight: PatternInput = None,
        conv_weight: PatternInput = None,
        a_log: PatternInput = None,
        dt_bias: PatternInput = None,
        norm_weight: PatternInput = None,
        output_weight: PatternInput = None,
        output_scale: PatternInput = None,
        *,
        num_key_heads: int | None = None,
        num_value_heads: int | None = None,
        key_head_dim: int | None = None,
        value_head_dim: int | None = None,
        conv_kernel_size: int | None = None,
        epsilon: float | None = None,
        weight_block_n: int | None = None,
        weight_block_k: int | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            GatedDeltaNet,
            (
                value, state, qkv_weight, qkv_scale, z_weight, z_scale, b_weight,
                a_weight, conv_weight, a_log, dt_bias, norm_weight, output_weight,
                output_scale,
            ),
            attributes=_gdn_attributes(
                num_key_heads, num_value_heads, key_head_dim, value_head_dim,
                conv_kernel_size, epsilon, weight_block_n, weight_block_k),
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(Qwen3PagedAttention)
    def is_qwen3_paged_attention(
        value: PatternInput = None,
        state: PatternInput = None,
        q_weight: PatternInput = None,
        k_weight: PatternInput = None,
        v_weight: PatternInput = None,
        q_norm_weight: PatternInput = None,
        k_norm_weight: PatternInput = None,
        output_weight: PatternInput = None,
        layer_id: PatternInput = None,
        advance_sequence: PatternInput = None,
        *,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        epsilon: float | None = None,
        rope_theta: float | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            Qwen3PagedAttention,
            (
                value, state, q_weight, k_weight, v_weight, q_norm_weight,
                k_norm_weight, output_weight, layer_id, advance_sequence,
            ),
            attributes={
                "num_attention_heads": num_attention_heads,
                "num_key_value_heads": num_key_value_heads,
                "head_dim": head_dim,
                "epsilon": epsilon,
                "rope_theta": rope_theta,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(RotaryEmbedding)
    def is_rotary_embedding(
        reference: PatternInput = None,
        state: PatternInput = None,
        *,
        head_dim: int | None = None,
        theta: float | None = None,
        attention_scaling: float | None = None,
        output_dtype: DType | str | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            RotaryEmbedding,
            (reference, state),
            attributes={
                "head_dim": head_dim,
                "theta": theta,
                "attention_scaling": attention_scaling,
                "output_dtype": None if output_dtype is None else DType(output_dtype).value,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(RoPE)
    def is_rope(
        input: PatternInput = None,
        cos: PatternInput = None,
        sin: PatternInput = None,
        *,
        rotary_dim: int | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            RoPE,
            (input, cos, sin),
            attributes={"rotary_dim": rotary_dim},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(QKVRoPEWithCache)
    def is_qkv_rope_with_cache(
        qkv: PatternInput = None,
        q_scale: PatternInput = None,
        k_scale: PatternInput = None,
        q_bias: PatternInput = None,
        k_bias: PatternInput = None,
        cos: PatternInput = None,
        sin: PatternInput = None,
        state: PatternInput = None,
        layer_id: PatternInput = None,
        advance_sequence: PatternInput = None,
        *,
        q_axis: int | None = None,
        q_epsilon: float | None = None,
        q_use_mean: bool | None = None,
        q_round_before_scale: bool | None = None,
        k_axis: int | None = None,
        k_epsilon: float | None = None,
        k_use_mean: bool | None = None,
        k_round_before_scale: bool | None = None,
        round_qk_intermediates: bool | None = None,
        rotary_dim: int | None = None,
        qkv_layout: tuple[str, str, str] | None = None,
        attention_layout: tuple[str, str, str] | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            QKVRoPEWithCache,
            (
                qkv,
                q_scale,
                k_scale,
                q_bias,
                k_bias,
                cos,
                sin,
                state,
                layer_id,
                advance_sequence,
            ),
            attributes={
                "q_axis": q_axis,
                "q_epsilon": q_epsilon,
                "q_use_mean": q_use_mean,
                "q_round_before_scale": q_round_before_scale,
                "k_axis": k_axis,
                "k_epsilon": k_epsilon,
                "k_use_mean": k_use_mean,
                "k_round_before_scale": k_round_before_scale,
                "round_qk_intermediates": round_qk_intermediates,
                "rotary_dim": rotary_dim,
                "qkv_layout": qkv_layout,
                "attention_layout": attention_layout,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(UpdatePagedAttentionKVCache)
    def is_update_paged_attention_kv_cache(
        slots: PatternInput = None,
        state: PatternInput = None,
        layer_id: PatternInput = None,
        advance_sequence: PatternInput = None,
        *,
        cache_kind: str | None = None,
        layout: tuple[str, str, str] | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            UpdatePagedAttentionKVCache,
            (slots, state, layer_id, advance_sequence),
            attributes={"cache_kind": cache_kind, "layout": layout},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(PagedAttention)
    def is_paged_attention(
        q: PatternInput = None,
        state: PatternInput = None,
        layer_id: PatternInput = None,
        *,
        scale: float | None = None,
        layout: tuple[str, str, str] | None = None,
        hidden_size: int | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            PagedAttention,
            (q, state, layer_id),
            attributes={
                "scale": scale,
                "layout": layout,
                "hidden_size": hidden_size,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(PackedQwen3PagedAttention)
    def is_packed_qwen3_paged_attention(
        value: PatternInput = None,
        state: PatternInput = None,
        packed_qkv_weight: PatternInput = None,
        q_norm_weight: PatternInput = None,
        k_norm_weight: PatternInput = None,
        output_weight: PatternInput = None,
        layer_id: PatternInput = None,
        advance_sequence: PatternInput = None,
        *,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        epsilon: float | None = None,
        rope_theta: float | None = None,
        context_mesh_size: int | None = None,
        head_mesh_size: int | None = None,
        block_k: int | None = None,
        n_lane: int | None = None,
        k_lane: int | None = None,
        packed_layout: str | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            PackedQwen3PagedAttention,
            (
                value,
                state,
                packed_qkv_weight,
                q_norm_weight,
                k_norm_weight,
                output_weight,
                layer_id,
                advance_sequence,
            ),
            attributes={
                "num_attention_heads": num_attention_heads,
                "num_key_value_heads": num_key_value_heads,
                "head_dim": head_dim,
                "epsilon": epsilon,
                "rope_theta": rope_theta,
                "packed_layout": packed_layout,
                "context_mesh_size": context_mesh_size,
                "head_mesh_size": head_mesh_size,
                "block_k": block_k,
                "n_lane": n_lane,
                "k_lane": k_lane,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(GatedDeltaNetConvolution)
    def is_gated_delta_net_convolution(
        qkv: PatternInput = None,
        state: PatternInput = None,
        conv_weight: PatternInput = None,
        *,
        conv_kernel_size: int | None = None,
        round_products: bool | None = None,
        round_before_activation: bool | None = None,
        accumulation_order: str | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            GatedDeltaNetConvolution,
            (qkv, state, conv_weight),
            attributes={
                "conv_kernel_size": conv_kernel_size,
                "round_products": round_products,
                "round_before_activation": round_before_activation,
                "accumulation_order": accumulation_order,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(GatedDeltaNetRecurrentCore)
    def is_gated_delta_net_recurrent_core(
        state: PatternInput = None,
        qkv: PatternInput = None,
        z: PatternInput = None,
        projection_input: PatternInput = None,
        b_weight: PatternInput = None,
        a_weight: PatternInput = None,
        a_log: PatternInput = None,
        dt_bias: PatternInput = None,
        norm_weight: PatternInput = None,
        *,
        num_key_heads: int | None = None,
        num_value_heads: int | None = None,
        key_head_dim: int | None = None,
        value_head_dim: int | None = None,
        epsilon: float | None = None,
        qk_norm_mode: str | None = None,
        qk_norm_epsilon: float | None = None,
        round_normalized_qk: bool | None = None,
        round_beta: bool | None = None,
        round_core: bool | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            GatedDeltaNetRecurrentCore,
            (
                state, qkv, z, projection_input, b_weight, a_weight, a_log,
                dt_bias, norm_weight,
            ),
            attributes={
                "num_key_heads": num_key_heads,
                "num_value_heads": num_value_heads,
                "key_head_dim": key_head_dim,
                "value_head_dim": value_head_dim,
                "epsilon": epsilon,
                "qk_norm_mode": qk_norm_mode,
                "qk_norm_epsilon": qk_norm_epsilon,
                "round_normalized_qk": round_normalized_qk,
                "round_beta": round_beta,
                "round_core": round_core,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )


class _ntt:
    @staticmethod
    @_op_pattern_function(GatherReduceAddNormApply)
    def is_gather_reduce_add_norm_apply(
        input: PatternInput = None,
        addend: PatternInput = None,
        scale: PatternInput = None,
        bias: PatternInput = None,
        *,
        axis: int | None = None,
        epsilon: float | None = None,
        use_mean: bool | None = None,
        round_before_scale: bool | None = None,
        output_dtype: DType | str | None = None,
        has_bias: bool | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            GatherReduceAddNormApply,
            (input, addend, scale, bias),
            attributes={
                "axis": axis,
                "epsilon": epsilon,
                "use_mean": use_mean,
                "round_before_scale": round_before_scale,
                "output_dtype": output_dtype,
                "has_bias": has_bias,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(GatherReduceNormApply)
    def is_gather_reduce_norm_apply(
        partial_stats: PatternInput = None,
        input: PatternInput = None,
        scale: PatternInput = None,
        bias: PatternInput = None,
        *,
        materialized_stats_type: IRType | None = None,
        axis: int | None = None,
        epsilon: float | None = None,
        use_mean: bool | None = None,
        round_before_scale: bool | None = None,
        output_dtype: DType | str | None = None,
        has_bias: bool | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            GatherReduceNormApply,
            (partial_stats, input, scale, bias),
            attributes={
                "materialized_stats_type": materialized_stats_type,
                "axis": axis,
                "epsilon": epsilon,
                "use_mean": use_mean,
                "round_before_scale": round_before_scale,
                "output_dtype": output_dtype,
                "has_bias": has_bias,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(GatherReduceQKVRoPEWithCache)
    def is_gather_reduce_qkv_rope_with_cache(
        qkv: PatternInput = None,
        q_scale: PatternInput = None,
        k_scale: PatternInput = None,
        q_bias: PatternInput = None,
        k_bias: PatternInput = None,
        cos: PatternInput = None,
        sin: PatternInput = None,
        state: PatternInput = None,
        layer_id: PatternInput = None,
        advance_sequence: PatternInput = None,
        *,
        materialized_qkv_type: IRType | None = None,
        logical_qkv_type: IRType | None = None,
        q_axis: int | None = None,
        q_epsilon: float | None = None,
        q_use_mean: bool | None = None,
        q_round_before_scale: bool | None = None,
        k_axis: int | None = None,
        k_epsilon: float | None = None,
        k_use_mean: bool | None = None,
        k_round_before_scale: bool | None = None,
        round_qk_intermediates: bool | None = None,
        rotary_dim: int | None = None,
        qkv_layout: tuple[str, str, str] | None = None,
        attention_layout: tuple[str, str, str] | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            GatherReduceQKVRoPEWithCache,
            (
                qkv,
                q_scale,
                k_scale,
                q_bias,
                k_bias,
                cos,
                sin,
                state,
                layer_id,
                advance_sequence,
            ),
            attributes={
                "materialized_qkv_type": materialized_qkv_type,
                "logical_qkv_type": logical_qkv_type,
                "q_axis": q_axis,
                "q_epsilon": q_epsilon,
                "q_use_mean": q_use_mean,
                "q_round_before_scale": q_round_before_scale,
                "k_axis": k_axis,
                "k_epsilon": k_epsilon,
                "k_use_mean": k_use_mean,
                "k_round_before_scale": k_round_before_scale,
                "round_qk_intermediates": round_qk_intermediates,
                "rotary_dim": rotary_dim,
                "qkv_layout": qkv_layout,
                "attention_layout": attention_layout,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(VectorizedRoPE)
    def is_vectorized_rope(
        input: PatternInput = None,
        cos: PatternInput = None,
        sin: PatternInput = None,
        *,
        rotary_dim: int | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            VectorizedRoPE,
            (input, cos, sin),
            attributes={"rotary_dim": rotary_dim},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(PackedMatMul)
    def is_packed_matmul(
        lhs: PatternInput = None,
        rhs: PatternInput = None,
        scale: PatternInput = None,
        addend: PatternInput = None,
        *,
        fused_reduce: bool | None = None,
        output_data_type=None,
        rhs_layout: str | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            PackedMatMul,
            (lhs, rhs, scale, addend),
            attributes={
                "fused_reduce": fused_reduce,
                "output_data_type": output_data_type,
                "rhs_layout": rhs_layout,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(PagedAttentionPartial)
    def is_paged_attention_partial(
        q: PatternInput = None,
        state: PatternInput = None,
        layer_id: PatternInput = None,
        *,
        scale: float | None = None,
        layout: tuple[str, str, str] | None = None,
        hidden_size: int | None = None,
        split_hierarchy_axis: int | None = None,
        split_count: int | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            PagedAttentionPartial,
            (q, state, layer_id),
            attributes={
                "scale": scale,
                "layout": layout,
                "hidden_size": hidden_size,
                "split_hierarchy_axis": split_hierarchy_axis,
                "split_count": split_count,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(PagedAttentionCombine)
    def is_paged_attention_combine(
        max_state: PatternInput = None,
        sum_state: PatternInput = None,
        acc_state: PatternInput = None,
        *,
        layout: tuple[str, str, str] | None = None,
        hidden_size: int | None = None,
        output_data_type=None,
        output_type: IRType | None = None,
        split_hierarchy_axis: int | None = None,
        split_count: int | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            PagedAttentionCombine,
            (max_state, sum_state, acc_state),
            attributes={
                "layout": layout,
                "hidden_size": hidden_size,
                "output_data_type": output_data_type,
                "output_type": output_type,
                "split_hierarchy_axis": split_hierarchy_axis,
                "split_count": split_count,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(PackedQKVParallelLinearCombine)
    def is_packed_qkv_parallel_linear_combine(
        qkv: PatternInput = None,
        output_type: IRType | None = None,
        *,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            PackedQKVParallelLinearCombine,
            (qkv,),
            attributes={"output_type": output_type},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(PackedQKVParallelLinear)
    def is_packed_qkv_parallel_linear(
        input: PatternInput = None,
        q_weight: PatternInput = None,
        k_weight: PatternInput = None,
        v_weight: PatternInput = None,
        q_bias: PatternInput = None,
        k_bias: PatternInput = None,
        v_bias: PatternInput = None,
        q_input_scale: PatternInput = None,
        k_input_scale: PatternInput = None,
        v_input_scale: PatternInput = None,
        q_weight_scale: PatternInput = None,
        k_weight_scale: PatternInput = None,
        v_weight_scale: PatternInput = None,
        *,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
        output_data_type: str | None = None,
        rhs_layout: str | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            PackedQKVParallelLinear,
            (
                input,
                q_weight,
                k_weight,
                v_weight,
                q_bias,
                k_bias,
                v_bias,
                q_input_scale,
                k_input_scale,
                v_input_scale,
                q_weight_scale,
                k_weight_scale,
                v_weight_scale,
            ),
            attributes={
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "output_data_type": output_data_type,
                "rhs_layout": rhs_layout,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(MatMulNormStats)
    def is_matmul_norm_stats(
        lhs: PatternInput = None,
        rhs: PatternInput = None,
        addend: PatternInput = None,
        *,
        transpose_a: bool | None = None,
        transpose_b: bool | None = None,
        rhs_layout: str | None = None,
        axis: int | None = None,
        use_mean: bool | None = None,
        addend_cast_dtypes: tuple[DType | str, ...] | None = None,
        output_data_type: DType | str | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            MatMulNormStats,
            (lhs, rhs, addend),
            attributes={
                "transpose_a": transpose_a,
                "transpose_b": transpose_b,
                "rhs_layout": rhs_layout,
                "axis": axis,
                "use_mean": use_mean,
                "addend_cast_dtypes": addend_cast_dtypes,
                "output_data_type": output_data_type,
            },
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(MatMulNormStatsCombine)
    def is_matmul_norm_stats_combine(
        input: PatternInput = None,
        addend: PatternInput = None,
        *,
        axis: int | None = None,
        use_mean: bool | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            MatMulNormStatsCombine,
            (input, addend),
            attributes={"axis": axis, "use_mean": use_mean},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(VectorizedCast)
    def is_vectorized_cast(
        value: PatternInput = None,
        *,
        new_type=None,
        vectorize_axes: tuple[int, ...] | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        attributes: dict[str, object] = {}
        if new_type is not None:
            attributes["new_type"] = VectorizedCast.normalize_attrs(
                {"new_type": new_type, "vectorize_axes": vectorize_axes or (0,)}
            )["new_type"]
        if vectorize_axes is not None:
            attributes["vectorize_axes"] = tuple(vectorize_axes)
        return _call_pattern(
            VectorizedCast,
            (value,),
            attributes=attributes,
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )


class _tensors:

    @staticmethod
    @_op_pattern_function(Slice)
    def is_slice(value: PatternInput = None, *, starts: tuple[int | None, ...] | None = None,
                 ends: tuple[int | None, ...] | None = None, axes: tuple[int, ...] | None = None,
                 steps: tuple[int, ...] | None = None, target_name: str | None = None, call_name: str | None = None,
                 condition: Condition = None) -> CallPattern:
        return _call_pattern(Slice, (value, ),
                             attributes={"starts": starts, "ends": ends, "axes": axes, "steps":
                                         steps}, target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(BroadcastTo)
    def is_broadcast_to(value: PatternInput = None, *, shape: tuple[int, ...] | None = None,
                        target_name: str | None = None, call_name: str | None = None,
                        condition: Condition = None) -> CallPattern:
        return _call_pattern(BroadcastTo, (value, ), attributes={"shape": shape}, target_name=target_name,
                             call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(TopK)
    def is_top_k(value: PatternInput = None, *, k: int | None = None, axis: int | None = None,
                 largest: bool | None = None, sorted: bool | None = None, index_dtype: DType | str | None = None,
                 target_name: str | None = None, call_name: str | None = None,
                 condition: Condition = None) -> CallPattern:
        return _call_pattern(
            TopK, (value, ),
            attributes={"k": k, "axis": axis, "largest": largest, "sorted": sorted, "index_dtype":
                        index_dtype}, target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(Bitcast)
    def is_bitcast(
        value: PatternInput = None,
        *,
        dtype=None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        normalized = None if dtype is None else Bitcast.normalize_attrs(
            {"dtype": dtype})["dtype"]
        return _call_pattern(
            Bitcast, (value,), attributes={"dtype": normalized},
            target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(Cast)
    def is_cast(
        value: PatternInput = None,
        *,
        dtype=None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        normalized = None if dtype is None else Cast.normalize_attrs({"dtype": dtype})["dtype"]
        return _call_pattern(
            Cast, (value,), attributes={"dtype": normalized},
            target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(Concat)
    def is_concat(
        *values: PatternInput,
        axis: int | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            Concat,
            values,
            attributes={"axis": axis},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(Reshape)
    def is_reshape(
        value: PatternInput = None,
        *,
        shape: tuple[int, ...] | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            Reshape,
            (value,),
            attributes={"shape": shape},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(Permute)
    def is_permute(
        value: PatternInput = None,
        *,
        axes: tuple[int, ...] | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            Permute,
            (value,),
            attributes={"axes": axes},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(Pack)
    def is_pack(
        value: PatternInput = None,
        *,
        lanes: tuple[int, ...] | None = None,
        axes: tuple[int, ...] | None = None,
        axis: int | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            Pack,
            (value,),
            attributes={"axes": axes, "axis": axis, "lanes": lanes},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(Unpack)
    def is_unpack(
        value: PatternInput = None,
        *,
        axes: tuple[int, ...] | None = None,
        axis: int | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            Unpack,
            (value,),
            attributes={"axes": axes, "axis": axis},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    @_op_pattern_function(Pad)
    def is_pad(
        value: PatternInput = None,
        *,
        pad_end: tuple[int, ...] | None = None,
        pad_value: int | float | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            Pad, (value,), attributes={"pad_end": pad_end, "pad_value": pad_value},
            target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(SliceToShape)
    def is_slice_to_shape(
        value: PatternInput = None,
        *,
        shape: tuple[int, ...] | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            SliceToShape, (value,), attributes={"shape": shape},
            target_name=target_name, call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(GetItem)
    def is_get_item(
        value: PatternInput = None,
        index: int | None = None,
        *,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            GetItem,
            (value,),
            attributes={"index": index},
            target_name=target_name,
            call_name=call_name,
            condition=condition,
        )

    @staticmethod
    def is_get_item_at(
        index: int,
        value: PatternInput = None,
        *,
        call_name: str | None = None,
    ) -> CallPattern:
        """Convenience pattern for one exact tuple index."""

        return _tensors.is_get_item(value, index, call_name=call_name)


class _tir:
    @staticmethod
    @_op_pattern_function(TIRCall)
    def is_call(
        *arguments: PatternInput,
        result_type: IRType | None = None,
        callee: str | None = None,
        effect: Effect | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            TIRCall,
            arguments,
            attributes={"callee": callee},
            target_name=target_name,
            call_name=call_name,
            condition=_with_node_constraints(
                condition, result_type=result_type, effect=effect
            ),
        )

    @staticmethod
    @_op_pattern_function(TIRScalarConst)
    def is_scalar_const(
        result_type: IRType | None = None,
        value: bool | int | float | None = None,
        *,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            TIRScalarConst,
            (),
            attributes={"value": value},
            target_name=target_name,
            call_name=call_name,
            condition=_with_node_constraints(condition, result_type=result_type),
        )

    @staticmethod
    @_op_pattern_function(Barrier)
    def is_barrier(
        result_type: IRType | None = None,
        *,
        attrs: Mapping[str, Any] | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            Barrier,
            (),
            attributes=attrs,
            target_name=target_name,
            call_name=call_name,
            condition=_with_node_constraints(condition, result_type=result_type),
        )

    @staticmethod
    @_op_pattern_function(Buffer)
    def is_buffer(
        result_type: IRType | None = None,
        *,
        weight_name: str | None = None,
        source: str | None = None,
        key: str | None = None,
        storage: str | None = None,
        alignment: int | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            Buffer,
            (),
            attributes={
                "name": weight_name,
                "source": source,
                "key": key,
                "storage": storage,
                "alignment": alignment,
            },
            target_name=target_name,
            call_name=call_name,
            condition=_with_node_constraints(condition, result_type=result_type),
        )

    @staticmethod
    @_op_pattern_function(BufferView)
    def is_buffer_view(
        value: PatternInput = None,
        new_type: IRType | None = None,
        alias_kind: str | None = None,
        *,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            BufferView,
            (value,),
            attributes={"alias_kind": alias_kind},
            target_name=target_name,
            call_name=call_name,
            condition=_with_node_constraints(condition, result_type=new_type),
        )

    @staticmethod
    @_op_pattern_function(RefSlice)
    def is_ref_slice(value: PatternInput = None, index: PatternInput = None, length: int | None = None, *,
                     target_name: str | None = None, call_name: str | None = None,
                     condition: Condition = None) -> CallPattern:
        return _call_pattern(RefSlice, (value, index), attributes={"length": length}, target_name=target_name,
                             call_name=call_name, condition=condition)

    @staticmethod
    @_op_pattern_function(Kernel)
    def is_kernel(
        *arguments: PatternInput,
        result_type: IRType | None = None,
        semantic_op: str | None = None,
        candidate: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        facts: Mapping[str, Any] | None = None,
        semantic_attrs: Mapping[str, Any] | None = None,
        effect: Effect | None = None,
        target_name: str | None = None,
        call_name: str | None = None,
        condition: Condition = None,
    ) -> CallPattern:
        return _call_pattern(
            Kernel,
            arguments,
            attributes={
                "semantic_op": semantic_op,
                "candidate": candidate,
                "parameters": parameters,
                "facts": facts,
                "semantic_attrs": semantic_attrs,
            },
            target_name=target_name,
            call_name=call_name,
            condition=_with_node_constraints(
                condition,
                result_type=result_type,
                effect=effect,
            ),
        )


def _gdn_attributes(
    num_key_heads,
    num_value_heads,
    key_head_dim,
    value_head_dim,
    conv_kernel_size,
    epsilon,
    weight_block_n,
    weight_block_k,
) -> dict[str, object]:
    return {
        "num_key_heads": num_key_heads,
        "num_value_heads": num_value_heads,
        "key_head_dim": key_head_dim,
        "value_head_dim": value_head_dim,
        "conv_kernel_size": conv_kernel_size,
        "epsilon": epsilon,
        "weight_block_n": weight_block_n,
        "weight_block_k": weight_block_k,
    }


def _with_node_constraints(
    condition: Condition,
    *,
    result_type: IRType | None = None,
    effect: Effect | None = None,
) -> Condition:
    if result_type is None and effect is None:
        return condition

    def combined(node: Node) -> bool:
        return (
            (result_type is None or node.type == result_type)
            and (effect is None or node.effect == effect)
            and (condition is None or condition(node))
        )

    return combined


class F:
    """Static PatternMatch namespaces with op and convenience patterns."""

    builtin = _builtin
    distributed = _distributed
    math = _math
    nn = _nn
    ntt = _ntt
    tensors = _tensors
    tir = _tir


__all__ = ["F"]
