# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Statically declared functional constructors for FlagMega IR.

These functions are ordinary Python source rather than methods installed with
``setattr``. IDEs can therefore complete, navigate and inspect ``F.math.*``
directly, while op definitions remain responsible for construction,
type/effect inference and verification.
"""

from __future__ import annotations

from typing import Any, Mapping

from triton.flagmega.ir.model import DType, Effect, IRType, Node, PURE
from triton.flagmega.ir.ops.builtin.call import Call as BuiltinCall
from triton.flagmega.ir.ops.builtin.get_item import GetItem
from triton.flagmega.ir.ops.builtin.none import NoneValue
from triton.flagmega.ir.ops.builtin.const_asset import ConstAsset
from triton.flagmega.ir.ops.builtin.scalar_const import ScalarConst as BuiltinScalarConst
from triton.flagmega.ir.ops.builtin.splat_const import SplatConst
from triton.flagmega.ir.ops.builtin.tuple import TupleValue
from triton.flagmega.ir.ops.core import construction_scope
from triton.flagmega.ir.ops.distributed.boxing import Boxing
from triton.flagmega.ir.ops.distributed.force_boxing import ForceBoxing
from triton.flagmega.ir.ops.distributed.materialize_local_shards import (
    MaterializeLocalShards,
)
from triton.flagmega.ir.ops.distributed.sharded_view import ShardedView
from triton.flagmega.ir.ops.math.add import Add
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
from triton.flagmega.ir.ops.nn.dense_matmul_glu import DenseMatMulGlu
from triton.flagmega.ir.ops.nn.gated_delta_net import GatedDeltaNet
from triton.flagmega.ir.ops.nn.gdn_convolution import GatedDeltaNetConvolution
from triton.flagmega.ir.ops.nn.gdn_recurrent_core import GatedDeltaNetRecurrentCore
from triton.flagmega.ir.ops.nn.greedy_sample import GreedySample
from triton.flagmega.ir.ops.nn.matmul_glu import MatMulGlu
from triton.flagmega.ir.ops.nn.packed_matmul_glu import PackedMatMulGlu
from triton.flagmega.ir.ops.nn.packed_dense_matmul_glu import PackedDenseMatMulGlu
from triton.flagmega.ir.ops.nn.packed_qwen3_paged_attention import PackedQwen3PagedAttention
from triton.flagmega.ir.ops.nn.rms_norm import RMSNorm
from triton.flagmega.ir.ops.nn.qkv_parallel_linear import QKVParallelLinear
from triton.flagmega.ir.ops.nn.qkv_rope_with_cache import QKVRoPEWithCache
from triton.flagmega.ir.ops.nn.qwen3_paged_attention import Qwen3PagedAttention
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
from triton.flagmega.ir.ops.ntt.gather_reduce_qkv_rope_with_cache import (
    GatherReduceQKVRoPEWithCache,
)
from triton.flagmega.ir.ops.ntt.gather_reduce_norm_apply import (
    GatherReduceNormApply,
)
from triton.flagmega.ir.ops.ntt.gather_reduce_add_norm_apply import (
    GatherReduceAddNormApply,
)
from triton.flagmega.ir.ops.ntt.matmul_norm_stats_combine import MatMulNormStatsCombine
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
from triton.flagmega.ir.ops.tir.kernel import Kernel
from triton.flagmega.ir.ops.tir.call import Call
from triton.flagmega.ir.ops.tir.scalar_const import ScalarConst


Metadata = Mapping[str, Any] | None


def _op_function(definition):
    """Attach introspection metadata without changing a static declaration."""

    def decorate(function):
        function.__flagmega_op_definition__ = definition
        return function

    return decorate


class _math:
    @staticmethod
    @_op_function(Add)
    def add(lhs: Node, rhs: Node, *, name: str | None = None, metadata: Metadata = None) -> Node:
        """Elementwise add of two identically typed tensors."""

        return Add.construct(lhs, rhs, name=name, metadata=metadata)

    @staticmethod
    @_op_function(Mul)
    def mul(lhs: Node, rhs: Node, *, name: str | None = None, metadata: Metadata = None) -> Node:
        """Elementwise multiply of two identically typed tensors."""

        return Mul.construct(lhs, rhs, name=name, metadata=metadata)

    @staticmethod
    @_op_function(Silu)
    def silu(value: Node, *, name: str | None = None, metadata: Metadata = None) -> Node:
        """Apply the SiLU activation."""

        return Silu.construct(value, name=name, metadata=metadata)

    @staticmethod
    def silu_mul(
        value: Node,
        multiplier: Node,
        *,
        silu_name: str | None = None,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Convenience graph builder for ``mul(silu(value), multiplier)``."""

        activated = Silu.construct(value, name=silu_name, metadata=metadata)
        return Mul.construct(activated, multiplier, name=name, metadata=metadata)

    @staticmethod
    @_op_function(MatMul)
    def matmul(
        lhs: Node,
        rhs: Node,
        *,
        transpose_a: bool = False,
        transpose_b: bool = False,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Rank-2 matrix multiplication."""

        return MatMul.construct(
            lhs,
            rhs,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(PackedDenseMatMul)
    def packed_dense_matmul(
        lhs: Node,
        weight: Node,
        *,
        packed_layout: str = "k_major_n8_k16",
        logical_n: int | None = None,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Dense projection over an offline K-major/N8/K16 BF16 weight."""

        return PackedDenseMatMul.construct(
            lhs,
            weight,
            packed_layout=packed_layout,
            logical_n=logical_n,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(VectorizedBinary)
    def vectorized_binary(
        lhs: Node,
        rhs: Node,
        *,
        binary_op: str,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Binary arithmetic over typed-vector elements."""

        return VectorizedBinary.construct(lhs, rhs, binary_op=binary_op, name=name, metadata=metadata)

    @staticmethod
    @_op_function(VectorizedUnary)
    def vectorized_unary(
        value: Node,
        *,
        unary_op: str,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Unary arithmetic over typed-vector elements."""

        return VectorizedUnary.construct(value, unary_op=unary_op, name=name, metadata=metadata)

    @staticmethod
    @_op_function(VectorizedMatMul)
    def vectorized_matmul(
        lhs: Node,
        rhs: Node,
        *,
        lhs_axes: tuple[int, ...] | list[int],
        rhs_axes: tuple[int, ...] | list[int],
        output_axes: tuple[int, ...] | list[int],
        output_lanes: tuple[int, ...] | list[int],
        transpose_a: bool = False,
        transpose_b: bool = False,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Matrix multiplication with independently vectorized input axes."""

        return VectorizedMatMul.construct(
            lhs,
            rhs,
            lhs_axes=lhs_axes,
            rhs_axes=rhs_axes,
            output_axes=output_axes,
            output_lanes=output_lanes,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(BlockScaledMatMul)
    def block_scaled_matmul(
        value: Node,
        weight: Node,
        weight_scale: Node,
        *,
        weight_block_n: int,
        weight_block_k: int,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Construct a block-scaled FP8 matrix multiplication."""

        return BlockScaledMatMul.construct(
            value,
            weight,
            weight_scale,
            weight_block_n=weight_block_n,
            weight_block_k=weight_block_k,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(PackedBlockScaledMatMul)
    def packed_block_scaled_matmul(
        value: Node,
        weight: Node,
        weight_scale: Node,
        *,
        weight_block_n: int,
        weight_block_k: int,
        k_pack: int,
        k_vector: int,
        packed_layout: str = "n_major_k_packed",
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Block-scaled matmul over an N-major/K-vector packed weight."""

        return PackedBlockScaledMatMul.construct(
            value,
            weight,
            weight_scale,
            weight_block_n=weight_block_n,
            weight_block_k=weight_block_k,
            k_pack=k_pack,
            k_vector=k_vector,
            packed_layout=packed_layout,
            name=name,
            metadata=metadata,
        )


class _nn:
    @staticmethod
    @_op_function(NormStats)
    def norm_stats(
        input: Node,
        *,
        axis: int,
        use_mean: bool,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Compute additive sum/sum-of-squares normalization statistics."""

        return NormStats.construct(
            input, axis=axis, use_mean=use_mean, name=name, metadata=metadata)

    @staticmethod
    @_op_function(BindNormStats)
    def bind_norm_stats(
        input: Node,
        stats: Node,
        *,
        axis: int,
        use_mean: bool,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Bind materialized statistics to their source distribution relation."""

        return BindNormStats.construct(
            input, stats, axis=axis, use_mean=use_mean, name=name, metadata=metadata)

    @staticmethod
    @_op_function(NormApply)
    def norm_apply(
        input: Node,
        stats: Node,
        scale: Node,
        bias: Node,
        *,
        axis: int,
        epsilon: float,
        use_mean: bool,
        round_before_scale: bool = False,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Apply LayerNorm/RMSNorm from explicit additive statistics."""

        return NormApply.construct(
            input,
            stats,
            scale,
            bias,
            axis=axis,
            epsilon=epsilon,
            use_mean=use_mean,
            round_before_scale=round_before_scale,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(LayerNorm)
    def layer_norm(
        input: Node,
        scale: Node,
        bias: Node,
        *,
        axis: int,
        epsilon: float,
        use_mean: bool = True,
        round_before_scale: bool = False,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Construct fused LayerNorm; TargetIndependent decomposes it."""

        return LayerNorm.construct(
            input,
            scale,
            bias,
            axis=axis,
            epsilon=epsilon,
            use_mean=use_mean,
            round_before_scale=round_before_scale,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(GreedySample)
    def greedy_sample(
        logits: Node,
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Select the maximum-logit token for every batch row."""

        return GreedySample.construct(logits, name=name, metadata=metadata)

    @staticmethod
    @_op_function(DenseMatMulGlu)
    def dense_matmul_glu(
        value: Node,
        gate_weight: Node,
        up_weight: Node,
        *,
        activation: str = "silu",
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Dense BF16 gate/up projections followed by SiLU GLU."""

        return DenseMatMulGlu.construct(
            value, gate_weight, up_weight,
            activation=activation, name=name, metadata=metadata)

    @staticmethod
    @_op_function(Embedding)
    def embedding(
        indices: Node,
        weight: Node,
        *,
        padding_idx: int | None = None,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Gather token embeddings, optionally zeroing the padding row."""

        return Embedding.construct(
            indices,
            weight,
            padding_idx=padding_idx,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(PackedDenseMatMulGlu)
    def packed_dense_matmul_glu(
        value: Node,
        gate_weight: Node,
        up_weight: Node,
        *,
        activation: str = "silu",
        packed_layout: str = "k_major_n8_k16",
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """GLU over K-major/N8/K16 offline-packed BF16 weights."""

        return PackedDenseMatMulGlu.construct(
            value,
            gate_weight,
            up_weight,
            activation=activation,
            packed_layout=packed_layout,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(VectorizedRMSNorm)
    def vectorized_rms_norm(
        value: Node,
        weight: Node,
        *,
        value_axes: tuple[int, ...] | list[int],
        weight_axes: tuple[int, ...] | list[int],
        logical_extent: int,
        epsilon: float,
        weight_bias: float = 1.0,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """RMSNorm whose reduction axis is represented by VectorType lanes."""

        return VectorizedRMSNorm.construct(
            value,
            weight,
            value_axes=value_axes,
            weight_axes=weight_axes,
            logical_extent=logical_extent,
            epsilon=epsilon,
            weight_bias=weight_bias,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(RMSNorm)
    def rms_norm(
        value: Node,
        weight: Node,
        *,
        epsilon: float,
        weight_bias: float = 1.0,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """RMSNorm with an optional additive offset applied to the scale."""

        return RMSNorm.construct(
            value,
            weight,
            epsilon=epsilon,
            weight_bias=weight_bias,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(MatMulGlu)
    def matmul_glu(
        value: Node,
        gate_weight: Node,
        up_weight: Node,
        gate_scale: Node,
        up_scale: Node,
        *,
        activation: str,
        weight_block_n: int,
        weight_block_k: int,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Fused block-scaled gate/up projections and GLU activation."""

        return MatMulGlu.construct(
            value,
            gate_weight,
            up_weight,
            gate_scale,
            up_scale,
            activation=activation,
            weight_block_n=weight_block_n,
            weight_block_k=weight_block_k,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(PackedMatMulGlu)
    def packed_matmul_glu(
        value: Node,
        gate_weight: Node,
        up_weight: Node,
        gate_scale: Node,
        up_scale: Node,
        *,
        activation: str,
        weight_block_n: int,
        weight_block_k: int,
        k_pack: int,
        k_vector: int,
        packed_layout: str = "n_major_k_packed",
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Fused GLU over two N-major/K-vector packed weights."""

        return PackedMatMulGlu.construct(
            value,
            gate_weight,
            up_weight,
            gate_scale,
            up_scale,
            activation=activation,
            weight_block_n=weight_block_n,
            weight_block_k=weight_block_k,
            k_pack=k_pack,
            k_vector=k_vector,
            packed_layout=packed_layout,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(GatedDeltaNet)
    def gated_delta_net(
        value: Node,
        state: Node,
        qkv_weight: Node,
        qkv_scale: Node,
        z_weight: Node,
        z_scale: Node,
        b_weight: Node,
        a_weight: Node,
        conv_weight: Node,
        a_log: Node,
        dt_bias: Node,
        norm_weight: Node,
        output_weight: Node,
        output_scale: Node,
        *,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_size: int,
        epsilon: float,
        weight_block_n: int,
        weight_block_k: int,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Construct the fused stateful Gated DeltaNet layer operation."""

        return GatedDeltaNet.construct(
            value,
            state,
            qkv_weight,
            qkv_scale,
            z_weight,
            z_scale,
            b_weight,
            a_weight,
            conv_weight,
            a_log,
            dt_bias,
            norm_weight,
            output_weight,
            output_scale,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            conv_kernel_size=conv_kernel_size,
            epsilon=epsilon,
            weight_block_n=weight_block_n,
            weight_block_k=weight_block_k,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(QKVParallelLinear)
    def qkv_parallel_linear(
        input: Node,
        q_weight: Node,
        k_weight: Node,
        v_weight: Node,
        q_bias: Node,
        k_bias: Node,
        v_bias: Node,
        q_input_scale: Node,
        k_input_scale: Node,
        v_input_scale: Node,
        q_weight_scale: Node,
        k_weight_scale: Node,
        v_weight_scale: Node,
        *,
        num_heads: int,
        num_kv_heads: int,
        output_data_type: str,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Construct three logical Q/K/V projections with ``[K,N]`` weights."""

        return QKVParallelLinear.construct(
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
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            output_data_type=output_data_type,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(RotaryEmbedding)
    def rotary_embedding(
        reference: Node,
        state: Node,
        *,
        head_dim: int,
        theta: float,
        attention_scaling: float = 1.0,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Build position-dependent cos/sin tensors for a decode span."""

        return RotaryEmbedding.construct(
            reference,
            state,
            head_dim=head_dim,
            theta=theta,
            attention_scaling=attention_scaling,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(RoPE)
    def rope(
        input: Node,
        cos: Node,
        sin: Node,
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Apply rotary position embedding to a rank-three tensor."""

        return RoPE.construct(input, cos, sin, name=name, metadata=metadata)

    @staticmethod
    @_op_function(QKVRoPEWithCache)
    def qkv_rope_with_cache(
        qkv: Node,
        q_scale: Node,
        k_scale: Node,
        q_bias: Node,
        k_bias: Node,
        cos: Node,
        sin: Node,
        state: Node,
        layer_id: Node,
        advance_sequence: Node,
        *,
        q_axis: int,
        q_epsilon: float,
        q_use_mean: bool,
        q_round_before_scale: bool = False,
        k_axis: int,
        k_epsilon: float,
        k_use_mean: bool,
        k_round_before_scale: bool = False,
        qkv_layout: tuple[str, str, str],
        attention_layout: tuple[str, str, str],
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Normalize Q/K, apply RoPE, and update both paged-cache slots."""

        return QKVRoPEWithCache.construct(
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
            q_axis=q_axis,
            q_epsilon=q_epsilon,
            q_use_mean=q_use_mean,
            k_axis=k_axis,
            k_epsilon=k_epsilon,
            k_use_mean=k_use_mean,
            qkv_layout=qkv_layout,
            attention_layout=attention_layout,
            q_round_before_scale=q_round_before_scale,
            k_round_before_scale=k_round_before_scale,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(UpdatePagedAttentionKVCache)
    def update_paged_attention_kv_cache(
        slots: Node,
        state: Node,
        layer_id: Node,
        advance_sequence: Node,
        *,
        cache_kind: str,
        layout: tuple[str, str, str],
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Write one semantic K or V tensor into a paged cache."""

        return UpdatePagedAttentionKVCache.construct(
            slots,
            state,
            layer_id,
            advance_sequence,
            cache_kind=cache_kind,
            layout=layout,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(PagedAttention)
    def paged_attention(
        q: Node,
        state: Node,
        layer_id: Node,
        *,
        scale: float,
        layout: tuple[str, str, str],
        hidden_size: int,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Read an explicitly updated paged cache and compute attention."""

        return PagedAttention.construct(
            q,
            state,
            layer_id,
            scale=scale,
            layout=layout,
            hidden_size=hidden_size,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(Qwen3PagedAttention)
    def qwen3_paged_attention(
        value: Node,
        state: Node,
        q_weight: Node,
        k_weight: Node,
        v_weight: Node,
        q_norm_weight: Node,
        k_norm_weight: Node,
        output_weight: Node,
        layer_id: Node,
        advance_sequence: Node,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        epsilon: float,
        rope_theta: float,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Qwen3 QK-normalized RoPE attention over a mutable paged KV cache."""

        return Qwen3PagedAttention.construct(
            value, state, q_weight, k_weight, v_weight, q_norm_weight,
            k_norm_weight, output_weight, layer_id, advance_sequence,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            epsilon=epsilon,
            rope_theta=rope_theta,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(PackedQwen3PagedAttention)
    def packed_qwen3_paged_attention(
        value: Node,
        state: Node,
        packed_qkv_weight: Node,
        q_norm_weight: Node,
        k_norm_weight: Node,
        output_weight: Node,
        layer_id: Node,
        advance_sequence: Node,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        epsilon: float,
        rope_theta: float,
        context_mesh_size: int,
        head_mesh_size: int,
        block_k: int,
        n_lane: int,
        k_lane: int,
        packed_layout: str,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Qwen3 attention consuming an offline split-K K-major QKV asset."""

        return PackedQwen3PagedAttention.construct(
            value,
            state,
            packed_qkv_weight,
            q_norm_weight,
            k_norm_weight,
            output_weight,
            layer_id,
            advance_sequence,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            epsilon=epsilon,
            rope_theta=rope_theta,
            packed_layout=packed_layout,
            context_mesh_size=context_mesh_size,
            head_mesh_size=head_mesh_size,
            block_k=block_k,
            n_lane=n_lane,
            k_lane=k_lane,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(GatedDeltaNetConvolution)
    def gated_delta_net_convolution(
        qkv: Node,
        state: Node,
        conv_weight: Node,
        *,
        conv_kernel_size: int,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Apply stateful depthwise convolution to projected QKV."""

        return GatedDeltaNetConvolution.construct(
            qkv,
            state,
            conv_weight,
            conv_kernel_size=conv_kernel_size,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(GatedDeltaNetRecurrentCore)
    def gated_delta_net_recurrent_core(
        state: Node,
        qkv: Node,
        z: Node,
        projection_input: Node,
        b_weight: Node,
        a_weight: Node,
        a_log: Node,
        dt_bias: Node,
        norm_weight: Node,
        *,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        epsilon: float,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Update GDN state and produce the local gated value activation."""

        return GatedDeltaNetRecurrentCore.construct(
            state,
            qkv,
            z,
            projection_input,
            b_weight,
            a_weight,
            a_log,
            dt_bias,
            norm_weight,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            epsilon=epsilon,
            name=name,
            metadata=metadata,
        )


class _tensors:
    @staticmethod
    @_op_function(Bitcast)
    def bitcast(
        value: Node,
        dtype,
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Reinterpret tensor storage as another scalar or vector dtype."""

        return Bitcast.construct(
            value, dtype=dtype, name=name, metadata=metadata)

    @staticmethod
    @_op_function(Cast)
    def cast(
        value: Node,
        dtype,
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Convert tensor elements to another scalar or vector data type."""

        return Cast.construct(value, dtype=dtype, name=name, metadata=metadata)

    @staticmethod
    @_op_function(Concat)
    def concat(
        *values: Node,
        axis: int,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Concatenate tensors along one logical axis."""

        return Concat.construct(*values, axis=axis, name=name, metadata=metadata)

    @staticmethod
    @_op_function(Reshape)
    def reshape(
        value: Node,
        shape: tuple[int, ...] | list[int],
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Reshape a static logical tensor without changing element order."""

        return Reshape.construct(value, shape=shape, name=name, metadata=metadata)

    @staticmethod
    @_op_function(Permute)
    def permute(
        value: Node,
        axes: tuple[int, ...] | list[int],
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Permute logical tensor axes."""

        return Permute.construct(value, axes=axes, name=name, metadata=metadata)

    @staticmethod
    @_op_function(Pack)
    def pack(
        value: Node,
        lanes: int | tuple[int, ...] | list[int],
        *,
        axes: int | tuple[int, ...] | list[int] | None = None,
        axis: int | None = None,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Pack logical axes into a typed vector payload."""

        return Pack.construct(value, axes=axes, axis=axis, lanes=lanes, name=name, metadata=metadata)

    @staticmethod
    @_op_function(Unpack)
    def unpack(
        value: Node,
        *,
        axes: int | tuple[int, ...] | list[int] | None = None,
        axis: int | None = None,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Expand leading typed-vector lanes back into logical axes."""

        return Unpack.construct(value, axes=axes, axis=axis, name=name, metadata=metadata)

    @staticmethod
    @_op_function(Pad)
    def pad(
        value: Node,
        pad_end: tuple[int, ...] | list[int],
        *,
        pad_value: int | float = 0.0,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Pad the end of logical tensor axes."""

        return Pad.construct(value, pad_end=pad_end, pad_value=pad_value, name=name, metadata=metadata)

    @staticmethod
    @_op_function(SliceToShape)
    def slice_to_shape(
        value: Node,
        shape: tuple[int, ...] | list[int],
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Remove vectorization padding and restore a static shape."""

        return SliceToShape.construct(value, shape=shape, name=name, metadata=metadata)

    @staticmethod
    @_op_function(GetItem)
    def get_item(
        value: Node,
        index: int,
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Project one field from a tuple-valued expression."""

        return GetItem.construct(value, index, name=name, metadata=metadata)

    @staticmethod
    def get_items(
        value: Node,
        *indices: int,
        name_prefix: str | None = None,
        metadata: Metadata = None,
    ) -> tuple[Node, ...]:
        """Convenience projection of several tuple fields."""

        return tuple(
            GetItem.construct(
                value,
                index,
                name=None if name_prefix is None else f"{name_prefix}_{index}",
                metadata=metadata,
            )
            for index in indices
        )


class _ntt:
    @staticmethod
    @_op_function(GatherReduceAddNormApply)
    def gather_reduce_add_norm_apply(
        input: Node,
        addend: Node,
        scale: Node,
        bias: Node,
        *,
        axis: int,
        epsilon: float,
        use_mean: bool,
        round_before_scale: bool = False,
        has_bias: bool = True,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Materialize a Sum-partial residual and apply normalization."""

        return GatherReduceAddNormApply.construct(
            input,
            addend,
            scale,
            bias,
            axis=axis,
            epsilon=epsilon,
            use_mean=use_mean,
            has_bias=has_bias,
            round_before_scale=round_before_scale,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(GatherReduceNormApply)
    def gather_reduce_norm_apply(
        partial_stats: Node,
        input: Node,
        scale: Node,
        bias: Node,
        *,
        materialized_stats_type: IRType,
        axis: int,
        epsilon: float,
        use_mean: bool,
        round_before_scale: bool = False,
        has_bias: bool = True,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Reduce Sum-partial statistics while applying normalization."""

        return GatherReduceNormApply.construct(
            partial_stats,
            input,
            scale,
            bias,
            materialized_stats_type=materialized_stats_type,
            axis=axis,
            epsilon=epsilon,
            use_mean=use_mean,
            has_bias=has_bias,
            round_before_scale=round_before_scale,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(GatherReduceQKVRoPEWithCache)
    def gather_reduce_qkv_rope_with_cache(
        qkv: Node,
        q_scale: Node,
        k_scale: Node,
        q_bias: Node,
        k_bias: Node,
        cos: Node,
        sin: Node,
        state: Node,
        layer_id: Node,
        advance_sequence: Node,
        *,
        materialized_qkv_type: IRType,
        logical_qkv_type: IRType,
        q_axis: int,
        q_epsilon: float,
        q_use_mean: bool,
        q_round_before_scale: bool = False,
        k_axis: int,
        k_epsilon: float,
        k_use_mean: bool,
        k_round_before_scale: bool = False,
        qkv_layout: tuple[str, str, str],
        attention_layout: tuple[str, str, str],
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Reduce partial Q/K/V inside normalization, RoPE, and cache IO."""

        return GatherReduceQKVRoPEWithCache.construct(
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
            materialized_qkv_type=materialized_qkv_type,
            logical_qkv_type=logical_qkv_type,
            q_axis=q_axis,
            q_epsilon=q_epsilon,
            q_use_mean=q_use_mean,
            k_axis=k_axis,
            k_epsilon=k_epsilon,
            k_use_mean=k_use_mean,
            qkv_layout=qkv_layout,
            attention_layout=attention_layout,
            q_round_before_scale=q_round_before_scale,
            k_round_before_scale=k_round_before_scale,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(VectorizedRoPE)
    def vectorized_rope(
        input: Node,
        cos: Node,
        sin: Node,
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Apply RoPE to a final-axis typed-vector representation."""

        return VectorizedRoPE.construct(
            input, cos, sin, name=name, metadata=metadata
        )

    @staticmethod
    @_op_function(PackedMatMul)
    def packed_matmul(
        lhs: Node,
        rhs: Node,
        scale: Node,
        addend: Node,
        *,
        fused_reduce: bool = False,
        output_data_type: DType | str = DType.FLOAT32,
        rhs_layout: str = "k_major",
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """MatMul over a typed-vector packed RHS."""

        return PackedMatMul.construct(
            lhs,
            rhs,
            scale,
            addend,
            fused_reduce=fused_reduce,
            output_data_type=output_data_type,
            rhs_layout=rhs_layout,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(PagedAttentionPartial)
    def paged_attention_partial(
        q: Node,
        state: Node,
        layer_id: Node,
        *,
        scale: float,
        layout: tuple[str, str, str],
        hidden_size: int,
        split_hierarchy_axis: int,
        split_count: int,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Build explicit per-partition paged-attention softmax states."""

        return PagedAttentionPartial.construct(
            q,
            state,
            layer_id,
            scale=scale,
            layout=layout,
            hidden_size=hidden_size,
            split_hierarchy_axis=split_hierarchy_axis,
            split_count=split_count,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(PagedAttentionCombine)
    def paged_attention_combine(
        max_state: Node,
        sum_state: Node,
        acc_state: Node,
        *,
        layout: tuple[str, str, str],
        hidden_size: int,
        output_data_type,
        output_type: IRType,
        split_hierarchy_axis: int,
        split_count: int,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Discharge paged-attention partial states into one output tensor."""

        return PagedAttentionCombine.construct(
            max_state,
            sum_state,
            acc_state,
            layout=layout,
            hidden_size=hidden_size,
            output_data_type=output_data_type,
            output_type=output_type,
            split_hierarchy_axis=split_hierarchy_axis,
            split_count=split_count,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(PackedQKVParallelLinearCombine)
    def packed_qkv_parallel_linear_combine(
        qkv: Node,
        output_type: IRType,
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Materialize coupled split-K Q/K/V sums into an output layout."""

        return PackedQKVParallelLinearCombine.construct(
            qkv,
            output_type,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(PackedQKVParallelLinear)
    def packed_qkv_parallel_linear(
        input: Node,
        q_weight: Node,
        k_weight: Node,
        v_weight: Node,
        q_bias: Node,
        k_bias: Node,
        v_bias: Node,
        q_input_scale: Node,
        k_input_scale: Node,
        v_input_scale: Node,
        q_weight_scale: Node,
        k_weight_scale: Node,
        v_weight_scale: Node,
        *,
        num_heads: int,
        num_kv_heads: int,
        output_data_type: str,
        rhs_layout: str = "k_major",
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Q/K/V projection over three independent K-major vector weights."""

        return PackedQKVParallelLinear.construct(
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
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            output_data_type=output_data_type,
            rhs_layout=rhs_layout,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(MatMulNormStats)
    def matmul_norm_stats(
        lhs: Node,
        rhs: Node,
        addend: Node,
        *,
        transpose_a: bool = False,
        transpose_b: bool = False,
        rhs_layout: str | None = None,
        axis: int,
        use_mean: bool,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Dense matmul, residual addition, and explicit value/stats results."""

        return MatMulNormStats.construct(
            lhs,
            rhs,
            addend,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            rhs_layout=rhs_layout,
            axis=axis,
            use_mean=use_mean,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(MatMulNormStatsCombine)
    def matmul_norm_stats_combine(
        input: Node,
        addend: Node,
        *,
        axis: int,
        use_mean: bool,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Materialize a matmul partial, add a residual, and return value/stats."""

        return MatMulNormStatsCombine.construct(
            input,
            addend,
            axis=axis,
            use_mean=use_mean,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(VectorizedCast)
    def vectorized_cast(
        value: Node,
        new_type,
        vectorize_axes: int | tuple[int, ...] | list[int],
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Cast a typed vector while preserving its scalar logical shape."""

        return VectorizedCast.construct(
            value,
            new_type=new_type,
            vectorize_axes=vectorize_axes,
            name=name,
            metadata=metadata,
        )


class _tir:
    @staticmethod
    @_op_function(Call)
    def call(
        *arguments: Node,
        result_type: IRType,
        callee: str,
        effect: Effect = PURE,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Call a named TIR function through its typed value ABI."""

        return Call.construct(
            *arguments,
            callee=callee,
            result_type=result_type,
            effect=effect,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(ScalarConst)
    def scalar_const(
        result_type: IRType,
        value: bool | int | float,
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Construct a rank-zero compile-time TIR value."""

        return ScalarConst.construct(
            result_type,
            value,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(Barrier)
    def barrier(
        result_type: IRType,
        *,
        attrs: Mapping[str, Any],
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Construct an explicit TIR synchronization barrier."""

        return Barrier.construct(result_type, attrs=attrs, name=name, metadata=metadata)

    @staticmethod
    @_op_function(Buffer)
    def buffer(
        result_type: IRType,
        *,
        weight_name: str,
        source: str,
        key: str,
        storage: str,
        alignment: int,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Construct a semantic TIR buffer/weight reference."""

        return Buffer.construct(
            result_type,
            weight_name=weight_name,
            source=source,
            key=key,
            storage=storage,
            alignment=alignment,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(BufferView)
    def buffer_view(
        value: Node,
        new_type: IRType,
        *,
        alias_kind: str = "sharded_view",
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Construct a typed zero-copy view of an existing physical buffer."""

        return BufferView.construct(
            value,
            new_type,
            alias_kind=alias_kind,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(Kernel)
    def kernel(
        *arguments: Node,
        result_type: IRType,
        semantic_op: str,
        candidate: str,
        parameters: Mapping[str, Any],
        facts: Mapping[str, Any],
        semantic_attrs: Mapping[str, Any],
        effect: Effect = PURE,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Construct one selected semantic TIR kernel call."""

        return Kernel.construct(
            *arguments,
            result_type=result_type,
            semantic_op=semantic_op,
            candidate=candidate,
            parameters=parameters,
            facts=facts,
            semantic_attrs=semantic_attrs,
            effect=effect,
            name=name,
            metadata=metadata,
        )


class _builtin:
    @staticmethod
    @_op_function(NoneValue)
    def none(
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Construct the first-class optional-input sentinel."""

        return NoneValue.construct(name=name, metadata=metadata)

    @staticmethod
    @_op_function(TupleValue)
    def tuple(
        *fields: Node,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Construct a tuple from ordinary IR values."""

        return TupleValue.construct(*fields, name=name, metadata=metadata)

    @staticmethod
    @_op_function(BuiltinCall)
    def call(
        *arguments: Node,
        result_type: IRType,
        callee: str,
        effect: Effect = PURE,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Call a named imported function through its typed value ABI."""

        return BuiltinCall.construct(
            *arguments,
            callee=callee,
            result_type=result_type,
            effect=effect,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(BuiltinScalarConst)
    def scalar_const(
        result_type: IRType,
        value: bool | int | float,
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Construct a rank-zero argument value in high-level IR."""

        return BuiltinScalarConst.construct(
            result_type,
            value,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(SplatConst)
    def splat_const(
        result_type: IRType,
        value: bool | int | float,
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Construct a compressed, scalar-filled compile-time tensor."""

        return SplatConst.construct(
            result_type,
            value,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(ConstAsset)
    def const_asset(
        result_type: IRType,
        *,
        recipe: str,
        output: str,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Reference one output of a frozen compile-time constant recipe."""

        return ConstAsset.construct(
            result_type,
            recipe=recipe,
            output=output,
            name=name,
            metadata=metadata,
        )


class _distributed:
    @staticmethod
    @_op_function(MaterializeLocalShards)
    def materialize_local_shards(
        value: Node,
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Offline-materialize dense local shards in placement owner order."""

        return MaterializeLocalShards.construct(
            value,
            name=name,
            metadata=metadata,
        )

    @staticmethod
    @_op_function(Boxing)
    def boxing(
        value: Node,
        new_type: IRType,
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Materialize a legal reshard or a distributed/logical boundary."""

        return Boxing.construct(value, new_type, name=name, metadata=metadata)

    @staticmethod
    @_op_function(ShardedView)
    def sharded_view(
        value: Node,
        new_type: IRType,
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Create a read-only distributed alias without copying storage."""

        return ShardedView.construct(value, new_type, name=name, metadata=metadata)

    @staticmethod
    @_op_function(ForceBoxing)
    def force_boxing(
        value: Node,
        new_type: IRType,
        *,
        name: str | None = None,
        metadata: Metadata = None,
    ) -> Node:
        """Force a boxing edge for rule/pass tests."""

        return ForceBoxing.construct(value, new_type, name=name, metadata=metadata)


class F:
    """Static functional namespaces used by editable Python IR."""

    builtin = _builtin
    distributed = _distributed
    math = _math
    nn = _nn
    ntt = _ntt
    tensors = _tensors
    tir = _tir


__all__ = ["F", "construction_scope"]
