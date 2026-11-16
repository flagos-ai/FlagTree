# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Candidates for the explicit six-input RMS/BF16/GLU contract."""
from dataclasses import replace
from math import copysign

from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.ntt.rms_packed_dense_matmul_glu import composition_nodes
from triton.flagmega.passes.tir.norm_consumer import _constant_splat_value
from .core import TritonCandidateProposal
from .dense_matmul_glu import dense_glu_implementation_applicable


class PartialRmsPackedDenseMatmulGluCandidateProvider:
    op_names = frozenset({'ntt.partial_rms_packed_dense_matmul_glu'})

    def propose(self, node, context):
        inputs = tuple(context.module.node_map[key] for key in node.inputs)
        _, normalized, glu = composition_nodes(inputs[:5], node.attrs)
        # This proxy is used only by the existing physical-shape proof. The
        # actual proposed node, dispatch schema and input ABI remain six-input.
        proxy = replace(context.module, nodes=(*context.module.nodes, normalized))
        raw_type = tensor_of(inputs[0].type)
        if raw_type.shape[0].fixed_value != 1:
            return None
        k = raw_type.shape[1].fixed_value
        partial_type = inputs[5].type
        if (k != 2048 or node.attrs['norm_tree_group_size'] != 16
                or node.attrs['norm_partial_group_size'] != 32
                or node.attrs['round_before_scale']
                or partial_type.partial.axes != (0, 1)
                or partial_type.placement.hierarchy != (8, 16)
                or partial_type.placement.hierarchy_levels != 'bb'):
            return None
        bias_value = _constant_splat_value(inputs[2], context.module)
        zero = bias_value == 0.0 and copysign(1.0, bias_value) == 1.0
        variants = []
        for implementation in context.implementations(
                'dense_matmul_glu', input_kind='packed', fusion='explicit_partial_rms',
                packed_layout=node.attrs['packed_layout']):
            pipeline = implementation.transfer_pipeline
            if (pipeline is None or len(pipeline.channels) != 1
                    or pipeline.channels[0].source_argument_indices != (3, 4)
                    or pipeline.channels[0].shared_workspace_indices != (0, 1)
                    or pipeline.consumer_shared_workspace_indices != (2,)
                    or implementation.parameters.get('lhs_stage_extent') != k
                    or not implementation.parameters.get('paired_weight_fields')
                    or implementation.parameters.get('descriptor_kind') != 'table'
                    or node.attrs['norm_math_mode'] not in implementation.contract.get('norm_math_modes', ())
                    or not dense_glu_implementation_applicable(glu, proxy, implementation)):
                continue
            variants.append(context.configure_implementation(implementation,
                semantic_parameters={'packed_layout': node.attrs['packed_layout'],
                                     'norm_bias_is_zero': zero},
                facts={'explicit_partial_rms_input': True, 'materializes_input_norm': False}))
        if not variants:
            return None
        variants = tuple(variants)
        return TritonCandidateProposal(variants, context.choose_default('dense_matmul_glu', variants))


__all__ = ['PartialRmsPackedDenseMatmulGluCandidateProvider']
