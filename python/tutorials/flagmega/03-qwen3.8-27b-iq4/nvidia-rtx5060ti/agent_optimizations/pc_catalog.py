"""Typed native-IQ4 producer/consumer candidates scoped to this tutorial."""
from dataclasses import replace
from math import prod

from triton.flagmega.codegen.triton.candidates import (
    TritonCandidateProposal,
    TritonCandidateProviderRegistry,
    BlockFp8CandidateProvider,
    MatmulGluCandidateProvider,
)
from triton.flagmega.codegen.triton.implementation import TritonImplementation
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.ir import DType, tensor_type
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.tir import (
    TIRSharedWorkspaceDescriptor,
    TIRTransferPipelineChannel,
    TIRTransferPipelineContract,
)
from triton.flagmega.targets import NvidiaSm120Target


def _power(value):
    return 1 << (value - 1).bit_length()


def _candidate(family, k, *, rows, leaf_inline=False, segmented=False, dual_activation=False, dense_dual=False,
               global_activation=False, packed_activation=False, down_chunk=1024):
    glu = family == 'matmul_glu'
    fields = ('gate_stage', 'up_stage') if glu else ('weight_stage', )
    stages = 1 if dense_dual and not glu and not global_activation else 2
    segments = ((2048, 512, 128, 32) if dense_dual else
                (2048, 672)) if glu else (2048, 264) if rows == 1 else (4096, 512, 16)
    if segmented:
        assert (glu and k == 5120 and rows == 4) or (not glu and k == 17408 and rows in (1, 2))
        assert sum(segments) == rows * k // 256 * 34
        weights = tuple(
            TIRSharedWorkspaceDescriptor(f'{name}_{i}', tensor_type('int32', (
                stages, _power(size))), 128, matrix_compatible=False)
            for name in fields
            for i, size in enumerate(segments))
    else:
        weights = tuple(
            TIRSharedWorkspaceDescriptor(name, tensor_type('int32', (stages, _power(rows * k // 256 *
                                                                                    34))), 128, matrix_compatible=False)
            for name in fields)
    split = 512 if k == 17408 else 128 if dense_dual else 0
    consumer = (
        TIRSharedWorkspaceDescriptor('q8_words', tensor_type('int32', (split * 8 if split else _power(k // 4), )), 128,
                                     matrix_compatible=False),
        TIRSharedWorkspaceDescriptor('q8_scales', tensor_type('float16', (split if split else _power(k // 32), )), 128,
                                     matrix_compatible=False),
    )
    if split:
        consumer += (
            TIRSharedWorkspaceDescriptor('q8_words_tail', tensor_type('int32', (_power(k // 4 - split * 8), )), 128,
                                         matrix_compatible=False),
            TIRSharedWorkspaceDescriptor('q8_scales_tail', tensor_type('float16', (_power(k // 32 - split), )), 128,
                                         matrix_compatible=False),
        )
    if dual_activation:
        consumer += (
            TIRSharedWorkspaceDescriptor('q8_residual_words',
                                         tensor_type('int32', (split * 8 if split else _power(k // 4), )), 128,
                                         matrix_compatible=False),
            TIRSharedWorkspaceDescriptor('q8_residual_scales',
                                         tensor_type('float16', (split if split else _power(k // 32), )), 128,
                                         matrix_compatible=False),
        )
        if split:
            consumer += (
                TIRSharedWorkspaceDescriptor('q8_residual_words_tail',
                                             tensor_type('int32', (_power(k // 4 - split * 8), )), 128,
                                             matrix_compatible=False),
                TIRSharedWorkspaceDescriptor('q8_residual_scales_tail',
                                             tensor_type('float16', (_power(k // 32 - split), )), 128,
                                             matrix_compatible=False),
            )
    quantization = 'q8_1_rne_residual2' if dual_activation else 'q8_1_rne'
    if global_activation:
        assert dual_activation
        consumer = ()
    return TritonImplementation(
        f'tir.{family}.iq4_pc_tma.k{k}' + ('.inline' if leaf_inline else '') + ('.segments' if segmented else '') +
        ('.q8x2' if dual_activation else '') + ('.dense' if dense_dual else '') +
        ('.global' if global_activation else '') + ('.packed' if packed_activation else '') +
        (f'.c{down_chunk}' if global_activation and k == 17408 and down_chunk != 1024 else ''),
        family,
        'simt',
        parameters={
            'tile_n': rows, 'compute_num_warps': 8, 'consumer_warps': 8, 'producer_warps': 1, 'producer_registers': 32,
            'num_stages': stages, 'iq4_chunk':
            down_chunk if global_activation and k == 17408 else 512 if rows >= 4 else 1024 if rows == 2 else 2048,
            'q8_split_groups': split, 'required_row_bytes': k // 256 * 136,
            **({'weight_segments': segments} if segmented else {}),
            **({'activation_parts': 2} if dual_activation else {})
        },
        contract={'weight_encoding': 'IQ4_XS', 'activation_quantization': quantization},
        requires=('tma', 'warp_specialize'),
        facts={
            'transfer_pipeline':
            True, 'native_quantized_weights':
            True, 'activation_quantization':
            quantization, 'transport':
            'bulk_tma', **({'activation_parts': 2} if dual_activation else {}), **({
                'activation_storage':
                'cta_global_interleaved' if packed_activation else 'cta_global'
            } if global_activation else {}), **({'dense_dual_buffers': True} if dense_dual else {}),
            **({'leaf_inline': True} if leaf_inline else {}), **({'weight_segments': True} if segmented else {})
        },
        shared_workspaces=(*weights, *consumer),
        transfer_pipeline=TIRTransferPipelineContract(
            (TIRTransferPipelineChannel('weights' if glu else 'weight', source_argument_indices=(1, 2) if glu else
                                        (1, ), shared_workspace_indices=tuple(range(
                                            len(weights))), source_alignment_bytes=16), ),
            consumer_shared_workspace_indices=tuple(range(len(weights),
                                                          len(weights) + len(consumer))), capacity=stages),
    )


class NativeProvider:

    def __init__(self, original, family, leaf_inline=False, segmented=False, dual_activation=False, dual_all=False,
                 dense_dual=False, global_activation=False, packed_activation=False, down_chunk=1024,
                 precision_scope='all', nibble_pair=False):
        self.original, self.family = original, family
        self.leaf_inline = leaf_inline
        self.segmented = segmented
        self.dual_activation = dual_activation
        self.dual_all = dual_all
        self.dense_dual = dense_dual
        self.global_activation = global_activation
        self.packed_activation = packed_activation
        self.down_chunk = down_chunk
        self.precision_scope = precision_scope
        self.nibble_pair = nibble_pair
        self.op_names = original.op_names

    def propose(self, node, context):
        encoding = node.attrs.get('weight_encoding') if self.family == 'block_fp8' else node.attrs.get(
            'gate_weight_encoding')
        weight = tensor_of(context.module.node_map[node.inputs[1]].type)
        row_bytes = weight.shape[-1].fixed_value if encoding == 'IQ4_XS' else None
        segmented = self.segmented and row_bytes == (2720 if self.family == 'matmul_glu' else 9248)
        output_dtype = tensor_of(node.type).dtype
        dual_activation = (self.dual_activation and self.family == 'block_fp8' and row_bytes in (2720, 3264)
                           and getattr(output_dtype, 'elem_type', output_dtype) != DType.FLOAT32)
        if self.dual_all and getattr(output_dtype, 'elem_type', output_dtype) != DType.FLOAT32:
            dual_activation = True
            scope = 'glu' if self.family == 'matmul_glu' else 'down' if row_bytes == 9248 else 'attention'
            dual_activation = (self.precision_scope == 'all' or self.precision_scope == scope
                               or (self.precision_scope == 'ffn' and scope in ('glu', 'down')))
            if self.family == 'matmul_glu' and dual_activation:
                segmented = False
            elif row_bytes == 9248 and dual_activation:
                segmented = True
        dense_dual = self.dense_dual and dual_activation and (self.family == 'matmul_glu' or row_bytes == 9248)
        if dense_dual:
            segmented = True
        implementations = tuple(
            value for value in context.implementations(self.family)
            if value.facts.get('transport') == 'bulk_tma' and bool(value.facts.get('leaf_inline', False)) ==
            self.leaf_inline and bool(value.facts.get('activation_nibble_pairs', False)) == self.nibble_pair and bool(
                value.facts.get('weight_segments', False)) == segmented and (value.facts.get('activation_parts', 1) == (
                    2 if dual_activation else 1)) and bool(value.facts.get('dense_dual_buffers', False)) == dense_dual
            and value.facts.get('activation_storage') == (
                ('cta_global_interleaved' if self.packed_activation else 'cta_global'
                 ) if self.global_activation and dual_activation else None) and (row_bytes != 9248 or not (
                     self.global_activation and dual_activation) or value.parameters['iq4_chunk'] == self.down_chunk)
            and value.parameters['required_row_bytes'] == row_bytes)
        if not implementations:
            baseline = tuple(
                context.configure_implementation(value)
                for value in context.implementations(self.family)
                if value.facts.get('transport') != 'bulk_tma' and (encoding is None or value.variant == 'simt'))
            return None if not baseline else TritonCandidateProposal(baseline,
                                                                     context.choose_default(self.family, baseline))
        if self.family == 'matmul_glu' and node.attrs.get('up_weight_encoding') != encoding:
            return self.original.propose(node, context)
        mesh_size = prod(context.target.distributed_placements(context.module)[0].hierarchy)
        candidates = []
        for value in implementations:
            semantic = {}
            if value.facts.get('activation_storage') in ('cta_global', 'cta_global_interleaved'):
                k = row_bytes // 136 * 256
                formats = (('q8_words', 'int64', k // 4),
                           ('q8_scales', 'int32',
                            k // 32)) if self.packed_activation else (('q8_words', 'int32',
                                                                       k // 4), ('q8_scales', 'float16', k // 32),
                                                                      ('q8_residual_words', 'int32', k // 4),
                                                                      ('q8_residual_scales', 'float16', k // 32))
                semantic['workspaces'] = tuple({
                    'name': name, 'type': tensor_type(dtype, (
                        mesh_size, width)), 'memory_space': 'workspace', 'alignment': 128, 'lifetime': 'invocation'
                } for name, dtype, width in formats)
            candidates.append(context.configure_implementation(value, semantic_parameters=semantic))
        candidates = tuple(candidates)
        return TritonCandidateProposal(candidates, candidates[0].id)


def configure(target, *, leaf_inline=False, segmented=False, dual_activation=False, dual_all=False, dense_dual=False,
              global_activation=False, packed_activation=False, down_chunk=1024, precision_scope='all',
              nibble_pair=False):
    old_registry = target.tir_selection_policy.registry
    registry = TritonCandidateProviderRegistry()
    for provider in old_registry.providers:
        if isinstance(provider, BlockFp8CandidateProvider):
            provider = NativeProvider(provider, 'block_fp8', leaf_inline, segmented, dual_activation, dual_all,
                                      dense_dual, global_activation, packed_activation, down_chunk, precision_scope,
                                      nibble_pair)
        elif isinstance(provider, MatmulGluCandidateProvider):
            provider = NativeProvider(provider, 'matmul_glu', leaf_inline, segmented, dual_activation, dual_all,
                                      dense_dual, global_activation, packed_activation, down_chunk, precision_scope,
                                      nibble_pair)
        registry.add(provider)
    capability = replace(target.machine.capability, supports_tma=True, supports_warp_specialize=True)
    configured = NvidiaSm120Target(capability, options=target.options)
    model = target.triton_implementation_model
    added = tuple(_candidate('block_fp8', k, rows=1 if k == 17408 else 4)
                  for k in (5120, 6144, 17408)) + (_candidate('matmul_glu', 5120, rows=2), )
    added += tuple(
        _candidate('block_fp8', k, rows=1 if k == 17408 else 4, leaf_inline=True)
        for k in (5120, 6144, 17408)) + (_candidate('matmul_glu', 5120, rows=2, leaf_inline=True), )
    added += (_candidate('block_fp8', 17408, rows=2,
                         segmented=True), _candidate('matmul_glu', 5120, rows=4, segmented=True))
    added += tuple(_candidate('block_fp8', k, rows=4, dual_activation=True) for k in (5120, 6144))
    added += (_candidate('block_fp8', 17408, rows=1, segmented=True,
                         dual_activation=True), _candidate('matmul_glu', 5120, rows=2, dual_activation=True))
    added += (_candidate('block_fp8', 17408, rows=2, segmented=True, dual_activation=True, dense_dual=True),
              _candidate('matmul_glu', 5120, rows=4, segmented=True, dual_activation=True, dense_dual=True))
    added += tuple(
        _candidate('block_fp8', k, rows=4, dual_activation=True, global_activation=True) for k in (5120, 6144))
    added += (_candidate('block_fp8', 17408, rows=2, segmented=True, dual_activation=True, dense_dual=True,
                         global_activation=True),
              _candidate('matmul_glu', 5120, rows=4, segmented=True, dual_activation=True, dense_dual=True,
                         global_activation=True))
    added += tuple(
        _candidate('block_fp8', k, rows=4, dual_activation=True, global_activation=True, packed_activation=True)
        for k in (5120, 6144))
    added += (_candidate('matmul_glu', 5120, rows=4, segmented=True, dual_activation=True, dense_dual=True,
                         global_activation=True, packed_activation=True), )
    added += tuple(
        _candidate('block_fp8', 17408, rows=2, segmented=True, dual_activation=True, dense_dual=True,
                   global_activation=True, packed_activation=packed, down_chunk=chunk)
        for packed in (False, True)
        for chunk in (1024, 2048, 4096)
        if packed or chunk != 1024)
    added += tuple(
        replace(value, id=value.id + '.pair', facts={**value.facts, 'activation_nibble_pairs': True},
                contract={**value.contract, 'activation_layout': 'iq4_nibble_pairs'}) for value in added)
    configured.triton_implementation_model = replace(model, implementations=(*model.implementations, *added))
    configured.tir_selection_policy = TritonTirSelectionPolicy(configured.machine.annotate_workspaces, registry)
    target.__dict__.update(configured.__dict__)
