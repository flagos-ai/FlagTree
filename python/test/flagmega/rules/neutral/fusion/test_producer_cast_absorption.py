# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Preserve materialized values while absorbing private producer casts."""
from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.fusion import fusion_rules
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator, PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.passes.functions.post_boundary_pack import post_function_boundary_pack_propagation
from triton.flagmega.targets import NvidiaSm90Target


def embedding_graph(*, shared=None):
    mesh = fm.Placement((2, 2), 'yx', 'bb')
    broad = fm.SBP.broadcast()
    state_config = PagedAttentionStateConfig(1, 2, 32)

    class Graph(fm.Module):
        def forward(self):
            ids = self.input('ids', fm.DistributedType(fm.tensor_type('int64', (2,)), (broad,), mesh))
            weight = self.input('weight', fm.DistributedType(
                fm.tensor_type(fm.vector_type('bfloat16', (4,)), (7, 4)),
                (broad, fm.SBP.split_contiguous((0, 1), 1)), mesh))
            state = self.input('state', state_config.ref_type)
            embedded = fm.F.nn.embedding(ids, weight, padding_idx=0, name='embedded')
            target = replace(embedded.type, axis_policies=(broad, broad))
            view = fm.F.distributed.sharded_view(embedded, target, name='view')
            positions = fm.F.nn.rotary_embedding(view, state, head_dim=32, theta=10000., name='positions')
            cosine = fm.F.tensors.get_item(positions, index=0)
            wide = fm.F.ntt.vectorized_cast(view, fm.vector_type('float32', (4,)), (1,), name='wide')
            outputs = [wide, cosine]
            if shared == 'exported':
                outputs.append(view)
            elif shared == 'numeric':
                outputs.append(fm.F.math.add(view, view))
            self.function('main', (ids, weight, state), tuple(outputs))

    return Graph(dialect='ntt', stage='frozen_constants', entry='main',
                 metadata={'auto_distribution': {'placement': mesh.to_data()}}).build(), state_config


def dense_graph(*, shared=False, partial=False, k=32, n=32):
    mesh = fm.Placement((2, 2), 'yx', 'bb')
    class Graph(fm.Module):
        def forward(self):
            x_type = fm.tensor_type('bfloat16', (1, k))
            weight_type = fm.tensor_type(fm.vector_type('bfloat16', (8, 2, 8)), (k // 16, n // 8))
            if partial:
                x_type = fm.DistributedType(x_type, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 16)), mesh)
                weight_type = fm.DistributedType(weight_type,
                    (fm.SBP.split_contiguous((0,), 1), fm.SBP.split_contiguous((1,), 1)), mesh)
            x = self.input('x', x_type)
            weight = self.input('weight', weight_type)
            none = fm.F.builtin.none()
            projection = fm.F.ntt.packed_matmul(x, weight, none, none, output_data_type='bfloat16', name='projection',
                metadata={'selected_vectorization': 'vectorization.matmul.n', 'selected_vector_axes': (1,),
                          'selected_vector_lanes': (8,)})
            view = projection if partial else fm.F.tensors.bitcast(projection, 'bfloat16', name='view')
            wide = fm.F.tensors.cast(view, fm.vector_type('float32', (8,)) if partial else 'float32', name='wide')
            self.function('main', (x, weight), (wide, view) if shared else (wide,))
    return Graph(dialect='ntt', stage='frozen_constants', entry='main').build()


def rotary_graph(*, mixed=False, exported=False, cached=False, axes=(2,), mixed_layout=False):
    config = PagedAttentionStateConfig(1, 2, 32)
    class Graph(fm.Module):
        def forward(self):
            x = self.input('x', fm.tensor_type('bfloat16', (2, 16)))
            state = self.input('state', config.ref_type)
            cache = (self.input('cache', fm.tensor_type('float32', (config.max_sequence_length, 2, 32))),) if cached else ()
            pair = fm.F.nn.rotary_embedding(x, state, *cache, head_dim=32, theta=10000., output_lanes=(2, 2, 4), name='rotary')
            fields = [fm.F.tensors.get_item(pair, index=i, name=f'field{i}') for i in range(2)]
            narrow = [fm.F.ntt.vectorized_cast(field, fm.vector_type('bfloat16', (4, 4) if mixed_layout and i else (2, 8)), axes,
                      rounding_contract='preserve', name=f'narrow{i}') for i, field in enumerate(fields)]
            self.function('main', (x, state, *cache), tuple(narrow + ([fields[0]] if mixed else []) + ([pair] if exported else [])))
    return Graph(dialect='high_level', stage='packed', entry='main').build(), config


def test_embedding_shape_only_shared_view_absorbs_widening(tmp_path):
    source, config = embedding_graph()
    result = DataflowPass('Fuse', fusion_rules(), rewrite_constants=False).run(source)
    assert not any(n.op in {'tensors.cast', 'ntt.vectorized_cast'} for n in result.nodes)
    assert len([n for n in result.nodes if n.op == 'nn.embedding']) == 1
    assert result.node_map['positions'].effect == source.node_map['positions'].effect
    assert fm.load_module(fm.emit_module(result, tmp_path/'embedding.py')).semantic_hash == result.semantic_hash
    evaluator = TorchEvaluator(DictWeightResolver({}))
    values = {'ids': torch.tensor([0, 6]), 'weight': torch.arange(112).reshape(7, 4, 4).bfloat16()/32,
              'state': create_paged_attention_state(config, device='cpu')}
    torch.testing.assert_close(evaluator.run(result, values), evaluator.run(source, values), rtol=0, atol=0)


def test_dense_bf16_materialization_survives_widening_epilogue(tmp_path):
    source = dense_graph()
    result = DataflowPass('Fuse', fusion_rules(), rewrite_constants=False).run(source)
    assert not any(n.op in {'tensors.cast', 'ntt.vectorized_cast'} for n in result.nodes)
    producer, = (n for n in result.nodes if n.op == 'ntt.packed_matmul')
    assert producer.attrs['output_data_type'] == 'bfloat16'
    assert producer.attrs['post_ops'][0].nodes[-1].op == 'tensors.cast'
    assert fm.load_module(fm.emit_module(result, tmp_path/'dense.py')).semantic_hash == result.semantic_hash
    weight = (torch.arange(1024).reshape(32, 32)%29-14).bfloat16()/32
    packed = weight.reshape(2, 2, 8, 4, 8).permute(0, 3, 4, 1, 2).contiguous()
    values = {'x': torch.linspace(-1.125, 1.1875, 32).reshape(1, 32).bfloat16(), 'weight': packed}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    actual, = evaluator.run(result, values)
    expected, = evaluator.run(source, values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, (values['x'].float()@weight.float()).bfloat16().float(), rtol=0, atol=0)


def test_rotary_absorbs_both_fields_at_single_original_state_read(tmp_path):
    source, config = rotary_graph()
    result = post_function_boundary_pack_propagation(source, NvidiaSm90Target())
    assert not any(n.op in {'tensors.cast', 'ntt.vectorized_cast'} for n in result.nodes)
    old, = (n for n in source.nodes if n.op == 'nn.rotary_embedding')
    new, = (n for n in result.nodes if n.op == 'nn.rotary_embedding')
    assert new.id == old.id and new.effect == old.effect and new.inputs == old.inputs
    assert new.attrs['output_dtype'] == 'bfloat16' and tuple(new.attrs['output_lanes']) == (2, 8)
    assert post_function_boundary_pack_propagation(result, NvidiaSm90Target()).semantic_hash == result.semantic_hash
    assert fm.load_module(fm.emit_module(result, tmp_path/'rotary.py')).semantic_hash == result.semantic_hash
    values = {'x': torch.zeros((2, 16), dtype=torch.bfloat16), 'state': create_paged_attention_state(config, device='cpu')}
    values['state'].seq_lens.fill_(127)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(result, values), evaluator.run(source, values), rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['exported', 'numeric'])
def test_embedding_numeric_or_exported_old_values_are_preserved(kind):
    source, _ = embedding_graph(shared=kind)
    result = DataflowPass('Fuse', fusion_rules(), rewrite_constants=False).run(source)
    assert result.node_map['wide'].op == 'ntt.vectorized_cast'


def test_dense_exported_bf16_view_is_preserved():
    source = dense_graph(shared=True)
    result = DataflowPass('Fuse', fusion_rules(), rewrite_constants=False).run(source)
    assert result.node_map['wide'].op == 'tensors.cast'


def test_partial_dense_accumulators_are_not_rounded_by_post_ops():
    source = dense_graph(partial=True)
    result = DataflowPass('Fuse', fusion_rules(), rewrite_constants=False).run(source)
    assert result.node_map['projection'].type.partial is not None
    assert not result.node_map['projection'].attrs.get('post_ops')
    assert result.node_map['wide'].op == 'tensors.cast'


@pytest.mark.parametrize('kind', ['mixed', 'exported', 'cached', 'mixed_layout'])
def test_rotary_observable_fp32_values_are_preserved(kind):
    source, _ = rotary_graph(**{kind: True})
    result = post_function_boundary_pack_propagation(source, NvidiaSm90Target())
    producer, = (n for n in result.nodes if n.op == 'nn.rotary_embedding')
    assert producer.attrs.get('output_dtype', 'float32') == 'float32'
    assert any(n.op in {'tensors.cast', 'ntt.vectorized_cast'} for n in result.nodes)


def test_rotary_nontrailing_lane_cast_is_not_absorbed():
    source, _ = rotary_graph(axes=(0,))
    result = post_function_boundary_pack_propagation(source, NvidiaSm90Target())
    producer, = (n for n in result.nodes if n.op == 'nn.rotary_embedding')
    assert producer.attrs.get('output_dtype', 'float32') == 'float32'


@pytest.mark.parametrize('kind', ['embedding', 'dense', 'rotary'])
def test_ordinary_codegen_absorbs_cast_with_declared_store_dtype(kind, tmp_path):
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    source = embedding_graph()[0] if kind == 'embedding' else dense_graph() if kind == 'dense' else rotary_graph()[0]
    compiled = Compiler().compile(source).module
    assert not any(k.dispatch.semantic_op in {'tensors.cast', 'ntt.vectorized_cast'} for k in compiled.kernel_definitions)
    artifact = write_artifact(compiled, tmp_path/kind, target='nvidia-sm90', emit_executable=True)
    text = (artifact/'generated_kernels.py').read_text()
    if kind in {'embedding', 'dense'}:
        assert '_fm_producer_post_0' in text
        assert '.to(tl.bfloat16)).to(tl.float32)' in text
    else:
        kernel, = (k for k in compiled.kernel_definitions if k.dispatch.semantic_op == 'nn.rotary_embedding')
        for name in kernel.dispatch.outputs:
            dtype = fm.logical_type(kernel.parameter_map[name].type).dtype
            assert (dtype.elem_type if isinstance(dtype, fm.VectorType) else dtype) == fm.DType.BFLOAT16


def test_ordinary_tma_codegen_preserves_projection_rounding(tmp_path):
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.codegen.triton import describe_tir_package
    from triton.flagmega.compiler import Compiler
    compiled = Compiler().compile(dense_graph(k=2048, n=512)).module
    call, = (call for call in describe_tir_package(compiled)['render_calls'] if call['semantic_op'] == 'ntt.packed_matmul')
    assert call['variant'] == 'packed_tensor_descriptor_smem_pipeline_gemv'
    assert not any(k.dispatch.semantic_op in {'tensors.cast', 'ntt.vectorized_cast'} for k in compiled.kernel_definitions)
    artifact = write_artifact(compiled, tmp_path/'tma', target='nvidia-sm90', emit_executable=True)
    source = (artifact/'generated_kernels.py').read_text()
    assert '_fm_producer_post_0 = ((dense_accumulator).to(tl.bfloat16)).to(tl.float32)' in source
    assert source.index('dense_accumulator = tl.sum(dense_partial, axis=1)') < source.index('_fm_producer_post_0 =')


@pytest.mark.parametrize('n', [256, 512])
def test_corex_dense_override_preserves_projection_rounding(n):
    from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry
    call = dict(family='dense_matmul', variant='packed_k_major_gemv', symbol='run', signature='X,W,Y',
                local_m_capacity=1, local_n_capacity=n, local_k_capacity=128, tile_n=128, block_k=64,
                source='X', weight='W', result='Y', source_active='True', result_active='True', output_type='tl.float32',
                global_n='dense_local_n_offsets', global_k='dense_local_k_offsets', source_offset='dense_local_k_offsets',
                result_offset='dense_local_n_offsets', weight_offset='0',
                corex_flat_sme_gemv=dict(source='X', weight='W', result='Y', weight_k_group_stride=n * 16),
                fusion_post_lines=('_fm_producer_post_0 = ((dense_result).to(tl.bfloat16)).to(tl.float32)',),
                fusion_post_value='_fm_producer_post_0')
    source = TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec('dense_matmul', 'packed_k_major_gemv', 'corex', 'bi_v150'),
        dict(render_calls=[call], num_warps=16, distributed_entry=False, mesh_hierarchy=(1, 1)),
    ).source
    compile(source, 'corex_projection_cast.py', 'exec')
    assert '_fm_producer_post_0 = ((dense_result).to(tl.bfloat16)).to(tl.float32)' in source
    assert source.index('dense_result=partials.reshape(') < source.index('_fm_producer_post_0 =')
    assert ')),_fm_producer_post_0,mask=' in source


@pytest.mark.parametrize('variant', ['mma', 'packed_k_major_mma', 'packed_k_major_threaded_fma'])
def test_unsupported_dense_variants_reject_widening_before_abi_access(variant):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _dense_matmul_call
    from triton.flagmega.errors import CodegenError
    with pytest.raises(CodegenError, match='has no BF16 projection widening epilogue'):
        _dense_matmul_call({'semantic_op': 'ntt.packed_matmul', 'variant': variant,
                            'semantic_attrs': {'output_data_type': 'bfloat16', 'post_ops': ('declared',)}})
