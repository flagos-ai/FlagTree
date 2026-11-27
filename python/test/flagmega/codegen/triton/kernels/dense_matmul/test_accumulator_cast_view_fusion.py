# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Preserve accumulator rounding while fusing across a pure ownership view."""
from dataclasses import replace
from math import prod
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.tir.fuse_packed_accumulator_cast import fuse_packed_accumulator_casts
from triton.flagmega.rules.neutral._utility import make_node


def _projection_graph(mesh, axes):
    broadcast = fm.SBP.broadcast()
    groups = (32 + prod(mesh.hierarchy[a] for a in axes) - 1) // prod(mesh.hierarchy[a] for a in axes)
    block_axes = tuple(i for i, level in enumerate(mesh.hierarchy_levels) if level == 'b')
    sentinel_groups = (128 + prod(mesh.hierarchy[a] for a in block_axes) - 1) // prod(mesh.hierarchy[a] for a in block_axes)

    class Graph(fm.Module):
        def forward(self):
            x = self.input('x', fm.DistributedType(fm.tensor_type('bfloat16', (1, 256)), (broadcast, broadcast), mesh), id='x')
            w = self.input('w', fm.DistributedType(
                fm.tensor_type(fm.vector_type('bfloat16', (8, 2, 8)), (16, 32)),
                (broadcast, fm.SBP.split_contiguous(axes, groups) if axes else broadcast), mesh), id='w')
            no = fm.F.builtin.none()
            projection = fm.F.ntt.packed_matmul(x, w, no, no, output_data_type='float32', name='projection',
                metadata={'selected_vectorization': 'vectorization.matmul.n', 'selected_vector_axes': (1,), 'selected_vector_lanes': (8,)})
            sentinel = self.input('sentinel', fm.tensor_type('float32', (128,)), id='sentinel')
            split = fm.DistributedType(sentinel.type, (fm.SBP.split_contiguous(block_axes, sentinel_groups),), mesh)
            local = fm.F.distributed.force_boxing(sentinel, split)
            copied = fm.F.distributed.force_boxing(fm.F.math.add(local, local), sentinel.type)
            self.function('main', (x, w, sentinel), (projection, copied))

    return Graph(dialect='distributed', stage='frozen_constants', entry='main',
                 metadata={'auto_distribution': {'placement': mesh.to_data()}}).build()


def graph(case, *, shared=None, exported=None, boxing=False):
    if case == 'refine':
        mesh = fm.Placement((4, 4), 'yx', 'bb')
        producer_axes = ()
        target_policy = fm.SBP.split_contiguous((0, 1), 2)
    elif case == 'block_coarsen':
        mesh = fm.Placement((4, 4), 'yx', 'bb')
        producer_axes = (0, 1)
        target_policy = fm.SBP.broadcast()
    elif case == 'coarsen':
        mesh = fm.Placement((2, 4, 4), 'cyx', 'cbb')
        producer_axes = (0, 1, 2)
        target_policy = fm.SBP.split_contiguous((0,), 16)
    else:
        mesh = fm.Placement((2, 4, 4), 'ycx', 'bcb')
        producer_axes = (1, 0, 2)
        target_policy = fm.SBP.split_contiguous((1,), 16)
    source = _projection_graph(mesh, producer_axes)
    producer = source.node_map['projection']
    producer = replace(producer, metadata={**producer.metadata, 'offline_projection_weight_decode': {'block_k': 128}})
    assert fm.logical_type(producer.type).dtype.lanes == (2, 4)
    target = replace(producer.type, axis_policies=(fm.SBP.broadcast(), target_policy))
    view = make_node('distributed.boxing' if boxing else 'distributed.sharded_view', 'view', (producer,), {'new_type': target}, {})
    cast = make_node('ntt.vectorized_cast', 'rounded', (view,),
                     {'new_type': fm.vector_type('bfloat16', (8,)), 'vectorize_axes': (1, 1)}, {})
    nodes = [producer if n.id == producer.id else n for n in source.nodes] + [view, cast]
    outputs = [cast.id, source.functions[0].outputs[1]]
    if exported:
        outputs.append({'producer': producer, 'view': view}[exported].id)
    if shared:
        value = {'producer': producer, 'view': view}[shared]
        extra = make_node('math.add', 'other_use', (value, value), {}, {})
        nodes.append(extra)
        outputs.append(extra.id)
    return fm.verify_module(replace(source, nodes=tuple(nodes), functions=(replace(source.functions[0], outputs=tuple(outputs)),)))


@pytest.mark.parametrize('case', ['refine', 'coarsen', 'nonfirst_chip'])
def test_rounding_fuses_at_producer_before_ownership_view(case, tmp_path):
    source = graph(case)
    result = fm.verify_module(fuse_packed_accumulator_casts(source, predicate=lambda n: 'offline_projection_weight_decode' in n.metadata))
    assert not any(n.op in {'tensors.cast', 'ntt.vectorized_cast'} for n in result.nodes)
    output = result.node_map['rounded']
    assert output.op == 'distributed.sharded_view'
    assert output.type == source.node_map['rounded'].type
    producer = result.node_map[output.inputs[0]]
    assert producer.op == 'ntt.packed_matmul'
    assert producer.attrs['output_data_type'] == 'float32'
    assert producer.attrs['post_ops'][0].nodes[-1].op == 'ntt.vectorized_cast'
    assert producer.type.axis_policies == source.node_map['projection'].type.axis_policies
    assert fm.logical_type(producer.type).dtype == fm.vector_type('bfloat16', (8,))
    assert fuse_packed_accumulator_casts(result).semantic_hash == result.semantic_hash
    restored = fm.load_module(fm.emit_module(result, tmp_path / 'roundtrip.py'))
    assert restored.semantic_hash == result.semantic_hash
    import torch
    generator = torch.Generator().manual_seed(3847)
    for _ in range(3):
        x = torch.randn((1, 256), generator=generator).bfloat16()
        w = (torch.randn((256, 256), generator=generator) / 16).bfloat16()
        packed = w.reshape(16, 2, 8, 32, 8).permute(0, 3, 4, 1, 2).contiguous()
        inputs = {'x': x, 'w': packed, 'sentinel': torch.arange(128, dtype=torch.float32)}
        expected, _ = TorchEvaluator(DictWeightResolver({})).run(source, inputs)
        actual, _ = TorchEvaluator(DictWeightResolver({})).run(restored, inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual.reshape(1, 256), (x.float() @ w.float()).bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize('shared,exported', [('producer', None), ('view', None), (None, 'producer'), (None, 'view')])
def test_observable_wide_values_keep_the_original_contract(shared, exported):
    source = graph('coarsen', shared=shared, exported=exported)
    result = fm.verify_module(fuse_packed_accumulator_casts(source))
    assert result.node_map['rounded'].op == 'ntt.vectorized_cast'
    assert result.node_map['projection'].type == source.node_map['projection'].type
    assert result.node_map['view'].type == source.node_map['view'].type


def test_collective_boxing_is_not_a_coordinate_view():
    source = graph('coarsen', boxing=True)
    result = fm.verify_module(fuse_packed_accumulator_casts(source))
    assert result.node_map['rounded'].op == 'ntt.vectorized_cast'
    assert not result.node_map['projection'].attrs.get('post_ops')


def test_rejected_projection_predicate_preserves_the_chain():
    source = graph('coarsen')
    result = fm.verify_module(fuse_packed_accumulator_casts(source, predicate=lambda n: False))
    assert result.node_map['rounded'].op == 'ntt.vectorized_cast'
    assert not result.node_map['projection'].attrs.get('post_ops')


@pytest.mark.parametrize('case', ['refine', 'block_coarsen'])
def test_device_rounding_across_view_matches_unfused_kernel(case, tmp_path):
    import torch
    from math import prod
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.options import CompileOptions
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Run
    from triton.flagmega.selection import apply_plan, override_plan
    from triton.flagmega.targets import NvidiaSm90Target
    from triton.flagmega.targets.nvidia.dense_matmul import packed_mma_accumulator_implementations

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires SM90')
    source = graph(case)
    mesh = source.node_map['projection'].type.placement

    class Target(NvidiaSm90Target):
        def propose_tir(self, module):
            module = super().propose_tir(fuse_packed_accumulator_casts(module) if self.fused else module)
            return apply_plan(module, override_plan(module,
                [('tir.projection', 'tir.dense_matmul.packed_k_major_mma_accumulator')],
                policy='test.cast_view', rationale='Same ordered FP32 accumulator and BF16 rounding.'))

    base = NvidiaSm90Target().triton_implementation_model
    catalog = replace(base, implementations=(*base.implementations, *packed_mma_accumulator_implementations()))
    runtimes = []
    outputs = []
    arguments = []
    guards = []
    inputs = {}

    def allocate(kind):
        kind = fm.logical_type(kind)
        shape = tuple(d.fixed_value for d in kind.shape) + tuple(getattr(kind.dtype, 'lanes', ()))
        dtype = getattr(torch, getattr(kind.dtype, 'elem_type', kind.dtype).value)
        raw = torch.full((prod(shape) + 32,), 123, dtype=dtype, device='cuda')
        guards.append(raw)
        return raw[16:-16].reshape(shape)

    try:
        for fused in (False, True):
            compiler = Compiler(CompileOptions(placement=mesh))
            target = Target(triton_implementation_model=catalog).with_placement(mesh)
            target.fused = fused
            compiler.target = target
            module = compiler.compile(source).module
            dense = [d for d in module.kernel_definitions if d.dispatch.microkernel.family == 'dense_matmul']
            assert len(dense) == 1
            assert bool(dense[0].dispatch.semantic_attrs.get('post_ops')) == fused
            assert (not any(d.dispatch.semantic_op == 'ntt.vectorized_cast' for d in module.kernel_definitions)) == fused
            runtime = load(write_artifact(module, tmp_path / str(fused), target='nvidia-sm90', emit_executable=True), device='cuda:0')
            runtimes.append(runtime)
            values = {}
            binding = runtime.buffer_plan.function_map[module.entry]
            for value, names in binding.parameters:
                name = module.node_map[value].attrs['name']
                if name not in inputs:
                    inputs[name] = allocate(module.node_map[value].type)
                values[names[0]] = inputs[name]
            results = []
            for value, names in binding.outputs:
                result = allocate(module.node_map[value].type)
                values[names[0]] = result
                results.append(result)
            outputs.append(results)
            args = tuple(values[spec['buffer']] for spec in runtime.external_arguments)
            arguments.append(args)
            Run.prepare(runtime, *args)
            assert runtime.resource_report['spill_bytes'] == 0
        generator = torch.Generator().manual_seed(3847)
        for mutation in range(6):
            x = (torch.randint(-16, 17, (1, 256), generator=generator).float() / 128).bfloat16()
            w = (torch.randint(-16, 17, (256, 256), generator=generator).float() / 128).bfloat16()
            inputs['x'].copy_(x)
            inputs['w'].copy_(w.reshape(16, 2, 8, 32, 8).permute(0, 3, 4, 1, 2).contiguous())
            inputs['sentinel'].copy_(torch.arange(128, dtype=torch.float32))
            for runtime, args, results in zip(runtimes, arguments, outputs, strict=True):
                for result in results:
                    result.fill_(-17)
                Run.run_into(runtime, *args)
            torch.cuda.synchronize()
            expected = (x.float() @ w.float()).bfloat16().reshape(-1)
            for a, b in zip(outputs[0], outputs[1], strict=True):
                assert torch.equal(a.view(torch.uint8).flatten(), b.view(torch.uint8).flatten())
            torch.testing.assert_close(outputs[1][0].cpu().reshape(-1), expected, rtol=0, atol=0)
            torch.testing.assert_close(outputs[1][1].cpu(), 2 * torch.arange(128, dtype=torch.float32), rtol=0, atol=0)
            for raw in guards:
                assert bool((raw[:16] == 123).all()) and bool((raw[-16:] == 123).all())
    finally:
        for runtime in runtimes:
            runtime.close()
