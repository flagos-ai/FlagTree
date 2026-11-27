# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Non-pointwise owners preserve numeric conversion boundaries in registers."""
from dataclasses import replace
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.fusion import fusion_rules
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.neutral._utility import make_node


def graph(mode, *, shared=False):
    mesh = fm.Placement((2, 2), 'yx', 'bb')
    lanes = (8,) if mode == 'broadcast_vector' else ()
    tensor = fm.tensor_type(fm.vector_type('bfloat16', lanes) if lanes else 'bfloat16', (1, 1, 8 if lanes else 64))
    if mode.startswith('reduce'):
        tensor = fm.tensor_type('float32', (1, 4, 64))
    source = fm.Node('x', 'builtin.var', (), fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 3, mesh), attrs={'name': 'x'})
    if mode.startswith('broadcast'):
        base = make_node('tensors.broadcast_to', 'broadcast', (source,), {'shape': (1, 4, 8 if lanes else 64)}, {})
        root = make_node('tensors.cast', 'wide', (base,), {'dtype': fm.vector_type('float32', lanes) if lanes else 'float32'}, {})
        nodes = [source, base, root]
    else:
        narrow = make_node('tensors.cast', 'narrow', (source,), {'dtype': 'bfloat16'}, {})
        base = make_node('math.reduce_sum', 'sum', (narrow,), {'axes': () if 'identity' in mode else (1,), 'keep_dims': False, 'output_lanes': (8,)}, {})
        root = base
        nodes = [source, narrow, base]
        if mode.endswith('_post'):
            root = make_node('tensors.cast', 'wide', (base,), {'dtype': fm.vector_type('float32', (8,))}, {})
            nodes.append(root)
    outputs = [root.id]
    if shared:
        outputs.append(base.id if mode.startswith('broadcast') else 'narrow')
    return fm.verify_module(fm.IRModule('distributed', 'frozen_constants', tuple(nodes),
        (fm.Function('main', ('x',), tuple(outputs)),), 'main', metadata={'auto_distribution': {'placement': mesh.to_data()}}))


@pytest.mark.parametrize('mode', ['broadcast', 'broadcast_vector', 'reduce', 'reduce_post', 'reduce_identity', 'reduce_identity_post'])
def test_conversion_is_in_owning_kernel_contract(mode, tmp_path):
    source = graph(mode)
    result = fm.verify_module(DataflowPass('Fuse', fusion_rules()).run(source))
    assert not any(n.op in {'tensors.cast', 'ntt.vectorized_cast'} for n in result.nodes)
    restored = fm.load_module(fm.emit_module(result, tmp_path / 'module.py'))
    assert restored.semantic_hash == result.semantic_hash
    from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
    import torch
    shape = tuple(d.fixed_value for d in fm.logical_type(source.node_map['x'].type).shape) + tuple(getattr(fm.logical_type(source.node_map['x'].type).dtype, 'lanes', ()))
    g = torch.Generator().manual_seed(284)
    for _ in range(3):
        x = torch.randn(shape, generator=g) * 2
        if mode.startswith('broadcast'):
            x = x.bfloat16()
        before = TorchEvaluator(DictWeightResolver({})).run(source, {'x': x})
        after = TorchEvaluator(DictWeightResolver({})).run(restored, {'x': x})
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        expected = (x.expand(1, 4, *x.shape[2:]).float() if mode.startswith('broadcast')
                    else x.bfloat16().reshape(1, 4, 8, 8) if 'identity' in mode
                    else x.bfloat16().float().sum(1).bfloat16().reshape(1, 8, 8))
        if mode.endswith('_post'):
            expected = expected.float()
        torch.testing.assert_close(after[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['broadcast', 'reduce'])
def test_shared_observable_conversion_is_not_erased(mode):
    source = graph(mode, shared=True)
    result = fm.verify_module(DataflowPass('Fuse', fusion_rules()).run(source))
    assert any(n.op == 'tensors.cast' for n in result.nodes)


def test_lane_regrouping_is_not_admitted_as_coordinate_preserving():
    source = graph('broadcast_vector')
    x = replace(source.node_map['x'], type=replace(source.node_map['x'].type,
        tensor=fm.tensor_type(fm.vector_type('bfloat16', (2, 4)), (1, 1, 8))))
    broadcast = make_node('tensors.broadcast_to', 'broadcast', (x,), {'shape': (1, 4, 8)}, {})
    wide = make_node('ntt.vectorized_cast', 'wide', (broadcast,),
                     {'new_type': fm.vector_type('float32', (8,)), 'vectorize_axes': (2, 2)}, {})
    source = fm.verify_module(replace(source, nodes=(x, broadcast, wide)))
    result = fm.verify_module(DataflowPass('Fuse', fusion_rules()).run(source))
    assert result.node_map['wide'].op == 'ntt.vectorized_cast'


def test_partial_reduction_does_not_absorb_numeric_boundaries():
    source = graph('reduce_post')
    x = replace(source.node_map['x'], type=replace(source.node_map['x'].type,
        axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 2), fm.SBP.broadcast())))
    narrow = make_node('tensors.cast', 'narrow', (x,), {'dtype': 'bfloat16'}, {})
    sum_ = make_node('math.reduce_sum', 'sum', (narrow,), {'axes': (1,), 'keep_dims': False, 'output_lanes': (8,)}, {})
    wide = make_node('tensors.cast', 'wide', (sum_,), {'dtype': fm.vector_type('float32', (8,))}, {})
    source = fm.verify_module(replace(source, nodes=(x, narrow, sum_, wide)))
    assert sum_.type.partial is not None
    result = fm.verify_module(DataflowPass('Fuse', fusion_rules()).run(source))
    assert result.node_map['narrow'].op == result.node_map['wide'].op == 'tensors.cast'


@pytest.mark.parametrize('mode,tiled', [('broadcast', False), ('broadcast_vector', False),
    ('reduce', False), ('reduce_post', False), ('reduce', True), ('reduce_post', True),
    ('reduce_identity', False), ('reduce_identity_post', True)])
def test_device_numeric_boundaries_preserve_all_output_bits(mode, tiled, tmp_path):
    import torch
    from math import prod
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.options import CompileOptions
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Run
    from triton.flagmega.targets import NvidiaSm90Target
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('SM90 required')
    source = graph(mode)
    mesh = source.node_map['x'].type.placement
    catalog = NvidiaSm90Target().triton_implementation_model
    if tiled:
        base = catalog.implementation('tir.reduce_sum.local')
        impl = replace(base, id='tir.reduce_sum.tiled_rows', variant='tiled_rows',
                       parameters={**base.parameters, 'row_tile': 16})
        catalog = replace(catalog, implementations=(*catalog.implementations, impl),
                          preferences={**catalog.preferences, 'reduce_sum': (impl.id,)})

    class Target(NvidiaSm90Target):
        def pre_post_ops_rules(self):
            return fusion_rules() if self.fused else ()

    runtimes = []
    outputs = []
    arguments = []
    guards = []
    shared = {}
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
            compiler.target = Target(triton_implementation_model=catalog).with_placement(mesh)
            compiler.target.fused = fused
            compiled = compiler.compile(source).module
            assert (not any(k.dispatch.semantic_op == 'tensors.cast' for k in compiled.kernel_definitions)) == fused
            runtime = load(write_artifact(compiled, tmp_path / str(fused), target='nvidia-sm90', emit_executable=True), device='cuda:0')
            runtimes.append(runtime)
            binding = runtime.buffer_plan.function_map[compiled.entry]
            values = {}
            for value, names in binding.parameters:
                if value not in shared:
                    shared[value] = allocate(compiled.node_map[value].type)
                values[names[0]] = shared[value]
            value, names = binding.outputs[0]
            result = allocate(compiled.node_map[value].type)
            values[names[0]] = result
            outputs.append(result)
            args = tuple(values[spec['buffer']] for spec in runtime.external_arguments)
            arguments.append(args)
            Run.prepare(runtime, *args)
            assert runtime.resource_report['spill_bytes'] == 0
        generator = torch.Generator().manual_seed(4218)
        for mutation in range(6):
            x = torch.randn(shared['x'].shape, generator=generator) * 2
            if mutation == 1:
                x = torch.ldexp(x, (torch.arange(x.numel()).reshape(x.shape) % 31 - 15).int())
            if mode.startswith('broadcast'):
                x = x.bfloat16()
            shared['x'].copy_(x)
            for runtime, args, output in zip(runtimes, arguments, outputs, strict=True):
                output.fill_(-17)
                Run.run_into(runtime, *args)
            torch.cuda.synchronize()
            assert torch.equal(outputs[0].view(torch.uint8).flatten(), outputs[1].view(torch.uint8).flatten())
            expected = (x.expand(1, 4, *x.shape[2:]).float() if mode.startswith('broadcast')
                        else x.bfloat16().reshape(1, 4, 8, 8) if 'identity' in mode
                        else x.bfloat16().float().sum(1).bfloat16().reshape(1, 8, 8))
            if mode.endswith('_post'):
                expected = expected.float()
            torch.testing.assert_close(outputs[1].cpu(), expected, rtol=0, atol=0)
            for raw in guards:
                assert bool((raw[:16] == 123).all()) and bool((raw[-16:] == 123).all())
    finally:
        for runtime in runtimes:
            runtime.close()
