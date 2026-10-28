# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Ragged cyclic GDN state writers execute through the normal compiled ABI."""
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig, create_gdn_state
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from python.test.flagmega.devices import executable_target


@pytest.mark.parametrize('family', ['convolution', 'recurrent'])
def test_ragged_cyclic_state_and_output_match_logical_reference(tmp_path, family):
    torch = pytest.importorskip('torch')
    target = executable_target()
    if target is None: pytest.skip('requires a GPU with an executable FlagMega target')
    cfg = GatedDeltaNetStateConfig(1, 3, 6, 128, 128, 4, 1536)
    mesh = fm.Placement((3, 3), 'yx', 'bb')
    b = fm.SBP.broadcast()
    cy = fm.SBP.split_block_cyclic((0, 1), 64)

    def replicated(dtype, shape):
        tensor = fm.tensor_type(dtype, shape)
        return fm.DistributedType(tensor, (b, ) * len(shape), mesh)

    qtype = replicated('bfloat16', (1, 1536))
    ztype = fm.DistributedType(fm.tensor_type('bfloat16', (1, 768)), (b, cy), mesh)
    if family == 'convolution': qtype = fm.DistributedType(qtype.tensor, (b, cy), mesh)
    types = {'qkv': qtype}
    if family == 'convolution':
        types['weight'] = fm.DistributedType(fm.tensor_type('bfloat16', (1536, 1, 4)), (cy, b, b), mesh)
    else:
        types.update(z=ztype, source=replicated('bfloat16', (1, 128)), b_weight=replicated('bfloat16', (6, 128)),
                     a_weight=replicated('bfloat16', (6, 128)), a_log=replicated('float32', (6, )),
                     dt_bias=replicated('float32', (6, )), norm=replicated('bfloat16', (128, )))

    class Graph(fm.Module):

        def forward(self):
            values = {name: self.input(name, ty, id=name) for name, ty in types.items()}
            state = self.input('state', cfg.ref_type, id='state')
            if family == 'convolution':
                result = fm.F.nn.gated_delta_net_convolution(values['qkv'], state, values['weight'], conv_kernel_size=4,
                                                             name='gdn')
            else:
                result = fm.F.nn.gated_delta_net_recurrent_core(state, values['qkv'], values['z'], values['source'],
                                                                values['b_weight'], values['a_weight'], values['a_log'],
                                                                values['dt_bias'], values['norm'], num_key_heads=3,
                                                                num_value_heads=6, key_head_dim=128, value_head_dim=128,
                                                                epsilon=1e-6, name='gdn')
            self.function('main', (*values.values(), state), (fm.F.tensors.get_item(result, 0), state))

    module = Graph(dialect='distributed', stage='frozen_constants', entry='main',
                   metadata={'auto_distribution': {'placement': mesh.to_data()}}).build()
    fm.verify_module(module)
    compiled = Compiler(CompileOptions(target=target)).compile(module).module
    artifact = write_artifact(compiled, tmp_path / 'artifact', target=target, emit_executable=True)
    runtime = load(artifact, device='cuda:0')
    torch.manual_seed(163)
    state = create_gdn_state(cfg, device='cuda:0')
    state.convolution.normal_(0, .1)
    state.recurrent.normal_(0, .01)
    reference_state = state.clone()
    inputs = {
        name:
        torch.randn(tuple(d.fixed_value
                          for d in ty.tensor.shape), device='cuda', dtype=getattr(torch, ty.tensor.dtype.value)) * .1
        for name, ty in types.items()
    }
    if family == 'recurrent':
        inputs['a_log'].fill_(-1)
        inputs['norm'].fill_(1)
    output = torch.empty((1, 1536 if family == 'convolution' else 768), device='cuda', dtype=torch.bfloat16)
    state_buffers = dict(runtime.buffer_plan.function_map['main'].parameters)['state']
    state_values = dict(zip(state_buffers, (state.convolution, state.recurrent)))

    def arguments():
        vals = []
        for spec in runtime.external_arguments:
            value = output if spec['role'] == 'result' else state_values[
                spec['buffer']] if spec['value'] == 'state' else inputs[spec['value']]
            buf = runtime.buffer_plan.buffer_map[spec['buffer']]
            dtype = buf.dtype
            lanes = dtype.lanes if isinstance(dtype, fm.VectorType) else ()
            vals.append(value.reshape((*buf.shape, *lanes)))
        return tuple(vals)

    GeneratedTirCallGraphModule.prepare(runtime, *arguments())
    evaluator = TorchEvaluator(DictWeightResolver({}))
    for step in range(3):
        inputs['qkv'].add_(.01)
        expected = evaluator.run(module, {**inputs, 'state': reference_state})[0]
        GeneratedTirCallGraphModule.run_into(runtime, *arguments())
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=.02, atol=.002)
        torch.testing.assert_close(state.recurrent, reference_state.recurrent, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(state.convolution, reference_state.convolution, rtol=0, atol=0)
    assert runtime.resource_report['spill_bytes'] == 0
