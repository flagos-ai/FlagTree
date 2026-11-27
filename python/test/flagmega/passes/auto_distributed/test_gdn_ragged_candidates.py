# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A target-owned ragged map must keep GDN distributed with unique state writers."""
from collections import Counter
from itertools import product
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext
from triton.flagmega.passes.auto_distributed.providers import GdnConvolutionCandidateProvider, GdnRecurrentCandidateProvider
from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider
from .test_candidate_provider import _provider_module


def context(family, mesh, physical):
    placement = fm.Placement(mesh, 'yx', 'bb')
    if family == 'convolution':
        state = fm.RefType('state', (('convolution', fm.tensor_type('bfloat16', (1, 10240, 3))), ))
        inputs = (fm.tensor_type('bfloat16', (1, 10240)), state, fm.tensor_type('bfloat16', (10240, 1, 4)))
        output = fm.TupleType((inputs[0], state))
        op = 'nn.gdn_convolution'
        attrs = {'conv_kernel_size': 4}
        provider = GdnConvolutionCandidateProvider()
        reason = 'gdn-convolution-channel-sbp'
    else:
        state = fm.RefType('state', (('recurrent', fm.tensor_type('float32', (1, 48, 128, 128))), ))
        inputs = (state, fm.tensor_type('bfloat16', (1, 10240)), fm.tensor_type('bfloat16', (1, 6144)),
                  fm.tensor_type('bfloat16', (1, 5120)), fm.tensor_type('bfloat16', (48, 5120)),
                  fm.tensor_type('bfloat16', (48, 5120)), fm.tensor_type('float32', (48, )),
                  fm.tensor_type('float32', (48, )), fm.tensor_type('bfloat16', (128, )))
        output = fm.TupleType((inputs[2], state))
        op = 'nn.gdn_recurrent_core'
        attrs = {'num_value_heads': 48, 'value_head_dim': 128}
        provider = GdnRecurrentCandidateProvider()
        reason = 'gdn-recurrent-head-sbp'
    module, call = _provider_module(op, inputs, output)
    kwargs = {'split_candidate_provider': PyNttDistributedSplitCandidateProvider()} if physical else {}
    ctx = DistributedCandidateContext(module, call, placement, tuple((value, ) for value in inputs), **kwargs)
    return ctx, provider, reason, attrs, state


@pytest.mark.parametrize('family', ['convolution', 'recurrent'])
@pytest.mark.parametrize('mesh', [(6, 6), (3, 3)])
@pytest.mark.parametrize('physical', [True, False])
def test_ragged_gdn_obeys_target_policy_and_state_ownership(family, mesh, physical):
    ctx, provider, reason, attrs, state = context(family, mesh, physical)
    choices = [value for value in provider.get_candidates(ctx) if value.reason == reason]
    if not physical:
        assert not choices  # contiguous backends still require equal shards
        return
    assert choices, 'target exposes a legal ragged split but GDN became replicated'
    for choice in choices:
        args = [fm.Node(f'arg{i}', 'builtin.var', (), value) for i, value in enumerate(choice.input_types)]
        assert fm.get_definition(ctx.source_call.op).infer_type(args, attrs) == choice.return_type
        assert choice.return_type.fields[1] == state
        output = choice.return_type.fields[0]
        ownership = Counter()
        for owner in product(*(range(n) for n in mesh)):
            axis = fm.local_shard_descriptor(output, owner).axes[1]
            for i in range(axis.active_extent.fixed_value):
                ownership[axis.map_local_to_global(i).fixed_value] += 1
        extent = output.tensor.shape[1].fixed_value
        assert ownership == Counter(range(extent))
