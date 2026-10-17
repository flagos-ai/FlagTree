# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Static slices of one weight can supply several packed callee arguments."""

import torch
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator, materialize_constant_assets
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.passes.functions import function_nodes


class PartitionedProjection(fm.Module):
    def forward(self):
        activation = fm.tensor_type('bfloat16', (1, 64))
        weight = fm.tensor_type('bfloat16', (128, 64))
        x = self.input('worker_x', activation, id='worker_x', metadata={'function_parameter':'worker'})
        w = self.input('worker_w', weight, id='worker_w', metadata={'function_parameter':'worker'})
        outputs = []
        for p in range(2):
            a = fm.F.tensors.slice(x, starts=(32*p,), ends=(32*(p+1),), axes=(1,))
            b = fm.F.tensors.slice(w, starts=(32*p,), ends=(32*(p+1),), axes=(1,))
            outputs.append(fm.F.math.matmul(a, b, transpose_b=True, name=f'dot_{p}'))
        self.function('worker', (x,w), outputs, attrs={'reusable':True, 'calling_convention':'device'})
        value = self.input('value', activation, id='value')
        actual_w = self.weight('weight', weight, source='memory', key='weight', id='weight')
        call = fm.F.builtin.call(value, actual_w, callee='worker', result_type=fm.TupleType(tuple(n.type for n in outputs)))
        self.function('main', (value,), fm.F.tensors.get_items(call, 0, 1))


def test_sliced_weight_packs_offline_and_lifts_both_representations():
    source = PartitionedProjection(dialect='high_level', stage='imported', entry='main').build()
    packed = Compiler().compile(source, stop_after='apply-packing').module
    worker = packed.function_map['worker']
    assert sum(n.op == 'ntt.packed_matmul' for n in function_nodes(packed, worker)) == 2
    assert not any(n.op == 'tensors.pack' for n in function_nodes(packed, worker))
    assert len(worker.parameters) == 3
    # Constant freezing must materialize the actual slices, not assume that
    # matching weight storage sizes imply the same payload.
    generator = torch.Generator().manual_seed(20261016)
    weights = DictWeightResolver({'weight':torch.randn(128,64,generator=generator).bfloat16()})
    values = {'value':torch.randn(1,64,generator=generator).bfloat16()}
    evaluator = TorchEvaluator(weights)
    torch.testing.assert_close(evaluator.run(packed,values),evaluator.run(source,values),rtol=0,atol=0)
    frozen = freeze_constant_islands(packed)
    assets = materialize_constant_assets(frozen,weights)
    assert len(assets) == 2
    torch.testing.assert_close(evaluator.run(frozen,values),evaluator.run(source,values),rtol=0,atol=0)


def test_slice_of_runtime_tensor_does_not_acquire_offline_weight_provenance():
    class RuntimeProjection(fm.Module):
        def forward(self):
            value = self.input('value', fm.tensor_type('bfloat16', (1,32)))
            weight = self.input('weight', fm.tensor_type('bfloat16', (128,64)))
            sliced = fm.F.tensors.slice(weight, starts=(0,), ends=(32,), axes=(1,))
            output = fm.F.math.matmul(value, sliced, transpose_b=True, name='projection')
            self.function('main', (value,weight), (output,))

    source = RuntimeProjection(dialect='high_level', stage='imported', entry='main').build()
    proposed = Compiler().compile(source, stop_after='propose-packing').module
    assert not any(p.id == 'packing.projection' for p in proposed.selection_points)
