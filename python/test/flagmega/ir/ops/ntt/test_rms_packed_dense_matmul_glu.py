# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit RMS/BF16/packed-GLU semantics (an extension, not an old-op bug)."""
from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.core import get_definition


def _arguments(k=128, n=64):
    return tuple(fm.Node(name, 'builtin.var', (), ty, attrs={'name': name}) for name, ty in (
        ('raw', fm.tensor_type('float32', (1, k))),
        ('scale', fm.tensor_type('bfloat16', (k,))),
        ('bias', fm.tensor_type('bfloat16', (k,))),
        ('gate_weight', fm.tensor_type('bfloat16', (k // 16, n // 8, 2, 64))),
        ('up_weight', fm.tensor_type('bfloat16', (k // 16, n // 8, 2, 64))),
    ))


def test_explicit_raw_rms_glu_has_a_registered_five_input_contract():
    definition = get_definition('ntt.rms_packed_dense_matmul_glu')
    assert definition is not None, 'Missing explicit RMS-to-GLU semantic operation'
    attrs = definition.normalize_attrs({'epsilon': 1e-6, 'norm_tree_group_size': 16,
                                        'norm_partial_group_size': 4,
                                        'round_activation': False})
    prepared = definition.prepare(_arguments(), attrs)
    assert prepared.result_type == fm.tensor_type('bfloat16', (1, 64))
    assert tuple(p.name for p in definition.input_parameters) == (
        'raw', 'scale', 'bias', 'gate_weight', 'up_weight')


def test_ordinary_glu_correctly_rejects_unnormalized_fp32():
    from triton.flagmega.ir.ops.nn.packed_dense_matmul_glu import PackedDenseMatMulGlu
    raw, _, _, gate, up = _arguments()
    with pytest.raises(IRSchemaError, match='dtype/K'):
        PackedDenseMatMulGlu.prepare((raw, gate, up), {})


def _module():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect='high_level', stage='imported', entry='main')

        def forward(self):
            values = tuple(self.input(v.id, v.type, id=v.id) for v in _arguments())
            result = fm.F.ntt.rms_packed_dense_matmul_glu(*values, epsilon=1e-6,
                norm_tree_group_size=16, norm_partial_group_size=4, round_activation=False, name='result')
            self.function('main', values, (result,))
    return Graph().build()


def test_evaluator_preserves_bf16_normalized_and_projection_boundaries():
    from triton.flagmega.ir.ops.tensors._k_major import unpack_k_major_weight
    torch.manual_seed(42)
    module = _module()
    values = {n.id: torch.randn(tuple(d.fixed_value for d in n.type.shape), dtype=getattr(torch, n.type.dtype.value))
              for n in module.nodes if n.op == 'builtin.var'}
    raw = values['raw']
    square = raw.square().reshape(1, 8, 16)
    for group in (16, 4, 2):
        square = square.reshape(1, -1, group)
        for step in (1 << bit for bit in range(group.bit_length() - 2, -1, -1)):
            square = square + square[..., torch.arange(group) ^ step]
        square = square[..., 0]
    normalized = ((raw * torch.rsqrt((square + 0.0) / 128 + 1e-6)) * values['scale'].float()
                  + values['bias'].float()).to(torch.bfloat16)
    gate = torch.nn.functional.linear(normalized, unpack_k_major_weight(values['gate_weight'], 'k_major_n8_k16'))
    up = torch.nn.functional.linear(normalized, unpack_k_major_weight(values['up_weight'], 'k_major_n8_k16'))
    expected = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
    actual, = TorchEvaluator(DictWeightResolver({})).run(module, values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_three_level_reduction_matches_independent_separate_rn_hierarchy_witness():
    from pathlib import Path
    import numpy as np
    from triton.flagmega.ir.ops.ntt.rms_packed_dense_matmul_glu import hierarchical_square_sum
    fixture = np.load(Path(__file__).parents[3] / 'data/native_post_rms_three_level.npz')
    actual = hierarchical_square_sum(torch.from_numpy(fixture['raw']),
        {'norm_tree_group_size': 16, 'norm_partial_group_size': 32}, torch)
    assert actual.reshape(-1).view(torch.int32).tolist() == [0x45795ac0]
    np.testing.assert_array_equal(actual.numpy().reshape(-1), fixture['expected'])


@pytest.mark.parametrize('field,value', [('norm_tree_group_size', 3), ('norm_partial_group_size', 0),
                                        ('epsilon', float('nan')), ('norm_math_mode', 'unknown')])
def test_invalid_numerical_contract_is_rejected(field, value):
    definition = get_definition('ntt.rms_packed_dense_matmul_glu')
    with pytest.raises(IRSchemaError):
        definition.prepare(_arguments(), {'epsilon': 1e-6, 'norm_tree_group_size': 16,
                                         'norm_partial_group_size': 4, field: value})


def test_registered_function_roundtrips_and_preserves_memory_effects(tmp_path):
    module = _module()
    restored = fm.load_module(fm.emit_module(module, tmp_path/'rms_glu.py'))
    assert restored.semantic_hash == module.semantic_hash
    definition = get_definition('ntt.rms_packed_dense_matmul_glu')
    assert all(p.memory_effect == fm.MemoryEffect.CHIP_READ for p in definition.input_parameters[:3])
