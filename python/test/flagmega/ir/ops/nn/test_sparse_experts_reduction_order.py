# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Expert decomposition retains the source's BF16 K accumulation order."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.sparse_experts import SparseExperts
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from python.test.flagmega.codegen.triton.kernels.sparse_experts.helpers import stage_module


@pytest.mark.parametrize('definition', (SparseExperts, SparseExpertsGateUp, SparseExpertsDown))
def test_ordered_projection_contract_survives_stage_construction(definition, tmp_path):
    module = stage_module(definition, hidden=128, intermediate=64, reduction_order='mma_k16')
    node = next(n for n in module.nodes if n.op == definition.op_name)
    assert node.attrs['reduction_order'] == 'mma_k16'
    if definition is SparseExperts:
        inputs = tuple(module.node_map[n] for n in node.inputs)
        _, gate, down, _ = definition.stage_calls(inputs, node.attrs)
        assert gate.attrs['reduction_order'] == down.attrs['reduction_order'] == 'mma_k16'
    fm.emit_module(module, tmp_path / 'ir.py')
    assert fm.load_module(tmp_path / 'ir.py').semantic_hash == module.semantic_hash


@pytest.mark.parametrize('definition', (SparseExperts, SparseExpertsGateUp, SparseExpertsDown))
def test_relaxed_order_remains_implicit_in_existing_ir(definition):
    module = stage_module(definition)
    assert 'reduction_order' not in next(n for n in module.nodes if n.op == definition.op_name).attrs


def test_ordered_down_rejects_partitioned_k():
    with pytest.raises(IRSchemaError, match='unpartitioned K'):
        stage_module(SparseExpertsDown, hidden=128, intermediate=64, distribution='split_k', reduction_order='mma_k16')


@pytest.mark.parametrize('definition', (SparseExpertsGateUp, SparseExpertsDown))
def test_ordered_mma_requires_bf16_projection_operands(definition):
    with pytest.raises(IRSchemaError, match='BF16'):
        stage_module(definition, weight_dtype='float8_e4m3fn', block_scaled=True, reduction_order='mma_k16')
