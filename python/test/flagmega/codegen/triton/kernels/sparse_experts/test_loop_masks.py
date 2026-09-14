# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import ast

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.codegen.triton.sparse_experts.common import _axis_capacity_is_active
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.ir.ops.ntt.sparse_experts import SparseExpertsDownCombine
from python.test.flagmega.codegen.triton.kernels.sparse_experts.helpers import stage_module
from python.test.flagmega.codegen.triton.kernels.sparse_experts.test_routes import route_module


@pytest.mark.parametrize("definition", [SparseExpertsGateUp, SparseExpertsDown, SparseExpertsDownCombine, None])
@pytest.mark.parametrize("cyclic", [False, True])
def test_route_loop_masks_only_owner_padding(tmp_path, definition, cyclic):
    module = stage_module(definition, tokens=1) if definition is not None else route_module(cyclic=cyclic)
    compiled = Compiler().compile(module).module
    render_triton_package(compiled, tmp_path / "generated")
    tree = ast.parse((tmp_path / "generated" / "generated_kernels.py").read_text())
    route_comparisons = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "_fm_route"
    ]
    assert bool(route_comparisons) == (definition is None and cyclic)


@pytest.mark.parametrize("routes,policy,expected", [
    (9, fm.SBP.broadcast(), True),
    (4, fm.SBP.split_contiguous((0,)), True),
    (6, fm.SBP.split_contiguous((0,), 4), False),
    (4, fm.SBP.split_contiguous((0,), 4), False),
    (8, fm.SBP.split_block_cyclic((0,), 2), True),
    (3, fm.SBP.split_block_cyclic((0,), 2), False),
    (fm.dim("routes", minimum=1, maximum=9), fm.SBP.broadcast(), False),
])
def test_full_axis_proof_uses_active_extent_not_global_divisibility(routes, policy, expected):
    value_type = fm.DistributedType(fm.tensor_type("int32", (1, routes)),
                                   (fm.SBP.broadcast(), policy), fm.Placement((2,), "x", "b"))
    capacity = fm.local_shape(value_type)[1]
    capacity = capacity.fixed_value if capacity.is_fixed else capacity.maximum
    assert _axis_capacity_is_active(value_type, 1, capacity) == expected
