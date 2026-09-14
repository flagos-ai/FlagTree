# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn.sparse_experts import SparseExperts
from triton.flagmega.ir.ops.nn.sparse_experts_dispatch import SparseExpertsDispatch
from triton.flagmega.ir.ops.nn.sparse_experts_combine import SparseExpertsCombine
from triton.flagmega.passes.target_independent import decompose_complex_ops
from python.test.flagmega.sparse_experts.helpers import operand_types
from python.test.flagmega.codegen.triton.kernels.sparse_experts.helpers import execute_and_reference


def route_module(*, mesh=(2, 2, 2), cyclic=True, exported=False):
    b = fm.SBP.broadcast()
    split = lambda axis: fm.SBP.split_block_cyclic((axis,), 2) if cyclic else fm.SBP.split_contiguous((axis,))
    r, k, h = split(0), split(1), split(2)
    placement = fm.Placement(mesh, "xyz", "bbb")
    types = operand_types(tokens=2, hidden=40, intermediate=24, routes=3 if cyclic else 4, experts=5)
    policies = {"q": (b, b), "router_expert_ids": (b, r), "router_expert_weights": (b, r),
                "gate_weight": (b, k, b), "up_weight": (b, k, b), "down_weight": (b, h, k)}

    class Graph(fm.Module):
        def forward(self):
            inputs = tuple(self.input(p.name, fm.DistributedType(types[p.name], policies.get(p.name, (b, b)), placement),
                                      id=p.name) for p in SparseExperts.input_parameters)
            result = SparseExperts.construct(*inputs, name="experts")
            result = fm.F.distributed.boxing(result, result.type.tensor, name="result")
            self.function("main", inputs, (result,))

    module = Graph(dialect="nn", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    module = decompose_complex_ops(module)
    if exported:
        module = decompose_complex_ops(module)
        # A second combine keeps the per-route projections public to another
        # consumer without adding an output to the runtime fixture.
        down = module.node_map["experts.down"]
        fn = module.functions[0]
        module = replace(module, functions=(replace(fn, outputs=(*fn.outputs, down.id)),))
    return module


@pytest.mark.parametrize("cyclic", [False, True])
@pytest.mark.parametrize("mesh", [(2, 2, 2), (4, 2, 2)])
def test_route_and_k_parallel_fused_pipeline_masks_inactive_slots(tmp_path, mesh, cyclic):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module = route_module(mesh=mesh, cyclic=cyclic)
    output, expected, _ = execute_and_reference(module, tmp_path, torch)
    torch.testing.assert_close(output, expected, rtol=0.016, atol=0.015625)


@pytest.mark.parametrize("definition", [SparseExpertsDispatch, SparseExpertsCombine])
@pytest.mark.parametrize("packed", [False, True])
def test_standalone_route_stages_with_cyclic_owners(tmp_path, definition, packed):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    b, r, h = fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 2), fm.SBP.split_contiguous((1,))
    placement = fm.Placement((4, 2), "yx", "bb")
    types = operand_types(tokens=2, hidden=64 if packed else 40, routes=3, experts=5)
    types["value"] = types["q"]
    attrs = {}
    if packed:
        types["value"] = fm.tensor_type(fm.vector_type("bfloat16", (2, 2)), (2, 16))
        types["projections"] = fm.tensor_type(fm.vector_type("float32", (2, 2)), (2, 3, 16))
        if definition is SparseExpertsCombine:
            attrs = {"output_dtype": fm.vector_type("bfloat16", 8), "round_weighted_output": True}
    policies = {"value": (b, h), "router_expert_ids": (b, r), "projections": (b, r, h), "router_expert_weights": (b, r)}

    class Graph(fm.Module):
        def forward(self):
            inputs = tuple(self.input(p.name, fm.DistributedType(types[p.name], policies[p.name], placement), id=p.name)
                           for p in definition.input_parameters)
            result = definition.construct(*inputs, **attrs, name="stage")
            result = fm.F.distributed.boxing(result, result.type.tensor)
            self.function("main", inputs, (result,))

    module = Graph(dialect="nn", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    output, expected, _ = execute_and_reference(module, tmp_path, torch)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
