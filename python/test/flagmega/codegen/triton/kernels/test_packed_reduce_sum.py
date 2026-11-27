# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A reduction over tensor axes retains each packet component independently."""

from types import SimpleNamespace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.reduction_domain import local_reduction_domain
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.math.reduce_sum import ReduceSum


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("lanes", [(8,), (2, 4)])
@pytest.mark.parametrize("axes", [(0,), (1, 2), (-1,), ()])
@pytest.mark.parametrize("keep_dims", [False, True])
def test_typed_component_sum_matches_independent_physical_reference(dtype, lanes, axes, keep_dims):
    shape = (4, 3, 5)
    kind = fm.tensor_type(fm.vector_type(dtype, lanes), shape)
    source = fm.Node("x", "builtin.var", (), kind)
    attrs = ReduceSum.normalize_attrs({"axes": axes, "keep_dims": keep_dims})
    result = ReduceSum.infer_type((source,), attrs)
    assert result.dtype == kind.dtype
    resolved = tuple(axis % len(shape) for axis in axes)
    expected_shape = tuple(
        1 if axis in resolved else extent for axis, extent in enumerate(shape) if keep_dims or axis not in resolved
    )
    assert tuple(d.fixed_value for d in result.shape) == expected_shape
    # Random input, plus a BF16 rounding midpoint and a cancellation in each lane.
    values = torch.randn((*shape, *lanes), generator=torch.Generator().manual_seed(20261018)).to(getattr(torch, dtype))
    values[0, 0, 0].fill_(1.0)
    values[1, 0, 0].fill_(1 / 256)
    values[2, 0, 0].fill_(-1.0)
    node = fm.Node("sum", "math.reduce_sum", ("x",), result, attrs=attrs)
    got = ReduceSum.evaluate(node, (values,), SimpleNamespace(types={"x": kind}))
    want = values if not axes else values.float().sum(resolved, keepdim=keep_dims).to(values.dtype)
    assert got.shape == (*expected_shape, *lanes)
    assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


def test_packet_input_retains_owner_units_and_partial_axes():
    mesh = fm.Placement((4, 8, 16), "cyx", "cbb")
    kind = fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (2, 4)), (4, 1, 512)),
        (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast(), fm.SBP.split_block_cyclic((2,), 1)),
        mesh,
    )
    attrs = ReduceSum.normalize_attrs({"axes": (0,), "keep_dims": False})
    output = ReduceSum.infer_type((fm.Node("x", "builtin.var", (), kind),), attrs)
    assert output.tensor == fm.tensor_type(fm.vector_type("bfloat16", (2, 4)), (1, 512))
    assert output.axis_policies == kind.axis_policies[1:]
    assert output.partial == fm.SBP.partial((0,))


def test_packet_input_rejects_an_implicit_repacking():
    kind = fm.tensor_type(fm.vector_type("bfloat16", (2, 4)), (4, 1, 512))
    attrs = ReduceSum.normalize_attrs({"axes": (0,), "keep_dims": False, "output_lanes": (4,)})
    with pytest.raises(IRSchemaError, match="input lane"):
        ReduceSum.infer_type((fm.Node("x", "builtin.var", (), kind),), attrs)


def test_explicit_component_domain_preserves_tensor_coordinates():
    abi = {"local_capacity_shape": (4, 3, 5), "scalar_lane_count": 8, "active_shape_expressions": ("4", "3", "5")}
    domain = local_reduction_domain(abi, (0,), 256, lane_wise=True)
    assert domain["rows"] == 3 * 5 * 8 and domain["capacity"] == 4
    for row in range(domain["rows"]):
        for offset in range(4):
            env = {"_fm_row": row, "_fm_offsets": offset}
            actual = tuple(eval(value, {"__builtins__": {}}, env) for value in domain["coordinates"])
            assert actual == (offset, row // 8 // 5, row // 8 % 5)
            assert eval(domain["lane_coordinate"], {"__builtins__": {}}, env) == row % 8


@pytest.mark.parametrize("lanes", [(8,), (2, 4)])
@pytest.mark.parametrize("groups", [1, 2])
def test_producer_can_prepend_output_groups_without_splitting_input_components(lanes, groups):
    kind = fm.tensor_type(fm.vector_type("bfloat16", lanes), (4, 1, 12))
    attrs = ReduceSum.normalize_attrs({"axes": (0,), "keep_dims": False, "output_lanes": (groups, *lanes)})
    output = ReduceSum.infer_type((fm.Node("x", "builtin.var", (), kind),), attrs)
    assert output == fm.tensor_type(fm.vector_type("bfloat16", (groups, *lanes)), (1, 12 // groups))
    value = torch.randn((4, 1, 12, *lanes), generator=torch.Generator().manual_seed(20261018)).bfloat16()
    node = fm.Node("sum", "math.reduce_sum", ("x",), output, attrs=attrs)
    got = ReduceSum.evaluate(node, (value,), SimpleNamespace(types={"x": kind}))
    expected = value.float().sum(0).bfloat16().reshape(1, 12 // groups, groups, *lanes)
    assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("lanes", [(8,), (2, 4)])
def test_packet_input_producer_demand_reaches_one_shared_reduction(lanes):
    from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
    from triton.flagmega.passes.functions import post_function_boundary_pack_propagation
    from triton.flagmega.targets import NvidiaSm90Target

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type(fm.vector_type("bfloat16", lanes), (1, 4, 12)), id="x")
            reduction = fm.F.math.reduce_sum(x, axes=(1,), keep_dims=False, name="sum")
            vector = fm.F.tensors.pack(reduction, (2,), axes=(-1,))
            scaled = fm.F.math.vectorized_binary(vector, fm.F.builtin.splat_const(vector.type, 0.25), binary_op="mul")
            self.function("main", (x,), (scaled, reduction))

    original = Graph(dialect="high_level", stage="packed", entry="main").build()
    rewritten = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    (reduction,) = (node for node in rewritten.nodes if node.op == "math.reduce_sum")
    assert reduction.attrs["output_lanes"] == (2, *lanes)
    assert not any(node.op == "tensors.pack" for node in rewritten.nodes)
    value = torch.randn((1, 4, 12, *lanes), generator=torch.Generator().manual_seed(20261018)).bfloat16()
    evaluator = TorchEvaluator(DictWeightResolver({}))
    before, after = evaluator.run(original, {"x": value}), evaluator.run(rewritten, {"x": value})
    assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(before, after, strict=True))
    assert (
        post_function_boundary_pack_propagation(rewritten, NvidiaSm90Target()).semantic_hash == rewritten.semantic_hash
    )


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("lanes", [(8,), (2, 4)])
@pytest.mark.parametrize("axes", [(1,), (-1,)])
@pytest.mark.parametrize("variant", ["local", "tiled_rows"])
@pytest.mark.parametrize("output_groups", [(), (2,)])
def test_device_component_loads_and_stores_are_guarded(tmp_path, dtype, lanes, axes, variant, output_groups):
    from math import prod
    from pathlib import Path
    import sys
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime
    from triton.flagmega.targets import NvidiaSm90Target

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    shape = (3, 4, 14)
    kind = fm.tensor_type(fm.vector_type(dtype, lanes), shape)
    mesh = fm.Placement((8, 16), "yx", "bb")

    class Graph(fm.Module):
        def forward(self):
            value = self.input("x", kind, id="x")
            self.function(
                "main",
                (value,),
                (
                    fm.F.math.reduce_sum(
                        value,
                        axes=axes,
                        keep_dims=False,
                        output_lanes=(*output_groups, *lanes) if output_groups else (),
                        name="sum",
                    ),
                ),
            )

    graph = Graph(
        dialect="distributed",
        stage="frozen_constants",
        entry="main",
        metadata={"auto_distribution": {"placement": mesh.to_data()}},
    ).build()
    compiler = Compiler()
    catalog = NvidiaSm90Target().triton_implementation_model
    if variant == "tiled_rows":
        tutorial = Path(__file__).resolve().parents[5] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
        sys.path.insert(0, str(tutorial))
        from local_optimizations.reduction_rows_schedule import implementation_model

        catalog = implementation_model(catalog)
    compiler.target = NvidiaSm90Target(triton_implementation_model=catalog)
    compiled = compiler.compile(graph).module
    (dispatch,) = (
        entry.dispatch for entry in compiled.kernel_definitions if entry.dispatch.semantic_op == "math.reduce_sum"
    )
    assert dispatch.microkernel.variant == variant
    assert fm.load_module(fm.emit_module(compiled, tmp_path / "compiled.py")).semantic_hash == compiled.semantic_hash
    runtime = load(
        write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True), device="cuda:0"
    )
    output_type = graph.node_map["sum"].type
    output_shape = tuple(d.fixed_value for d in output_type.shape) + output_type.dtype.lanes
    count = prod(output_shape)
    backing = torch.full((count + 128,), -123.0, dtype=getattr(torch, dtype), device="cuda")
    output = backing[64:-64].reshape(output_shape)
    value = (
        ((torch.arange(prod(shape) * prod(lanes), device="cuda") % 31 - 15).float() / 16)
        .reshape(*shape, *lanes)
        .to(getattr(torch, dtype))
    )
    try:
        for mutation in range(3):
            value.add_(1 / 128)
            arguments = tuple(output if spec["role"] == "result" else value for spec in runtime.external_arguments)
            Runtime.prepare(runtime, *arguments)
            Runtime.run_into(runtime, *arguments)
            torch.cuda.synchronize()
            expected = (
                value.cpu().float().sum(tuple(axis % len(shape) for axis in axes)).to(value.dtype).reshape(output_shape)
            )
            assert torch.equal(output.cpu().view(torch.uint8), expected.view(torch.uint8))
            assert bool((backing[:64] == -123.0).all()) and bool((backing[-64:] == -123.0).all())
            assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
