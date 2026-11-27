# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pure ownership copies own conversion without crossing a reduction."""

from dataclasses import replace
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.fusion import fusion_rules
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.neutral._utility import make_node


def graph(
    mode="direct",
    *,
    shared=False,
    exported=False,
    partial=False,
    exclusive=False,
    identity=False,
    levels="bb",
    remote=False,
):
    mesh = fm.Placement(
        (2, 4) if levels == "cb" else (4, 2) if levels == "bc" else (4, 4),
        "cy" if levels == "cb" else "yc" if levels == "bc" else "yx",
        levels,
    )
    b = fm.SBP.broadcast()
    tensor = fm.tensor_type(fm.vector_type("float32" if mode == "roundtrip" else "bfloat16", (8,)), (1, 32))
    owner_axis = levels.index("c") if remote else 1
    x_type = fm.DistributedType(
        tensor,
        (b, b) if exclusive else (b, fm.SBP.split_contiguous((owner_axis,), 32 // mesh.hierarchy[owner_axis])),
        mesh,
        partial=fm.SBP.partial((0,)) if partial else None,
        exclusive=fm.SBP.exclusive((0,)) if exclusive else None,
    )
    if identity:
        x_type = fm.DistributedType(tensor, (b, b), mesh)
    x = fm.Node("x", "builtin.var", (), x_type, attrs={"name": "x"})
    destination = fm.DistributedType(tensor, (b, b), mesh)
    nodes = [x]
    producer = x
    if remote:
        producer = make_node("math.add", "owned", (x, x), {}, {})
        nodes.append(producer)
    copy = make_node("distributed.force_boxing", "copy", (producer,), {"new_type": destination}, {})
    nodes.append(copy)
    value = copy
    if mode == "roundtrip":
        value = make_node("tensors.cast", "narrow", (value,), {"dtype": fm.vector_type("bfloat16", (8,))}, {})
        nodes.append(value)
    if mode == "scalar_view":
        value = make_node("tensors.bitcast", "scalar_view", (value,), {"dtype": "bfloat16"}, {})
        nodes.append(value)
    elif mode == "owner_view":
        axis = levels.index("b") if remote else 1
        view_type = replace(
            destination, axis_policies=(b, fm.SBP.split_contiguous((axis,), 32 // mesh.hierarchy[axis]))
        )
        value = make_node("distributed.sharded_view", "owner_view", (value,), {"new_type": view_type}, {})
        nodes.append(value)
    dtype = "float32" if mode == "scalar_view" else fm.vector_type("float32", (8,))
    result = make_node("tensors.cast", "result", (value,), {"dtype": dtype}, {})
    nodes.append(result)
    outputs = [result.id]
    if exported:
        outputs.append(copy.id)
    if shared:
        other = make_node("math.add", "other", (copy, copy), {}, {})
        nodes.append(other)
        outputs.append(other.id)
    return fm.verify_module(
        fm.IRModule(
            "distributed",
            "frozen_constants",
            tuple(nodes),
            (fm.Function("main", ("x",), tuple(outputs)),),
            "main",
            metadata={"auto_distribution": {"placement": mesh.to_data()}},
        )
    )


@pytest.mark.parametrize("mode", ["direct", "scalar_view", "owner_view", "roundtrip"])
def test_numeric_copy_epilogue_eliminates_standalone_conversion(mode, tmp_path):
    source = graph(mode)
    result = fm.verify_module(DataflowPass("Fuse", fusion_rules()).run(source))
    assert not any(n.op in {"tensors.cast", "ntt.vectorized_cast"} for n in result.nodes)
    assert result.node_map["result"].type == source.node_map["result"].type
    copies = [n for n in result.nodes if n.op == "distributed.force_boxing"]
    assert len(copies) == 1
    assert copies[0].attrs.get("post_ops")
    assert copies[0].attrs["new_type"] == source.node_map["copy"].type
    restored = fm.load_module(fm.emit_module(result, tmp_path / "module.py"))
    assert restored.semantic_hash == result.semantic_hash
    assert DataflowPass("Fuse", fusion_rules()).run(restored).semantic_hash == result.semantic_hash
    import torch
    from triton.flagmega.evaluator import TorchEvaluator, DictWeightResolver

    g = torch.Generator().manual_seed(3755)
    for _ in range(3):
        x = torch.randn((1, 32, 8), generator=g)
        if mode != "roundtrip":
            x = x.bfloat16()
        expected = TorchEvaluator(DictWeightResolver({})).run(source, {"x": x})
        actual = TorchEvaluator(DictWeightResolver({})).run(restored, {"x": x})
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[0].reshape(-1), x.bfloat16().float().reshape(-1), rtol=0, atol=0)


@pytest.mark.parametrize("reason", ["shared", "exported", "partial", "exclusive", "identity"])
def test_numeric_copy_does_not_cross_an_unproven_boundary(reason):
    source = graph(**{reason: True})
    result = fm.verify_module(DataflowPass("Fuse", fusion_rules()).run(source))
    assert result.node_map["result"].op == "tensors.cast"
    assert not result.node_map["copy"].attrs.get("post_ops")


def test_scalar_bit_reinterpretation_is_not_numeric_lane_regrouping():
    source = graph("direct")
    copy = source.node_map["copy"]
    bits = make_node("tensors.bitcast", "bits", (copy,), {"dtype": fm.vector_type("float16", (8,))}, {})
    wide = make_node("tensors.cast", "result", (bits,), {"dtype": fm.vector_type("float32", (8,))}, {})
    source = fm.verify_module(replace(source, nodes=(*source.nodes[:-1], bits, wide)))
    result = fm.verify_module(DataflowPass("Fuse", fusion_rules()).run(source))
    assert result.node_map["result"].op == "tensors.cast"


def test_shared_lane_view_preserves_the_unconverted_public_value():
    source = graph("scalar_view")
    fn = source.functions[0]
    source = fm.verify_module(replace(source, functions=(replace(fn, outputs=(*fn.outputs, "scalar_view")),)))
    result = fm.verify_module(DataflowPass("Fuse", fusion_rules()).run(source))
    assert result.node_map["result"].op == "tensors.cast"


@pytest.mark.parametrize("mode", ["direct", "scalar_view", "owner_view", "roundtrip"])
def test_device_copy_conversion_preserves_inputs_guards_and_rounding(mode, tmp_path):
    import torch
    from math import prod
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.options import CompileOptions
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Run
    from triton.flagmega.targets import NvidiaSm90Target

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    source = graph(mode)
    mesh = source.node_map["x"].type.placement

    class Target(NvidiaSm90Target):
        def pre_post_ops_rules(self):
            return fusion_rules() if self.fused else ()

    runtimes = []
    outputs = []
    arguments = []
    guards = []
    inputs = {}

    def allocate(kind):
        kind = fm.logical_type(kind)
        shape = tuple(d.fixed_value for d in kind.shape) + tuple(getattr(kind.dtype, "lanes", ()))
        dtype = getattr(torch, getattr(kind.dtype, "elem_type", kind.dtype).value)
        raw = torch.full((prod(shape) + 32,), 123, dtype=dtype, device="cuda")
        guards.append(raw)
        return raw[16:-16].reshape(shape)

    try:
        for fused in (False, True):
            c = Compiler(CompileOptions(placement=mesh))
            c.target = Target().with_placement(mesh)
            c.target.fused = fused
            m = c.compile(source).module
            assert (
                not any(k.dispatch.semantic_op in {"tensors.cast", "ntt.vectorized_cast"} for k in m.kernel_definitions)
            ) == fused
            r = load(
                write_artifact(m, tmp_path / str(fused), target="nvidia-sm90", emit_executable=True), device="cuda:0"
            )
            runtimes.append(r)
            binding = r.buffer_plan.function_map[m.entry]
            values = {}
            for value, names in binding.parameters:
                if value not in inputs:
                    inputs[value] = allocate(m.node_map[value].type)
                values[names[0]] = inputs[value]
            value, names = binding.outputs[0]
            output = allocate(m.node_map[value].type)
            outputs.append(output)
            values[names[0]] = output
            args = tuple(values[s["buffer"]] for s in r.external_arguments)
            arguments.append(args)
            Run.prepare(r, *args)
            assert r.resource_report["spill_bytes"] == 0
        g = torch.Generator().manual_seed(906)
        for mutation in range(6):
            x = torch.randn((1, 32, 8), generator=g)
            if mutation == 1:
                x = torch.ldexp(x, (torch.arange(x.numel()).reshape(x.shape) % 31 - 15).int())
            if mutation == 2:
                x.fill_(-0.0)
            if mode != "roundtrip":
                x = x.bfloat16()
            inputs["x"].copy_(x)
            before = inputs["x"].clone()
            for r, args, output in zip(runtimes, arguments, outputs, strict=True):
                output.fill_(-17)
                Run.run_into(r, *args)
            torch.cuda.synchronize()
            expected = x.bfloat16().float().reshape(-1)
            for output in outputs:
                torch.testing.assert_close(output.cpu().reshape(-1), expected, rtol=0, atol=0)
            assert torch.equal(outputs[0].view(torch.uint8).flatten(), outputs[1].view(torch.uint8).flatten())
            assert torch.equal(inputs["x"].view(torch.uint8), before.view(torch.uint8))
            for raw in guards:
                assert bool((raw[:16] == 123).all()) and bool((raw[-16:] == 123).all())
    finally:
        for r in runtimes:
            r.close()
