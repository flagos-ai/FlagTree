# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A BF16 epilogue consumes the FP32 accumulator without another call."""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.passes.tir.fuse_packed_accumulator_cast import fuse_packed_accumulator_casts
from triton.flagmega.rules.neutral._utility import make_node
from triton.flagmega.selection import apply_plan, override_plan
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.nvidia.dense_matmul import packed_mma_accumulator_implementations

from python.test.flagmega.codegen.triton.kernels.dense_matmul.test_small_bf16_schedule import _graph

IDENTITY = "tir.dense_matmul.packed_k_major_mma_accumulator"


def _module(mesh, *, explicit_cast=False):
    source = _graph(1, 256, 24, mesh, output="float32")
    producer = source.node_map["projection"]
    parameter = fm.Node("accumulator", "builtin.var", (), fm.logical_type(producer.type), attrs={"name": "accumulator"})
    cast = make_node("ntt.vectorized_cast", "rounded", (parameter,),
                     {"new_type": fm.vector_type("bfloat16", (8,)), "vectorize_axes": (1, 1)}, {})
    if explicit_cast:
        cast = make_node(cast.op, cast.id, (producer,), cast.attrs, {})
        function = source.functions[0]
        outputs = tuple(cast.id if value == producer.id else value for value in function.outputs)
        return fm.verify_module(replace(source, nodes=(*source.nodes, cast),
                                       functions=(replace(function, outputs=outputs),)))
    fusion = fm.Fusion("round_accumulator", (parameter, cast), cast.id)
    folded = make_node(producer.op, producer.id, tuple(source.node_map[i] for i in producer.inputs),
                       {**producer.attrs, "post_ops": (fusion,)}, producer.metadata)
    return fm.verify_module(replace(source, nodes=tuple(folded if n.id == producer.id else n for n in source.nodes)))


def _compile(mesh):
    class Target(NvidiaSm90Target):
        def propose_tir(self, module):
            proposed = super().propose_tir(fuse_packed_accumulator_casts(module))
            return apply_plan(proposed, override_plan(proposed, [("tir.rounded", IDENTITY)],
                policy="test.accumulator_cast", rationale="Round only the explicit FP32 projection accumulator."))

    base = NvidiaSm90Target().triton_implementation_model
    catalog = replace(base, implementations=(*base.implementations, *packed_mma_accumulator_implementations()))
    compiler = Compiler(CompileOptions(placement=mesh))
    compiler.target = Target(triton_implementation_model=catalog)
    return compiler.compile(_module(mesh, explicit_cast=True)).module


def test_fp32_accumulator_cast_is_one_normal_kernel_and_roundtrips(tmp_path):
    result = _compile(fm.Placement((1, 1), "yx", "bb"))
    definitions = [d for d in result.kernel_definitions if d.dispatch.microkernel.family == "dense_matmul"]
    assert len(definitions) == 1
    assert definitions[0].dispatch.semantic_attrs["output_data_type"] == "float32"
    assert definitions[0].dispatch.semantic_attrs["post_ops"]
    assert not any(d.dispatch.semantic_op == "ntt.vectorized_cast" for d in result.kernel_definitions)
    restored = fm.load_module(fm.emit_module(result, tmp_path / "module.py"))
    assert restored.semantic_hash == result.semantic_hash
    write_artifact(restored, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)


def test_post_cast_keeps_the_fp32_semantic_accumulator():
    import torch
    from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator

    module = _module(fm.Placement((1, 1), "yx", "bb"))
    x = (((torch.arange(256) % 13) - 6) / 128).reshape(1, 256).bfloat16()
    w = (((torch.arange(256 * 24) % 17) - 8) / 128).reshape(256, 24).bfloat16()
    packed = w.reshape(16, 2, 8, 3, 8).permute(0, 3, 4, 1, 2).contiguous()
    result, _ = TorchEvaluator(DictWeightResolver({})).run(module,
        {"x": x, "w": packed, "sentinel": torch.arange(128, dtype=torch.float32)})
    torch.testing.assert_close(result.reshape(1, 24), (x.float() @ w.float()).bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("shared,exported", [(False, False), (True, False), (False, True)])
def test_folding_preserves_observable_fp32_values(shared, exported):
    source = _graph(1, 256, 24, fm.Placement((1, 1), "yx", "bb"), output="float32")
    producer = source.node_map["projection"]
    cast = make_node("ntt.vectorized_cast", "rounded", (producer,),
                     {"new_type": fm.vector_type("bfloat16", (8,)), "vectorize_axes": (1, 1)}, {})
    nodes = list(source.nodes)
    nodes.append(cast)
    outputs = [cast.id]
    if exported:
        outputs.append(producer.id)
    if shared:
        other = make_node("math.add", "wide_use", (producer, producer), {}, {})
        nodes.append(other)
        outputs.append(other.id)
    source = fm.verify_module(replace(source, nodes=tuple(nodes), functions=(replace(source.functions[0], outputs=tuple(outputs)),)))
    result = fm.verify_module(fuse_packed_accumulator_casts(source))
    assert ("projection" in result.node_map) == (shared or exported)
    if not shared and not exported:
        assert result.node_map["rounded"].op == "ntt.packed_matmul"
        assert result.node_map["rounded"].attrs["output_data_type"] == "float32"
        assert result.node_map["rounded"].type == cast.type


def test_arbitrary_dense_post_ops_are_rejected():
    from triton.flagmega.codegen.triton.fusion import require_fusion
    from triton.flagmega.errors import CodegenError

    source = _module(fm.Placement((1, 1), "yx", "bb"))
    producer = source.node_map["projection"]
    body = producer.attrs["post_ops"][0]
    tail = make_node("math.vectorized_unary", "sigmoid", (body.nodes[-1],), {"unary_op": "sigmoid"}, {})
    body = fm.Fusion(body.name, (*body.nodes, tail), tail.id)
    node = replace(producer, attrs={**producer.attrs, "post_ops": (body,)})
    with pytest.raises(CodegenError, match="No PreOps/PostOps boundary emitter"):
        require_fusion(node, tuple(source.node_map[value] for value in node.inputs))


def test_cast_is_not_moved_across_boxing():
    source = _graph(1, 256, 24, fm.Placement((1, 1), "yx", "bb"), output="float32")
    producer = source.node_map["projection"]
    boxed = make_node("distributed.boxing", "boxed", (producer,), {"new_type": producer.type}, {})
    cast = make_node("ntt.vectorized_cast", "rounded", (boxed,),
                     {"new_type": fm.vector_type("bfloat16", (8,)), "vectorize_axes": (1, 1)}, {})
    source = fm.verify_module(replace(source, nodes=(*source.nodes, boxed, cast),
        functions=(replace(source.functions[0], outputs=(cast.id,)),)))
    result = fm.verify_module(fuse_packed_accumulator_casts(source))
    assert result.node_map["projection"].attrs == producer.attrs
    assert result.node_map["rounded"].op == "ntt.vectorized_cast"


def test_partial_projection_remains_explicit_before_rounding():
    mesh = fm.Placement((8, 16), "yx", "bb")
    b = fm.SBP.broadcast()
    lhs = fm.Node("lhs", "builtin.var", (), fm.DistributedType(fm.tensor_type("bfloat16", (1, 256)),
        (b, fm.SBP.split_contiguous((1,), 16)), mesh), attrs={"name": "lhs"})
    rhs = fm.Node("rhs", "builtin.var", (), fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (16, 16)),
        (fm.SBP.split_contiguous((1,), 1), fm.SBP.split_contiguous((0,), 2)), mesh), attrs={"name": "rhs"})
    none = fm.Node("none", "builtin.none", (), fm.NoneType())
    producer = make_node("ntt.packed_matmul", "projection", (lhs, rhs, none, none), {"output_data_type": "float32"}, {})
    assert producer.type.partial is not None
    cast = make_node("ntt.vectorized_cast", "rounded", (producer,),
                     {"new_type": fm.vector_type("bfloat16", (8,)), "vectorize_axes": (1, 1)}, {})
    source = fm.verify_module(fm.IRModule("distributed", "distributed", (lhs, rhs, none, producer, cast),
        (fm.Function("main", (lhs.id, rhs.id), (cast.id,)),), "main"))
    result = fm.verify_module(fuse_packed_accumulator_casts(source))
    assert result.node_map["projection"].type == producer.type
    assert result.node_map["rounded"].op == "ntt.vectorized_cast"


def test_device_post_cast_matches_the_unrounded_accumulator(tmp_path):
    import torch
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

    from python.test.flagmega.codegen.triton.kernels.dense_matmul.test_mma_accumulator import _compile as plain_compile

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    mesh = fm.Placement((8, 16), "yx", "bb")
    modules = [plain_compile(mesh), _compile(mesh)]
    runtimes = [load(write_artifact(module, tmp_path / str(i), target="nvidia-sm90", emit_executable=True),
                     device="cuda:0") for i, module in enumerate(modules)]
    generator = torch.Generator(device="cuda").manual_seed(29197)
    try:
        for mutation in range(3):
            x = torch.randn((1, 256), generator=generator, device="cuda").bfloat16()
            w = (torch.randn((256, 24), generator=generator, device="cuda") / 16).bfloat16()
            packed = w.reshape(16, 2, 8, 3, 8).permute(0, 3, 4, 1, 2).contiguous()
            results = []
            for runtime, module in zip(runtimes, modules, strict=True):
                specs = [s for s in runtime.external_arguments if s["role"] == "result"]
                kind = fm.logical_type(module.node_map[specs[0]["value"]].type)
                dtype = getattr(torch, kind.dtype.elem_type.value)
                result = torch.full((24 + 32,), 123., dtype=dtype, device="cuda")
                copied = torch.full((128 + 32,), 123., device="cuda")
                shape = (*[d.fixed_value for d in kind.shape], *kind.dtype.lanes)
                values = {"x": x, "w": packed, "sentinel": torch.arange(128, dtype=torch.float32, device="cuda"),
                          specs[0]["value"]: result[:24].reshape(shape), specs[1]["value"]: copied[:128]}
                arguments = tuple(values[s["value"]] for s in runtime.external_arguments)
                GeneratedTirCallGraphModule.prepare(runtime, *arguments)
                GeneratedTirCallGraphModule.run_into(runtime, *arguments)
                torch.cuda.synchronize()
                assert torch.all(result[24:] == 123.) and torch.all(copied[128:] == 123.)
                assert runtime.resource_report["spill_bytes"] == 0
                results.append(result[:24].bfloat16())
            torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    finally:
        for runtime in runtimes:
            runtime.close()
