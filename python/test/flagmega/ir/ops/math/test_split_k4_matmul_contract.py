# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit four-partial K32 order survives typed IR without changing defaults."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.ops.math.matmul import MatMul, matmul_value
from triton.flagmega.ir.ops.ntt.packed_matmul import PackedMatMul
from triton.flagmega.ir.ops.ntt.matmul_norm_stats import MatMulNormStats
from triton.flagmega.ir.tir.serialization import tir_from_data, tir_to_data
from triton.flagmega.rules.ntt.vectorize.matmul import VectorizeMatMul

ORDER = "split_k4_k32_mma_k16_pairtree"


def node(name, kind):
    return fm.Node(name, "builtin.var", (), kind, attrs={"name": name})


def operands(k=128, dtype="bfloat16", packed=False):
    lhs = node("lhs", fm.tensor_type(dtype, (32, k)))
    if packed:
        rhs = node("rhs", fm.tensor_type(fm.vector_type(dtype, (8, 2, 8)), (k // 16, 8)))
        none = fm.Node("none", "builtin.none", (), fm.NoneType())
        return lhs, rhs, none, none
    return lhs, node("rhs", fm.tensor_type(dtype, (64, k)))


@pytest.mark.parametrize("k", [128, 256, 6144])
@pytest.mark.parametrize("packed", [False, True])
def test_split_order_prepares_rank2_without_workload_specific_shape(k, packed):
    definition = PackedMatMul if packed else MatMul
    attrs = {"reduction_order": ORDER, "output_data_type": "bfloat16"}
    if not packed:
        attrs["transpose_b"] = True
    prepared = definition.prepare(operands(k, packed=packed), attrs)
    assert prepared.attrs["reduction_order"] == ORDER
    assert prepared.result_type.rank == 2
    assert definition.ir_attrs(prepared.attrs)["reduction_order"] == ORDER


@pytest.mark.parametrize("k", [0, 64, 192, "K"])
def test_split_order_requires_complete_static_scalar_k128(k):
    with pytest.raises(IRSchemaError, match="K128"):
        MatMul.prepare(operands(k), {"transpose_b": True, "reduction_order": ORDER})


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_split_order_requires_bf16(dtype):
    with pytest.raises(IRSchemaError, match="BF16"):
        MatMul.prepare(operands(dtype=dtype), {"transpose_b": True, "reduction_order": ORDER})


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("invalid", ["k_split", "lhs_partial", "rhs_partial"])
def test_split_order_rejects_independent_k_shards_and_partial_inputs(packed, invalid):
    args = list(operands(packed=packed))
    placement = fm.Placement((2, 2), "yx", "bb")
    b = fm.SBP.broadcast()
    split = fm.SBP.split_contiguous((0, ), 1)
    for i in range(2):
        policies = (b, b)
        if invalid == "k_split":
            policies = (split, b) if i == 1 and packed else (b, split)
        partial = fm.SBP.partial((0, )) if invalid == ("lhs_partial" if i == 0 else "rhs_partial") else None
        args[i] = replace(args[i], type=fm.DistributedType(args[i].type, policies, placement, partial=partial))
    attrs = {"reduction_order": ORDER, **({} if packed else {"transpose_b": True})}
    with pytest.raises(IRSchemaError, match="ordered|partial|K shards"):
        (PackedMatMul if packed else MatMul).prepare(args, attrs)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("axis", ["M", "N"])
def test_complete_k_keeps_legal_output_sharding(packed, axis):
    args = list(operands(packed=packed))
    placement = fm.Placement((2, 2), "yx", "bb")
    b, split = fm.SBP.broadcast(), fm.SBP.split_contiguous((1, ), 1)
    lhs_policy = (split, b) if axis == "M" else (b, b)
    rhs_policy = ((b, split) if packed else (split, b)) if axis == "N" else (b, b)
    args[0] = replace(args[0], type=fm.DistributedType(args[0].type, lhs_policy, placement))
    args[1] = replace(args[1], type=fm.DistributedType(args[1].type, rhs_policy, placement))
    attrs = {"reduction_order": ORDER, **({} if packed else {"transpose_b": True})}
    result = (PackedMatMul if packed else MatMul).prepare(args, attrs).result_type
    assert result.partial is None
    assert result.axis_policies == ((split, b) if axis == "M" else (b, split))


def test_new_order_vectorization_validates_logical_k_not_packed_group_count():
    lhs, rhs = operands()
    prepared = MatMul.prepare((lhs, rhs), {"transpose_b": True, "reduction_order": ORDER})
    result = fm.Node("result", MatMul.op_name, (lhs.id, rhs.id), prepared.result_type, prepared.effect, prepared.attrs)
    module = fm.IRModule(dialect="hl", stage="imported", nodes=(lhs, rhs, result),
                         functions=(fm.Function("main", (lhs.id, rhs.id), (result.id, )), ), entry="main")
    rule = VectorizeMatMul()
    candidates = rule.candidates(result, module)
    assert {c.parameters["kind"] for c in candidates} == {"n", "mn", "mkn"}
    for candidate in candidates:
        rewritten = rule.rewrite(result, module, candidate)
        compute = next(n for n in rewritten.prefix_nodes if n.op == "math.vectorized_matmul")
        assert compute.attrs["reduction_order"] == ORDER


def test_semantic_module_and_selected_dispatch_roundtrip_keep_distinct_order(tmp_path):

    class Module(fm.Module):

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type("bfloat16", (1, 128)))
            rhs = self.input("rhs", fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (8, 2)))
            none = fm.F.builtin.none()
            out = fm.F.ntt.packed_matmul(lhs, rhs, none, none, reduction_order=ORDER, output_data_type="bfloat16")
            self.function("main", (lhs, rhs), (out, ))

    module = Module(dialect="ntt", stage="packed", entry="main").build()
    restored = fm.load_module(fm.emit_module(module, tmp_path / "split.py"))
    assert restored.semantic_hash == module.semantic_hash
    semantic = next(n for n in restored.nodes if n.op == "ntt.packed_matmul")
    assert semantic.attrs["reduction_order"] == ORDER
    old = replace(
        module, nodes=tuple(
            replace(n, attrs={**n.attrs, "reduction_order": "mma_k16"}) if n.id == semantic.id else n
            for n in module.nodes))
    assert old.semantic_hash != module.semantic_hash
    dispatch = fm.KernelDispatch("ntt.packed_matmul", semantic_candidate="test.split", arguments=("lhs", "rhs"),
                                 outputs=("result", ), semantic_attrs=semantic.attrs)
    assert tir_from_data(tir_to_data(dispatch)).semantic_attrs["reduction_order"] == ORDER


def test_matmul_norm_stats_keeps_pairtree_projection_boundary():
    lhs, rhs, _, _ = operands(packed=True)
    addend = node("addend", fm.tensor_type(fm.vector_type("bfloat16", (8, )), (32, 8)))
    prepared = MatMulNormStats.prepare((lhs, rhs, addend), {
        "rhs_layout": "k_major",
        "axis": 1,
        "use_mean": False,
        "output_data_type": "bfloat16",
        "reduction_order": ORDER,
    })
    assert prepared.attrs["reduction_order"] == ORDER
    assert prepared.result_type.fields[0] == addend.type


@pytest.mark.parametrize("order", [None, "relaxed", "mma_k16"])
def test_old_ir_defaults_are_unchanged(order):
    attrs = {} if order is None else {"reduction_order": order}
    normalized = MatMul.normalize_attrs(attrs)
    assert normalized.get("reduction_order") == ("mma_k16" if order == "mma_k16" else None)


@pytest.mark.parametrize("op,attrs", [
    ("nn.qkv_parallel_linear", {"num_heads": 8, "num_kv_heads": 8, "output_data_type": "bfloat16"}),
    ("ntt.packed_qkv_parallel_linear", {"num_heads": 8, "num_kv_heads": 8, "output_data_type": "bfloat16"}),
    ("nn.dense_matmul_glu", {}),
    ("nn.packed_dense_matmul_glu", {}),
])
def test_unimplemented_semantic_families_do_not_inherit_new_order(op, attrs):
    with pytest.raises(IRSchemaError, match="reduction_order"):
        fm.get_definition(op).normalize_attrs({**attrs, "reduction_order": ORDER})


def test_expert_normalizer_does_not_silently_accept_split_order():
    from triton.flagmega.ir.ops.nn._sparse_experts import normalize_projection_order
    with pytest.raises(IRSchemaError, match="reduction_order"):
        normalize_projection_order({"reduction_order": ORDER})


def test_exact_evaluation_does_not_enter_torch_gemm():
    with pytest.raises(EvaluationError, match="exact.*split|split.*exact"):
        matmul_value(None, None, None, None, reduction_order=ORDER)


def _logical_weight_module():

    class Module(fm.Module):

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type("bfloat16", (1, 128)), id="lhs")
            rhs = self.weight("rhs", fm.tensor_type("bfloat16", (64, 128)), source="memory", key="rhs", id="rhs")
            result = fm.F.math.matmul(lhs, rhs, transpose_b=True, reduction_order=ORDER, name="result")
            self.function("main", (lhs, ), (result, ))

    return Module(dialect="high_level", stage="imported", entry="main").build()


@pytest.mark.parametrize("vectorized", [False, True])
def test_normal_packing_preserves_order_on_both_scalar_and_vectorized_routes(tmp_path, vectorized):
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.rules.ntt.packing import NttPackingPolicy
    from triton.flagmega.targets import NvidiaSm90Target
    module = _logical_weight_module()
    if vectorized:
        # CPU IR passes only; no executable/native compiler is invoked.
        packed = Compiler().compile(module, stop_after="apply-packing").module
        expected = "ntt.packed_matmul"
    else:
        target = NvidiaSm90Target()
        policy = NttPackingPolicy(vector_bytes=16, k_pack=2)
        packed = policy.apply(policy.propose(module, target), target)
        expected = "math.packed_dense_matmul"
    compute = next(n for n in packed.nodes if n.op == expected)
    assert compute.attrs["reduction_order"] == ORDER
    assert fm.load_module(fm.emit_module(packed, tmp_path / "packed.py")).semantic_hash == packed.semantic_hash


@pytest.mark.parametrize("packed", [False, True])
def test_actual_distribution_provider_keeps_complete_k(packed):
    from triton.flagmega.passes.auto_distributed import DistributedCandidateContext
    from triton.flagmega.passes.auto_distributed.providers import MatMulCandidateProvider
    from triton.flagmega.passes.auto_distributed.packed_matmul_provider import PackedMatMulCandidateProvider
    from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider
    args = operands(packed=packed)
    definition = PackedMatMul if packed else MatMul
    prepared = definition.prepare(args, {"reduction_order": ORDER, **({} if packed else {"transpose_b": True})})
    compute = fm.Node("result", definition.op_name, tuple(n.id for n in args), prepared.result_type, prepared.effect,
                      prepared.attrs)
    nodes = tuple({n.id: n for n in (*args, compute)}.values())
    module = fm.IRModule(dialect="ntt", stage="packed", nodes=nodes,
                         functions=(fm.Function("main", ("lhs", "rhs"), ("result", )), ), entry="main")
    context = DistributedCandidateContext(module, compute, fm.Placement((2, 2), "yx", "bb"),
                                          tuple((arg.type, ) for arg in args),
                                          PyNttDistributedSplitCandidateProvider(4))
    candidates = (PackedMatMulCandidateProvider() if packed else MatMulCandidateProvider()).get_candidates(context)
    assert candidates
    assert all(c.input_types[0].axis_policies[1] == fm.SBP.broadcast() and c.return_type.partial is None
               for c in candidates)


@pytest.mark.parametrize("export_projection", [False, True])
def test_existing_norm_stats_fusion_preserves_order_and_shared_result_boundary(export_projection):
    from triton.flagmega.passes.norm_stats import lower_add_norm_stats

    class Module(fm.Module):

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type("bfloat16", (1, 128)))
            rhs = self.input("rhs", fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (8, 2)))
            residual = self.input("residual", fm.tensor_type(fm.vector_type("bfloat16", (8, )), (1, 2)))
            none = fm.F.builtin.none()
            projection = fm.F.ntt.packed_matmul(lhs, rhs, none, none, output_data_type="bfloat16",
                                                reduction_order=ORDER, name="projection")
            combined = fm.F.ntt.add_norm_stats(projection, residual, axis=1, use_mean=False, name="combined")
            outputs = list(fm.F.tensors.get_items(combined, 0, 1))
            if export_projection:
                outputs.append(projection)
            self.function("main", (lhs, rhs, residual), outputs)

    before = Module(dialect="ntt", stage="norm_bindings_finalized", entry="main").build()
    after = lower_add_norm_stats(before)
    fused = [n for n in after.nodes if n.op == "ntt.matmul_norm_stats"]
    if export_projection:
        assert not fused
        assert after.node_map["projection"].attrs["reduction_order"] == ORDER
    else:
        assert len(fused) == 1 and fused[0].attrs["reduction_order"] == ORDER
        assert fused[0].attrs["output_data_type"] == "bfloat16"
    assert after.function_map["main"].outputs == before.function_map["main"].outputs


@pytest.mark.parametrize("invalid", ["partial", "Ksplit"])
def test_scalar_packed_intermediate_rejects_incomplete_weights(invalid):
    definition = fm.get_definition("math.packed_dense_matmul")
    placement = fm.Placement((2, 2), "yx", "bb")
    b = fm.SBP.broadcast()
    lhs = node("lhs", fm.DistributedType(fm.tensor_type("bfloat16", (1, 128)), (b, b), placement))
    policies = (fm.SBP.split_contiguous((0, ), 1), b, b, b) if invalid == "Ksplit" else (b, ) * 4
    rhs = node(
        "rhs",
        fm.DistributedType(fm.tensor_type("bfloat16", (8, 2, 2, 64)), policies, placement, partial=fm.SBP.partial(
            (0, )) if invalid == "partial" else None))
    with pytest.raises(IRSchemaError, match="partial|K-sharded"):
        definition.prepare((lhs, rhs), {"reduction_order": ORDER})
