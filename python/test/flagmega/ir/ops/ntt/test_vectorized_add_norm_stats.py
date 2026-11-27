# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Packed residual arithmetic and explicit owner-local statistics stay paired."""
from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import EvaluationError, IRSchemaError, IRVerificationError

ORDER = "square_rn_xor8_fma_xor4_2_1_lane0"


def operands(pre_ops=False, round_lhs=False):
    mesh = fm.Placement((8, 16), "yx", "bb")
    broad = fm.SBP.broadcast()
    lhs_type = fm.DistributedType(fm.tensor_type(fm.vector_type("float32", (4,)), (1, 512)),
                                 (broad, fm.SBP.split_contiguous((0, 1), 4)), mesh)
    rhs_type = fm.DistributedType(fm.tensor_type("bfloat16" if pre_ops else "float32", (1, 2048)),
                                 (broad, fm.SBP.split_contiguous((0, 1), 16)), mesh)
    args = tuple(fm.Node(name, "builtin.var", (), ty, attrs={"name": name})
                 for name, ty in (("lhs", lhs_type), ("rhs", rhs_type)))
    attrs = {"binary_op": "add", "rhs_pack": {"axes": (1,), "lanes": (4,)},
             "axis": -1, "use_mean": False, "local_reduction_order": ORDER}
    if pre_ops:
        @fm.fusion(rhs_type)
        def widen(x):
            return fm.F.tensors.cast(x, "float32")
        attrs["pre_ops"] = {"rhs": widen}
    if round_lhs:
        @fm.fusion(lhs_type)
        def rounded(x):
            return fm.F.tensors.cast(fm.F.tensors.cast(x, fm.vector_type("bfloat16", (4,))),
                                     fm.vector_type("float32", (4,)))
        attrs.setdefault("pre_ops", {})["lhs"] = rounded
    return args, attrs


@pytest.mark.parametrize("pre_ops,round_lhs", [(False, False), (True, False), (True, True)])
def test_real_mixed_packing_and_preops_produce_original_raw_and_partial(pre_ops, round_lhs):
    args, attrs = operands(pre_ops, round_lhs)
    op = fm.get_definition("ntt.vectorized_add_norm_stats")
    result = op.prepare(args, attrs)
    assert result.result_type.fields[0] == args[0].type
    stats = result.result_type.fields[1]
    assert stats.tensor == fm.tensor_type("float32", (1, 1, 1))
    assert stats.partial == fm.SBP.partial((0, 1))
    assert result.attrs["local_reduction_order"] == ORDER
    assert not op.inplace_output_parameters
    assert result.attrs.get("pre_ops", {}) == attrs.get("pre_ops", {})


@pytest.mark.parametrize("change", ["input_partial", "axis_partial", "wrong_placement", "wrong_pack",
                                    "postops", "mean", "mul", "order", "missing_order"])
def test_invalid_boundaries_remain_rejected(change):
    args, attrs = operands(True, True)
    if change == "input_partial":
        args = (replace(args[0], type=replace(args[0].type, partial=fm.SBP.partial((0,)))), args[1])
    elif change == "axis_partial":
        args = (replace(args[0], type=replace(args[0].type, axis_policies=(fm.SBP.broadcast(), fm.SBP.partial((0,))))), args[1])
    elif change == "wrong_placement":
        args = (args[0], replace(args[1], type=replace(args[1].type, placement=fm.Placement((4, 32), "yx", "bb"))))
    elif change == "wrong_pack":
        attrs["rhs_pack"] = {"axes": (1,), "lanes": (8,)}
    elif change == "postops":
        attrs["post_ops"] = (attrs["pre_ops"]["lhs"], None)
    elif change == "mean":
        attrs["use_mean"] = True
    elif change == "mul":
        attrs["binary_op"] = "mul"
    elif change == "order":
        attrs["local_reduction_order"] = "group_fma16"
    else:
        attrs.pop("local_reduction_order")
    with pytest.raises(IRSchemaError):
        fm.get_definition("ntt.vectorized_add_norm_stats").prepare(args, attrs)


def test_semantic_module_emit_reload_preserves_two_results_and_preops(tmp_path):
    args, attrs = operands(True, True)
    prepared = fm.get_definition("ntt.vectorized_add_norm_stats").prepare(args, attrs)
    result = fm.Node("paired", "ntt.vectorized_add_norm_stats", tuple(n.id for n in args),
                     prepared.result_type, prepared.effect, prepared.attrs)
    module = fm.IRModule(dialect="ntt", stage="attention_gate_fused", nodes=(*args, result),
                         functions=(fm.Function("main", tuple(n.id for n in args), (result.id,)),), entry="main")
    fm.verify_module(module)
    restored = fm.load_module(fm.emit_module(module, tmp_path / "paired.py"))
    assert restored.semantic_hash == module.semantic_hash
    assert fm.IRModule.from_data(module.to_data()).semantic_hash == module.semantic_hash


def test_dense_evaluator_does_not_pretend_generic_sum_is_exact():
    with pytest.raises(EvaluationError, match="owner|exact"):
        fm.get_definition("ntt.vectorized_add_norm_stats").evaluate(
            fm.Node("unsupported", "builtin.none", (), fm.NoneType()), (), None)


def test_dynamic_outer_rows_have_unknown_cost_without_adding_none():
    args, attrs = operands()
    args = tuple(replace(n, type=replace(n.type, tensor=fm.tensor_type(
        n.type.tensor.dtype, ("rows", n.type.tensor.shape[-1])))) for n in args)
    op = fm.get_definition("ntt.vectorized_add_norm_stats")
    p = op.prepare(args, attrs)
    result = fm.Node("dynamic", op.op_name, tuple(n.id for n in args), p.result_type, p.effect, p.attrs)
    assert op.cost(result).bytes_written is None
    assert op.cost_factors(args, attrs, p.result_type) is None


def test_distributed_logical_cost_and_owner_local_cost_are_distinct():
    args, attrs = operands()
    op = fm.get_definition("ntt.vectorized_add_norm_stats")
    p = op.prepare(args, attrs)
    result = fm.Node("cost", op.op_name, tuple(n.id for n in args), p.result_type, p.effect, p.attrs)
    assert op.cost(result).bytes_written == 8192 + 4
    assert op.cost(result).flops == 3 * 2048
    assert op.cost_factors(args, attrs, p.result_type).block_local_memory_store_bytes == 64 + 4


@pytest.mark.parametrize("lanes", [4, 8])
def test_logical_cost_counts_each_scalar_once_for_different_packets(lanes):
    args = tuple(fm.Node(name, "builtin.var", (), ty, attrs={"name": name}) for name, ty in (
        ("lhs", fm.tensor_type(fm.vector_type("float32", (lanes,)), (1, 16 // lanes))),
        ("rhs", fm.tensor_type("float32", (1, 16)))))
    attrs = {"binary_op": "add", "rhs_pack": {"axes": (1,), "lanes": (lanes,)},
             "axis": -1, "use_mean": False, "local_reduction_order": ORDER}
    op = fm.get_definition("ntt.vectorized_add_norm_stats")
    p = op.prepare(args, attrs)
    node = fm.Node("cost", op.op_name, tuple(n.id for n in args), p.result_type, p.effect, p.attrs)
    assert op.cost(node).flops == 48


def test_legacy_add_norm_stats_does_not_accept_mixed_pack_or_partial_preops():
    args, attrs = operands(True)
    with pytest.raises(IRSchemaError):
        fm.get_definition("ntt.add_norm_stats").prepare(args, attrs)


def residual_module():
    args, attrs = operands(True, True)
    attrs = {k: v for k, v in attrs.items() if k not in {"axis", "use_mean", "local_reduction_order"}}
    p = fm.get_definition("math.vectorized_binary").prepare(args, attrs)
    raw = fm.Node("raw", "math.vectorized_binary", tuple(n.id for n in args), p.result_type, p.effect, p.attrs)
    p = fm.get_definition("nn.norm_stats").prepare((raw,), {"axis": -1, "use_mean": False})
    stats = fm.Node("stats", "nn.norm_stats", (raw.id,), p.result_type, p.effect, p.attrs)
    return fm.IRModule(dialect="ntt", stage="attention_gate_fused", nodes=(*args, raw, stats),
        functions=(fm.Function("main", tuple(n.id for n in args), ("raw", "stats")),), entry="main")


def test_formation_keeps_shared_raw_and_exported_partial_identities(tmp_path):
    from triton.flagmega.passes.functions.form_vectorized_add_norm_stats import form_vectorized_add_norm_stats
    old = residual_module()
    fused = form_vectorized_add_norm_stats(old, pairs=(("raw", "stats"),))
    assert fused.functions == old.functions
    assert fused.node_map["raw"].type == old.node_map["raw"].type
    assert fused.node_map["stats"].type == old.node_map["stats"].type
    raw, stats = (fused.node_map[k] for k in ("raw", "stats"))
    assert raw.inputs == stats.inputs
    assert (raw.attrs["index"], stats.attrs["index"]) == (0, 1)
    assert fused.node_map[raw.inputs[0]].attrs["pre_ops"] == old.node_map["raw"].attrs["pre_ops"]
    assert fm.load_module(fm.emit_module(fused, tmp_path / "formed.py")).semantic_hash == fused.semantic_hash


@pytest.mark.parametrize("change", ["ordered", "mean", "wrong_input", "overlap"])
def test_formation_rejects_other_reduction_contracts(change):
    from triton.flagmega.passes.functions.form_vectorized_add_norm_stats import form_vectorized_add_norm_stats
    old = residual_module()
    stats = old.node_map["stats"]
    if change == "ordered":
        stats = replace(stats, attrs={**stats.attrs, "reduction_order": "tree"})
    elif change == "mean":
        stats = replace(stats, attrs={**stats.attrs, "use_mean": True})
    elif change == "wrong_input":
        stats = replace(stats, inputs=("lhs",))
    old = replace(old, nodes=tuple(stats if n.id == "stats" else n for n in old.nodes))
    pairs = (("raw", "stats"),) * (2 if change == "overlap" else 1)
    with pytest.raises((IRSchemaError, IRVerificationError, ValueError)):
        form_vectorized_add_norm_stats(old, pairs=pairs)


@pytest.mark.parametrize("variant,tile,shared,cooperative", [
    ("local16_xor_fma", 16, False, True), ("other", 16, False, True),
    ("local16_xor_fma", 32, False, True), ("local16_xor_fma", 16, True, True),
    ("local16_xor_fma", 16, False, False),
])
def test_candidate_selects_only_direct_local16_without_new_shared(variant, tile, shared, cooperative):
    from types import SimpleNamespace
    from triton.flagmega.codegen.triton.candidates.vectorized_add_norm_stats import VectorizedAddNormStatsCandidateProvider
    args, attrs = operands(True, True)
    p = fm.get_definition("ntt.vectorized_add_norm_stats").prepare(args, attrs)
    node = fm.Node("paired", "ntt.vectorized_add_norm_stats", tuple(n.id for n in args), p.result_type, p.effect, p.attrs)
    implementation = SimpleNamespace(id="test.local16", variant=variant, parameters={"tile": tile},
                                     shared_workspaces=("forbidden",) if shared else (), transfer_pipeline=None)
    context = SimpleNamespace(module=SimpleNamespace(node_map={n.id: n for n in args}),
        cooperative_grid=cooperative, implementations=lambda family, **kw: (implementation,),
        configure_implementation=lambda i, **kw: fm.Candidate(i.id, kw["semantic_parameters"], kw["facts"]),
        choose_default=lambda family, candidates: candidates[0].id)
    proposal = VectorizedAddNormStatsCandidateProvider().propose(node, context)
    if variant == "local16_xor_fma" and tile == 16 and not shared and cooperative:
        assert proposal.candidates[0].parameters["vector_schedule"]["physical"]["elements_per_program"] == 16
    else:
        assert proposal is None
