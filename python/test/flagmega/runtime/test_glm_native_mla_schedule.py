# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The native MLA profile includes both outputs of fused Q/KV_A GEMM."""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from triton.flagmega import ir as fm


@dataclass(frozen=True)
class Dispatch:
    microkernel: object
    semantic_attrs: object


@dataclass(frozen=True)
class Definition:
    name: str
    dispatch: Dispatch


def test_native_q_a_uses_fused_projection_partitions_and_q_b_keeps_running_accumulator(monkeypatch):
    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    spec = importlib.util.spec_from_file_location("glm_native_mla_schedule", tutorial / "local_optimizations/native_tp4.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    prefix = "decode_mla_attention_sparse_self_attn_"
    names = [
        *(prefix + f"o_proj_weight_apply.partition{p}.dot" for p in range(4)),
        prefix + "kv_a_proj_with_mqa_weight_apply.packed",
        prefix + "q_a_proj_weight_apply.packed",
        prefix + "q_b_proj_weight_apply.packed",
    ]
    definitions = tuple(
        Definition(
            f"projection_{i}",
            Dispatch(
                fm.TIRMicroKernelSelection(
                    f"test_{i}", "block_fp8", "mma", {"dequant_mma_splits": 1, "compute_num_warps": 8}
                ),
                {"activation_quantization": "none"},
            ),
        )
        for i in range(len(names))
    )
    source = SimpleNamespace(
        nodes=tuple(
            SimpleNamespace(id=name, op="tir.call", attrs={"callee": definition.name})
            for name, definition in zip(names, definitions, strict=True)
        ),
        kernel_definitions=definitions,
    )
    result = profile.rewrite_mla_schedules(source)
    actual = {
        name: definition.dispatch.microkernel.parameters["dequant_mma_splits"]
        for name, definition in zip(names, result, strict=True)
    }
    assert actual[prefix + "q_a_proj_weight_apply.packed"] == 2
    assert actual[prefix + "q_b_proj_weight_apply.packed"] == 1
    assert all(actual[name] == 2 for name in names[:-1])
    assert all(value.dispatch.microkernel.parameters["compute_num_warps"] == 8 for value in result)
    assert all(value.dispatch.microkernel.parameters["dequant_mma_splits"] == 1 for value in definitions)


def test_native_scaled_promotion_preserves_dequantized_and_expert_schedules(monkeypatch):
    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    spec = importlib.util.spec_from_file_location("glm_native_scaled_promotion", tutorial / "local_optimizations/native_tp4.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    kinds = (
        ("block_fp8", "mma", "dynamic_fp8_sm90"),
        ("matmul_glu", "mma", "dynamic_fp8_sm90"),
        ("block_fp8", "mma", "none"),
        ("sparse_experts_gate_up", "wgmma", "dynamic_fp8_ue8m0"),
    )
    definitions = tuple(
        Definition(
            f"kernel_{i}",
            Dispatch(
                fm.TIRMicroKernelSelection(f"test_{i}", family, variant, {"dequant_mma_splits": 2}),
                {"activation_quantization": quantization},
            ),
        )
        for i, (family, variant, quantization) in enumerate(kinds)
    )
    result = profile.rewrite_fp8_scaled_accumulation(SimpleNamespace(kernel_definitions=definitions))
    assert all(value.dispatch.microkernel.parameters["scaled_accumulation"] == "fma" for value in result[:2])
    assert result[2:] == definitions[2:]
    assert all("scaled_accumulation" not in value.dispatch.microkernel.parameters for value in definitions)
    catalog = profile.native_implementation_model()
    selected = [value for value in catalog.implementations if value.id in {"tir.block_fp8.mma", "tir.matmul_glu.mma"}]
    assert len(selected) == 2
    assert all(value.parameters["scaled_accumulation"] == "fma" for value in selected)


def test_native_attention_uses_continuous_bf16_mma(monkeypatch):
    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    spec = importlib.util.spec_from_file_location("glm_native_attention_schedule", tutorial / "local_optimizations/native_tp4.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    definitions = tuple(
        Definition(
            f"kernel_{i}",
            Dispatch(
                fm.TIRMicroKernelSelection(f"test_{i}", family, "decode", {}), {"probability_rounding": "online_bf16"}
            ),
        )
        for i, family in enumerate(("dsa_paged_attention", "dsa_indexer"))
    )
    result = profile.rewrite_attention_schedules(SimpleNamespace(kernel_definitions=definitions))
    assert result[0].dispatch.microkernel.parameters == {"decode_schedule": "bf16_mma", "compute_num_warps": 8}
    assert result[0].dispatch.semantic_attrs["index_traversal"] == "compact"
    assert result[1] == definitions[1]
    assert definitions[0].dispatch.microkernel.parameters == {}


def test_native_qkv_statistics_keep_contiguous_lanes_and_one_global_tree(monkeypatch):
    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    spec = importlib.util.spec_from_file_location("glm_native_qkv_norm", tutorial / "local_optimizations/native_tp4.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    definitions = tuple(
        Definition(name, Dispatch(fm.TIRMicroKernelSelection(name, family, variant, {"block_size": 128}), {}))
        for name, family, variant in (
            ("bf16_stats", "norm_stats", "local"),
            ("fp32_stats", "norm_stats", "local"),
            ("bf16_finalize", "gather_reduce_norm_apply", "sum"),
            ("fp32_finalize", "gather_reduce_norm_apply", "sum"),
        )
    )

    def call(name, callee, source):
        return SimpleNamespace(id=name, op="tir.call", attrs={"callee": callee}, inputs=(source,))

    nodes = (
        SimpleNamespace(id="bf16", op="builtin.var", type=fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 64))),
        SimpleNamespace(id="fp32", op="builtin.var", type=fm.DistributedType(
            fm.tensor_type("float32", (1, 4096)),
            (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 2), 64)),
            fm.Placement((4, 8, 16), "cyx", "cbb"),
        )),
        call("bstat", "bf16_stats", "bf16"),
        call("fstat", "fp32_stats", "fp32"),
        call("bfinal", "bf16_finalize", "bstat"),
        call("ffinal", "fp32_finalize", "fstat"),
        call("function_call", "decode_linear_attention_dense", "bf16"),
    )
    result = profile.rewrite_qkv_norm_schedules(
        SimpleNamespace(nodes=nodes, node_map={node.id: node for node in nodes}, kernel_definitions=definitions)
    )
    assert result[0].dispatch.microkernel.parameters == {"block_size": 2048, "reduction_layout": "contiguous_8"}
    assert result[2].dispatch.microkernel.parameters == {"block_size": 128, "partial_reduction": "global_tree"}
    assert result[1] == definitions[1] and result[3] == definitions[3]
    assert all(value.dispatch.microkernel.parameters == {"block_size": 128} for value in definitions)
    assert "index_traversal" not in definitions[0].dispatch.semantic_attrs
    selected = [
        v for v in profile.native_implementation_model().implementations if v.id == "tir.dsa_paged_attention.decode"
    ]
    assert len(selected) == 1
    assert selected[0].parameters["decode_schedule"] == "bf16_mma"


@pytest.mark.parametrize("packed,split_outer,distributed", (
    (False, False, False), (False, False, True), (True, False, True), (True, True, True),
))
def test_native_complete_fp32_statistics_select_the_official_reduction(monkeypatch, packed, split_outer,
                                                                     distributed):
    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    spec = importlib.util.spec_from_file_location("glm_native_complete_rms", tutorial / "local_optimizations/native_tp4.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    mesh = fm.Placement((4, 8, 16), "cyx", "cbb")
    dtype = fm.vector_type("float32", 4) if packed else fm.DType.FLOAT32
    logical = fm.tensor_type(dtype, (8 if split_outer else 1, 1024 if packed else 4096))
    source_type = fm.DistributedType(logical, (
        fm.SBP.split_contiguous((1,), 1) if split_outer else fm.SBP.broadcast(), fm.SBP.broadcast(),
    ), mesh) if distributed else logical
    definition = Definition("statistics", Dispatch(
        fm.TIRMicroKernelSelection("statistics", "norm_stats", "local", {"block_size": 128}),
        {"axis": -1, "use_mean": False},
    ))
    source = SimpleNamespace(id="source", op="builtin.var", type=source_type)
    call = SimpleNamespace(id="stats", op="tir.call", attrs={"callee": definition.name}, inputs=(source.id,))
    module = SimpleNamespace(nodes=(source, call), node_map={source.id: source}, kernel_definitions=(definition,))
    selected, = profile.rewrite_qkv_norm_schedules(module)
    assert selected.dispatch.microkernel.parameters.get("reduction_algorithm", "tree") == "fma_1024"
    assert "reduction_algorithm" not in definition.dispatch.microkernel.parameters


@pytest.mark.parametrize("kind", ("split_reduction", "partial", "exclusive", "mean"))
def test_native_ordered_fp32_statistics_require_complete_rms_input(monkeypatch, kind):
    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    spec = importlib.util.spec_from_file_location("glm_native_incomplete_rms", tutorial / "local_optimizations/native_tp4.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    mesh = fm.Placement((4, 8, 16), "cyx", "cbb")
    source_type = fm.DistributedType(
        fm.tensor_type("float32", (1, 4096)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 2), 64)
         if kind == "split_reduction" else fm.SBP.broadcast()), mesh,
        partial=fm.SBP.partial((0,)) if kind == "partial" else None,
        exclusive=fm.SBP.exclusive((1, 2)) if kind == "exclusive" else None,
    )
    definition = Definition("statistics", Dispatch(
        fm.TIRMicroKernelSelection("statistics", "norm_stats", "local", {"block_size": 128}),
        {"axis": -1, "use_mean": kind == "mean"},
    ))
    source = SimpleNamespace(id="source", op="builtin.var", type=source_type)
    call = SimpleNamespace(id="stats", op="tir.call", attrs={"callee": definition.name}, inputs=(source.id,))
    selected, = profile.rewrite_qkv_norm_schedules(SimpleNamespace(
        nodes=(source, call), node_map={source.id: source}, kernel_definitions=(definition,),
    ))
    assert selected is definition


def test_native_kda_profile_keeps_raw_gate_parameters_and_selects_warp_k(monkeypatch):
    from python.test.flagmega.importer.glm5_moe.test_kda_fused_gate import _imported

    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    spec = importlib.util.spec_from_file_location("glm_native_gate_inputs", tutorial / "local_optimizations/native_profile.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    _, _, original, _ = _imported()
    transformed = profile._keep_raw_kda_decay(original)
    assert transformed.metadata["native_kda_gate"]["source_maps"] > 0
    assert not any(n.op == "math.exp" for n in transformed.nodes)
    assert all(n.attrs["decay_input"] == "log_rate" for n in transformed.nodes if n.op == "nn.kda_gated_recurrent_core")
    assert all("decay_input" not in n.attrs for n in original.nodes if n.op == "nn.kda_gated_recurrent_core")
    spec = importlib.util.spec_from_file_location("glm_native_gate_schedule", tutorial / "local_optimizations/native_tp4.py")
    schedule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schedule)
    selected = [
        v for v in schedule.native_implementation_model().implementations if v.family == "kda_gated_recurrent_core"
    ]
    assert selected and all(v.parameters["reduction_layout"] == "warp_k" for v in selected)


def test_native_gated_norm_uses_sqrt_div_without_changing_plain_norm(monkeypatch):
    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    spec = importlib.util.spec_from_file_location("glm_native_gated_norm_schedule", tutorial / "local_optimizations/native_tp4.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    definitions = tuple(
        Definition(
            f"kernel_{i}",
            Dispatch(fm.TIRMicroKernelSelection(f"test_{i}", family, "local", {}), {}),
        )
        for i, family in enumerate(("gated_norm_apply", "norm_apply", "norm_stats"))
    )
    result = profile.rewrite_gated_norm_schedules(SimpleNamespace(kernel_definitions=definitions))
    assert result[0].dispatch.microkernel.parameters == {"inverse_std_algorithm": "sqrt_div"}
    assert result[1:] == definitions[1:]
    assert all(value.dispatch.microkernel.parameters == {} for value in definitions)
    selected = [
        value for value in profile.native_implementation_model().implementations if value.family == "gated_norm_apply"
    ]
    assert selected and all(value.parameters["inverse_std_algorithm"] == "sqrt_div" for value in selected)
