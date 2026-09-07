# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Ownership contracts between NTT policies and concrete target machines."""

from pathlib import Path
import inspect

from triton.flagmega.codegen.triton.lowering import TritonTirLoweringPolicy
from triton.flagmega.targets.base import Target
from triton.flagmega.targets.ntt import NttTarget
from triton.flagmega.targets.pyntt import PyNttTarget


FLAGMEGA = Path(__file__).parents[3] / "triton" / "flagmega"


def _python_sources(root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    )


def test_ntt_rules_do_not_depend_on_a_vendor_or_architecture():
    source = _python_sources(FLAGMEGA / "rules" / "ntt")

    assert "targets.nvidia" not in source
    assert "NvidiaSm90" not in source
    assert "sm90" not in source.lower()
    # AutoPacking describes tensor layouts.  Instructions and memory-transfer
    # mechanisms are selected later by the target's TIR implementation model.
    for spelling in ("mma", "tma", "smem", "warp_specialize"):
        assert spelling not in source.lower()
    for spelling in (
        "allowed_kinds",
        "contiguous_last_axis_only",
        "current_triton",
    ):
        assert spelling not in source


def test_nvidia_target_has_no_graph_policy_modules():
    target_root = FLAGMEGA / "targets" / "nvidia"

    assert not (target_root / "packing.py").exists()
    assert not (target_root / "selection.py").exists()
    assert not (target_root / "distribution.py").exists()
    source = _python_sources(target_root)
    for spelling in (
        "NvidiaSm90PackingPolicy",
        "NvidiaSm90SelectionPolicy",
        "NvidiaSm90DistributedSplitCandidateProvider",
        "target_asset_abi",
    ):
        assert spelling not in source


def test_generic_passes_do_not_depend_on_the_sm90_target():
    source = _python_sources(FLAGMEGA / "passes")

    assert "targets.nvidia" not in source
    assert "NvidiaSm90" not in source


def test_target_uses_a_generic_post_auto_packing_registration_hook():
    protocol_source = inspect.getsource(Target)
    ntt_source = inspect.getsource(NttTarget)
    pyntt_source = inspect.getsource(PyNttTarget)

    assert "register_post_auto_packing_passes" in protocol_source
    assert "register_post_auto_packing_passes" in ntt_source
    assert "register_post_auto_packing_passes" in pyntt_source
    assert "def decompose_paged_attention" not in protocol_source
    assert "def decompose_paged_attention" not in ntt_source
    assert "def decompose_paged_attention" not in pyntt_source


def test_egraph_infrastructure_does_not_depend_on_a_vendor_or_architecture():
    source = _python_sources(FLAGMEGA / "egraph")

    assert "targets.nvidia" not in source
    assert "NvidiaSm90" not in source
    assert "sm90" not in source.lower()


def test_generic_triton_candidate_providers_use_capabilities_not_architecture_names():
    source = _python_sources(FLAGMEGA / "codegen" / "triton" / "candidates")

    assert "targets.nvidia" not in source
    assert "NvidiaSm90" not in source
    assert "sm90" not in source.lower()
    assert "configure_candidate" not in source
    assert '"tir.' not in source
    # Tile, pipeline, and resource geometry belongs to the injected concrete
    # implementation model. Generic providers may inspect resolved values but
    # must not define a target's numerical kernel configuration.
    for spelling in (
        '"block_k": 128',
        '"block_k": 512',
        '"block_k": 1024',
        '"tile_n": 16',
        '"stages": 2',
        '"producer_registers": 24',
        '"vocab_tile": 1024',
        "n8_k16",
        "tn16",
        "tn32",
        "tn64",
        "bk512",
        "bn32",
        "bn64",
        "bn128",
        "mma",
        "tma",
        "smem",
        "warp_specialize",
        "producer_registers",
        "kv_tile_bytes",
        "aliased_kv_stage_bytes",
    ):
        assert spelling not in source


def test_generic_tir_lowering_delegates_launch_geometry_to_the_target():
    source = inspect.getsource(TritonTirLoweringPolicy)

    assert "target.plan_launch" in source
    assert "entry_num_warps" not in source
    assert "uses_warp_specialized_pipeline" not in source


def test_generic_tir_lowering_delegates_package_resources_to_the_target():
    source = inspect.getsource(TritonTirLoweringPolicy)

    assert "target.plan_codegen_package" in source
    assert "shared_memory" not in source
    assert "producer_registers" not in source


def test_codegen_package_composition_is_model_independent():
    renderer = (
        FLAGMEGA / "codegen" / "triton" / "tir_package.py"
    ).read_text(encoding="utf-8")
    target_plan = (
        FLAGMEGA / "targets" / "nvidia" / "package.py"
    ).read_text(encoding="utf-8")

    assert 'require_package_plan(module, "tir_call_graph")' in renderer
    assert '"kind": "tir_call_graph"' in target_plan
    for source in (renderer, target_plan):
        assert 'metadata.get("architecture")' not in source
        assert "output_boundary" not in source
        assert "role_ids" not in source
        assert "Qwen3ForCausalLM" not in source
        assert "Qwen3_5ForConditionalGeneration" not in source


def test_codegen_has_no_model_named_package_renderer_or_entrypoint():
    codegen = FLAGMEGA / "codegen" / "triton"

    assert not (codegen / "qwen3.py").exists()
    assert not (codegen / "qwen3_model.py").exists()
    assert not (codegen / "qwen3_5.py").exists()
    assert tuple((codegen / "entrypoints").glob("*qwen*")) == ()


def test_model_neutral_compiler_layers_have_no_model_identity():
    roots = tuple(
        FLAGMEGA / name
        for name in ("passes", "rules", "targets", "codegen", "runtime")
    )

    sources = tuple(
        path
        for root in roots
        for path in root.rglob("*")
        if path.suffix in {".py", ".jinja"}
    )
    assert not any("qwen" in path.name.lower() for path in sources)
    assert not any("qwen" in path.read_text(encoding="utf-8").lower() for path in sources)
