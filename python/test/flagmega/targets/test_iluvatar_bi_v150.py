from pathlib import Path
import runpy

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry
from triton.flagmega.targets import get_target, target_names
from triton.flagmega.targets.iluvatar_bi_v150 import IluvatarBiV150Capability
from triton.flagmega.runtime.nvidia_launch import compilation_options
from triton.flagmega.runtime.module import _resource_contract
from triton.flagmega.runtime.prepared import ResourceContract, _validate_resources
from triton.experimental.tle.language.distributed import device_mesh, _apply_mesh_grid_launch


def test_bi_v150_target_is_registered_with_corex_template_identity():
    target = get_target("iluvatar-bi-v150")
    assert "iluvatar-bi-v150" in target_names()
    assert target.codegen_platform == "corex"
    assert target.codegen_architecture == "bi_v150"
    assert target.options.placements[0].hierarchy == (4, 4)
    assert target.capability.compute_capability == (7, 1)
    assert target.capability.warp_size == 64
    assert TritonTemplateRegistry().resolve(
        KernelTemplateSpec("elementwise", "add", "corex", "bi_v150")
    ) == "kernels/elementwise/add.py.jinja"


def test_bi_v150_capability_round_trip_and_feature_filtering():
    capability = IluvatarBiV150Capability()
    assert IluvatarBiV150Capability.from_data(capability.to_data()) == capability
    assert capability.supports(("async_copy", "warp_specialize"))
    assert capability.missing(("mma_v3",)) == ("mma_v3",)


def test_bi_v150_uses_atomic_barrier_launch_contract():
    options = compilation_options(
        ResourceContract(compute_num_warps=4, resident_blocks_per_sm=1, warp_size=64),
        "iluvatar-bi-v150",
    )
    assert options["num_warps"] == 4
    assert "launch_cooperative_grid" not in options


def test_corex_mesh_barrier_does_not_request_unsupported_cooperative_launch():
    class Options:
        num_ctas = 1
        cluster_dims = (1, 1, 1)
        launch_cooperative_grid = False
        backend_name = "corex"

    class Semantic:
        builder = type("Builder", (), {"options": Options()})()

    mesh = device_mesh({"block": [("block_y", 4), ("block_x", 4)]})
    _apply_mesh_grid_launch(mesh, Semantic())
    assert Semantic.builder.options.launch_cooperative_grid is False


def test_bi_v150_exposes_raw_pointer_async_qkv_pipeline_without_selecting_it():
    from triton.flagmega.targets.iluvatar_bi_v150 import corex_triton_implementation_model

    model = corex_triton_implementation_model()
    candidate = "tir.qkv_parallel_linear.packed_gemv_async_smem_pipeline"
    implementations = {value.id: value for value in model.implementations}
    assert candidate in implementations
    assert implementations[candidate].parameters["descriptor_kind"] == "pointer"
    assert implementations[candidate].parameters["inter_stage_buffer_slots"] == 2
    assert implementations[candidate].requires == ("async_copy", "warp_specialize")
    assert model.preferences["qkv_parallel_linear"][0] == "tir.qkv_parallel_linear.packed_fused_gemv_tn256"


def test_bi_v150_qkv_pipeline_stage_depth_updates_all_storage_contracts(monkeypatch):
    from triton.flagmega.targets.iluvatar_bi_v150 import corex_triton_implementation_model

    candidate = "tir.qkv_parallel_linear.packed_gemv_async_smem_pipeline"
    for stages in (2, 4):
        monkeypatch.setenv("FLAGMEGA_QKV_PIPELINE_STAGES", str(stages))
        model = corex_triton_implementation_model()
        implementation = model.implementation(candidate)
        assert implementation.parameters["num_stages"] == stages
        assert implementation.shared_workspaces[0].type.shape[0].fixed_value == stages
        assert implementation.transfer_pipeline.capacity == stages

    monkeypatch.setenv("FLAGMEGA_QKV_PIPELINE_STAGES", "3")
    with pytest.raises(ValueError, match="power-of-two stage extent"):
        corex_triton_implementation_model()


def test_bi_v150_qkv_pipeline_block_n_updates_copy_geometry_and_shared_abi(monkeypatch):
    from triton.flagmega.targets.iluvatar_bi_v150 import corex_triton_implementation_model

    candidate = "tir.qkv_parallel_linear.packed_gemv_async_smem_pipeline"
    for block_n, shared_width in ((32, 512), (64, 1024)):
        monkeypatch.setenv("FLAGMEGA_QKV_BLOCK_N", str(block_n))
        model = corex_triton_implementation_model()
        implementation = model.implementation(candidate)
        assert implementation.parameters["block_n"] == block_n
        assert implementation.shared_workspaces[0].type.shape[2].fixed_value == shared_width

    monkeypatch.setenv("FLAGMEGA_QKV_BLOCK_N", "48")
    with pytest.raises(ValueError, match="FLAGMEGA_QKV_BLOCK_N must be 16, 32, or 64"):
        corex_triton_implementation_model()


def test_bi_v150_tutorial_target_selects_async_qkv_pipeline():
    target_path = (
        Path(__file__).resolve().parents[4]
        / "python/tutorials/flagmega/01-qwen3-1.7b-bf16/"
        "iluvator-bi-v150/local_optimizations/target.py"
    )
    namespace = runpy.run_path(str(target_path))
    target = namespace["create_target"]()
    candidate = "tir.qkv_parallel_linear.packed_gemv_async_smem_pipeline"
    assert target.triton_implementation_model.preferences["qkv_parallel_linear"][0] == candidate
    implementation = target.triton_implementation_model.implementation(candidate)
    assert implementation.parameters["inter_stage_buffer_slots"] == 2
    assert implementation.parameters["producer_warps"] == 16
    assert implementation.parameters["producer_registers"] == 24
    assert implementation.parameters["consumer_warps"] == 8
    assert implementation.parameters["compute_num_warps"] == 32


def test_bi_v150_resource_contract_accepts_verified_256_register_kernel():
    contract = _resource_contract({"num_warps": 12}, "iluvatar-bi-v150")
    assert contract.registers_per_thread_limit == 256

    class Compiled:
        n_regs = 256
        n_spills = 22
        run = lambda self: None
        metadata = type("Metadata", (), {"num_warps": 12, "shared": 14060})()

    _validate_resources(Compiled(), contract, (1,))
