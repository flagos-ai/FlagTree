# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
)
from triton.flagmega.targets.portable_triton_implementations import (
    portable_triton_implementation_model,
)


def test_hybrid_split_k_n_uses_a_mesh_parameterized_generic_template():
    implementation = portable_triton_implementation_model().implementation(
        "tir.dense_matmul.split_k_n_packed_k_major_gemv"
    )

    assert implementation is not None
    assert implementation.contract["distribution_kind"] == (
        "output_reduction_split"
    )
    assert implementation.facts["portable_triton"] is True
    registry = TritonTemplateRegistry()
    spec = KernelTemplateSpec(
        implementation.family,
        implementation.variant,
        "nvidia",
        "sm90",
    )
    assert registry.resolve(spec) == (
        "kernels/dense_matmul/split_k_n_packed_k_major_gemv.py.jinja"
    )
    source = registry.render_kernel(spec, {}).source
    compile(source, "split_k_n_packed_k_major_gemv.py", "exec")

    assert (
        "def _flagmega_dense_matmul_split_k_n_packed_k_major_gemv_accumulate("
        in source
    )
    assert "source_offsets" in source
    assert "weight_offsets" in source
    assert "mesh_size" not in source
    assert "mesh_hierarchy" not in source
    assert "sm90" not in source.lower()


def _split_k_n_call(weight_k_contiguity):
    return {
        "family": "dense_matmul", "variant": "split_k_n_packed_k_major_gemv",
        "symbol": "run", "signature": "X, W, Y", "local_m_capacity": 1,
        "local_n_capacity": 9496, "tile_n": 64, "global_n": "dense_local_n_offsets",
        "result_active": "(dense_local_n_offsets // 8) < 1187", "block_k": 64,
        "local_k_capacity": 256, "global_k": "dense_local_k_offsets",
        "source_active": "True", "source": "X", "source_offset": "dense_local_k_offsets",
        "weight": "W", "weight_offset": "dense_global_k[None, :] * 9496 + dense_global_n[:, None]",
        "weight_k_contiguity": weight_k_contiguity, "result": "Y",
        "result_offset": "dense_local_n_offsets", "output_type": "tl.float32",
    }


def _render(spec, call):
    context = {"render_calls": [call], "distributed_entry": False, "mesh_hierarchy": (1, 1)}
    return TritonTemplateRegistry().render_kernel(spec, context).source


def test_amd_split_k_n_asserts_only_a_proven_k_run():
    generic = KernelTemplateSpec("dense_matmul", "split_k_n_packed_k_major_gemv")
    amd = KernelTemplateSpec("dense_matmul", "split_k_n_packed_k_major_gemv", "amd", "gfx1201")
    assert TritonTemplateRegistry().resolve(amd) == (
        "kernels/dense_matmul/platforms/amd/split_k_n_packed_k_major_gemv.py.jinja"
    )
    hinted = _render(amd, _split_k_n_call(16))
    compile(hinted, "amd_split_k_n.py", "exec")
    assert "tl.max_contiguous(tl.multiple_of(" in hinted and "[1, 16]" in hinted
    # Other platforms never see the hint, and AMD drops it when unproven.
    assert "max_contiguous" not in _render(generic, _split_k_n_call(16))
    unproven = _render(amd, _split_k_n_call(1))
    assert "max_contiguous" not in unproven
    assert unproven.splitlines()[1:] == _render(generic, _split_k_n_call(1)).splitlines()[1:]


def test_weight_tile_prefetch_renders_only_in_the_amd_template():
    plan = {"distance": 2, "tiles": 4, "k_tiles": 2, "base": "shard_coord1 * 4096",
            "k_tile_stride": 4864, "n_tile_stride": 128, "chunks": 1}
    call = {**_split_k_n_call(16), "local_n_capacity": 128, "local_k_capacity": 128,
            "result_active": "True", "weight_tile_prefetch": plan}
    generic = KernelTemplateSpec("dense_matmul", "split_k_n_packed_k_major_gemv")
    amd = KernelTemplateSpec("dense_matmul", "split_k_n_packed_k_major_gemv", "amd", "gfx1201")
    planned = _render(amd, call)
    compile(planned, "amd_split_k_n_prefetch.py", "exec")
    # Prologue for the first `distance` tiles, then `distance` ahead in the loop.
    assert "_flagmega_amd_prefetch_slab(W + (shard_coord1 * 4096 + 0), 1)" in planned
    assert "_flagmega_amd_prefetch_slab(W + (shard_coord1 * 4096 + 4864), 1)" in planned
    assert "dense_prefetch_tile = (dense_local_n_start // 64) * 2 + dense_local_k_start // 64 + 2" in planned
    assert "if dense_prefetch_tile < 4:" in planned
    unplanned = {key: value for key, value in call.items() if key != "weight_tile_prefetch"}
    assert "prefetch" not in _render(amd, unplanned)
    assert _render(generic, call) == _render(generic, unplanned)
