# Copyright 2026- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""gfx1201 s_prefetch_data: next-invocation QKV slab and split-K GEMV weight tiles."""

import copy

import pytest

from triton.flagmega.codegen.triton.amd_prefetch import (
    add_next_invocation_qkv_prefetch,
    add_split_k_weight_tile_prefetch,
)


def _layer(*, offsets=(0, 8, 16), families=None, column=1):
    families = families or ["gather_reduce_norm_apply", "qkv_parallel_linear", "dense_matmul",
                            "distributed_boxing", "add_norm_stats"]
    kinds = {"distributed_boxing": "collective"}
    events = []
    for index, family in enumerate(families):
        events.append({"call": f"c{index}", "family": family,
                       "execution_kind": kinds.get(family, "local_shard")})
        events.append({"kind": "barrier", "scope": "grid"})
    binding = {"formal": "qkv_w", "storage": "rdata", "byte_offset": 0}
    if column is not None:
        binding["record_column"] = column
    definition = {
        "function": "layer",
        "schedule": {"consumer_events": events},
        "function_pool_abi": {
            "bindings": [binding],
            "record_width": 8,
            "calls": [{"caller": "main", "call": f"x{i}", "record_offset": o} for i, o in enumerate(offsets)],
        },
    }
    qkv = [{"call": f"c{i}", "inputs": [{"formal": "q_weight_qkv_fused", "buffers": [{
        "runtime_argument": "qkv_w", "abi": {"scalar_storage_strides": (65536, 8, 1), "scalar_itemsize": 2},
    }]}]} for i, family in enumerate(families) if family == "qkv_parallel_linear"]
    return definition, {"layer": qkv}


def test_inserts_before_the_last_collective_on_gfx1201():
    definition, calls = _layer()
    assert add_next_invocation_qkv_prefetch([definition], calls, platform="amd", architecture="gfx1201")
    events = definition["schedule"]["consumer_events"]
    index = next(i for i, e in enumerate(events) if e.get("kind") == "prefetch")
    assert events[index + 1]["family"] == "distributed_boxing"
    event = events[index]
    assert event["guard"] == (
        "_flagmega_call_offsets.to(tl.int64, bitcast=True) != "
        "(_flagmega_function_offsets.to(tl.pointer_type(tl.int64)) + 16).to(tl.int64, bitcast=True)"
    )
    assert event["base"] == (
        "_flagmega_call_pool_rdata.to(tl.pointer_type(tl.uint8)) "
        "+ tl.load(_flagmega_call_offsets + 9, volatile=True)"
    )
    assert event["slab_bytes"] == 131072 and event["chunks"] == 32


@pytest.mark.parametrize("platform,architecture", [("amd", "gfx1100"), ("nvidia", "sm90")])
def test_other_targets_are_untouched(platform, architecture):
    definition, calls = _layer()
    before = copy.deepcopy(definition)
    assert not add_next_invocation_qkv_prefetch([definition], calls, platform=platform, architecture=architecture)
    assert definition == before


@pytest.mark.parametrize("kwargs", [
    {"families": ["gather_reduce_norm_apply", "dense_matmul", "distributed_boxing"]},
    {"families": ["qkv_parallel_linear", "qkv_parallel_linear", "dense_matmul", "distributed_boxing"]},
    {"families": ["qkv_parallel_linear", "add_norm_stats", "distributed_boxing"]},
    {"offsets": (0, 8, 24)},
    {"offsets": (0,)},
    {"column": None},
])
def test_unproven_shapes_are_left_alone(kwargs):
    definition, calls = _layer(**kwargs)
    before = copy.deepcopy(definition)
    assert not add_next_invocation_qkv_prefetch([definition], calls, platform="amd", architecture="gfx1201")
    assert definition == before


def test_rendered_prefetch_compiles_and_runs_on_gfx12(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not str(
            getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).startswith("gfx12"):
        pytest.skip("s_prefetch_data requires a gfx12 GPU")
    import importlib.util

    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

    definition, calls = _layer()
    assert add_next_invocation_qkv_prefetch([definition], calls, platform="amd", architecture="gfx1201")
    event = next(e for e in definition["schedule"]["consumer_events"] if e.get("kind") == "prefetch")
    template = TritonTemplateRegistry().environment.get_template("entrypoints/_amd_prefetch.py.jinja")
    source = (
        "import triton\nimport triton.language as tl\n" + template.render()
        + "\n@triton.jit\n"
        + "def run(_flagmega_function_offsets, _flagmega_call_offsets, _flagmega_call_pool_rdata, out):\n"
        + "    shard_index = tl.program_id(0).to(tl.int64)\n"
        + str(template.module.prefetch_event(event)) + "\n"
        + "    tl.store(out + shard_index, shard_index + 1)\n"
    )
    path = tmp_path / "amd_prefetch_kernel.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("amd_prefetch_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = torch.zeros(24, dtype=torch.int64, device="cuda")
    rdata = torch.zeros(2 * event["slab_bytes"] + 4096, dtype=torch.uint8, device="cuda")
    for record in (0, 16):  # a middle record prefetches; the last record must not
        out = torch.zeros(2, dtype=torch.int64, device="cuda")
        kernel = module.run[(2,)](table, table[record:], rdata, out)
        torch.cuda.synchronize()
        assert out.tolist() == [1, 2]
    assert "s_prefetch_data" in kernel.asm["amdgcn"]


# --- split-K GEMV weight-tile prefetch -------------------------------------

_MESH = {"mesh_hierarchy": (8, 16), "mesh_levels": ("b", "b")}
_GLOBAL_N = "((((shard_x) + ((((dense_local_n_offsets) // 8)) * 16))) * 8 + (((dense_local_n_offsets) % 8)))"
_WEIGHT = ("((dense_global_k[None, :]) // 16) * 32768 + ((dense_global_n[:, None]) // 8) * 128"
           " + (((dense_global_n[:, None]) % 8) * 16 + ((dense_global_k[None, :]) % 16))")
# Qwen3-1.7B decode on the 8x16 mesh: K split block-cyclically over y, N over x.
_DOWN_K = "(((shard_y) * 64) + ((((dense_local_k_offsets)) // (64)) * 512) + (((dense_local_k_offsets)) % (64)))"
_OPROJ_K = "(((shard_y) * 256) + ((((dense_local_k_offsets)) // (256)) * 2048) + (((dense_local_k_offsets)) % (256)))"


def _split_k_call(global_k=_DOWN_K, local_k=768, **overrides):
    call = {
        "family": "dense_matmul", "variant": "split_k_n_packed_k_major_gemv", "symbol": "run",
        "signature": "X, W, Y", "local_m_capacity": 1, "local_n_capacity": 128, "tile_n": 64,
        "global_n": _GLOBAL_N, "result_active": "True", "block_k": 64, "local_k_capacity": local_k,
        "global_k": global_k, "source_active": "True", "source": "X", "source_offset": "dense_global_k",
        "weight": "W", "weight_offset": _WEIGHT, "weight_k_contiguity": 16, "weight_itemsize": 2,
        "packed_layout": "k_major_n8_k16", "result": "Y",
        "result_offset": "shard_index * 128 + dense_local_n_offsets", "output_type": "tl.float32",
    }
    call.update(overrides)
    return call


@pytest.mark.parametrize("global_k,local_k,expected", [
    (_DOWN_K, 768, {"distance": 4, "tiles": 24, "k_tiles": 12, "k_tile_stride": 1048576,
                    "n_tile_stride": 16384, "chunks": 2,
                    "base": "131072 * shard_coord0 + (shard_coord1 // 4) * 32768 + (shard_coord1 % 4) * 4096"}),
    (_OPROJ_K, 256, {"distance": 2, "tiles": 8, "k_tiles": 4, "k_tile_stride": 131072,
                     "n_tile_stride": 16384, "chunks": 2,
                     "base": "524288 * shard_coord0 + (shard_coord1 // 4) * 32768 + (shard_coord1 % 4) * 4096"}),
])
def test_split_k_row_shares_its_tile_region(global_k, local_k, expected):
    # A row's 16 owners read 4 k16 groups x 32 KB per tile: 128 KB, 8 KB each.
    call = _split_k_call(global_k, local_k)
    assert add_split_k_weight_tile_prefetch([call], _MESH, platform="amd", architecture="gfx1201")
    assert call["weight_tile_prefetch"] == expected


@pytest.mark.parametrize("platform,architecture", [("amd", "gfx1100"), ("nvidia", "sm90")])
def test_split_k_prefetch_is_gfx1201_only(platform, architecture):
    call = _split_k_call()
    assert not add_split_k_weight_tile_prefetch([call], _MESH, platform=platform, architecture=architecture)
    assert "weight_tile_prefetch" not in call


@pytest.mark.parametrize("overrides", [
    {"weight_k_contiguity": 1},
    {"local_m_capacity": 2},
    {"weight_itemsize": None},
    {"local_n_capacity": 96},
    {"local_k_capacity": 64},  # one K tile: nothing to run ahead of
    {"variant": "split_k_packed_k_major_gemv"},
    # Every owner reads its own K slice: no row shares a tile.
    {"global_k": "shard_index * 768 + dense_local_k_offsets"},
    {"weight_offset": "dense_global_k[None, :] * 2048 + dense_global_n[:, None]", "packed_layout": None},
])
def test_split_k_prefetch_skips_unproven_calls(overrides):
    call = _split_k_call(**overrides)
    assert not add_split_k_weight_tile_prefetch([call], _MESH, platform="amd", architecture="gfx1201")
    assert "weight_tile_prefetch" not in call


def _load_source(tmp_path, name, source):
    import importlib.util

    path = tmp_path / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rendered_split_k_prefetch_runs_the_full_mesh_on_gfx12(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not str(
            getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).startswith("gfx12"):
        pytest.skip("s_prefetch_data requires a gfx12 GPU")
    from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry

    registry = TritonTemplateRegistry()
    spec = KernelTemplateSpec("dense_matmul", "split_k_n_packed_k_major_gemv", "amd", "gfx1201")
    helper = registry.environment.get_template("entrypoints/_amd_prefetch.py.jinja").render()
    # Launch the 8x16 mesh as a 2-D grid so every owner runs its own addressing.
    mesh_ids = (
        ("    shard_coord0 = tl.full((), 0, tl.int64)\n", "    shard_coord0 = tl.program_id(1).to(tl.int64)\n"),
        ("    shard_coord1 = tl.full((), 0, tl.int64)\n", "    shard_coord1 = tl.program_id(0).to(tl.int64)\n"),
        ("    shard_index = 0\n", "    shard_index = shard_coord0 * 16 + shard_coord1\n"),
    )
    modules = {}
    for name, prefetch in (("plain", False), ("prefetch", True)):
        call = _split_k_call()
        if prefetch:
            assert add_split_k_weight_tile_prefetch([call], _MESH, platform="amd", architecture="gfx1201")
        source = registry.render_kernel(
            spec, {"render_calls": [call], "distributed_entry": False, "mesh_hierarchy": (8, 16)}).source
        for old, new in mesh_ids:
            assert old in source
            source = source.replace(old, new)
        assert ("_flagmega_amd_prefetch_slab(" in source) == prefetch
        modules[name] = _load_source(tmp_path, f"split_k_{name}",
                                     "import triton\nimport triton.language as tl\n" + helper + "\n" + source)
    torch.manual_seed(7)
    n, k = 2048, 6144
    x = torch.randn(k, device="cuda").bfloat16()
    dense = torch.randn(n, k, device="cuda").bfloat16()
    packed = dense.reshape(n // 8, 8, k // 16, 16).permute(2, 0, 1, 3).contiguous()
    outputs = {}
    for name, module in modules.items():
        outputs[name] = torch.full((128 * 128,), float("nan"), device="cuda")
        kernel = module.run[(16, 8)](x, packed, outputs[name])
    torch.cuda.synchronize()
    assert "s_prefetch_data" in kernel.asm["amdgcn"]
    torch.testing.assert_close(outputs["prefetch"], outputs["plain"], rtol=0, atol=0)
    local = torch.arange(128, device="cuda")
    for y in range(8):
        for x_owner in range(16):
            rows = (x_owner + (local // 8) * 16) * 8 + local % 8
            cols = torch.cat([y * 64 + t * 512 + torch.arange(64, device="cuda") for t in range(12)])
            expected = (dense[rows][:, cols].float() * x[cols].float()).sum(-1)
            owner = y * 16 + x_owner
            torch.testing.assert_close(outputs["prefetch"][owner * 128:(owner + 1) * 128], expected,
                                       rtol=1e-4, atol=1e-3)
