# Copyright 2026- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""gfx1201 next-invocation QKV prefetch insertion."""

import copy

import pytest

from triton.flagmega.codegen.triton.amd_prefetch import add_next_invocation_qkv_prefetch


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
