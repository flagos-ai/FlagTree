# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Complete shared statistics elect computation without narrowing barriers."""

import ast
from copy import deepcopy
from itertools import product
from math import prod

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _canonical_writer_active,
    _exclusive_participation_active,
    _norm_stats_call,
    prepare_kernel_calls,
)
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.codegen.triton.tir_package import _kernel_source_event, _mesh_context
from triton.flagmega.errors import CodegenError


def _abi(shape, hierarchy, levels, lanes=1):
    return {
        "storage": "workspace",
        "storage_kind": "canonical_global",
        "memory_sharing_scope": "chip",
        "coordinate_space": "canonical_global",
        "pool_byte_offset": 0,
        "scalar_dtype": "float32",
        "scalar_itemsize": 4,
        "logical_shape": shape,
        "local_capacity_shape": shape,
        "active_shape_expressions": tuple(map(str, shape)),
        "logical_coordinate_expressions": tuple(f"local_coord_{i}" for i in range(len(shape))),
        "scalar_storage_strides": tuple(prod(shape[i + 1:]) * lanes for i in range(len(shape))),
        "scalar_lane_count": lanes,
        "scalar_lane_shape": (lanes, ) if lanes != 1 else (),
        "component_stride_scalar_elements": 0,
        "distributed_type": {
            "placement": {
                "hierarchy": hierarchy, "hierarchy_levels": levels, "name":
                "".join(chr(97 + i) for i in range(len(hierarchy)))
            },
            "axis_policies": [{"kind": "broadcast"} for _ in shape],
            "partial": None,
            "exclusive": None,
        },
    }


def _raw(shape=(2, 8), axis=1, mean=False, lanes=1, hierarchy=(2, 3), levels="bb"):
    result = (2 if mean else 1, *shape[:axis], *(1 for _ in shape[axis:]))
    bindings = []
    for formal, geometry, width in [("input", shape, lanes), ("result", result, 1)]:
        bindings.append({
            "formal":
            formal, "buffers": [{
                "formal": formal,
                "actual": formal,
                "runtime_argument": formal,
                "abi": _abi(geometry, hierarchy, levels, width),
            }]
        })
    return {
        "call": "stats",
        "family": "norm_stats",
        "variant": "local",
        "implementation": "tir.norm_stats.local",
        "execution_kind": "local_shard",
        "semantic_op": "nn.norm_stats",
        "semantic_attrs": {"axis": axis, "use_mean": mean},
        "parameters": {"block_size": 128},
        "facts": {"local_shard_reduction": True},
        "inputs": [bindings[0]],
        "outputs": [bindings[1]],
        "workspaces": [],
        "shared_workspaces": [],
    }


def _binding(raw, section):
    return raw[section][0]["buffers"][0]["abi"]


def _prepare(raw):
    return prepare_kernel_calls([raw], function_name="main")[0]


def _owners(predicate, hierarchy):
    owners = []
    for coordinates in product(*(range(extent) for extent in hierarchy)):
        names = {f"shard_coord{i}": value for i, value in enumerate(coordinates)}
        if len(hierarchy) == 2:
            names.update(shard_y=coordinates[0], shard_x=coordinates[1])
        if eval(predicate, {"__builtins__": {}}, names):
            owners.append(coordinates)
    return owners


@pytest.mark.parametrize("shape,axis,mean,lanes", [
    ((8, ), 0, False, 1),
    ((1, ), 0, True, 1),
    ((2, 8), 1, False, 4),
    ((2, 3, 5), 1, True, 1),
])
@pytest.mark.parametrize("hierarchy", [(3, ), (2, 3), (2, 2, 3), (1, 3, 1)])
def test_complete_shared_norm_executes_on_its_existing_writer(shape, axis, mean, lanes, hierarchy):
    raw = _raw(shape, axis, mean, lanes, hierarchy, "b" * len(hierarchy))
    call = _prepare(raw)
    assert call["participation_active"] == _canonical_writer_active(_binding(raw, "outputs"))
    assert _owners(call["participation_active"], hierarchy) == [(0, ) * len(hierarchy)]
    leaf = _norm_stats_call(raw)
    assert all(call[key] == value for key, value in leaf.items())


@pytest.mark.parametrize("order", ["tile_fma", "group_fma"])
def test_declared_ordered_tree_keeps_the_same_leaf_on_the_elected_writer(order):
    raw = _raw()
    raw["semantic_attrs"].update({
        "reduction_group_size": 2,
        "reduction_order": order,
        "reduction_block_size": 8,
        "reduction_group_seed_index": 0,
    })
    call = _prepare(raw)
    assert call["participation_active"] == call["writer_active"]
    leaf = _norm_stats_call(raw)
    assert leaf["ordered_tree"] and all(call[key] == value for key, value in leaf.items())


@pytest.mark.parametrize("levels", ["cbb", "bdb", "cdb"])
def test_every_chip_node_instance_keeps_one_writer_and_E_domain(levels):
    raw = _raw(hierarchy=(2, 3, 2), levels=levels)
    exclusive = {
        "axes": tuple(i for i, level in enumerate(levels) if level != "b"), "owner_coordinates":
        tuple(1 for level in levels if level != "b")
    }
    for section in ("inputs", "outputs"):
        _binding(raw, section)["distributed_type"]["exclusive"] = deepcopy(exclusive)
    call = _prepare(raw)
    owners = _owners(call["participation_active"], (2, 3, 2))
    assert owners == [tuple(1 if level != "b" else 0 for level in levels)]
    unrestricted = _raw(hierarchy=(2, 3, 2), levels=levels)
    all_owners = _owners(_prepare(unrestricted)["participation_active"], (2, 3, 2))
    assert len(all_owners) == prod(extent for extent, level in zip((2, 3, 2), levels) if level != "b")


def test_real_package_guard_keeps_arguments_leaf_and_external_grid_barriers():
    raw = _raw()
    call = _prepare(raw)
    event = _kernel_source_event(call, {}, {}, [], (), memory_dependencies=({"scope": "grid"}, ))
    assert event["participation_active"] == call["writer_active"]
    assert event["arguments"] == call["signature"]
    assert event["barrier_before"] and event["memory_dependencies"]
    next_call = {**event, "call": "consumer", "symbol": "consume", "participation_active": "True"}
    registry = TritonTemplateRegistry()
    mesh = _mesh_context({"hierarchy": (2, 3), "name": "yx", "hierarchy_levels": "bb"})
    source = registry.render(
        "entrypoints/call_graph.py.jinja", {
            **mesh,
            "symbol": "run",
            "signature": call["signature"],
            "distributed_entry": True,
            "entry_events": [event, next_call],
        })
    tree = ast.parse(source)
    entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    guarded = next(node for node in entry.body if isinstance(node, ast.If))
    assert len(guarded.body) == 1 and isinstance(guarded.body[0], ast.Expr)
    assert guarded.body[0].value.func.id == call["symbol"]
    barriers = [
        node for node in entry.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "distributed_barrier"
    ]
    assert len(barriers) == 2 and barriers[0].lineno < guarded.lineno < barriers[1].lineno
    leaf_context = {**mesh, "distributed_entry": True, "num_warps": 4, "render_calls": [call]}
    leaf = registry.render("kernels/norm_stats/_function.py.jinja", leaf_context)
    leaf_context["render_calls"] = [{**call, "participation_active": "True"}]
    assert registry.render("kernels/norm_stats/_function.py.jinja", leaf_context) == leaf


@pytest.mark.parametrize("mutation", [
    "input_private",
    "result_private",
    "compact_input",
    "compact_result",
    "split",
    "partial",
    "scratch",
    "shared_scratch",
    "pipeline",
    "side_effect",
    "internal_barrier",
    "collective",
    "other_semantic_op",
    "other_variant",
    "different_placement",
    "incomplete_capacity",
])
def test_ineligible_calls_preserve_the_required_participants(mutation):
    raw = _raw()
    if mutation.endswith("private"):
        _binding(raw, "inputs" if mutation.startswith("input") else "outputs")["memory_sharing_scope"] = "block"
    elif mutation.startswith("compact"):
        _binding(raw, "inputs" if mutation.endswith("input") else "outputs")["storage_kind"] = "compact_local"
    elif mutation == "split":
        _binding(raw, "inputs")["distributed_type"]["axis_policies"][1] = {
            "kind": "split", "stages": [{"hierarchy_axes": (0, )}]
        }
    elif mutation == "partial":
        _binding(raw, "outputs")["distributed_type"]["partial"] = {"axes": (0, ), "reduce_op": "sum"}
    elif mutation == "scratch":
        raw["workspaces"] = [{
            "formal": "scratch", "buffers": [{"runtime_argument": "scratch", "abi": _abi((1, ), (2, 3), "bb")}]
        }]
    elif mutation == "shared_scratch":
        raw["shared_workspaces"] = [{"name": "scratch", "shape": (8, )}]
    elif mutation == "pipeline":
        raw["pipeline_channels"] = ["producer"]
    elif mutation == "side_effect":
        raw["facts"]["side_effects"] = ["write_other_storage"]
    elif mutation == "internal_barrier":
        raw["facts"]["internal_grid_barriers"] = 1
    elif mutation == "collective":
        raw["execution_kind"] = "collective"
    elif mutation == "other_semantic_op":
        raw["semantic_op"] = "nn.layer_norm"
    elif mutation == "other_variant":
        raw["variant"] = "other"
    elif mutation == "different_placement":
        _binding(raw, "inputs")["distributed_type"]["placement"]["name"] = "different"
    elif mutation == "incomplete_capacity":
        _binding(raw, "inputs")["logical_shape"] = (2, 16)
    assert _prepare(raw)["participation_active"] == _exclusive_participation_active(raw)


@pytest.mark.parametrize("mutation", ["empty_mesh", "zero_extent", "levels", "axis_policy", "E_owner"])
def test_malformed_complete_ABI_remains_rejected(mutation):
    raw = _raw()
    abi = _binding(raw, "inputs")
    if mutation == "empty_mesh":
        abi["distributed_type"]["placement"]["hierarchy"] = ()
    elif mutation == "zero_extent":
        abi["distributed_type"]["placement"]["hierarchy"] = (0, 3)
    elif mutation == "levels":
        abi["distributed_type"]["placement"]["hierarchy_levels"] = "b"
    elif mutation == "axis_policy":
        abi["distributed_type"]["axis_policies"] = ({"kind": "unknown"}, {"kind": "broadcast"})
    elif mutation == "E_owner":
        abi["distributed_type"]["exclusive"] = {"axes": (0, ), "owner_coordinates": (3, )}
    with pytest.raises(CodegenError):
        _prepare(raw)
