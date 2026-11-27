# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from copy import deepcopy

import pytest

from triton.flagmega.codegen.triton.descriptor_resources import DescriptorResources
from triton.flagmega.codegen.triton.pipeline_function_abi import descriptor_requests
from triton.flagmega.codegen.triton.tir_package import _instantiate_descriptor_requests
from triton.flagmega.errors import CodegenError


def _table(offset=0, **changes):
    return {
        "parameter": "weight_descriptor", "source": "weight", "kind": "table",
        "dtype": "bfloat16", "block_shape": (2, 2, 2, 64),
        "padding": "zero", "swizzle_mode": 3, "entry_size_bytes": 128,
        "rebase_axis": -1,
        "entries": tuple({
            "offset_bytes": offset + owner * 65536, "shape": (8, 8, 2, 64),
            "strides": (1024, 128, 64, 1), "source_shape_axes": ((), (), (), ()),
        } for owner in range(2)),
        **changes,
    }


def _bind(resources, request, call):
    return _instantiate_descriptor_requests(
        {"host_tensor_descriptor_requests": (request,)},
        {"weight": ("rdata", 0)}, resources, (call,),
    )


def test_uniform_owner_rebase_shares_backing_and_grows_only_contiguous_extent():
    resources = DescriptorResources()
    first = _bind(resources, _table(), "first")
    second = _bind(resources, _table(131072), "second")
    third = _bind(resources, _table(262144), "third")
    assert len(resources) == 1
    assert first[0] == second[0] == third[0]
    assert first[1] == "tl.full((), 0, tl.int32)"
    assert second[1] == "tl.full((), 65536, tl.int32)"
    assert third[1] == "tl.full((), 131072, tl.int32)"
    assert tuple(entry["shape"] for entry in resources[0]["entries"]) == ((8, 8, 2, 131136),) * 2
    assert tuple(entry["offset_bytes"] for entry in resources[0]["entries"]) == (0, 65536)
    assert "rebase_axis" not in resources[0]


@pytest.mark.parametrize("field,value", [
    ("dtype", "float16"), ("swizzle_mode", 2),
    ("padding", "nan"), ("block_shape", (4, 2, 2, 64)),
])
def test_incompatible_hardware_contracts_have_distinct_backing(field, value):
    resources = DescriptorResources()
    _bind(resources, _table(), "first")
    _bind(resources, _table(131072, **{field: value}), "second")
    assert len(resources) == 2


@pytest.mark.parametrize("mutation", ["nonuniform", "shape", "stride", "axes", "overflow", "negative", "unaligned"])
def test_rebase_requires_one_legal_affine_origin_for_all_owners(mutation):
    resources = DescriptorResources()
    _bind(resources, _table(131072), "first")
    candidate = deepcopy(_table(262144))
    if mutation == "nonuniform":
        candidate["entries"][1]["offset_bytes"] += 128
    elif mutation in ("overflow", "negative", "unaligned"):
        delta = {"overflow": 2**33, "negative": -262144, "unaligned": 1}[mutation]
        for entry in candidate["entries"]:
            entry["offset_bytes"] += delta
    else:
        field, value = {
            "shape": ("shape", (16, 8, 2, 64)),
            "stride": ("strides", (2048, 128, 64, 1)),
            "axes": ("source_shape_axes", ((0,), (), (), ())),
        }[mutation]
        candidate["entries"][0][field] = value
    _bind(resources, candidate, "second")
    assert len(resources) == 2


def test_formal_origin_composes_through_nested_function_abi():
    inner = DescriptorResources(formal_origins=True)
    first = _bind(inner, _table(), "first")
    second = _bind(inner, _table(131072), "second")
    assert first[1] == first[0] + "__origin_elements"
    assert second[1] == f"({first[1]} + tl.full((), 65536, tl.int32))"
    assert inner.parameters() == (first[0], first[1])
    outer = DescriptorResources()
    request, = descriptor_requests(inner)
    result = _instantiate_descriptor_requests(
        {"host_tensor_descriptor_requests": (request,)},
        {"rdata": ("rdata", 524288)}, outer, ("outer",),
    )
    assert result[1] == "tl.full((), 0, tl.int32)"
    assert outer[0]["entries"][0]["offset_bytes"] == 524288
    assert outer[0]["entries"][0]["shape"][-1] == 65600


def test_rebasing_never_implicitly_enabled_on_a_masked_view():
    resources = DescriptorResources()
    for index in range(2):
        request = _table(131072 * index)
        request.pop("rebase_axis")
        _bind(resources, request, f"call{index}")
    assert len(resources) == 2
    assert len(resources.parameters()) == 2


def test_noncontiguous_origin_is_rejected():
    request = _table()
    request["entries"][0]["strides"] = (1024, 128, 64, 2)
    with pytest.raises(CodegenError, match="stride-one"):
        _bind(DescriptorResources(), request, "bad")


def test_dynamic_last_coordinate_cannot_replace_the_backing_extent_at_runtime():
    request = _table()
    for entry in request["entries"]:
        entry["source_shape_axes"] = ((), (), (), (0,))
    with pytest.raises(CodegenError, match="fixed last coordinate"):
        _bind(DescriptorResources(), request, "dynamic")
