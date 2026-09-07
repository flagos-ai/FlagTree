# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Descriptor resources are storage views, not call-site identities."""

from triton.flagmega.codegen.triton.tir_package import _instantiate_descriptor_requests


def _request(source="cache", offset=0):
    return {
        "parameter": "key_descriptor", "source": source, "kind": "single",
        "offset_bytes": offset, "dtype": "bfloat16", "shape": (16, 64),
        "strides": (64, 1), "block_shape": (1, 64),
        "source_shape_axes": ((), ()), "padding": "zero",
    }


def test_identical_views_share_resource_across_reusable_calls():
    specs = []
    bindings = [
        _instantiate_descriptor_requests(
            {"host_tensor_descriptor_requests": (_request(),)},
            {"cache": ("cache", 0)}, specs, (f"layer_{index}",),
        )
        for index in range(28)
    ]
    assert len(specs) == 1
    assert len(set(bindings)) == 1


def test_distinct_sources_do_not_share_resources():
    specs = []
    for source in ("key", "value"):
        _instantiate_descriptor_requests(
            {"host_tensor_descriptor_requests": (_request(source),)},
            {source: (source, 0)}, specs, (source,),
        )
    assert len(specs) == 2


def test_reuse_is_decided_after_resolving_storage_roots():
    specs = []
    bindings = [
        _instantiate_descriptor_requests(
            {"host_tensor_descriptor_requests": (_request(source, offset),)},
            {source: ("arena", root_offset)}, specs, (source,),
        )
        for source, offset, root_offset in (("first", 128, 256), ("second", 0, 384))
    ]
    assert len(specs) == 1
    assert bindings[0] == bindings[1]
