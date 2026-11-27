# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import json
from types import MappingProxyType

from triton.flagmega.codegen.triton.package_plan import plain_package_value
from triton.flagmega import ir as fm


def test_frozen_nested_package_plan_becomes_json_safe_descriptor_data():
    frozen = MappingProxyType({
        "pipeline": MappingProxyType({
            "dense": MappingProxyType({
                "stage_shape": (2, 16, 1024),
            }),
        }),
        "templates": (
            MappingProxyType({"kernel": "dense_matmul", "variant": "gemv"}),
        ),
    })

    plain = plain_package_value(frozen)

    assert plain == {
        "pipeline": {"dense": {"stage_shape": [2, 16, 1024]}},
        "templates": [{"kernel": "dense_matmul", "variant": "gemv"}],
    }
    json.dumps(plain)


def test_package_plan_serializes_nested_ir_types_for_artifact_manifests():
    value = fm.DistributedType(
        fm.tensor_type("bfloat16", (8, 16)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,))),
        fm.Placement((2,), "x", "b"),
    )

    plain = plain_package_value({"distribution": {"input_types": (value,)}})

    assert plain["distribution"]["input_types"][0]["kind"] == "distributed"
    json.dumps(plain)
