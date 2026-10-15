# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Packed collapsed addressing retains scalar order and rejects ABI mismatches."""

from math import prod

import pytest

from triton.flagmega.codegen.triton.mhc_renderer import mhc_pre_call
from triton.flagmega.errors import CodegenError


def _raw(lanes=(2, 4)):
    tokens, hc, hidden = 3, 4, 72
    coefficients = (2 + hc) * hc
    shapes = {"streams": (tokens, hc, hidden), "projection": (coefficients, hc * hidden),
              "base": (coefficients,), "scale": (3,), "result_0": (tokens, hc),
              "result_1": (tokens, hc, hc), "result_2": (tokens, hidden // prod(lanes))}

    def parameter(name, shape):
        lane_shape = lanes if name == "result_2" else ()
        abi = {"local_capacity_shape": shape,
               "active_shape_expressions": tuple(str(value) for value in shape),
               "scalar_dtype": "bfloat16" if name in {"streams", "result_2"} else "float32",
               "scalar_lane_shape": lane_shape, "scalar_lane_count": prod(lane_shape),
               "scalar_storage_strides": tuple(prod(shape[index + 1:]) * prod(lane_shape)
                                                 for index in range(len(shape))),
               "coordinate_space": "local", "storage_kind": "compact_local",
               "storage": "external", "distributed_type": None}
        return {"formal": name, "buffers": [{"runtime_argument": name, "abi": abi}]}

    return {"inputs": [parameter(name, shape) for name, shape in shapes.items() if not name.startswith("result")],
            "outputs": [parameter(name, shape) for name, shape in shapes.items() if name.startswith("result")],
            "parameters": {"elements_per_program": 256},
            "semantic_attrs": {"epsilon": 1e-5, "sinkhorn_epsilon": 1e-6, "post_multiplier": 2.,
                               "sinkhorn_iterations": 20, "output_lanes": lanes}}


@pytest.mark.parametrize("lanes", [(), (8,), (2, 4), (4, 2), (1,)])
def test_collapsed_scalar_offsets_preserve_dense_hidden_order(lanes):
    rendered = mhc_pre_call(_raw(lanes))
    for token in range(3):
        for hidden in range(72):
            offset = eval(rendered["collapsed_offset"], {"_fm_token": token, "_fm_hidden": hidden})
            assert offset == token * 72 + hidden


@pytest.mark.parametrize("changes,message", [
    ({"scalar_lane_shape": (4, 2)}, "ABI disagrees"),
    ({"scalar_lane_count": 4}, "ABI disagrees"),
    ({"scalar_dtype": "float32"}, "ABI disagrees"),
    ({"local_capacity_shape": (3, 72)}, "stream capacity"),
    ({"distributed_type": {"partial": "sum", "axis_policies": []}}, "partial"),
    ({"distributed_type": {"partial": None, "axis_policies": [{"kind": "split"}]}}, "split"),
])
def test_renderer_rejects_inconsistent_or_nonreplicated_packed_output(changes, message):
    raw = _raw()
    raw["outputs"][2]["buffers"][0]["abi"].update(changes)
    with pytest.raises(CodegenError, match=message):
        mhc_pre_call(raw)


def test_packed_collapsed_support_does_not_relax_coefficient_layout():
    raw = _raw()
    raw["outputs"][0]["buffers"][0]["abi"].update(scalar_lane_shape=(2,), scalar_lane_count=2)
    with pytest.raises(CodegenError, match="post.*vector lanes"):
        mhc_pre_call(raw)
