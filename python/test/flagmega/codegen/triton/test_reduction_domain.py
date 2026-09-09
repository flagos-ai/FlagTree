# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.reduction_domain import local_reduction_domain
from triton.flagmega.errors import CodegenError


def abi(shape, *, lanes=1):
    return {
        "local_capacity_shape": shape, "scalar_lane_count": lanes, "active_shape_expressions": tuple(map(str, shape))
    }


@pytest.mark.parametrize("axes,coordinates", [((0, 2), (1, 4, 1)), ((2, 0), (0, 4, 2))])
def test_reduction_domain_separates_outer_rows_from_ordered_reduction_axes(axes, coordinates):
    domain = local_reduction_domain(abi((2, 7, 3)), axes, 256)
    assert (domain["rows"], domain["capacity"], domain["tile"]) == (7, 6, 8)
    values = {"_fm_row": 4, "_fm_offsets": 4}
    actual = tuple(eval(value, {"__builtins__": {}}, values) for value in domain["coordinates"])
    assert actual == coordinates


@pytest.mark.parametrize("shape,axes,rows,capacity", [
    ((2, 0, 7), (1, ), 14, 0),
    ((0, 7), (1, ), 0, 7),
    ((), (), 1, 1),
    ((3, 513), (1, ), 3, 513),
])
def test_reduction_domain_empty_shapes_and_tiling_do_not_divide_by_zero(shape, axes, rows, capacity):
    domain = local_reduction_domain(abi(shape), axes, 256)
    assert (domain["rows"], domain["capacity"]) == (rows, capacity)
    assert domain["tile"] <= 256
    for coordinate in domain["coordinates"]:
        eval(coordinate, {"__builtins__": {}}, {"_fm_row": 0, "_fm_offsets": 0})


@pytest.mark.parametrize("tile", [0, -1, 3])
def test_reduction_domain_rejects_invalid_tile(tile):
    with pytest.raises(CodegenError, match="positive power of two"):
        local_reduction_domain(abi((3, 7)), (1, ), tile)


def test_reduction_domain_does_not_flatten_vector_lanes_into_a_scalar_axis():
    with pytest.raises(CodegenError, match="vector element"):
        local_reduction_domain(abi((3, 7), lanes=8), (1, ), 256)
