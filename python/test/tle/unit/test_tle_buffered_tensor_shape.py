# flagtree tle
"""MemDesc shape rules are weaker than register block-shape rules."""

import pytest
import triton.language as tl
import triton.experimental.tle.language as tle


class _Semantic:
    builder = object()


def _type(shape):
    return tle.gpu.buffered_tensor_type(
        tl.uint8,
        shape,
        tle.gpu.smem,
        tle.gpu.swizzled_shared_layout.make_default(len(shape)),
        _Semantic(),
    )


def test_buffered_tensor_accepts_non_power_of_two_leading_allocation_extent():
    value = _type((135168,))

    assert value.shape == (135168,)
    assert value.alloc_shape == [135168]
    assert value.numel == 135168


def test_buffered_tensor_still_requires_power_of_two_trailing_dimensions():
    with pytest.raises(ValueError, match="Shape element 1 must be a power of 2"):
        _type((3, 96))
