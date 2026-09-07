# flagtree tle
"""Validate explicit arena alignment before creating compiler IR."""

import pytest
import triton.language as tl
import triton.experimental.tle.language as tle


class _Semantic:
    builder = object()


@pytest.mark.parametrize("alignment", (0, -1, 3, True, 1.5, 1 << 31))
def test_invalid_shared_allocation_alignment_is_rejected(alignment):
    with pytest.raises(ValueError, match="alignment_bytes must"):
        tle.gpu.alloc((128,), tl.uint8, alignment_bytes=alignment, _semantic=_Semantic())


def test_additional_alignment_cannot_be_applied_to_a_view():
    with pytest.raises(ValueError, match="not an alias"):
        tle.gpu.alloc((128,), tl.uint8, alias=object(), alignment_bytes=1024, _semantic=_Semantic())
