# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError


def test_shared_workspace_descriptor_requires_a_materializable_tensor():
    descriptor = fm.T.shared_workspace_descriptor(
        "rhs_stage",
        fm.tensor_type("bfloat16", (2, 64, 16)),
        16,
    )

    assert descriptor.maximum_nbytes == 4096
    assert descriptor.alignment_bytes == 16

    with pytest.raises(IRSchemaError, match="non-empty name"):
        fm.T.shared_workspace_descriptor(
            "", fm.tensor_type("bfloat16", (16,)), 16
        )
    with pytest.raises(IRSchemaError, match="positive power of two"):
        fm.T.shared_workspace_descriptor(
            "rhs_stage", fm.tensor_type("bfloat16", (16,)), 12
        )
    with pytest.raises(IRSchemaError, match="at least its element size"):
        fm.T.shared_workspace_descriptor(
            "rhs_stage", fm.tensor_type("float32", (16,)), 2
        )


def test_shared_workspace_descriptor_requires_finite_positive_bounds():
    with pytest.raises(IRSchemaError, match="finite positive maximum size"):
        fm.T.shared_workspace_descriptor(
            "dynamic",
            fm.tensor_type("bfloat16", (fm.dim("tokens"), 16)),
            16,
        )
    with pytest.raises(IRSchemaError, match="finite positive maximum size"):
        fm.T.shared_workspace_descriptor(
            "empty", fm.tensor_type("bfloat16", (0, 16)), 16
        )
