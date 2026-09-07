# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.targets.nvidia import sm90_launch_parameters


def _kernel(semantic_op="math.add", *, requires=()):
    return fm.Node(
        "kernel",
        "tir.kernel",
        (),
        fm.tensor_type("bfloat16", (1, 8)),
        attrs={"semantic_op": semantic_op, "facts": {"requires": requires}},
    )


@pytest.mark.parametrize(
    ("kernels", "expected"),
    (
        ((_kernel(),), 4),
        ((_kernel(requires=("warp_specialize",)),), 8),
    ),
)
def test_sm90_launch_policy_owns_entry_warp_geometry(kernels, expected):
    assert sm90_launch_parameters(None, kernels) == {"num_warps": expected}


def test_sm90_launch_policy_rejects_an_empty_selected_program():
    with pytest.raises(IRVerificationError, match="requires selected TIR kernels"):
        sm90_launch_parameters(None, ())
