# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.targets.nvidia import sm90_launch_parameters


def _kernel(semantic_op="math.add", *, requires=(), parameters=None):
    return fm.Node(
        "kernel",
        "tir.kernel",
        (),
        fm.tensor_type("bfloat16", (1, 8)),
        attrs={"semantic_op": semantic_op, "facts": {"requires": requires}, "parameters": parameters or {}},
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


@pytest.mark.parametrize("required,expected", ((1, 4), (2, 4), (4, 4), (8, 8), (16, 16), (32, 32)))
def test_selected_implementation_declares_compute_warp_requirement(required, expected):
    kernels = (_kernel(parameters={"compute_num_warps": required}), _kernel())
    assert sm90_launch_parameters(None, kernels) == {"num_warps": expected}


def test_compute_warp_requirements_combine_with_warp_specialization():
    kernels = (_kernel(parameters={"compute_num_warps": 4}), _kernel(requires=("warp_specialize",)))
    assert sm90_launch_parameters(None, kernels) == {"num_warps": 8}
    kernels += (_kernel(parameters={"compute_num_warps": 16}),)
    assert sm90_launch_parameters(None, kernels) == {"num_warps": 16}


@pytest.mark.parametrize("invalid", (None, True, 0, -1, 3, 64, 8.0, "8"))
def test_invalid_compute_warp_requirement_is_not_silently_ignored(invalid):
    with pytest.raises(IRVerificationError, match="compute_num_warps"):
        sm90_launch_parameters(None, (_kernel(parameters={"compute_num_warps": invalid}),))
