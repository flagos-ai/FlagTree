# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""SM90 entry launch policy over selected Triton implementations."""

from triton.flagmega.errors import IRVerificationError


def sm90_launch_parameters(module, kernel_nodes) -> dict[str, object]:
    """Resolve target-owned entry resources without leaking them into lowering."""

    del module
    if any(
        "warp_specialize" in node.attrs.get("facts", {}).get("requires", ())
        for node in kernel_nodes
    ):
        num_warps = 8
    else:
        num_warps = 4
    if not kernel_nodes:
        raise IRVerificationError("SM90 launch planning requires selected TIR kernels.")
    return {"num_warps": num_warps}


__all__ = ["sm90_launch_parameters"]
