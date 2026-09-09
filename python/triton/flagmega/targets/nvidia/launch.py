# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""SM90 entry launch policy over selected Triton implementations."""

from triton.flagmega.errors import IRVerificationError


def sm90_launch_parameters(module, kernel_nodes) -> dict[str, object]:
    """Resolve target-owned entry resources without leaking them into lowering."""

    del module
    if not kernel_nodes:
        raise IRVerificationError("SM90 launch planning requires selected TIR kernels.")
    if any(
        "warp_specialize" in node.attrs.get("facts", {}).get("requires", ())
        for node in kernel_nodes
    ):
        num_warps = 8
    else:
        num_warps = 4
    # A selected implementation may need more compute threads for its tiles
    # even when it has no separate warp-specialized producer. The shared
    # entry must satisfy every callee's minimum, without changing old defaults.
    for node in kernel_nodes:
        required = node.attrs.get("parameters", {}).get("compute_num_warps", 1)
        if type(required) is not int or required not in (1, 2, 4, 8, 16, 32):
            raise IRVerificationError(
                f"SM90 kernel {node.id!r} compute_num_warps must be an integer power of two from 1 to 32; got {required!r}."
            )
        num_warps = max(num_warps, required)
    return {"num_warps": num_warps}


__all__ = ["sm90_launch_parameters"]
