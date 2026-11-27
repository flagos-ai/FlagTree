# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Verify caller-owned Shared payloads and reusable schedule interference.

Shared addresses may be relocated. A callee's already selected synchronization
cannot, however, protect a new overlap introduced only in its caller binding.
Removing overlap is safe: it leaves a conservative drain in the callee. Leaf
kernel workspaces coexist for the whole operation and must remain disjoint.
"""

from math import gcd

from triton.flagmega.errors import IRVerificationError


def verify_shared_call_abi(call, formals, *, stage, leaf, maximum_bytes=None):
    actuals = call.shared_workspace_buffers

    def fail(message):
        raise IRVerificationError(
            f"Execution call {call.call_id!r} Shared {message}",
            stage=stage, node_id=call.call_id,
        )

    if len(formals) != len(actuals):
        fail(f"ABI arity differs from @{call.callee}: expected {len(formals)}, got {len(actuals)}.")
    if len({buffer.name for buffer in actuals}) != len(actuals):
        fail("ABI buffer names must be unique.")
    actual_ranges = []
    formal_ranges = []
    for formal, actual in zip(formals, actuals, strict=True):
        if (actual.type != formal.type or actual.strides != formal.strides
                or actual.mem_span.size != formal.mem_span.size):
            fail(f"payload {actual.name!r} differs from formal {formal.name!r}.")
        try:
            formal_start = formal.mem_span.absolute_start.fixed_value
            formal_end = formal.mem_span.absolute_end.fixed_value
            relative = formal.mem_span.start.fixed_value
            actual_start = actual.mem_span.absolute_start.fixed_value
            actual_end = actual.mem_span.absolute_end.fixed_value
            physical_start = actual.mem_span.buffer.start.fixed_value
        except ValueError as error:
            fail(f"ABI requires fixed post-Bufferize byte ranges: {error}")
        # Allocation alignment does not imply alignment of an arbitrary view.
        # Preserve the formal view's proven alignment, not a guessed arena floor.
        required_alignment = gcd(formal.mem_span.buffer.alignment, relative)
        physical_alignment = actual.mem_span.buffer.alignment
        if (physical_start % physical_alignment
                or physical_alignment < required_alignment
                or actual_start % required_alignment):
            fail(f"payload {actual.name!r} violates its allocation/formal alignment.")
        if maximum_bytes is not None and actual_end > maximum_bytes:
            fail(f"payload {actual.name!r} exceeds the memory-space capacity {maximum_bytes}.")
        actual_ranges.append((actual_start, actual_end))
        formal_ranges.append((formal_start, formal_end))

    # Compare only actually overlapping pairs. Uniform frame rebases and legal
    # callee temporal reuse remain valid; no exact-offset ABI is imposed.
    frontier = []
    for index in sorted(range(len(actuals)), key=lambda i: actual_ranges[i][0]):
        start, end = actual_ranges[index]
        if start == end:
            continue
        frontier = [other for other in frontier if actual_ranges[other][1] > start]
        for other in frontier:
            formal_start, formal_end = formal_ranges[index]
            other_start, other_end = formal_ranges[other]
            if leaf or formal_start >= other_end or other_start >= formal_end:
                fail(f"overlap between {actuals[other].name!r} and {actuals[index].name!r} "
                     "is not protected by the callee resource schedule.")
        frontier.append(index)


__all__ = ["verify_shared_call_abi"]
