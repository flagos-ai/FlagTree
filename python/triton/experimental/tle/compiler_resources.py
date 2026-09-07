# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Assembler evidence for TLE's out-of-line device-call resource contract."""

import re


def parse_ptxas_resource_usage(log: str) -> dict[str, int]:
    """Match nncase's per-function accounting, including non-entry callees.

    CUDA LOCAL_SIZE_BYTES includes the device-call stack; it cannot identify
    allocator spills. Store/load counts are assembler instruction byte counts,
    not a local allocation size. Keep all three quantities separate.
    """
    records = re.findall(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log,
    )
    if not records:
        raise RuntimeError("ptxas did not report per-function resource usage")
    values = [tuple(map(int, record)) for record in records]
    return {
        "ptxas_stack_frame_bytes": max(stack for stack, _, _ in values),
        "ptxas_spill_store_bytes": sum(stores for _, stores, _ in values),
        "ptxas_spill_load_bytes": sum(loads for _, _, loads in values),
    }
