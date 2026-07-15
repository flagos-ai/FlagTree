# Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.

# Declare the TLE features that the PPU backend does not (yet) support so the
# Python frontend can raise a clear error before MLIR-level legalization fails.
#
# We import from triton.experimental._tle_capabilities rather than
# triton.experimental.tle._capabilities so we do not trigger
# triton.experimental.tle/__init__.py — that module pulls in tle.language,
# which transitively needs triton.runtime.jit. PPU's __init__.py runs while
# triton itself is still mid-init (during backend discovery), so the deeper
# package would cause a circular import.
try:
    from triton.experimental._tle_capabilities import register_unsupported
except ImportError:
    pass
else:
    register_unsupported(
        "ppu",
        {
            # No TMA-equivalent hardware on PPU.
            "tle.copy.tma",
            # Distributed TLE primitives require multi-card cluster support
            # that PPU does not provide today.
            "tle.distributed_barrier",
            "tle.shard_id",
            "tle.remote",
            # warp_specialize relies on hardware/runtime warp specialization
            # (Hopper warpgroup partitioning); PPU has no equivalent today.
            "tle.warp_specialize",
        },
    )
