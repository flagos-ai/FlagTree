# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compilation switches owned by TLE.

They live here, not in ``triton.knobs``, so that the compiler front ends of the
backends read them through TLE and the upstream knob classes stay as they are.
"""

from triton.knobs import base_knobs, env_bool


class tle_compilation_knobs(base_knobs):
    # Keep a load of mutable shared memory before a write between it and its use
    # (``keep-smem-read-order`` of ``tritongpu-reorder-instructions`` and of
    # ``triton-tle-optimize-local-pointer-loads``). A build whose reorder pass
    # has no such option ignores the knob.
    keep_smem_read_order: env_bool = env_bool("FLAGTREE_KEEP_SMEM_READ_ORDER", True)


compilation = tle_compilation_knobs()
