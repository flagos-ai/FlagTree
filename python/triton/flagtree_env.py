"""FlagTree-owned environment toggles for compiler passes.

Kept out of ``triton.knobs`` (upstream) so upstream rebases of that file do not
conflict. The C++ side declares the same names in
``include/flagtree/Common/EnvVars.h``, which ``get_cache_invalidating_env_vars``
merges into the upstream cache-invalidating list, so flipping any of these
still invalidates the JIT cache.
"""

import os


def _flagtree_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "on", "yes")


def is_lane_vectorize_disabled() -> bool:
    """Whether the ``triton-lane-vectorize`` TTIR pass should be skipped.

    The pass only runs in the Ascend backend. It is enabled by default; set
    ``TRITON_DISABLE_LANE_VECTORIZE=1`` to skip it.
    """
    return _flagtree_bool("TRITON_DISABLE_LANE_VECTORIZE", False)
