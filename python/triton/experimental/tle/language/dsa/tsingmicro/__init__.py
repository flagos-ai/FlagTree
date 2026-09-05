# flagtree tle
"""TsingMicro (TX8) vendor namespace for DSA primitives.

Mirrors the upstream per-backend pattern (``dsa.ascend``): vendor-specific
primitives and address-space selectors are exposed here as
``tle.dsa.tsingmicro.*``, while the multi-vendor DSA API surface stays at the
``dsa`` root.
"""

from .core import SPM, randgen, rand, randn

__all__ = ["SPM", "randgen", "rand", "randn"]
