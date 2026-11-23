# flagtree tle
"""Multi-vendor DSA (Device-Specific API) extensions.

``dsa`` is the multi-vendor namespace for device-specific primitives: the
shared API surface (``alloc``/``copy``/``pipeline``/slices/arithmetic) lives
at this level, and each vendor's private primitives live in a per-backend
sub-namespace (see ``dsa.tsingmicro`` and ``dsa.ascend``).

The shared surface is the portable Triton 3.6 DSA API.  The Ascend backend
does not lower the 3.6 ``dsa`` dialect yet, so on Ascend the same public names
are provided by the 3.5 CommonIR frontend in :mod:`dsa.ascend.compat`, and the
Ascend-only helpers (``UB``/``L1``/..., pipe scheduling, ``raw`` custom ops)
live in :mod:`dsa.ascend`.
"""

from triton._flagtree_backend import FLAGTREE_BACKEND

if FLAGTREE_BACKEND == "ascend":
    # 3.5 CommonIR implementations of the shared surface, plus the extra
    # tile/workspace/subview primitives the Ascend tutorials use.
    from .ascend.compat import *  # noqa: F401,F403
    from .ascend.compat import __all__ as _ascend_compat_all

    from . import ascend

    __all__ = list(_ascend_compat_all) + ["ascend"]
else:
    from .core import (
        tile_load,
        tile_store,
        pipeline,
        alloc,
        copy,
        memory_space,
        local_ptr,
        to_tensor,
        to_buffer,
        add,
        sub,
        mul,
        max,
        min,
        div,
        extract_slice,
        extract_element,
        insert_slice,
    )
    from .types import (
        scope,
        local,
        buffered_tensor,
        buffered_tensor_type,
    )
    from . import tsingmicro
    from . import thrive
    from .semantic import DSASemantic, DSASemanticError

    try:
        # The Ascend vendor namespace is only importable where the CANN
        # language extension exists.
        from . import ascend
    except Exception:
        ascend = None

    __all__ = [
        "tile_load",
        "tile_store",
        "pipeline",
        "alloc",
        "copy",
        "memory_space",
        "local_ptr",
        "to_tensor",
        "to_buffer",
        "add",
        "sub",
        "mul",
        "max",
        "min",
        "div",
        "extract_slice",
        "extract_element",
        "insert_slice",
        "scope",
        "local",
        "ascend",
        "tsingmicro",
        "thrive",
        "buffered_tensor",
        "buffered_tensor_type",
        "DSASemantic",
        "DSASemanticError",
    ]
