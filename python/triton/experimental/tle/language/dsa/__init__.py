# flagtree tle
"""Multi-vendor DSA (Device-Specific API) extensions.

``dsa`` is the multi-vendor namespace for device-specific primitives: the
shared API surface (``alloc``/``copy``/``pipeline``/slices/arithmetic) lives
at this level, and each vendor's private primitives live in a per-backend
sub-namespace (see ``dsa.tsingmicro``, mirroring upstream ``dsa.ascend``).
"""

from .core import (
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
    insert_slice,
)
from .types import (
    scope,
    local,
    buffered_tensor,
    buffered_tensor_type,
)
from . import tsingmicro
from .semantic import DSASemantic, DSASemanticError

__all__ = [
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
    "insert_slice",
    "scope",
    "local",
    "tsingmicro",
    "buffered_tensor",
    "buffered_tensor_type",
    "DSASemantic",
    "DSASemanticError",
]
