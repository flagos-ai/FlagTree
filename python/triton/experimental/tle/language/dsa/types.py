from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

from triton.language.core import base_type, base_value, dtype, _unwrap_if_constexpr


@dataclass(frozen=True)
class scope:
    """
    Simple storage descriptor for DSA buffers.

    This is intentionally backend-agnostic. `name` / `value` / `memory_space`
    are carried through as metadata only; concrete lowering is handled by the
    DSA dialect and backend.
    """

    name: str
    value: str
    memory_space: str

    def __repr__(self) -> str:
        return self.name


# DSA storage scopes.
local = scope("local", "local", "local")

# Scratch Pad Memory – the on-chip SRAM exposed by TsingMicro TX8.
# This is the primary storage scope for DSA kernels and serves the same
# conceptual role as NVIDIA shared memory (smem).
spm = scope("spm", "spm", "spm")


class buffered_tensor(base_value):
    """
    Symbolic handle to a buffer allocated via DSA.

    This is a thin wrapper over an IR value plus a `buffered_tensor_type`
    describing shape / element dtype / memory space.
    """

    def __init__(self, handle: Any, ty: "buffered_tensor_type"):
        self.handle = handle
        self.type = ty
        self.shape = ty.shape
        self.dtype = ty.element_ty

    def _flatten_ir(self, handles: List[Any]) -> None:
        handles.append(self.handle)


class buffered_tensor_type(base_type):
    """
    Frontend description of a DSA buffer.

    - `shape`: logical block shape (may be None for deferred shapes)
    - `element_ty`: scalar dtype
    - `storage`: abstract storage scope (currently `local`)
    - `memory_space`: backend-visible memory space string, defaults to
      `storage.memory_space`
    """

    def __init__(
        self,
        shape,
        element_ty: dtype,
        storage: scope | None = None,
        memory_space: str = "",
    ):
        if shape is None:
            self.shape = None
        else:
            shape = _unwrap_if_constexpr(shape)
            self.shape = tuple(int(_unwrap_if_constexpr(x)) for x in shape)
        self.element_ty = _unwrap_if_constexpr(element_ty)
        self.storage = storage if storage is not None else local
        self.memory_space = (str(_unwrap_if_constexpr(memory_space)) if memory_space else self.storage.memory_space)

    @property
    def scalar(self) -> dtype:
        return self.element_ty

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, buffered_tensor_type):
            return False
        return (
            self.shape,
            self.element_ty,
            self.storage,
            self.memory_space,
        ) == (
            other.shape,
            other.element_ty,
            other.storage,
            other.memory_space,
        )

    def __repr__(self) -> str:
        shape = "?" if self.shape is None else "x".join(map(str, self.shape))
        return f"buffered_tensor_type<{shape}, {self.element_ty}, {self.memory_space}>"

    def _unflatten_ir(self, handles: List[Any], cursor: int) -> Tuple[base_value, int]:
        value = buffered_tensor(handles[cursor], self)
        # Preserve remote metadata if present on the type.
        if hasattr(self, "_tle_remote_shard_id"):
            shard_id = getattr(self, "_tle_remote_shard_id")
            scope = getattr(self, "_tle_remote_scope", None)
            setattr(value, "_tle_remote_shard_id", shard_id)
            setattr(value, "_tle_remote_scope", scope)
            setattr(value.type, "_tle_remote_shard_id", shard_id)
            setattr(value.type, "_tle_remote_scope", scope)
        return value, cursor + 1
