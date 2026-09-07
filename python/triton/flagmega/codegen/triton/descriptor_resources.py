# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Function-local tensor-map backing resources and scalar-origin call ABI.

Only a caller which proves complete accesses on the contiguous coordinate may
request rebasing. In particular, enlarging a masked tail axis is not legal.
The serialized list contains only runtime descriptor specs; proof/ABI records
are compile-time state, separate from the hardware tensor-map representation.
"""

from copy import deepcopy

from triton.flagmega.errors import CodegenError
from .tensor_descriptor_planner import _DTYPE_ITEM_SIZES


def origin_parameter(name):
    return f"{name}__origin_elements"


def _entries(spec):
    return spec["entries"] if spec["kind"] == "table" else (spec,)


def _geometry(spec):
    return {key: value for key, value in spec.items() if key != "name"}


class DescriptorResources(list):
    """Ordered runtime specs with explicit, non-serialized rebasing contracts."""

    def __init__(self, *, formal_origins=False):
        super().__init__()
        self.formal_origins = formal_origins
        self._contracts = {}

    def bind(self, spec, rebase_axis=None):
        if rebase_axis not in (None, -1):
            raise CodegenError("Only a proven contiguous last-axis descriptor rebase is supported.")
        if rebase_axis is not None and any(
            int(entry["strides"][-1]) != 1 for entry in _entries(spec)
        ):
            raise CodegenError("Descriptor origin requires a stride-one last coordinate.")
        if rebase_axis is not None and any(entry["source_shape_axes"][-1] for entry in _entries(spec)):
            raise CodegenError("Descriptor origin requires a fixed last coordinate, not a runtime shape override.")
        for resource in self:
            original, axis = self._contracts[resource["name"]]
            if axis != rebase_axis:
                continue
            origin = self._compatible_origin(original, spec, axis)
            if origin is None:
                continue
            if axis is not None:
                for physical, logical, candidate in zip(
                    _entries(resource), _entries(original), _entries(spec), strict=True,
                ):
                    physical["shape"] = (
                        *physical["shape"][:-1],
                        max(int(physical["shape"][-1]), int(candidate["shape"][-1]) + origin),
                    )
            return self._binding(resource["name"], origin, axis)
        if any(resource["name"] == spec["name"] for resource in self):
            raise CodegenError(f"Conflicting descriptor resource {spec['name']!r}.")
        self.append(deepcopy(spec))
        self._contracts[spec["name"]] = (deepcopy(spec), rebase_axis)
        return self._binding(spec["name"], 0, rebase_axis)

    @staticmethod
    def _compatible_origin(base, candidate, axis):
        if _geometry(base) == _geometry(candidate):
            return 0
        if axis is None:
            return None
        item_size = _DTYPE_ITEM_SIZES[str(base["dtype"])]
        lhs, rhs = deepcopy(_geometry(base)), deepcopy(_geometry(candidate))
        if lhs["kind"] != rhs["kind"]:
            return None
        left_entries, right_entries = _entries(lhs), _entries(rhs)
        if len(left_entries) != len(right_entries):
            return None
        deltas = set()
        for left, right in zip(left_entries, right_entries, strict=True):
            delta = int(right["offset_bytes"]) - int(left["offset_bytes"])
            if delta < 0 or delta % item_size:
                return None
            origin = delta // item_size
            # The dynamic coordinate and the entire box must fit signed i32.
            if origin + int(right["shape"][-1]) > 2**31 - 1:
                return None
            deltas.add(origin)
            left.pop("offset_bytes")
            right.pop("offset_bytes")
        if len(deltas) != 1 or lhs != rhs:
            return None
        return deltas.pop()

    def _binding(self, name, origin, axis):
        if axis is None:
            return (name,)
        # A Python integer at a noinline call would specialize every layer.
        value = f"tl.full((), {origin}, tl.int32)"
        if self.formal_origins:
            value = origin_parameter(name) if not origin else f"({origin_parameter(name)} + {value})"
        return (name, value)

    def parameters(self):
        return tuple(
            parameter
            for spec in self
            for parameter in (
                (spec["name"], origin_parameter(spec["name"]))
                if self._contracts[spec["name"]][1] is not None else (spec["name"],)
            )
        )

    def requests(self):
        return tuple({
            **_geometry(spec), "parameter": spec["name"],
            **({"rebase_axis": -1} if self._contracts[spec["name"]][1] is not None else {}),
        } for spec in self)


__all__ = ["DescriptorResources", "origin_parameter"]
