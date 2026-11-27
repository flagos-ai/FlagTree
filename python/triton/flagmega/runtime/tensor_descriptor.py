# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Validated, reusable host tensor descriptors for TMA kernels."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from triton.flagmega.errors import RuntimeContractError


class TensorDescriptorCache:
    """Materialize single descriptors and device tensor-map tables.

    Cache identity includes the storage address, device, dtype, byte offset,
    resolved shape/strides, block shape and padding. Reusing a descriptor name
    with a changed dynamic tensor therefore rebuilds it instead of retaining a
    stale pointer.
    """

    descriptor_alignment_bytes = 16
    tensor_map_alignment_bytes = 128
    maximum_block_elements = 1_048_576
    _single_fields = frozenset({
        "kind", "name", "source", "offset_bytes", "dtype", "shape",
        "strides", "block_shape", "source_shape_axes", "padding",
    })
    _table_fields = frozenset({
        "kind", "name", "source", "dtype", "block_shape", "padding",
        "swizzle_mode", "entry_size_bytes", "entries",
    })
    _table_entry_fields = frozenset({
        "offset_bytes", "shape", "strides", "source_shape_axes",
    })
    _host_dtype = {
        "uint8": 0,
        "uint16": 1,
        "uint32": 2,
        "int32": 3,
        "uint64": 4,
        "int64": 5,
        "float16": 6,
        "float32": 7,
        "float64": 8,
        "bfloat16": 9,
        "float8_e4m3fn": 0,
        "float8e4m3fn": 0,
        "float8_e5m2": 0,
        "float8e5m2": 0,
    }

    def __init__(self) -> None:
        self._entries: dict[str, tuple[tuple[object, ...], object]] = {}

    def clear(self) -> None:
        self._entries.clear()

    @classmethod
    def validate_specs(
        cls,
        specs: Sequence[Mapping[str, Any]],
        *,
        source_names: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        """Validate the serialized descriptor ABI without materializing CUDA state."""

        available = None if source_names is None else frozenset(source_names)
        names: list[str] = []
        for index, spec in enumerate(specs):
            if not isinstance(spec, Mapping):
                raise RuntimeContractError(
                    f"Host tensor descriptor spec {index} must be a mapping."
                )
            kind = str(spec.get("kind", ""))
            expected = (
                cls._single_fields if kind == "single"
                else cls._table_fields if kind == "table"
                else None
            )
            if expected is None:
                raise RuntimeContractError(
                    "Host tensor descriptor kind must be 'single' or 'table', "
                    f"got {kind!r}."
                )
            if kind == "single" and spec.get("storage") == "device":
                expected = expected | {"storage", "box_shape", "swizzle_mode"}
            _require_exact_fields(spec, expected, "host tensor descriptor")
            if kind == "single" and spec.get("storage") == "device":
                _validate_device_encoding(spec)
            name = str(spec["name"])
            source = str(spec["source"])
            if not name or not source:
                raise RuntimeContractError(
                    "Host tensor descriptor name/source must be non-empty."
                )
            if name in names:
                raise RuntimeContractError(
                    f"Duplicate host tensor descriptor name {name!r}."
                )
            if available is not None and source not in available:
                raise RuntimeContractError(
                    f"Host tensor descriptor {name!r} references unbound "
                    f"source {source!r}."
                )
            if kind == "table":
                entries = spec["entries"]
                if not isinstance(entries, (tuple, list)) or not entries:
                    raise RuntimeContractError(
                        f"Tensor-map table {name!r} requires at least one entry."
                    )
                for entry_index, entry in enumerate(entries):
                    if not isinstance(entry, Mapping):
                        raise RuntimeContractError(
                            f"Tensor-map table {name!r} entry {entry_index} "
                            "must be a mapping."
                        )
                    _require_exact_fields(
                        entry,
                        cls._table_entry_fields,
                        f"tensor-map table {name!r} entry {entry_index}",
                    )
            names.append(name)
        return tuple(names)

    def materialize_many(
        self,
        kernel_name: str,
        specs: Sequence[Mapping[str, Any]],
        sources: Mapping[str, Any],
    ) -> tuple[object, ...]:
        self.validate_specs(specs, source_names=tuple(sources))
        result = []
        for spec in specs:
            kind = str(spec.get("kind", ""))
            name = str(spec["name"])
            source_name = str(spec["source"])
            if not name or not source_name:
                raise RuntimeContractError(
                    "Host tensor descriptor name/source must be non-empty."
                )
            try:
                storage = sources[source_name]
            except KeyError as error:
                raise RuntimeContractError(
                    f"Host tensor descriptor {name!r} references unbound "
                    f"source {source_name!r}."
                ) from error
            slot = f"{kernel_name}:{name}"
            if kind == "single" and spec.get("storage") == "device":
                result.append(self._materialize_device(slot, storage, spec))
            elif kind == "single":
                result.append(self._materialize_single(slot, storage, spec))
            else:
                result.append(self._materialize_table(slot, storage, spec))
        return tuple(result)

    def _materialize_device(self, slot, storage, spec):
        # One device map, with the SAME base/shape/strides as the single map.
        # The table encoder is reused as storage machinery, not as an owner
        # coordinate transformation. The launch owns its cached CUDA tensor.
        table = {
            "kind": "table", "name": spec["name"], "source": spec["source"],
            "dtype": spec["dtype"], "padding": spec["padding"],
            "block_shape": spec["box_shape"], "swizzle_mode": spec["swizzle_mode"],
            "entry_size_bytes": self.tensor_map_alignment_bytes,
            "entries": ({key: spec[key] for key in self._table_entry_fields},),
        }
        return self._materialize_table(slot, storage, table)

    def _materialize_single(self, slot, storage, spec):
        shape = self._resolve_shape(
            slot,
            storage,
            tuple(int(value) for value in spec["shape"]),
            tuple(tuple(int(axis) for axis in axes) for axes in spec["source_shape_axes"]),
        )
        strides = tuple(int(value) for value in spec["strides"])
        block_shape = tuple(int(value) for value in spec["block_shape"])
        dtype = str(spec["dtype"])
        padding = str(spec["padding"])
        base, descriptor_signature = self._prepare_base(
            slot,
            storage,
            offset_bytes=int(spec["offset_bytes"]),
            dtype=dtype,
            shape=shape,
            strides=strides,
            block_shape=block_shape,
            padding=padding,
        )
        signature = ("single", *descriptor_signature)
        cached = self._entries.get(slot)
        if cached is not None and cached[0] == signature:
            return cached[1]
        from triton.tools.tensor_descriptor import TensorDescriptor

        descriptor = TensorDescriptor(
            base,
            shape=list(shape),
            strides=list(strides),
            block_shape=list(block_shape),
            padding=padding,
        )
        self._entries[slot] = (signature, descriptor)
        return descriptor

    def _materialize_table(self, slot, storage, spec):
        dtype = str(spec["dtype"])
        block_shape = tuple(int(value) for value in spec["block_shape"])
        padding = str(spec["padding"])
        swizzle = int(spec["swizzle_mode"])
        entry_size = int(spec["entry_size_bytes"])
        entries = tuple(spec["entries"])
        if entry_size != self.tensor_map_alignment_bytes:
            raise RuntimeContractError(
                f"Tensor-map table {slot} entry size must be "
                f"{self.tensor_map_alignment_bytes} bytes."
            )
        if not 0 <= swizzle <= 3 or not entries:
            raise RuntimeContractError(
                f"Tensor-map table {slot} requires entries and swizzle in [0, 3]."
            )
        try:
            host_dtype = self._host_dtype[dtype]
        except KeyError as error:
            raise RuntimeContractError(
                f"Tensor-map table {slot} does not support dtype {dtype!r}."
            ) from error

        prepared = []
        signatures = []
        for index, entry in enumerate(entries):
            _require_exact_fields(
                entry, self._table_entry_fields, f"tensor-map table {slot} entry {index}"
            )
            entry_slot = f"{slot}[{index}]"
            shape = self._resolve_shape(
                entry_slot,
                storage,
                tuple(int(value) for value in entry["shape"]),
                tuple(tuple(int(axis) for axis in axes) for axes in entry["source_shape_axes"]),
            )
            strides = tuple(int(value) for value in entry["strides"])
            base, signature = self._prepare_base(
                entry_slot,
                storage,
                offset_bytes=int(entry["offset_bytes"]),
                dtype=dtype,
                shape=shape,
                strides=strides,
                block_shape=block_shape,
                padding=padding,
            )
            prepared.append((base, shape, strides))
            signatures.append(signature)

        signature = (
            "table", dtype, block_shape, padding, swizzle, entry_size,
            tuple(signatures),
        )
        cached = self._entries.get(slot)
        if cached is not None and cached[0] == signature:
            return cached[1]
        device = getattr(storage, "device", None)
        if getattr(device, "type", None) != "cuda":
            raise RuntimeContractError(
                f"Tensor-map table {slot} requires CUDA storage, got {device}."
            )
        import torch
        import triton

        from triton.flagmega.runtime.nvidia_tensor_map import (
            encode_tma_descriptor_for_driver,
        )

        driver_utils = triton.runtime.driver.active.utils
        payload = bytearray()
        item_size = _dtype_item_size(dtype)
        for index, (base, shape, strides) in enumerate(prepared):
            encoded = encode_tma_descriptor_for_driver(
                driver_utils,
                int(base.data_ptr()),
                swizzle,
                item_size,
                host_dtype,
                block_shape,
                shape,
                strides,
                1 if padding == "nan" else 0,
            )
            if len(encoded) != entry_size:
                raise RuntimeContractError(
                    f"Tensor-map table {slot} entry {index} encoded to an "
                    f"invalid {len(encoded)}-byte payload."
                )
            payload.extend(encoded)
        table = torch.frombuffer(payload, dtype=torch.uint8).to(
            device=device, non_blocking=False
        )
        if int(table.data_ptr()) % self.tensor_map_alignment_bytes:
            raise RuntimeContractError(
                f"Tensor-map table {slot} device address is not "
                f"{self.tensor_map_alignment_bytes}-byte aligned."
            )
        self._entries[slot] = (signature, table)
        return table

    def _prepare_base(
        self,
        slot,
        storage,
        *,
        offset_bytes,
        dtype,
        shape,
        strides,
        block_shape,
        padding,
    ):
        if not 1 <= len(shape) <= 5:
            raise RuntimeContractError(
                f"Host tensor descriptor {slot} rank must be in [1, 5]."
            )
        if len(strides) != len(shape) or len(block_shape) != len(shape):
            raise RuntimeContractError(
                f"Host tensor descriptor {slot} shape/stride/block ranks differ."
            )
        if any(value <= 0 for value in (*shape, *strides, *block_shape)):
            raise RuntimeContractError(
                f"Host tensor descriptor {slot} extents and strides must be positive."
            )
        if strides[-1] != 1:
            raise RuntimeContractError(
                f"Host tensor descriptor {slot} last dimension is not contiguous."
            )
        item_size = _dtype_item_size(dtype)
        if any(
            (stride * item_size) % self.descriptor_alignment_bytes
            for stride in strides[:-1]
        ):
            raise RuntimeContractError(
                f"Host tensor descriptor {slot} outer stride is not "
                f"{self.descriptor_alignment_bytes}-byte aligned."
            )
        block_elements = 1
        for extent in block_shape:
            if extent & (extent - 1):
                raise RuntimeContractError(
                    f"Host tensor descriptor {slot} block extents must be powers of two."
                )
            block_elements *= extent
        if block_elements > self.maximum_block_elements:
            raise RuntimeContractError(
                f"Host tensor descriptor {slot} block exceeds the Triton limit."
            )
        if offset_bytes < 0 or padding not in {"zero", "nan"}:
            raise RuntimeContractError(
                f"Host tensor descriptor {slot} has invalid offset/padding."
            )
        if not hasattr(storage, "data_ptr") or not hasattr(storage, "device"):
            raise RuntimeContractError(
                f"Host tensor descriptor {slot} expects tensor storage."
            )
        span = 1 + sum(
            (extent - 1) * stride for extent, stride in zip(shape, strides)
        )
        base = _view_typed_buffer(storage, offset_bytes, span * item_size, dtype)
        if int(base.data_ptr()) % self.descriptor_alignment_bytes:
            raise RuntimeContractError(
                f"Host tensor descriptor {slot} base is not "
                f"{self.descriptor_alignment_bytes}-byte aligned."
            )
        if padding == "nan" and not base.dtype.is_floating_point:
            raise RuntimeContractError(
                f"Host tensor descriptor {slot} cannot NaN-pad non-floating data."
            )
        return base, (
            int(storage.data_ptr()), str(storage.device),
            str(getattr(storage, "dtype", "")), int(offset_bytes), dtype,
            shape, strides, block_shape, padding,
        )

    @staticmethod
    def _resolve_shape(slot, storage, static_shape, source_shape_axes):
        if len(static_shape) != len(source_shape_axes):
            raise RuntimeContractError(
                f"Host tensor descriptor {slot} static/source ranks differ."
            )
        storage_shape = tuple(int(value) for value in getattr(storage, "shape", ()))
        used = set()
        result = []
        for descriptor_axis, (static_extent, source_axes) in enumerate(
            zip(static_shape, source_shape_axes)
        ):
            extent = static_extent
            if source_axes:
                extent = 1
                for source_axis in source_axes:
                    if not 0 <= source_axis < len(storage_shape) or source_axis in used:
                        raise RuntimeContractError(
                            f"Host tensor descriptor {slot} has invalid/reused "
                            f"source axis {source_axis} at axis {descriptor_axis}."
                        )
                    used.add(source_axis)
                    extent *= storage_shape[source_axis]
            if extent <= 0:
                raise RuntimeContractError(
                    f"Host tensor descriptor {slot} resolved non-positive shape."
                )
            result.append(extent)
        return tuple(result)


def _require_exact_fields(value, expected, owner):
    fields = frozenset(value)
    if fields != expected:
        raise RuntimeContractError(
            f"{owner} fields differ: missing={sorted(expected - fields)}, "
            f"unexpected={sorted(fields - expected)}."
        )


def _validate_device_encoding(spec):
    block = spec.get("block_shape", ())
    box = spec.get("box_shape", ())
    swizzle = spec.get("swizzle_mode")
    if (not isinstance(block, (tuple, list)) or not block
            or not isinstance(box, (tuple, list)) or len(box) != len(block)
            or isinstance(swizzle, bool) or not isinstance(swizzle, int)
            or swizzle not in range(4)
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 or value & (value - 1) for value in (*block, *box))):
        raise RuntimeContractError("Device tensor-map block/box/swizzle is invalid.")
    expected = tuple(min(value, 256) for value in block)
    if swizzle:
        item_size = _dtype_item_size(str(spec["dtype"]))
        width = (0, 32, 64, 128)[swizzle] // item_size
        if block[-1] < width:
            raise RuntimeContractError("Device tensor-map swizzle exceeds its tile.")
        expected = (*expected[:-1], width)
    if tuple(box) != expected:
        raise RuntimeContractError(
            f"Device tensor-map box {tuple(box)} does not match its tile: {expected}."
        )


def _torch_dtype(dtype: str):
    import torch

    values = {
        "uint8": torch.uint8,
        "uint16": getattr(torch, "uint16", None),
        "uint32": getattr(torch, "uint32", None),
        "int32": torch.int32,
        "uint64": getattr(torch, "uint64", None),
        "int64": torch.int64,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "float8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
        "float8e4m3fn": getattr(torch, "float8_e4m3fn", None),
        "float8_e5m2": getattr(torch, "float8_e5m2", None),
        "float8e5m2": getattr(torch, "float8_e5m2", None),
    }
    try:
        result = values[dtype]
    except KeyError as error:
        raise RuntimeContractError(
            f"Unsupported tensor descriptor dtype {dtype!r}."
        ) from error
    if result is None:
        raise RuntimeContractError(
            f"Installed torch does not support tensor descriptor dtype {dtype!r}."
        )
    return result


def _dtype_item_size(dtype: str) -> int:
    import torch

    return torch.empty((), dtype=_torch_dtype(dtype)).element_size()


def _view_typed_buffer(storage, offset_bytes: int, size_bytes: int, dtype: str):
    import torch

    if not isinstance(storage, torch.Tensor) or not storage.is_contiguous():
        raise RuntimeContractError(
            "Tensor descriptor storage must be a contiguous torch.Tensor."
        )
    torch_dtype = _torch_dtype(dtype)
    item_size = torch.empty((), dtype=torch_dtype).element_size()
    if offset_bytes % item_size or size_bytes % item_size:
        raise RuntimeContractError(
            "Tensor descriptor byte view is not aligned to its element type."
        )
    byte_storage = storage.reshape(-1).view(torch.uint8).reshape(-1)
    end = offset_bytes + size_bytes
    if end > byte_storage.numel():
        raise RuntimeContractError(
            f"Tensor descriptor byte view [{offset_bytes}, {end}) exceeds "
            f"storage size {byte_storage.numel()}."
        )
    return byte_storage.narrow(0, offset_bytes, size_bytes).view(torch_dtype)


__all__ = ["TensorDescriptorCache"]
