# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Bounded NumPy materialization for readonly representation recipes.

This backend intentionally operates on physical storage scalars: bfloat16 is
represented by ``uint16`` and float8 by ``uint8``. That makes reshape, pack,
permute, sharding and concatenation exact byte transformations without loading
PyTorch. Numeric operations do not opt in and fall back one recipe at a time to
the semantic Torch evaluator.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from types import MappingProxyType
from typing import Any

from triton.flagmega.errors import (
    ArtifactError,
    EvaluationError,
    NumpyMaterializationUnsupported,
)
from triton.flagmega.importer.checkpoint import Checkpoint, TensorByteRange
from triton.flagmega.ir import DType, DistributedType, IRModule, TensorType, VectorType, verify_module
from triton.flagmega.ir.ops.core import get_definition


class NumpyMaterializationContext:
    def __init__(self, module: IRModule, checkpoint: Checkpoint) -> None:
        try:
            import numpy
        except ImportError as error:
            raise EvaluationError("NumPy is required for readonly-data materialization.") from error
        self.numpy = numpy
        self.module = module
        self.checkpoint = checkpoint
        self.types = MappingProxyType({
            node.id: node.type
            for recipe in module.constant_recipes
            for node in recipe.nodes
        })

    def storage_dtype(self, dtype):
        scalar = dtype.elem_type if isinstance(dtype, VectorType) else dtype
        mapping = {
            DType.BOOL: self.numpy.bool_,
            DType.INT32: self.numpy.int32,
            DType.INT64: self.numpy.int64,
            DType.BFLOAT16: self.numpy.uint16,
            DType.FLOAT16: self.numpy.float16,
            DType.FLOAT32: self.numpy.float32,
            DType.FLOAT8_E4M3FN: self.numpy.uint8,
        }
        try:
            return self.numpy.dtype(mapping[scalar])
        except KeyError as error:  # pragma: no cover - closed DType today.
            raise EvaluationError(f"NumPy materializer does not support {scalar}.") from error

    def weight_value(self, node):
        key = str(node.attrs["key"])
        value_type = _tensor_type(node.type)
        shape = _physical_shape(value_type)
        byte_range_of = getattr(self.checkpoint, "tensor_byte_range", None)
        byte_range = None if byte_range_of is None else byte_range_of(key)
        if byte_range is not None:
            if not isinstance(byte_range, TensorByteRange):
                raise ArtifactError(
                    f"Checkpoint tensor_byte_range({key!r}) returned "
                    f"{type(byte_range).__name__}, expected TensorByteRange or None."
                )
            expected = _shape_product(shape) * value_type.dtype.itemsize // _lane_count(value_type)
            if byte_range.nbytes != expected:
                raise ArtifactError(
                    f"Checkpoint byte range for {key!r} has {byte_range.nbytes} bytes; "
                    f"constant recipe requires {expected}."
                )
            return self.numpy.memmap(
                byte_range.path,
                dtype=self.storage_dtype(value_type.dtype),
                mode="r",
                offset=byte_range.offset,
                shape=shape,
                order="C",
            )
        return self._tensor_to_numpy(self.checkpoint.load_tensor(key, device="cpu"), value_type)

    def _tensor_to_numpy(self, value, value_type: TensorType):
        if isinstance(value, self.numpy.ndarray):
            result = value
        elif hasattr(value, "detach") and hasattr(value, "view"):
            detached = value.detach().cpu().contiguous()
            scalar = value_type.dtype.elem_type if isinstance(value_type.dtype, VectorType) else value_type.dtype
            if scalar in {DType.BFLOAT16, DType.FLOAT8_E4M3FN}:
                # NumPy 1.x has no native BF16/FP8 contract. Reinterpret through
                # a torch integer dtype only for in-memory checkpoints; local
                # safetensors use the mmap path above and never import torch.
                import torch

                storage = torch.uint16 if scalar == DType.BFLOAT16 else torch.uint8
                result = detached.view(storage).numpy()
            else:
                result = detached.numpy()
        else:
            raise EvaluationError(
                f"Checkpoint returned {type(value).__name__}; NumPy materialization "
                "requires an ndarray or CPU tensor."
            )
        shape = _physical_shape(value_type)
        if tuple(result.shape) != shape:
            if int(result.size) != _shape_product(shape):
                raise EvaluationError(
                    f"Checkpoint tensor has shape {tuple(result.shape)}, expected {shape}."
                )
            result = result.reshape(shape)
        return self.as_contiguous(result)

    def as_contiguous(self, value):
        return self.numpy.ascontiguousarray(value)

    def pack(self, value, outer_rank, lanes, axes, old_lanes=()):
        from collections import defaultdict
        from triton.flagmega.ir.ops.tensors.pack import normalize_axes

        normalized = normalize_axes(axes, outer_rank)
        by_axis = defaultdict(list)
        for lane_index, (axis, lane) in enumerate(zip(normalized, lanes)):
            by_axis[axis].append((lane_index, lane))
        split_shape = []
        outer_positions = []
        lane_positions = {}
        cursor = 0
        for axis in range(outer_rank):
            factors = by_axis.get(axis, ())
            product = _shape_product(tuple(lane for _, lane in factors))
            extent = int(value.shape[axis])
            if extent % product:
                raise EvaluationError(
                    f"NumPy pack axis {axis} extent {extent} is not divisible by {product}."
                )
            split_shape.append(extent // product)
            outer_positions.append(cursor)
            cursor += 1
            for lane_index, lane in factors:
                split_shape.append(int(lane))
                lane_positions[lane_index] = cursor
                cursor += 1
        old_positions = tuple(range(cursor, cursor + len(old_lanes)))
        split_shape.extend(int(lane) for lane in old_lanes)
        split = value.reshape(tuple(split_shape))
        permutation = (
            *outer_positions,
            *(lane_positions[index] for index in range(len(lanes))),
            *old_positions,
        )
        return self.as_contiguous(split.transpose(permutation))

    def full(self, value_type, value):
        tensor = _tensor_type(value_type)
        shape = _physical_shape(tensor)
        scalar = tensor.dtype.elem_type if isinstance(tensor.dtype, VectorType) else tensor.dtype
        if scalar == DType.BFLOAT16:
            # Round-to-nearest-even exactly as a float32 -> bfloat16 cast, but
            # retain the resulting storage bits in uint16.
            bits = self.numpy.asarray(value, dtype=self.numpy.float32).view(self.numpy.uint32)
            rounded = bits + self.numpy.uint32(0x7FFF) + ((bits >> 16) & 1)
            encoded = self.numpy.uint16(rounded >> 16)
            return self.numpy.full(shape, encoded, dtype=self.numpy.uint16)
        if scalar == DType.FLOAT8_E4M3FN:
            raise NumpyMaterializationUnsupported(
                "NumPy float8 splat encoding is not implemented."
            )
        return self.numpy.full(shape, value, dtype=self.storage_dtype(tensor.dtype))


def iter_numpy_materialized_constant_assets(
    module: IRModule,
    checkpoint: Checkpoint,
    *,
    outputs: Iterable[str] | None = None,
) -> Iterator[tuple[str, Any]]:
    """Materialize each requested recipe with bounded storage and lazy fallback."""

    verify_module(module)
    requested = None if outputs is None else frozenset(str(value) for value in outputs)
    available = {output for recipe in module.constant_recipes for output in recipe.outputs}
    if requested is not None and not requested <= available:
        missing = ", ".join(sorted(requested - available))
        raise EvaluationError(f"Unknown constant recipe outputs: {missing}.")
    torch_resolver = None
    for recipe in module.constant_recipes:
        selected = (
            recipe.outputs
            if requested is None
            else tuple(output for output in recipe.outputs if output in requested)
        )
        if not selected:
            continue
        context = NumpyMaterializationContext(module, checkpoint)
        local = {}
        supported = True
        for node in recipe.nodes:
            definition = get_definition(node.op)
            if not definition.numpy_materializable:
                supported = False
                break
            arguments = tuple(local[value] for value in node.inputs)
            try:
                result = definition.materialize_numpy(node, arguments, context)
            except NumpyMaterializationUnsupported:
                supported = False
                break
            _validate_storage_value(node, result, context)
            local[node.id] = result
        if supported:
            values = tuple(local[output] for output in recipe.outputs)
        else:
            # Importing Torch is intentionally delayed until the first numeric
            # recipe that needs semantic arithmetic rather than byte layout.
            from triton.flagmega.evaluator.torch_backend import (
                CheckpointWeightResolver,
                materialize_constant_recipe,
            )

            if torch_resolver is None:
                torch_resolver = CheckpointWeightResolver(checkpoint, cache=False)
            values = materialize_constant_recipe(module, recipe, torch_resolver)
        by_output = dict(zip(recipe.outputs, values, strict=True))
        for output in selected:
            yield output, by_output[output]


def _validate_storage_value(node, value, context: NumpyMaterializationContext) -> None:
    tensor = _tensor_type(node.type)
    expected_shape = _physical_shape(tensor)
    if not isinstance(value, context.numpy.ndarray):
        raise EvaluationError(
            f"NumPy materializer for {node.op!r} returned {type(value).__name__}.",
            node_id=node.id,
        )
    if tuple(value.shape) != expected_shape:
        raise EvaluationError(
            f"NumPy materializer value {node.id!r} has shape {tuple(value.shape)}, "
            f"expected {expected_shape}.",
            node_id=node.id,
        )
    expected_dtype = context.storage_dtype(tensor.dtype)
    if value.dtype != expected_dtype:
        raise EvaluationError(
            f"NumPy materializer value {node.id!r} has storage dtype {value.dtype}, "
            f"expected {expected_dtype}.",
            node_id=node.id,
        )


def _tensor_type(value_type) -> TensorType:
    tensor = value_type.tensor if isinstance(value_type, DistributedType) else value_type
    if not isinstance(tensor, TensorType):
        raise EvaluationError(f"NumPy constant materialization requires a tensor, got {tensor!r}.")
    return tensor


def _physical_shape(value_type: TensorType) -> tuple[int, ...]:
    if any(not dimension.is_fixed for dimension in value_type.shape):
        raise EvaluationError("NumPy constant materialization requires fixed tensor dimensions.")
    lanes = value_type.dtype.lanes if isinstance(value_type.dtype, VectorType) else ()
    return (*tuple(dimension.fixed_value for dimension in value_type.shape), *lanes)


def _shape_product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _lane_count(value_type: TensorType) -> int:
    return value_type.dtype.lane_count if isinstance(value_type.dtype, VectorType) else 1


__all__ = ["NumpyMaterializationContext", "iter_numpy_materialized_constant_assets"]
