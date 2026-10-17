# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Deterministic readonly-data image packing for bufferized FlagMega IR."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import struct
import errno
import re
from threading import Condition, Event, Lock
from typing import Mapping
from itertools import product

try:
    import fcntl
except ImportError:  # Range-copy assembly is also available without Unix ioctls.
    fcntl = None

from triton.flagmega.errors import ArtifactError
from triton.flagmega.importer.checkpoint import Checkpoint, TensorByteRange
from triton.flagmega.evaluator import (
    iter_numpy_materialized_constant_assets,
)
from triton.flagmega.evaluator.numpy_backend import NumpyStorageReader, materialize_numpy_recipe
from triton.flagmega.ir import IRModule, verify_buffer_plan, verify_module
from triton.flagmega.ir import DistributedBufferStorageKind, VectorType
from triton.flagmega.ir.distributed_type import (
    block_owner_axes,
    chip_owner_axes,
    launch_chip_count,
)
from triton.flagmega.ir.local_shard import local_shard_descriptor, unravel_placement_index
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.ir.types import data_type_to_data
from triton.flagmega.artifacts.manifest import hash_file, write_json_atomic
from triton.flagmega.artifacts.source_layout import (
    SOURCE_LAYOUT_SCHEMA, SOURCE_LAYOUT_DIGEST, prepare_programs,
    program_ranges, content_digest, validate_source_layout,
)


RDATA_INDEX_SCHEMA = "flagmega.rdata-index/v4"
RDATA_INDEX_SCHEMA_V3 = "flagmega.rdata-index/v3"
RDATA_INDEX_SCHEMA_V2 = "flagmega.rdata-index/v2"
# v1 lacks ``segment_sha256``.  Such an image can only be verified whole.
RDATA_INDEX_SCHEMA_V1 = "flagmega.rdata-index/v1"
_SUPPORTED_RDATA_INDEX_SCHEMAS = frozenset({SOURCE_LAYOUT_SCHEMA, RDATA_INDEX_SCHEMA, RDATA_INDEX_SCHEMA_V3, RDATA_INDEX_SCHEMA_V2, RDATA_INDEX_SCHEMA_V1})
RDATA_SEGMENT_DIGEST_KIND = "sha256-segments/v1"
RDATA_CHUNK_DIGEST_KIND = "sha256-chunks/v1"
RDATA_COMPOSITE_DIGEST_KINDS = frozenset({RDATA_SEGMENT_DIGEST_KIND, RDATA_CHUNK_DIGEST_KIND, SOURCE_LAYOUT_DIGEST})
RDATA_CACHE_SCHEMA = "flagmega.rdata-cache/v2"
# Writer scheduling, mappings and digest metadata do not change packed bytes.
# Keep the physical materializer ABI stable so verified v3 cache images remain
# reusable; the reader independently versions and validates index formats.
RDATA_MATERIALIZER_ABI = "flagmega.readonly-materialization/v2"
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_ZERO_HASH_CHUNK = bytes(_HASH_CHUNK_BYTES)
_FICLONE = 0x40049409
_FICLONERANGE = 0x4020940D

def _aliases_rdata_owner(buffer) -> bool:
    """True when ``buffer`` is a view over bytes another rdata buffer owns.

    A view carries its owner's ``weight_key`` so address resolution can find
    the tensor behind it, but it holds no storage of its own and must not be
    packed -- the owner already contributes those bytes.
    """
    alias = getattr(buffer, "alias", None)
    if alias is None:
        return False
    from triton.flagmega.ir.bufferization.alias import AliasKind

    return alias.kind == AliasKind.VIEW and bool(alias.source)


def _source_tensor(checkpoint, buffer, module=None):
    """Load a readonly buffer's contents from its checkpoint source(s).

    A parameter-bank field whose program fuses several tensors stores one
    buffer per row, so the sources are assembled here in the order that
    program concatenates them. The layout work itself stays offline.

    That program joins the sources along the trailing axis of each owner's
    already-sharded tile, so one owner holds its share of q, then of k, then
    of v -- not all of q followed by all of k. Both orders hold the same
    elements, which is why a permutation-invariant comparison cannot tell them
    apart, yet every consumer of the wrong order reads mismatched weights.
    """
    keys = getattr(buffer, "fused_weight_keys", ()) or (buffer.weight_key,)
    if len(keys) == 1:
        return checkpoint.load_tensor(keys[0], device="cpu")
    if module is None:
        raise ArtifactError(
            f"Fused readonly buffer {buffer.id!r} cannot be assembled without "
            "the module that declares its parameter-bank field program."
        )
    return _replay_bank_field(checkpoint, buffer, module)


def _replay_bank_field(checkpoint, buffer, module):
    """Assemble a fused buffer by replaying its bank field's own program.

    The program packs lanes, permutes the N and K axes and takes each owner's
    shard before concatenating, so the packed bytes are not a row split of the
    sources. Replaying it reuses the one authoritative implementation instead
    of restating those semantics here, where any restatement would be a
    different permutation of the same elements -- indistinguishable to a check
    that only compares element multisets.
    """
    from triton.flagmega.evaluator.context import EvaluationContext
    from triton.flagmega.evaluator.torch_backend import CheckpointWeightResolver
    from triton.flagmega.ir.ops.builtin.parameter_bank import ParameterBankValue

    group = str(buffer.rdata_group or buffer.source_node or "")
    node = next(
        (
            value
            for value in module.nodes
            if value.attrs.get("schema") is not None and group.startswith(f"{value.id}.")
        ),
        None,
    )
    if node is None:
        raise ArtifactError(
            f"Fused readonly buffer {buffer.id!r} names no parameter bank."
        )
    name = group[len(node.id) + 1:]
    index = next(
        (
            ordinal
            for ordinal, field in enumerate(node.attrs["schema"].fields)
            if field.name == name
        ),
        None,
    )
    if index is None:
        raise ArtifactError(
            f"Fused readonly buffer {buffer.id!r} matches no parameter-bank "
            f"field (looked for {name!r})."
        )
    import torch

    context = EvaluationContext(
        module,
        torch=torch,
        inputs={},
        weights=CheckpointWeightResolver(checkpoint, device="cpu"),
        constant_assets={},
    )
    return ParameterBankValue(node, context).row(int(buffer.group_index or 0))[index]


def pack_rdata(
    module: IRModule,
    checkpoint: Checkpoint,
    output_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> dict[str, object]:
    """Pack readonly data, optionally reusing a trusted local build cache."""

    verify_module(module)
    if cache_dir is not None:
        cache_key = _rdata_cache_key(module, checkpoint)
        if cache_key is not None:
            return _pack_rdata_cached(
                module,
                checkpoint,
                output_dir,
                Path(cache_dir),
                cache_key,
            )
    return _pack_rdata_uncached(module, checkpoint, output_dir)


def _pack_rdata_uncached(
    module: IRModule,
    checkpoint: Checkpoint,
    output_dir: str | Path,
) -> dict[str, object]:
    plan = verify_buffer_plan(module)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    # A view denotes bytes another buffer owns, so it names that buffer's
    # checkpoint tensor without claiming it. Packing a view double-counts the
    # tensor and trips the uniqueness check below.
    buffers = [
        buffer
        for buffer in plan.buffers
        if buffer.storage == "rdata" and not _aliases_rdata_owner(buffer)
    ]
    by_key = {}
    seen_keys: set[str] = set()
    for buffer in buffers:
        if not buffer.weight_key:
            raise ArtifactError(f"Rdata buffer {buffer.id!r} does not name a checkpoint tensor.")
        if buffer.weight_key in seen_keys:
            raise ArtifactError(f"Checkpoint tensor {buffer.weight_key!r} appears more than once in the rdata plan.")
        seen_keys.add(buffer.weight_key)
        by_key[buffer.weight_key] = buffer

    image_path = destination / "rdata.bin"
    index_path = destination / "rdata.index.json"
    temporary_image = image_path.with_name(f".{image_path.name}.{os.getpid()}.tmp")
    entries = []
    recipe_output_order = tuple(
        output
        for recipe in module.constant_recipes
        for output in recipe.outputs
    )
    recipe_outputs = frozenset(recipe_output_order)
    requested_outputs = recipe_outputs & set(by_key)
    direct_ranges: dict[str, TensorByteRange] = {}
    tensor_byte_range = getattr(checkpoint, "tensor_byte_range", None)
    if tensor_byte_range is not None:
        for output in recipe_output_order:
            if output not in requested_outputs:
                continue
            source_key = _checkpoint_storage_key(module, output)
            if source_key is None:
                continue
            byte_range = tensor_byte_range(source_key)
            if byte_range is not None:
                if not isinstance(byte_range, TensorByteRange):
                    raise ArtifactError(
                        f"Checkpoint tensor_byte_range({source_key!r}) returned "
                        f"{type(byte_range).__name__}, expected TensorByteRange or None."
                    )
                direct_ranges[output] = byte_range
    evaluated_outputs = requested_outputs - set(direct_ranges)
    ordered_buffers = sorted(
        buffers,
        key=lambda value: (value.offset, value.id),
    )
    programs, sources = prepare_programs(module, checkpoint, ordered_buffers, destination)
    buffer_by_id = {buffer.id: buffer for buffer in ordered_buffers}
    virtual_outputs = {buffer_by_id[program["buffer"]].weight_key for program in programs}
    for program in programs:
        program["ranges"] = [list(pair) for pair in program_ranges(buffer_by_id[program["buffer"]])]
    evaluated_outputs -= virtual_outputs
    evaluation_order = tuple(
        output for output in recipe_output_order if output in evaluated_outputs
    )
    buffer_evaluation_order = tuple(
        buffer.weight_key
        for buffer in ordered_buffers
        if buffer.weight_key in evaluated_outputs
    )
    can_hash_while_writing = evaluation_order == buffer_evaluation_order
    chip_count = (
        launch_chip_count(module)
        if any(
            buffer.distributed_storage_kind
            is DistributedBufferStorageKind.COMPACT_PER_CHIP
            for buffer in buffers
        )
        else 1
    )
    segment_nbytes = plan.rdata_bytes
    image_nbytes = segment_nbytes * chip_count
    try:
        with temporary_image.open("w+b") as stream:
            # A newly truncated file defines alignment gaps as zero without a
            # model-sized Python bytearray.  Individual tensors are then written
            # directly to their verified physical ranges.
            stream.truncate(image_nbytes)
            payload_hash = None
            segment_chunks = None
            if chip_count > 1 or programs:
                entries, segment_chunks = _pack_chip_segments(
                    stream, module, checkpoint, ordered_buffers, direct_ranges,
                    evaluated_outputs, recipe_outputs, segment_nbytes, chip_count,
                    virtual_outputs=virtual_outputs,
                    virtual_ranges=[tuple(pair) for program in programs for pair in program["ranges"]],
                )
            elif can_hash_while_writing:
                entries, payload_hash = _pack_in_offset_order(
                    stream, module, checkpoint, ordered_buffers, direct_ranges,
                    evaluated_outputs, recipe_outputs, image_nbytes,
                )
            else:
                entries = _pack_with_post_hash(
                    stream, module, checkpoint, by_key, direct_ranges,
                    evaluated_outputs, recipe_outputs,
                )
            stream.flush()
            os.fsync(stream.fileno())

        entries.sort(key=lambda value: (int(value["offset"]), str(value["buffer"])))
        if programs:
            segment_hashes = None
        elif segment_chunks is not None:
            segment_hashes = [_chunk_segment_digest(chunks, segment_nbytes)
                              for chunks in segment_chunks]
        elif payload_hash is None:
            # Segment digests cover every payload byte, including padding.
            # A second whole-image chain and per-entry chains add no coverage.
            segment_hashes = _segment_digests(
                temporary_image, segment_nbytes, chip_count
            )
        else:
            # Written and hashed in offset order, which only happens for a
            # single-segment image: its segment digest is the image digest.
            segment_hashes = [payload_hash]
        os.replace(temporary_image, image_path)
    except BaseException:
        # Failed multi-gigabyte packs must not strand a hidden sparse image in
        # the artifact directory.  Preserve the original exception even if a
        # best-effort cleanup races with external filesystem activity.
        try:
            temporary_image.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    index = {
        "schema": SOURCE_LAYOUT_SCHEMA if programs else (RDATA_INDEX_SCHEMA if segment_chunks is not None else RDATA_INDEX_SCHEMA_V3),
        "image": image_path.name,
        "nbytes": image_nbytes,
        "segment_nbytes": segment_nbytes,
        "chip_count": chip_count,
        "alignment": plan.alignment,
        "digest_kind": SOURCE_LAYOUT_DIGEST if programs else (RDATA_CHUNK_DIGEST_KIND if segment_chunks is not None else "sha256"),
        "segment_sha256": segment_hashes,
        "entries": entries,
    }
    if segment_chunks is not None:
        index["segment_chunks"] = segment_chunks
    if programs:
        index.pop("segment_sha256")
        index.update(programs=programs, sources=sources)
    index["sha256"] = content_digest(index) if programs else _segment_content_digest(index)
    write_json_atomic(index_path, index)
    return index


def _pack_chip_segments(
    stream,
    module,
    checkpoint,
    buffers,
    direct_ranges,
    evaluated_outputs,
    recipe_outputs,
    segment_nbytes,
    chip_count,
    *,
    virtual_outputs=frozenset(),
    virtual_ranges=None,
):
    """Write one chip-local image segment per launch chip.

    Device/node-sharded readonly buffers are packed owner-major over block
    coordinates. Other buffers are replicated in each segment; this keeps the
    image ABI valid for mixed replicated and sharded constants.
    """
    entries_by_key = {}
    buffer_by_key = {buffer.weight_key: buffer for buffer in buffers}
    segment_chunks = [[] for _ in range(chip_count)]
    result_lock = Lock()
    workers = min(4, os.cpu_count() or 1)
    budget = _MaterializationBudget()
    storage_reader = NumpyStorageReader()
    failed = Event()
    image = _StripedImage(stream, segment_nbytes * chip_count, preallocate=not virtual_outputs)

    def write_source(buffer, source, writer_pool):
        sharded = buffer.distributed_storage_kind is DistributedBufferStorageKind.COMPACT_PER_CHIP
        array = _as_numpy_array(source, buffer) if sharded else _tensor_buffer(source)
        if not sharded and array.nbytes != buffer.nbytes:
            raise ArtifactError(
                f"Checkpoint tensor {buffer.weight_key!r} has {array.nbytes} bytes; "
                f"IR requires {buffer.nbytes}."
            )

        def write_chip(chip):
            writer = _MappedChunkWriter(image, chip * segment_nbytes)
            base = chip * segment_nbytes
            if sharded:
                _write_chip_sharded_buffer(writer, buffer, array, chip, base)
            else:
                writer.seek(base + buffer.offset)
                _write_all(writer, array)
            with result_lock:
                segment_chunks[chip].extend(writer.chunks)

        # Recipe and writer pools are separate: a recipe waiting for its chip
        # writes cannot occupy the workers needed to finish those writes.
        list(writer_pool.map(write_chip, range(chip_count)))
        with result_lock:
            entries_by_key[buffer.weight_key] = _rdata_entry(buffer)

    recipes = [recipe for recipe in module.constant_recipes
               if any(output in evaluated_outputs for output in recipe.outputs)]
    with image, ThreadPoolExecutor(max_workers=max(chip_count, workers),
                            thread_name_prefix="flagmega-rdata-write") as writer_pool:
        def materialize(recipe):
            if failed.is_set():
                return
            selected = [output for output in recipe.outputs if output in evaluated_outputs]
            cost = _recipe_storage_budget(recipe, [buffer_by_key[key] for key in selected], chip_count)
            try:
                with budget.reserve(cost):
                    if failed.is_set():
                        return
                    values = materialize_numpy_recipe(module, recipe, checkpoint, storage_reader=storage_reader)
                    for key, source in zip(recipe.outputs, values, strict=True):
                        if failed.is_set():
                            return
                        if key in evaluated_outputs:
                            write_source(buffer_by_key[key], source, writer_pool)
            except BaseException:
                failed.set()
                raise

        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="flagmega-rdata-recipe") as recipe_pool:
            # Closed recipes have no inter-recipe data dependencies. Physical
            # output ranges were verified before any worker starts.
            list(recipe_pool.map(materialize, recipes))

        for buffer in buffers:
            key = buffer.weight_key
            if key in entries_by_key:
                continue
            if key in virtual_outputs:
                entries_by_key[key] = _rdata_entry(buffer)
                continue
            direct = direct_ranges.get(key)
            if direct is not None:
                write_source(buffer, _range_numpy_array(buffer, direct), writer_pool)
            elif key in recipe_outputs:
                raise ArtifactError(f"Constant recipe output {key!r} was not materialized.")
            else:
                write_source(buffer, checkpoint.load_tensor(key, device="cpu"), writer_pool)
        image.assemble()
    if set(entries_by_key) != set(buffer_by_key):
        raise ArtifactError("Rdata packing did not cover every planned buffer.")
    if virtual_ranges is None:
        virtual_ranges = [pair for buffer in buffers if buffer.weight_key in virtual_outputs
                          for pair in program_ranges(buffer)]
    for chunks in segment_chunks:
        chunks.sort(key=lambda chunk: chunk["offset"])
        cursor = 0
        padding = []
        coverage = [(chunk["offset"], chunk["nbytes"]) for chunk in chunks]
        coverage.extend(virtual_ranges)
        for start, count in sorted(coverage):
            if start < cursor:
                raise ArtifactError("Rdata writes overlap in a chip segment.")
            if start > cursor:
                padding.extend(_zero_chunks(cursor, start - cursor))
            cursor = start + count
        if cursor < segment_nbytes:
            padding.extend(_zero_chunks(cursor, segment_nbytes - cursor))
        chunks.extend(padding)
        chunks.sort(key=lambda chunk: chunk["offset"])
    return list(entries_by_key.values()), segment_chunks


class _MaterializationBudget:
    """Bound concurrent recipe storage, even when one recipe exceeds the budget."""

    def __init__(self, nbytes=None):
        if nbytes is None:
            available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            try:
                for line in Path("/proc/meminfo").read_text().splitlines():
                    if line.startswith("MemAvailable:"):
                        available = int(line.split()[1]) * 1024
                        break
            except OSError:
                pass
            nbytes = min(128 * 1024**3, max(1, available // 4))
        self.nbytes = nbytes
        self.used = 0
        self.condition = Condition()

    @contextmanager
    def reserve(self, nbytes):
        charge = min(self.nbytes, max(1, nbytes))
        with self.condition:
            self.condition.wait_for(lambda: self.used + charge <= self.nbytes)
            self.used += charge
        try:
            yield
        finally:
            with self.condition:
                self.used -= charge
                self.condition.notify_all()


def _recipe_storage_budget(recipe, buffers, chip_count):
    from triton.flagmega.ir.ops.core import tensor_nbytes
    from triton.flagmega.ir.distributed_inference import tensor_of
    from triton.flagmega.ir.model import DistributedType, TensorType, TupleType
    def size(value_type):
        if isinstance(value_type, (TensorType, DistributedType)):
            return tensor_nbytes(tensor_of(value_type)) or 0
        if isinstance(value_type, TupleType):
            return sum(size(item) for item in value_type.fields)
        return 0
    views = {"builtin.weight", "distributed.sharded_view", "tensors.reshape", "tensors.bitcast"}
    intermediates = sum(size(node.type) for node in recipe.nodes if node.op not in views)
    # Each chip writer may hold a dense owner component while the recipe's
    # materialized intermediates remain live. Numeric Torch recipes also need
    # conversion storage; a factor of two accounts for that transient lifetime.
    pieces = sum(buffer.nbytes for buffer in buffers) * chip_count
    return max(1, 2 * intermediates + pieces)


def _write_chip_sharded_buffer(stream, buffer, array, chip, segment_base):
    distributed = buffer.distributed_type
    if distributed is None:
        raise ArtifactError(f"Chip-sharded rdata buffer {buffer.id!r} has no DistributedType.")
    chip_axes = chip_owner_axes(distributed)
    chip_hierarchy = tuple(distributed.placement.hierarchy[axis] for axis in chip_axes)
    chip_coordinates = unravel_placement_index(chip, chip_hierarchy) if chip_hierarchy else ()
    coordinates = [0] * distributed.placement.rank
    for axis, value in zip(chip_axes, chip_coordinates, strict=True):
        coordinates[axis] = value
    block_axes = block_owner_axes(distributed)
    block_hierarchy = tuple(distributed.placement.hierarchy[axis] for axis in block_axes)
    stride = buffer.component_stride_bytes
    for owner, block_coordinates in enumerate(product(*(range(v) for v in block_hierarchy))):
        owner_coordinates = list(coordinates)
        for axis, value in zip(block_axes, block_coordinates, strict=True):
            owner_coordinates[axis] = value
        descriptor = local_shard_descriptor(distributed, tuple(owner_coordinates))
        piece = _extract_local_shard(array, descriptor, buffer.component_shape)
        raw = _tensor_buffer(piece)
        if raw.nbytes != buffer.nbytes:
            raise ArtifactError(
                f"Chip-local shard for {buffer.weight_key!r} has {raw.nbytes} bytes; "
                f"IR requires {buffer.nbytes}."
            )
        stream.seek(segment_base + buffer.offset + owner * stride)
        _write_all(stream, raw)


def _extract_local_shard(array, descriptor, component_shape):
    import numpy

    result = array
    for axis_index, axis in enumerate(descriptor.axes):
        active = int(axis.active_extent.fixed_value)
        capacity = int(axis.local_capacity.fixed_value)
        window = axis.local_slice
        if window is not None:
            # A strided slice is a view: an unsplit axis costs nothing, and a
            # split axis pays one copy in the trailing ascontiguousarray rather
            # than a model-sized Python index list plus a fancy-index gather.
            selected = result[(slice(None), ) * axis_index + (window, )]
        else:
            indices = [
                int(axis.map_local_to_global(index).fixed_value)
                for index in range(active)
            ]
            selected = numpy.take(result, indices, axis=axis_index) if indices else numpy.take(
                result, [], axis=axis_index
            )
        if active != capacity:
            shape = list(selected.shape)
            shape[axis_index] = capacity
            padded = numpy.zeros(shape, dtype=selected.dtype)
            if active:
                slices = [slice(None)] * selected.ndim
                slices[axis_index] = slice(0, active)
                padded[tuple(slices)] = selected
            selected = padded
        result = selected
    expected = (*tuple(int(value) for value in component_shape), *tuple(array.shape[len(descriptor.axes):]))
    if tuple(result.shape) != expected:
        raise ArtifactError(
            f"Chip-local shard shape {tuple(result.shape)} does not match expected {expected}."
        )
    return numpy.ascontiguousarray(result)


def _as_numpy_array(value, buffer):
    import numpy

    dtype = buffer.dtype
    scalar = dtype.elem_type if isinstance(dtype, VectorType) else dtype
    storage_dtype = {
        "bool": numpy.bool_,
        "int32": numpy.int32,
        "int64": numpy.int64,
        "bfloat16": numpy.uint16,
        "float16": numpy.float16,
        "float32": numpy.float32,
        "float8_e4m3fn": numpy.uint8,
    }[scalar.value]
    raw = _tensor_buffer(value)
    shape = (*buffer.shape, *(dtype.lanes if isinstance(dtype, VectorType) else ()))
    return numpy.frombuffer(raw, dtype=storage_dtype).reshape(shape)


def _range_numpy_array(buffer, source):
    import numpy

    dtype = buffer.dtype
    scalar = dtype.elem_type if isinstance(dtype, VectorType) else dtype
    storage_dtype = {
        "bool": numpy.bool_,
        "int32": numpy.int32,
        "int64": numpy.int64,
        "bfloat16": numpy.uint16,
        "float16": numpy.float16,
        "float32": numpy.float32,
        "float8_e4m3fn": numpy.uint8,
    }[scalar.value]
    shape = (*buffer.shape, *(dtype.lanes if isinstance(dtype, VectorType) else ()))
    return numpy.memmap(source.path, dtype=storage_dtype, mode="r", offset=source.offset, shape=shape, order="C")


def _rdata_cache_key(module: IRModule, checkpoint: Checkpoint) -> str | None:
    plan = verify_buffer_plan(module)
    buffers = sorted(
        (value for value in plan.buffers if value.storage == "rdata"),
        key=lambda value: (value.offset, value.id),
    )
    recipe_by_output = {
        output: recipe
        for recipe in module.constant_recipes
        for output in recipe.outputs
    }
    checkpoint_keys: set[str] = set()
    recipe_fingerprints = []
    for buffer in buffers:
        key = buffer.weight_key
        if not key:
            return None
        recipe = recipe_by_output.get(key)
        if recipe is None:
            checkpoint_keys.add(key)
            continue
        recipe_fingerprints.append({
            "output": key,
            "fingerprint": recipe.fingerprint,
        })
        checkpoint_keys.update(
            str(node.attrs["key"])
            for node in recipe.nodes
            if node.op == "builtin.weight"
        )
    identity_of = getattr(checkpoint, "cache_identity", None)
    if not callable(identity_of):
        return None
    identity = identity_of(checkpoint_keys)
    if identity is None:
        return None
    key_data = {
        "schema": RDATA_CACHE_SCHEMA,
        "materializer": RDATA_MATERIALIZER_ABI,
        **({"source_layout": SOURCE_LAYOUT_DIGEST}
           if callable(getattr(checkpoint, "source_file_digest", None)) else {}),
        "alignment": plan.alignment,
        "nbytes": plan.rdata_bytes,
        "chip_count": launch_chip_count(module),
        "buffers": [
            {
                "id": buffer.id,
                "key": buffer.weight_key,
                "dtype": data_type_to_data(buffer.dtype),
                "shape": list(buffer.shape),
                "offset": buffer.offset,
                "nbytes": buffer.nbytes,
                "strides": list(buffer.strides),
                "distributed_type": None if buffer.distributed_type is None else buffer.distributed_type.to_data(),
                "storage_kind": buffer.distributed_storage_kind.value,
                "backing_type": (None if buffer.distributed_backing_type is None
                                 else buffer.distributed_backing_type.to_data()),
                "component_stride_bytes": buffer.component_stride_bytes,
            }
            for buffer in buffers
        ],
        "recipes": sorted(
            recipe_fingerprints,
            key=lambda value: (value["output"], value["fingerprint"]),
        ),
        "checkpoint": identity,
    }
    encoded = json.dumps(
        key_data,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pack_rdata_cached(
    module: IRModule,
    checkpoint: Checkpoint,
    output_dir: str | Path,
    cache_root: Path,
    cache_key: str,
) -> dict[str, object]:
    cache_root.mkdir(parents=True, exist_ok=True)
    entry = cache_root / cache_key
    index = _load_rdata_cache_entry(entry, cache_key)
    if index is None:
        staging = Path(tempfile.mkdtemp(
            prefix=f".{cache_key}.",
            dir=cache_root,
        ))
        try:
            index = _pack_rdata_uncached(module, checkpoint, staging)
            image = staging / "rdata.bin"
            index_path = staging / "rdata.index.json"
            manifest = {
                "schema": RDATA_CACHE_SCHEMA,
                "key": cache_key,
                "image_nbytes": image.stat().st_size,
                "image_sha256": index["sha256"],
                "index_sha256": hash_file(index_path),
            }
            write_json_atomic(staging / "cache.json", manifest)
            for path in (image, index_path, staging / "cache.json"):
                path.chmod(0o444)
            for source in index.get("sources", []):
                (staging / source["path"]).chmod(0o444)
            try:
                staging.rename(entry)
            except FileExistsError:
                shutil.rmtree(staging)
                index = _load_rdata_cache_entry(entry, cache_key)
                if index is None:
                    raise ArtifactError(
                        f"Concurrent rdata cache entry {entry} is invalid."
                    )
            else:
                staging = None
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _clone_file(entry / "rdata.bin", destination / "rdata.bin")
    _clone_file(entry / "rdata.index.json", destination / "rdata.index.json")
    for source in index.get("sources", []):
        target = destination / source["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        _clone_file(entry / source["path"], target)
    return dict(index)


def _load_rdata_cache_entry(
    entry: Path,
    cache_key: str,
) -> dict[str, object] | None:
    manifest_path = entry / "cache.json"
    index_path = entry / "rdata.index.json"
    image_path = entry / "rdata.bin"
    if not (manifest_path.is_file() and index_path.is_file() and image_path.is_file()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(manifest, Mapping)
        or not isinstance(index, dict)
        or manifest.get("schema") != RDATA_CACHE_SCHEMA
        or manifest.get("key") != cache_key
        or index.get("schema") not in _SUPPORTED_RDATA_INDEX_SCHEMAS
        or manifest.get("image_nbytes") != image_path.stat().st_size
        or manifest.get("image_nbytes") != index.get("nbytes")
        or manifest.get("image_sha256") != index.get("sha256")
        or manifest.get("index_sha256") != hash_file(index_path)
    ):
        return None
    if index.get("schema") == SOURCE_LAYOUT_SCHEMA:
        try:
            validate_source_layout(entry, index, None, segment=None, verify_bytes=False)
        except (ArtifactError, OSError, TypeError, ValueError):
            return None
    return index


def _clone_file(source: Path, destination: Path) -> None:
    """Atomically materialize an independent reflink, with copy fallback."""

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        try:
            import fcntl

            with temporary.open("w+b", buffering=0) as output_stream:
                with source.open("rb", buffering=0) as input_stream:
                    try:
                        fcntl.ioctl(output_stream.fileno(), _FICLONE, input_stream.fileno())
                    except OSError as error:
                        if error.errno != errno.EXDEV:
                            raise
                        backing = _overlay_clone_source(source, os.fstat(output_stream.fileno()).st_dev)
                        backing = source if backing is None else backing
                        alias = _bind_clone_destination(temporary, backing)
                        if alias is None:
                            raise
                        with backing.open("rb", buffering=0) as backing_stream, alias.open("r+b", buffering=0) as alias_stream:
                            fcntl.ioctl(alias_stream.fileno(), _FICLONE, backing_stream.fileno())
        except (ImportError, OSError):
            temporary.unlink(missing_ok=True)
            shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _overlay_clone_source(source: Path, destination_device: int, *, mountinfo=None) -> Path | None:
    """Find an exposed upper inode when OverlayFS obscures a same-FS clone.

    This is a filesystem capability check, independent of container paths. Only
    regular upper files with exactly the same inode, size and timestamps are
    accepted. Lower files, metacopy/redirect files and unavailable backing
    mounts retain the ordinary copy implementation.
    """
    if mountinfo is None:
        try:
            mountinfo = Path("/proc/self/mountinfo").read_text()
        except OSError:
            return None
    def unescape(value):
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)
    source = source.resolve()
    matches = []
    for line in mountinfo.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
            mountpoint = Path(unescape(fields[4]))
            relative = source.relative_to(mountpoint)
        except (ValueError, IndexError):
            continue
        matches.append((len(mountpoint.parts), fields, separator, relative))
    if not matches:
        return None
    _, fields, separator, relative = max(matches, key=lambda item: item[0])
    if fields[separator + 1] != "overlay" or fields[3] != "/":
        return None
    options = dict(part.split("=", 1) for part in fields[separator + 3].split(",") if "=" in part)
    upper = options.get("upperdir")
    if upper is None:
        return None
    candidate = Path(unescape(upper)) / relative
    try:
        actual, underlying = source.stat(), candidate.stat()
        if (not candidate.is_file() or underlying.st_dev != destination_device
                or (actual.st_ino, actual.st_size, actual.st_mtime_ns, actual.st_ctime_ns)
                != (underlying.st_ino, underlying.st_size, underlying.st_mtime_ns, underlying.st_ctime_ns)
                or any(name.endswith(("overlay.metacopy", "overlay.redirect")) for name in os.listxattr(candidate))):
            return None
    except OSError:
        return None
    return candidate


def _bind_clone_destination(destination, source, *, mountinfo=None):
    """Open bind-mounted files through the source mount for VFS remap legality."""
    if mountinfo is None:
        try:
            mountinfo = Path("/proc/self/mountinfo").read_text()
        except OSError:
            return None
    records = []
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        decode = lambda value: re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)
        records.append((fields[2], Path(decode(fields[3])), Path(decode(fields[4]))))
    def containing(path):
        # Canonicalize the queried path once, rather than once per mount.
        # Large container mount tables made every source snapshot repeat
        # hundreds of filesystem walks without adding any identity proof.
        path = path.resolve()
        matches = []
        for record in records:
            try:
                relative = path.relative_to(record[2])
            except ValueError:
                continue
            matches.append((len(record[2].parts), record, relative))
        return max(matches, key=lambda item: item[0]) if matches else None
    source_mount, destination_mount = containing(source), containing(destination)
    if source_mount is None or destination_mount is None or source_mount[1][0] != destination_mount[1][0]:
        return None
    _, src, _ = source_mount
    _, dst, relative = destination_mount
    try:
        alias = src[2] / (dst[1] / relative).relative_to(src[1])
        if not alias.samefile(destination):
            return None
    except (ValueError, OSError):
        return None
    return alias


def _pack_in_offset_order(
    stream,
    module,
    checkpoint,
    buffers,
    direct_ranges,
    evaluated_outputs,
    recipe_outputs,
    image_nbytes,
):
    materialized = iter_numpy_materialized_constant_assets(
        module,
        checkpoint,
        outputs=evaluated_outputs,
    )
    image_hasher = hashlib.sha256()
    entries = []
    cursor = 0
    for buffer in buffers:
        if buffer.offset < cursor:  # pragma: no cover - buffer verifier owns this.
            raise ArtifactError(f"Rdata buffer {buffer.id!r} overlaps a prior buffer.")
        _hash_zeros(image_hasher, buffer.offset - cursor)
        key = buffer.weight_key
        if key in direct_ranges:
            entry = _write_rdata_range(
                stream, buffer, direct_ranges[key], image_hasher=image_hasher
            )
        elif key in evaluated_outputs:
            try:
                materialized_key, tensor = next(materialized)
            except StopIteration as error:
                raise ArtifactError(
                    f"Constant recipe output {key!r} was not materialized."
                ) from error
            if materialized_key != key:
                raise ArtifactError(
                    f"Constant recipe materialization order changed: expected {key!r}, "
                    f"got {materialized_key!r}."
                )
            entry = _write_rdata_entry(
                stream, buffer, tensor, image_hasher=image_hasher
            )
        elif key in recipe_outputs:
            raise ArtifactError(f"Constant recipe output {key!r} was not materialized.")
        else:
            tensor = _source_tensor(checkpoint, buffer, module)
            entry = _write_rdata_entry(
                stream, buffer, tensor, image_hasher=image_hasher
            )
        entries.append(entry)
        cursor = buffer.offset + buffer.nbytes
    try:
        unexpected, _ = next(materialized)
    except StopIteration:
        pass
    else:  # pragma: no cover - ordering equality proves exact coverage.
        raise ArtifactError(f"Unexpected materialized constant output {unexpected!r}.")
    _hash_zeros(image_hasher, image_nbytes - cursor)
    return entries, image_hasher.hexdigest()


def _pack_with_post_hash(
    stream,
    module,
    checkpoint,
    by_key,
    direct_ranges,
    evaluated_outputs,
    recipe_outputs,
):
    entries = []
    written: set[str] = set()
    for key, tensor in iter_numpy_materialized_constant_assets(
        module,
        checkpoint,
        outputs=evaluated_outputs,
    ):
        entries.append(_write_rdata_entry(stream, by_key[key], tensor))
        written.add(key)
    for key, byte_range in direct_ranges.items():
        entries.append(_write_rdata_range(stream, by_key[key], byte_range))
        written.add(key)
    for key, buffer in by_key.items():
        if key in written:
            continue
        if key in recipe_outputs:
            raise ArtifactError(f"Constant recipe output {key!r} was not materialized.")
        entries.append(
            _write_rdata_entry(stream, buffer, _source_tensor(checkpoint, buffer, module))
        )
    return entries


def _write_rdata_entry(stream, buffer, tensor, *, image_hasher=None) -> dict[str, object]:
    raw = _tensor_buffer(tensor)
    if raw.nbytes != buffer.nbytes:
        raise ArtifactError(
            f"Checkpoint tensor {buffer.weight_key!r} has {raw.nbytes} bytes; "
            f"IR requires {buffer.nbytes}."
        )
    stream.seek(buffer.offset)
    _write_all(stream, raw)
    if image_hasher is not None:
        image_hasher.update(raw)
    return _rdata_entry(buffer)


def _write_rdata_range(
    stream,
    buffer,
    source: TensorByteRange,
    *,
    image_hasher=None,
) -> dict[str, object]:
    if source.nbytes != buffer.nbytes:
        raise ArtifactError(
            f"Checkpoint byte range for {buffer.weight_key!r} has {source.nbytes} bytes; "
            f"IR requires {buffer.nbytes}."
        )
    remaining = source.nbytes
    scratch = bytearray(min(_HASH_CHUNK_BYTES, max(remaining, 1)))
    scratch_view = memoryview(scratch)
    stream.seek(buffer.offset)
    try:
        with source.path.open("rb", buffering=0) as input_stream:
            input_stream.seek(source.offset)
            while remaining:
                requested = min(remaining, len(scratch))
                count = input_stream.readinto(scratch_view[:requested])
                if count is None or count <= 0:
                    raise ArtifactError(
                        f"Checkpoint byte range for {buffer.weight_key!r} is truncated."
                    )
                chunk = scratch_view[:count]
                _write_all(stream, chunk)
                if image_hasher is not None:
                    image_hasher.update(chunk)
                remaining -= count
    except ArtifactError:
        raise
    except OSError as error:
        raise ArtifactError(
            f"Cannot stream checkpoint bytes for {buffer.weight_key!r} from "
            f"{source.path}: {error}."
        ) from error
    return _rdata_entry(buffer)


def _hash_zeros(hasher, nbytes: int) -> None:
    remaining = nbytes
    while remaining:
        count = min(remaining, len(_ZERO_HASH_CHUNK))
        hasher.update(memoryview(_ZERO_HASH_CHUNK)[:count])
        remaining -= count


def _write_all(stream, raw: memoryview) -> None:
    written = 0
    while written < raw.nbytes:
        count = stream.write(raw[written:])
        if count is None or count <= 0:
            raise ArtifactError("Rdata image write made no forward progress.")
        written += count


class _MappedChunkWriter:
    """Hash the same bounded, immutable bytes passed to positioned writes."""

    def __init__(self, image, segment_base):
        self.image = image
        self.position = 0
        self.segment_base = segment_base
        self.chunks = []

    def seek(self, position):
        self.position = position

    def write(self, raw):
        for start in range(0, raw.nbytes, _HASH_CHUNK_BYTES):
            chunk = raw[start:start + _HASH_CHUNK_BYTES]
            offset = self.position - self.segment_base
            # A positioned syscall still takes the inode's exclusive write
            # lock. Disjoint mapped ranges allow concurrent C-level copies;
            # fsync at the producer boundary retains durable publication.
            self.image.write_at(self.position, chunk)
            self.position += chunk.nbytes
            self.chunks.append({"offset": offset, "nbytes": chunk.nbytes,
                                "sha256": hashlib.sha256(chunk).hexdigest()})
        return raw.nbytes


class _StripedImage:
    """Construct disjoint physical extents without one inode's allocation lock.

    Extents are joined by reflink range cloning: assembly changes ownership
    metadata and reads no payload. Source maps live until all writers join.
    """

    def __init__(self, stream, nbytes, *, preallocate=True):
        self.stream = stream
        self.nbytes = nbytes
        self.preallocate = preallocate
        self.part_nbytes = max(256 * 1024**2, ((nbytes + 128 * 4096 - 1) // (128 * 4096)) * 4096)
        self.directory = tempfile.TemporaryDirectory(prefix=".rdata-extents-", dir=Path(stream.name).parent)
        self.parts = {}
        self.lock = Lock()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for path, array in self.parts.values():
            array._mmap.close()
        self.parts.clear()
        self.directory.cleanup()

    def write_at(self, position, raw):
        import numpy
        written = 0
        while written < raw.nbytes:
            part, offset = divmod(position + written, self.part_nbytes)
            value = self.parts.get(part)
            if value is None:
                with self.lock:
                    value = self.parts.get(part)
                    if value is None:
                        path = Path(self.directory.name) / f"{part}.bin"
                        size = min(self.part_nbytes, self.nbytes - part * self.part_nbytes)
                        # Page-aligned physical extents also allow the final
                        # partial image page to participate in range cloning.
                        size = ((size + 4095) // 4096) * 4096
                        with path.open("w+b") as extent:
                            if self.preallocate and hasattr(os, "posix_fallocate"):
                                os.posix_fallocate(extent.fileno(), 0, size)
                            else:
                                extent.truncate(size)
                            array = numpy.memmap(extent, dtype=numpy.uint8, mode="r+", shape=(size,))
                        value = path, array
                        self.parts[part] = value
            count = min(raw.nbytes - written, self.part_nbytes - offset)
            numpy.copyto(value[1][offset:offset + count],
                         numpy.frombuffer(raw[written:written + count], dtype=numpy.uint8))
            written += count

    def assemble(self):
        # Reflink range cloning flushes source pages before publishing shared
        # extents. The artifact boundary fsyncs the assembled destination.
        for part, (path, array) in sorted(self.parts.items()):
            with path.open("rb") as source:
                request = struct.pack("qQQQ", source.fileno(), 0, array.nbytes, part * self.part_nbytes)
                try:
                    if fcntl is None:
                        raise OSError(errno.EOPNOTSUPP, "shared extent cloning is unavailable")
                    fcntl.ioctl(self.stream.fileno(), _FICLONERANGE, request)
                except OSError as error:
                    if error.errno not in {errno.EOPNOTSUPP, errno.ENOTTY, errno.EXDEV}:
                        raise
                    # Filesystems without shared extents use the same layout
                    # producer and a bounded range-copy assembly backend.
                    # Materialization, byte coverage and digest checks are
                    # identical; only the filesystem's copy primitive differs.
                    self.stream.seek(part * self.part_nbytes)
                    while chunk := source.read(_HASH_CHUNK_BYTES):
                        _write_all(self.stream, memoryview(chunk))
            # Writers have joined. Consume each staging extent as soon as its
            # bytes belong to the image, so copy assembly needs one extra extent
            # instead of retaining a second complete image until context exit.
            array._mmap.close()
            path.unlink()
            del self.parts[part]
        self.stream.truncate(self.nbytes)


def _zero_chunks(offset, nbytes):
    for start in range(0, nbytes, _HASH_CHUNK_BYTES):
        count = min(nbytes - start, _HASH_CHUNK_BYTES)
        yield {"offset": offset + start, "nbytes": count,
               "sha256": hashlib.sha256(memoryview(_ZERO_HASH_CHUNK)[:count]).hexdigest()}


def _rdata_entry(buffer) -> dict[str, object]:
    return {
        "buffer": buffer.id,
        "key": buffer.weight_key,
        "dtype": data_type_to_data(buffer.dtype),
        "shape": list(buffer.shape),
        "offset": buffer.offset,
        "nbytes": buffer.nbytes,
    }


def _checkpoint_storage_key(module: IRModule, output: str) -> str | None:
    """Trace a frozen byte-preserving view to one checkpoint tensor.

    This consumes the op definition's named storage contract, rather than a
    list of artifact-specific op names. Any semantic transform terminates the
    trace and is evaluated normally.
    """

    recipe = next(
        (value for value in module.constant_recipes if output in value.outputs),
        None,
    )
    if recipe is None:
        return None
    current = recipe.node_map[output]
    visited: set[str] = set()
    while current.id not in visited:
        visited.add(current.id)
        from triton.flagmega.ir.op_fusion import has_ops
        if has_ops(current.attrs):
            return None
        if current.op == "builtin.weight":
            return str(current.attrs["key"])
        parameters = get_definition(current.op).byte_preserving_input_parameters
        if len(parameters) != 1:
            return None
        source_id = parameters[0].read(current.inputs)
        if not isinstance(source_id, str):  # pragma: no cover - decorator contract.
            return None
        try:
            current = recipe.node_map[source_id]
        except KeyError:
            return None
    return None


def read_rdata_index(path: str | Path) -> dict[str, object]:
    """Read an rdata index's metadata without hashing the image.

    Callers that must know ``chip_count`` before choosing a verification scope
    need the metadata first; the bytes are still verified afterwards.
    """

    index_path = Path(path) / "rdata.index.json"
    if not index_path.is_file():
        raise ArtifactError(f"Rdata index is missing: {index_path}.")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Cannot read rdata index {index_path}: {error}.") from error
    if not isinstance(index, dict):
        raise ArtifactError("Rdata index must be a JSON object.")
    if index.get("schema") not in _SUPPORTED_RDATA_INDEX_SCHEMAS:
        raise ArtifactError(f"Unsupported rdata index schema {index.get('schema')!r}.")
    return index


def verify_rdata(
    path: str | Path,
    *,
    module: IRModule | None = None,
    segment: int | None = None,
    verify_image_bytes: bool = True,
    reuse_source_proofs: bool = False,
) -> dict[str, object]:
    """Verify a packed rdata image, optionally only the caller's chip segment.

    ``segment`` narrows verification to the bytes that one launch chip maps.
    This is a scope reduction, not a weaker check: that segment's digest and
    every index entry landing inside it are still verified. Ranks are each
    responsible for their own segment, so the union over a launch still covers
    the whole image exactly once instead of ``chip_count`` times per rank.

    ``verify_image_bytes=False`` checks the index, its internal consistency and
    the module contract, and defers only the payload digest to the caller that
    maps those bytes.  It exists so one process hashes the payload once instead
    of once per layer that happens to inspect the artifact; the mapping caller
    must still verify before use.

    ``reuse_source_proofs=True`` lets a runtime reuse a host-local digest proof
    for the exact unchanged source-snapshot file version. Unknown or changed
    versions are hashed before use; physical image chunks still receive their
    normal byte audit. Explicit artifact audits keep the default fresh hashes.
    """

    if segment is not None and not verify_image_bytes:
        raise ArtifactError(
            "Rdata verification cannot both target a chip segment and defer the "
            "payload digest."
        )
    root = Path(path)
    index = read_rdata_index(root)
    image_name = str(index.get("image", ""))
    if not image_name or Path(image_name).is_absolute() or ".." in Path(image_name).parts:
        raise ArtifactError(f"Rdata image must be a safe relative path, got {image_name!r}.")
    image = root / image_name
    if not image.is_file():
        raise ArtifactError(f"Rdata image is missing: {image}.")
    image_nbytes = image.stat().st_size
    chip_count = int(index.get("chip_count", 1))
    segment_nbytes = int(index.get("segment_nbytes", index.get("nbytes", -1)))
    if chip_count <= 0 or segment_nbytes < 0:
        raise ArtifactError("Rdata chip segment metadata is invalid.")
    if image_nbytes != int(index.get("nbytes", -1)) or image_nbytes != segment_nbytes * chip_count:
        raise ArtifactError("Rdata image size does not match its index.")
    segment_format = index.get("schema") in {RDATA_INDEX_SCHEMA, RDATA_INDEX_SCHEMA_V3}
    if segment_format and index.get("sha256") != _segment_content_digest(index):
        raise ArtifactError("Rdata content digest does not match its chip segment metadata.")
    entries = index.get("entries")
    if not isinstance(entries, list):
        raise ArtifactError("Rdata index entries must be a list.")
    ranges = []
    buffer_ids: set[str] = set()
    weight_keys: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ArtifactError("Rdata entry must be an object.")
        try:
            buffer_id = str(entry["buffer"])
            weight_key = str(entry["key"])
            start = int(entry["offset"])
            nbytes = int(entry["nbytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactError(f"Malformed rdata entry: {error}.") from error
        if not buffer_id or buffer_id in buffer_ids:
            raise ArtifactError(f"Rdata buffer id {buffer_id!r} is empty or duplicated.")
        if not weight_key or weight_key in weight_keys:
            raise ArtifactError(f"Rdata weight key {weight_key!r} is empty or duplicated.")
        buffer_ids.add(buffer_id)
        weight_keys.add(weight_key)
        end = start + nbytes
        if start < 0 or end > segment_nbytes:
            raise ArtifactError(f"Rdata entry {weight_key!r} is out of bounds.")
        ranges.append((start, end, weight_key, str(entry.get("sha256", ""))))
    ranges.sort()
    for lhs, rhs in zip(ranges, ranges[1:]):
        if lhs[1] > rhs[0]:
            raise ArtifactError(f"Rdata entries {lhs[2]!r} and {rhs[2]!r} overlap.")
    if verify_image_bytes and index.get("schema") != SOURCE_LAYOUT_SCHEMA:
        if index.get("schema") == RDATA_INDEX_SCHEMA:
            _verify_chunk_hashes(image, index, segment)
        elif segment_format and segment is None:
            if _segment_digests(image, segment_nbytes, chip_count) != index["segment_sha256"]:
                raise ArtifactError("Rdata image hash does not match its chip segment index.")
        elif segment is None:
            _verify_image_hashes(image, str(index.get("sha256", "")), ranges)
        else:
            _verify_segment_hashes(image, index, [] if segment_format else ranges,
                                   segment, segment_nbytes, chip_count)
    if module is not None:
        _verify_module_contract(module, index, entries)
    if index.get("schema") == SOURCE_LAYOUT_SCHEMA:
        validate_source_layout(root, index, module, segment=segment, verify_bytes=verify_image_bytes,
                               reuse_source_proofs=reuse_source_proofs)
    return index


def _verify_segment_hashes(
    image: Path,
    index: Mapping[str, object],
    ranges: list[tuple[int, int, str, str]],
    segment: int,
    segment_nbytes: int,
    chip_count: int,
) -> None:
    """Verify one chip segment's digest and the entries it contains.

    Entry offsets in the index are already segment-relative, so the same ranges
    apply unchanged once the stream is windowed onto this segment.
    """

    if segment < 0 or segment >= chip_count:
        raise ArtifactError(
            f"Rdata segment {segment} is outside the chip range [0, {chip_count})."
        )
    digests = index.get("segment_sha256")
    if not isinstance(digests, list) or len(digests) != chip_count:
        raise ArtifactError(
            "Rdata index does not publish one digest per chip segment; "
            "repack the artifact to verify a single segment."
        )
    # An entry digest is the first component of chip 0 (see
    # ``_write_chip_sharded_buffer``), so it describes segment 0 only; another
    # segment holds that buffer's other shard, which the index does not digest
    # separately.  The segment digest covers every byte of the segment, which is
    # a superset of its entry ranges, so byte coverage stays complete either
    # way; entry digests additionally attribute a mismatch to one buffer, and
    # are checked wherever they apply.
    entry_ranges = ranges if segment == 0 else []
    window = _FileWindow(image, segment * segment_nbytes, segment_nbytes)
    _verify_image_hashes(window, str(digests[segment]), entry_ranges)


class _FileWindow:
    """Expose a byte range of a file as if it were the complete file.

    ``_verify_image_hashes`` already streams with bounded memory and hashes
    indexed ranges; scoping it to a segment only requires that it see that
    segment's bytes and nothing else.
    """

    def __init__(self, path: Path, offset: int, nbytes: int) -> None:
        self._path = path
        self._offset = offset
        self._nbytes = nbytes
        self.name = path.name

    def open(self, mode: str):
        stream = self._path.open(mode)
        try:
            stream.seek(self._offset)
        except BaseException:
            stream.close()
            raise
        return _BoundedStream(stream, self._nbytes)


class _BoundedStream:
    """A read-only stream that stops at a fixed number of bytes."""

    def __init__(self, stream, nbytes: int) -> None:
        self._stream = stream
        self._remaining = nbytes

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        count = self._remaining if size is None or size < 0 else min(size, self._remaining)
        chunk = self._stream.read(count)
        self._remaining -= len(chunk)
        return chunk

    def close(self) -> None:
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _segment_content_digest(index: Mapping[str, object]) -> str:
    """Identify payload content from its complete ordered segment coverage.

    v3 hashes each payload byte once. Its multi-chip image identity is SHA-256
    of this domain-separated segment table, rather than SHA-256 of the raw
    image; ``digest_kind`` makes that distinction explicit to readers.
    """
    count = int(index["chip_count"])
    digests = index.get("segment_sha256")
    if (not isinstance(digests, list) or len(digests) != count
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value) for value in digests)):
        raise ArtifactError("Rdata index must publish one valid digest per chip segment.")
    chunk_format = index.get("schema") == RDATA_INDEX_SCHEMA
    expected_kind = RDATA_CHUNK_DIGEST_KIND if chunk_format else (RDATA_SEGMENT_DIGEST_KIND if count > 1 else "sha256")
    if index.get("digest_kind") != expected_kind:
        raise ArtifactError("Rdata digest kind does not match its chip segment geometry.")
    if chunk_format:
        chunks = index.get("segment_chunks")
        if not isinstance(chunks, list) or len(chunks) != count:
            raise ArtifactError("Rdata index must publish one chunk table per chip segment.")
        if [_chunk_segment_digest(table, int(index["segment_nbytes"])) for table in chunks] != digests:
            raise ArtifactError("Rdata segment digest does not match its chunk table.")
    if count == 1 and not chunk_format:
        return digests[0]
    table = {"domain": "flagmega.rdata-segments/v2" if chunk_format else "flagmega.rdata-segments/v1", "chip_count": count,
             "segment_nbytes": int(index["segment_nbytes"]), "segment_sha256": digests}
    return hashlib.sha256(json.dumps(table, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _chunk_segment_digest(chunks, segment_nbytes):
    if not isinstance(chunks, list):
        raise ArtifactError("Rdata segment chunk table must be a list.")
    cursor = 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ArtifactError("Rdata segment chunk must be an object.")
        start, count, digest = chunk.get("offset"), chunk.get("nbytes"), chunk.get("sha256")
        if (type(start) is not int or type(count) is not int or start != cursor or count <= 0
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ArtifactError("Rdata chunk table has invalid geometry or digest.")
        cursor += count
        if cursor > segment_nbytes:
            raise ArtifactError("Rdata chunk table exceeds its chip segment.")
    if cursor != segment_nbytes:
        raise ArtifactError("Rdata chunk table does not cover its complete chip segment.")
    table = {"domain": "flagmega.rdata-chunks/v1", "segment_nbytes": segment_nbytes, "chunks": chunks}
    return hashlib.sha256(json.dumps(table, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _verify_chunk_hashes(image, index, segment):
    count = int(index["chip_count"])
    if segment is not None and (segment < 0 or segment >= count):
        raise ArtifactError(f"Rdata segment {segment} is outside the chip range [0, {count}).")
    selected = range(count) if segment is None else (segment,)
    size = int(index["segment_nbytes"])
    windows = ((chip * size + chunk["offset"], chunk["nbytes"], chunk["sha256"])
               for chip in selected for chunk in index["segment_chunks"][chip])

    def verify(window):
        start, nbytes, digest = window
        if _stream_digest(_FileWindow(image, start, nbytes)) != digest:
            raise ArtifactError(f"Rdata image hash does not match its chunk at byte {start}.")

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1),
                            thread_name_prefix="flagmega-rdata-audit") as worker:
        for _ in worker.map(verify, windows):
            pass


def _stream_digest(image) -> str:
    hasher = hashlib.sha256()
    with image.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def _segment_digests(
    image: Path,
    segment_nbytes: int,
    chip_count: int,
) -> list[str]:
    """Hash disjoint chip windows once, with bounded memory per worker."""
    windows = tuple(_FileWindow(image, chip * segment_nbytes, segment_nbytes) for chip in range(chip_count))
    if chip_count == 1:
        return [_stream_digest(windows[0])]
    with ThreadPoolExecutor(max_workers=min(chip_count, os.cpu_count() or 1),
                            thread_name_prefix="flagmega-rdata-segment") as worker:
        return list(worker.map(_stream_digest, windows))


def _verify_image_hashes(
    image: Path,
    expected_image_hash: str,
    ranges: list[tuple[int, int, str, str]],
) -> None:
    """Hash a potentially multi-gigabyte rdata image without materializing it.

    Runtime loading already uses ``torch.from_file`` after verification.  The
    verifier must preserve that bounded-memory contract: ``Path.read_bytes``
    made peak host memory proportional to the complete model and also copied
    every entry slice before hashing it. One sequential read supplies two
    independent digest chains: a worker updates the whole-image hash while
    this thread updates indexed ranges over the same immutable bytes. Join
    each chunk before the next read, bounding queued payloads and propagating
    worker failures before verification can succeed.
    """

    image_hasher = hashlib.sha256()
    entry_hashers = [hashlib.sha256() for _ in ranges]
    position = 0
    first_active = 0
    with image.open("rb") as stream, ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="flagmega-rdata"
    ) as worker:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            image_update = worker.submit(image_hasher.update, chunk)
            chunk_end = position + len(chunk)
            while (
                first_active < len(ranges)
                and ranges[first_active][1] <= position
            ):
                first_active += 1
            index = first_active
            view = memoryview(chunk)
            while index < len(ranges) and ranges[index][0] < chunk_end:
                start, end, _, _ = ranges[index]
                overlap_start = max(start, position) - position
                overlap_end = min(end, chunk_end) - position
                if overlap_start < overlap_end:
                    entry_hashers[index].update(view[overlap_start:overlap_end])
                index += 1
            image_update.result()
            position = chunk_end

    if image_hasher.hexdigest() != expected_image_hash:
        raise ArtifactError("Rdata image hash does not match its index.")
    for hasher, (_, _, weight_key, expected_hash) in zip(
        entry_hashers, ranges, strict=True
    ):
        if hasher.hexdigest() != expected_hash:
            raise ArtifactError(
                f"Rdata entry {weight_key!r} hash does not match."
            )


def _verify_module_contract(
    module: IRModule,
    index: Mapping[str, object],
    entries: list[object],
) -> None:
    plan = verify_buffer_plan(module)
    expected_chip_count = (
        launch_chip_count(module)
        if any(
            buffer.distributed_storage_kind
            is DistributedBufferStorageKind.COMPACT_PER_CHIP
            for buffer in plan.buffers
            if buffer.storage == "rdata"
        )
        else 1
    )
    if int(index.get("chip_count", 1)) != expected_chip_count:
        raise ArtifactError("Rdata chip count does not match the module launch contract.")
    if int(index.get("segment_nbytes", index.get("nbytes", -1))) != plan.rdata_bytes:
        raise ArtifactError("Rdata segment size does not match the module buffer plan.")
    if int(index.get("alignment", -1)) != plan.alignment:
        raise ArtifactError("Rdata alignment does not match the module buffer plan.")
    actual = {str(entry["buffer"]): entry for entry in entries if isinstance(entry, Mapping)}
    # Views are not packed -- their owner contributes those bytes -- so the
    # image is expected to cover only the buffers that own storage.
    expected = {
        buffer.id: buffer
        for buffer in plan.buffers
        if buffer.storage == "rdata" and not _aliases_rdata_owner(buffer)
    }
    if set(actual) != set(expected):
        raise ArtifactError("Rdata entries do not exactly cover the module readonly-data buffers.")
    for buffer_id, buffer in expected.items():
        entry = actual[buffer_id]
        contract = {
            "key": buffer.weight_key,
            "dtype": data_type_to_data(buffer.dtype),
            "shape": list(buffer.shape),
            "offset": buffer.offset,
            "nbytes": buffer.nbytes,
        }
        for field, value in contract.items():
            if entry.get(field) != value:
                raise ArtifactError(
                    f"Rdata entry {buffer_id!r} field {field!r} does not match the module buffer plan.")


def _tensor_buffer(tensor) -> memoryview:
    try:
        import numpy
    except ImportError as error:
        raise ArtifactError("Rdata packing requires NumPy storage access.") from error
    if isinstance(tensor, numpy.ndarray):
        contiguous = numpy.ascontiguousarray(tensor)
        return memoryview(contiguous.view(numpy.uint8)).cast("B")
    try:
        import torch
    except ImportError as error:
        raise ArtifactError("Rdata packing requires PyTorch tensor storage access.") from error
    if not isinstance(tensor, torch.Tensor):
        raise ArtifactError(
            f"Checkpoint returned {type(tensor).__name__}, expected ndarray or torch.Tensor."
        )
    contiguous = tensor.detach().cpu().contiguous()
    # The ndarray and memoryview both retain the tensor storage owner. Casting
    # to a flat byte view gives file.write/hashlib the buffer protocol directly
    # and avoids a second model-sized allocation from ndarray.tobytes().
    return memoryview(contiguous.view(torch.uint8).numpy()).cast("B")
