# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Source snapshots and byte-address programs for readonly tensor views.

No numeric operation is deferred. A program composes contiguous checkpoint
bytes, a concat and representation-preserving views with the existing dense
chip/block ABI. The device loader fuses this address map with host-to-device
transfer, avoiding both compiler-side reordering and an intermediate image.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import product
from math import prod
from pathlib import Path
import hashlib
import json
import os
import tempfile

from triton.flagmega.errors import ArtifactError
from triton.flagmega.ir import DistributedBufferStorageKind, VectorType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import block_owner_axes, chip_owner_axes, BlockCyclicSplit
from triton.flagmega.ir.local_shard import local_shard_descriptor, unravel_placement_index
from triton.flagmega.ir.op_fusion import has_ops
from triton.flagmega.ir.ops.core import get_definition, tensor_nbytes
from triton.flagmega.artifacts.manifest import hash_file, resolve_artifact_path

SOURCE_LAYOUT_SCHEMA = "flagmega.rdata-index/v5"
SOURCE_LAYOUT_DIGEST = "sha256-source-layout/v1"


def _physical_byte_shape(value_type):
    tensor = tensor_of(value_type)
    dtype = tensor.dtype
    scalar = dtype.elem_type if isinstance(dtype, VectorType) else dtype
    return [*(dim.fixed_value for dim in tensor.shape),
            *(dtype.lanes if isinstance(dtype, VectorType) else ()), scalar.itemsize]


def _strip_views(nodes, node):
    seen = set()
    while node.id not in seen:
        seen.add(node.id)
        if has_ops(node.attrs):
            return None
        if node.op in {"builtin.weight", "tensors.concat"}:
            return node
        contracts = get_definition(node.op).byte_preserving_input_parameters
        if len(contracts) != 1:
            return None
        source = contracts[0].read(node.inputs)
        if not isinstance(source, str) or source not in nodes:
            return None
        node = nodes[source]
    return None


def _strides(shape):
    return [prod(shape[axis + 1:]) for axis in range(len(shape))]


def _output_source(nodes, node):
    """Compose byte views and dense permutations without evaluating values."""
    transforms = []
    seen = set()
    while node.id not in seen:
        seen.add(node.id)
        if has_ops(node.attrs):
            return None, []
        if node.op in {"builtin.weight", "tensors.concat"}:
            return node, transforms
        if node.op in {"tensors.pack", "tensors.permute"}:
            source = nodes[node.inputs[0]]
            tensor = tensor_of(source.type)
            input_shape = _physical_byte_shape(source.type)
            if node.op == "tensors.permute":
                permutation = [*node.attrs["axes"], *range(tensor.rank, len(input_shape))]
                expanded = input_shape
            else:
                from triton.flagmega.ir.ops.tensors.pack import _pack_axes
                lanes = tuple(node.attrs["lanes"])
                axes = _pack_axes(node.attrs, len(lanes), tensor.rank)
                expanded, outer, lane_positions = [], [], {}
                for axis in range(tensor.rank):
                    factors = [(i, lane) for i, (packed_axis, lane) in enumerate(zip(axes, lanes, strict=True))
                               if packed_axis == axis]
                    outer.append(len(expanded))
                    expanded.append(input_shape[axis] // prod(lane for _, lane in factors))
                    for i, lane in factors:
                        lane_positions[i] = len(expanded)
                        expanded.append(lane)
                trailing = list(range(len(expanded), len(expanded) + len(input_shape) - tensor.rank))
                expanded.extend(input_shape[tensor.rank:])
                permutation = [*outer, *(lane_positions[i] for i in range(len(lanes))), *trailing]
            strides = _strides(expanded)
            transforms.append({"shape": _physical_byte_shape(node.type),
                               "source_strides": [strides[axis] for axis in permutation]})
            node = source
            continue
        contracts = get_definition(node.op).byte_preserving_input_parameters
        if len(contracts) != 1:
            return None, []
        source_id = contracts[0].read(node.inputs)
        if not isinstance(source_id, str) or source_id not in nodes:
            return None, []
        node = nodes[source_id]
    return None, []


def build_program(buffer, nodes, output, range_getter):
    """Prove that the complete output is an address-only source composition."""
    node, transforms = _output_source(nodes, nodes[output])
    if node is None:
        return None
    shape = _physical_byte_shape(node.type)
    if any(value <= 0 for value in shape):
        return None
    if node.op == "tensors.concat":
        axis = int(node.attrs["axis"]) % tensor_of(node.type).rank
        inputs = tuple(nodes[value] for value in node.inputs)
    else:
        # One contiguous source is the same program with one concat operand.
        axis, inputs = 0, (node,)
    operands = []
    for child in inputs:
        leaf = _strip_views(nodes, child)
        if leaf is None or leaf.op != "builtin.weight":
            return None
        key = str(leaf.attrs["key"])
        source = range_getter(key)
        if source is None:
            return None
        if source.nbytes != tensor_nbytes(tensor_of(child.type)):
            raise ArtifactError(f"Source byte count for {key!r} differs from its constant recipe.")
        operands.append({"key": key, "path": str(source.path), "offset": source.offset,
                         "nbytes": source.nbytes, "extent": _physical_byte_shape(child.type)[axis]})
    final_shape = [*buffer.shape,
                   *(buffer.dtype.lanes if isinstance(buffer.dtype, VectorType) else ()),
                   (buffer.dtype.elem_type if isinstance(buffer.dtype, VectorType) else buffer.dtype).itemsize]
    if prod(final_shape) != prod(shape):
        raise ArtifactError(f"Source layout for {buffer.id!r} changes its byte count.")
    return {"buffer": buffer.id, "shape": final_shape, "concat_shape": shape,
            "axis": axis, "operands": operands, "transforms": transforms}


def prepare_programs(module, checkpoint, buffers, destination):
    if not callable(getattr(checkpoint, "source_file_digest", None)):
        return [], []
    getter = getattr(checkpoint, "tensor_byte_range", None)
    if getter is None:
        return [], []
    recipes = {output: recipe for recipe in module.constant_recipes for output in recipe.outputs}
    programs = []
    for buffer in buffers:
        recipe = recipes.get(buffer.weight_key)
        if recipe is None:
            continue
        program = build_program(buffer, recipe.node_map, buffer.weight_key, getter)
        if program is not None:
            programs.append(program)
    paths = sorted({operand["path"] for p in programs for operand in p["operands"]})
    if not paths:
        return [], []
    from triton.flagmega.artifacts.rdata import _clone_file
    def snapshot(item, staging):
        index, name = item
        source = Path(name)
        stat = source.stat()
        before = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        digest = checkpoint.source_file_digest(source)
        relative = f"sources/{index:04d}.bin"
        _clone_file(source, staging / f"{index:04d}.bin")
        stat = source.stat()
        after = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if before != after:
            raise ArtifactError(f"Source {source} changed while snapshotting readonly data.")
        return {"path": relative, "nbytes": before[2], "sha256": digest}

    with tempfile.TemporaryDirectory(prefix=".rdata-source-snapshots-", dir=destination) as temporary:
        staging = Path(temporary)
        with ThreadPoolExecutor(max_workers=min(32, len(paths)), thread_name_prefix="flagmega-source-snapshot") as workers:
            sources = list(workers.map(lambda item: snapshot(item, staging), enumerate(paths)))
        (destination / "sources").mkdir(exist_ok=True)
        for index, source in enumerate(sources):
            os.replace(staging / f"{index:04d}.bin", destination / source["path"])
    source_indices = {path: index for index, path in enumerate(paths)}
    for program in programs:
        for operand in program["operands"]:
            operand["source"] = source_indices[operand.pop("path")]
    return programs, sources


def component_layouts(buffer, chip):
    """Yield dense byte-coordinate maps, in physical block-owner order."""
    if buffer.distributed_storage_kind is not DistributedBufferStorageKind.COMPACT_PER_CHIP:
        yield buffer.offset, [*buffer.shape,
                              *(buffer.dtype.lanes if isinstance(buffer.dtype, VectorType) else ()),
                              (buffer.dtype.elem_type if isinstance(buffer.dtype, VectorType) else buffer.dtype).itemsize], []
        return
    distributed = buffer.distributed_type
    chip_axes = chip_owner_axes(distributed)
    chip_coords = unravel_placement_index(chip, tuple(distributed.placement.hierarchy[a] for a in chip_axes))
    coordinates = [0] * distributed.placement.rank
    for axis, value in zip(chip_axes, chip_coords, strict=True):
        coordinates[axis] = value
    block_axes = block_owner_axes(distributed)
    for owner, block in enumerate(product(*(range(distributed.placement.hierarchy[a]) for a in block_axes))):
        coords = list(coordinates)
        for axis, value in zip(block_axes, block, strict=True):
            coords[axis] = value
        descriptor = local_shard_descriptor(distributed, tuple(coords))
        maps = []
        for axis in descriptor.axes:
            stages = []
            for stage in reversed(axis.stages):
                distribution = stage.stage.distribution
                stages.append((int(stage.linear_shard_index.fixed_value),
                               distribution.block_size if isinstance(distribution, BlockCyclicSplit) else 0,
                               stage.shard_count, int(stage.local_capacity.fixed_value)))
            maps.append((int(axis.active_extent.fixed_value), tuple(stages)))
        dtype = buffer.dtype
        shape = [*buffer.component_shape, *(dtype.lanes if isinstance(dtype, VectorType) else ()),
                 (dtype.elem_type if isinstance(dtype, VectorType) else dtype).itemsize]
        yield buffer.offset + owner * buffer.component_stride_bytes, shape, maps


def program_ranges(buffer):
    return [(offset, prod(shape)) for offset, shape, _ in component_layouts(buffer, 0)]


def content_digest(index):
    table = {key: value for key, value in index.items() if key != "sha256"}
    table["domain"] = SOURCE_LAYOUT_DIGEST
    return hashlib.sha256(json.dumps(table, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_source_layout(root, index, module, *, segment, verify_bytes):
    """Validate source/program geometry even when payload auditing is deferred."""
    if index.get("digest_kind") != SOURCE_LAYOUT_DIGEST or index.get("sha256") != content_digest(index):
        raise ArtifactError("Rdata source layout digest does not match its index.")
    count, size = int(index["chip_count"]), int(index["segment_nbytes"])
    if segment is not None and not 0 <= segment < count:
        raise ArtifactError("Rdata source layout chip segment is out of range.")
    sources, programs, chunks = index.get("sources"), index.get("programs"), index.get("segment_chunks")
    if not isinstance(sources, list) or not isinstance(programs, list) or not isinstance(chunks, list) or len(chunks) != count:
        raise ArtifactError("Rdata source layout tables are invalid.")
    paths = []
    for source in sources:
        if not isinstance(source, dict) or not _valid_digest(source.get("sha256")):
            raise ArtifactError("Rdata source digest is invalid.")
        path = resolve_artifact_path(root, str(source.get("path", "")))
        if not path.is_file() or path.stat().st_size != source.get("nbytes") or path in paths:
            raise ArtifactError("Rdata source snapshot is missing, duplicated or has an invalid size.")
        paths.append(path)
    ranges = []
    seen = set()
    for program in programs:
        buffer_id = program.get("buffer")
        if buffer_id in seen:
            raise ArtifactError("Rdata source program buffer is duplicated.")
        seen.add(buffer_id)
        shape, concat_shape, axis = program.get("shape"), program.get("concat_shape"), program.get("axis")
        if (not _valid_shape(shape) or not _valid_shape(concat_shape) or type(axis) is not int
                or not 0 <= axis < len(concat_shape) or prod(shape) != prod(concat_shape)):
            raise ArtifactError("Rdata source program shape or axis is invalid.")
        transforms = program.get("transforms")
        if not isinstance(transforms, list):
            raise ArtifactError("Rdata source transforms must be a list.")
        for transform in transforms:
            dims, strides = transform.get("shape"), transform.get("source_strides")
            if (not _valid_shape(dims) or not isinstance(strides, list) or len(dims) != len(strides)
                    or any(type(stride) is not int or stride <= 0 for stride in strides)
                    or prod(dims) != prod(shape)):
                raise ArtifactError("Rdata source transform has invalid byte geometry.")
            cursor = 1
            for stride, extent in sorted((stride, extent) for stride, extent in zip(strides, dims) if extent != 1):
                if stride != cursor:
                    raise ArtifactError("Rdata source transform is not a bijective byte layout.")
                cursor *= extent
        operands = program.get("operands")
        if not isinstance(operands, list) or not operands:
            raise ArtifactError("Rdata source program requires operands.")
        extent = 0
        for operand in operands:
            source, offset, nbytes, length = (operand.get(k) for k in ("source", "offset", "nbytes", "extent"))
            if (any(type(value) is not int for value in (source, offset, nbytes, length))
                    or not 0 <= source < len(sources) or offset < 0 or nbytes <= 0 or length <= 0
                    or offset + nbytes > sources[source]["nbytes"]
                    or nbytes * concat_shape[axis] != prod(concat_shape) * length):
                raise ArtifactError("Rdata source operand range is invalid.")
            extent += length
        if extent != concat_shape[axis]:
            raise ArtifactError("Rdata source concat operands do not cover its axis.")
        for offset, nbytes in program.get("ranges", []):
            if type(offset) is not int or type(nbytes) is not int or offset < 0 or nbytes <= 0 or offset + nbytes > size:
                raise ArtifactError("Rdata source destination range is invalid.")
            ranges.append((offset, offset + nbytes))
    for table in chunks:
        coverage = list(ranges)
        for chunk in table:
            offset, nbytes = chunk.get("offset"), chunk.get("nbytes")
            if (type(offset) is not int or type(nbytes) is not int or offset < 0 or nbytes <= 0
                    or offset + nbytes > size or not _valid_digest(chunk.get("sha256"))):
                raise ArtifactError("Rdata source materialized chunk is invalid.")
            coverage.append((offset, offset + nbytes))
        cursor = 0
        for start, end in sorted(coverage):
            if start != cursor:
                raise ArtifactError("Rdata source layout coverage has an overlap or gap.")
            cursor = end
        if cursor != size:
            raise ArtifactError("Rdata source layout does not cover the complete chip segment.")
    if module is not None:
        from triton.flagmega.ir import verify_buffer_plan
        from triton.flagmega.importer.checkpoint import TensorByteRange
        buffers = verify_buffer_plan(module).buffer_map
        recipes = {output: recipe for recipe in module.constant_recipes for output in recipe.outputs}
        for program in programs:
            buffer = buffers.get(program["buffer"])
            if buffer is None or buffer.weight_key not in recipes:
                raise ArtifactError("Rdata source program does not name a constant recipe buffer.")
            bindings = {operand["key"]: TensorByteRange(paths[operand["source"]], operand["offset"], operand["nbytes"])
                        for operand in program["operands"]}
            expected = build_program(buffer, recipes[buffer.weight_key].node_map, buffer.weight_key, bindings.get)
            if expected is None:
                raise ArtifactError("Rdata source program defers a numeric constant operation.")
            for operand in expected["operands"]:
                operand["source"] = paths.index(Path(operand.pop("path")))
            expected["ranges"] = [list(pair) for pair in program_ranges(buffer)]
            if expected != program:
                raise ArtifactError("Rdata source program differs from its constant recipe or buffer layout.")
    if verify_bytes:
        def audit(item):
            path, source = item
            if hash_file(path) != source["sha256"]:
                raise ArtifactError(f"Rdata source hash mismatch: {source['path']}.")
        with ThreadPoolExecutor(max_workers=min(32, max(1, len(sources)))) as workers:
            list(workers.map(audit, zip(paths, sources, strict=True)))
        from triton.flagmega.artifacts.rdata import _verify_chunk_hashes
        _verify_chunk_hashes(Path(root) / index["image"], index, segment)


def _valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _valid_shape(value):
    return isinstance(value, list) and value and all(type(v) is int and v > 0 for v in value)
