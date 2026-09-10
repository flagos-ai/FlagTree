# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Deterministic readonly-data image packing for bufferized FlagMega IR."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from triton.flagmega.errors import ArtifactError
from triton.flagmega.importer.checkpoint import Checkpoint, TensorByteRange
from triton.flagmega.evaluator import (
    iter_numpy_materialized_constant_assets,
)
from triton.flagmega.ir import IRModule, verify_buffer_plan
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.ir.types import data_type_to_data
from triton.flagmega.artifacts.manifest import hash_file, write_json_atomic


RDATA_INDEX_SCHEMA = "flagmega.rdata-index/v1"
RDATA_CACHE_SCHEMA = "flagmega.rdata-cache/v1"
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_ZERO_HASH_CHUNK = bytes(_HASH_CHUNK_BYTES)
_FICLONE = 0x40049409


def pack_rdata(
    module: IRModule,
    checkpoint: Checkpoint,
    output_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> dict[str, object]:
    """Pack readonly data, optionally reusing a trusted local build cache."""

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
    buffers = [buffer for buffer in plan.buffers if buffer.storage == "rdata"]
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
    evaluation_order = tuple(
        output for output in recipe_output_order if output in evaluated_outputs
    )
    buffer_evaluation_order = tuple(
        buffer.weight_key
        for buffer in ordered_buffers
        if buffer.weight_key in evaluated_outputs
    )
    can_hash_while_writing = evaluation_order == buffer_evaluation_order
    try:
        with temporary_image.open("w+b") as stream:
            # A newly truncated file defines alignment gaps as zero without a
            # model-sized Python bytearray.  Individual tensors are then written
            # directly to their verified physical ranges.
            stream.truncate(plan.rdata_bytes)
            if can_hash_while_writing:
                entries, payload_hash = _pack_in_offset_order(
                    stream,
                    module,
                    checkpoint,
                    ordered_buffers,
                    direct_ranges,
                    evaluated_outputs,
                    recipe_outputs,
                    plan.rdata_bytes,
                )
            else:
                entries = _pack_with_post_hash(
                    stream,
                    module,
                    checkpoint,
                    by_key,
                    direct_ranges,
                    evaluated_outputs,
                    recipe_outputs,
                )
                payload_hash = None
            stream.flush()
            os.fsync(stream.fileno())

        entries.sort(key=lambda value: (int(value["offset"]), str(value["buffer"])))
        if payload_hash is None:
            payload_hash = hash_file(temporary_image)
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
        "schema": RDATA_INDEX_SCHEMA,
        "image": image_path.name,
        "nbytes": plan.rdata_bytes,
        "alignment": plan.alignment,
        "sha256": payload_hash,
        "entries": entries,
    }
    write_json_atomic(index_path, index)
    return index


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
        "alignment": plan.alignment,
        "nbytes": plan.rdata_bytes,
        "buffers": [
            {
                "id": buffer.id,
                "key": buffer.weight_key,
                "dtype": data_type_to_data(buffer.dtype),
                "shape": list(buffer.shape),
                "offset": buffer.offset,
                "nbytes": buffer.nbytes,
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
        or index.get("schema") != RDATA_INDEX_SCHEMA
        or manifest.get("image_nbytes") != image_path.stat().st_size
        or manifest.get("image_nbytes") != index.get("nbytes")
        or manifest.get("image_sha256") != index.get("sha256")
        or manifest.get("index_sha256") != hash_file(index_path)
    ):
        return None
    return index


def _clone_file(source: Path, destination: Path) -> None:
    """Atomically materialize an independent reflink, with copy fallback."""

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        try:
            import fcntl

            with source.open("rb", buffering=0) as input_stream:
                with temporary.open("w+b", buffering=0) as output_stream:
                    fcntl.ioctl(output_stream.fileno(), _FICLONE, input_stream.fileno())
        except (ImportError, OSError):
            temporary.unlink(missing_ok=True)
            shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
            tensor = checkpoint.load_tensor(key, device="cpu")
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
            _write_rdata_entry(stream, buffer, checkpoint.load_tensor(key, device="cpu"))
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
    return _rdata_entry(buffer, hashlib.sha256(raw).hexdigest())


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
    hasher = hashlib.sha256()
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
                hasher.update(chunk)
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
    return _rdata_entry(buffer, hasher.hexdigest())


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


def _rdata_entry(buffer, digest: str) -> dict[str, object]:
    return {
        "buffer": buffer.id,
        "key": buffer.weight_key,
        "dtype": data_type_to_data(buffer.dtype),
        "shape": list(buffer.shape),
        "offset": buffer.offset,
        "nbytes": buffer.nbytes,
        "sha256": digest,
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


def verify_rdata(path: str | Path, *, module: IRModule | None = None) -> dict[str, object]:
    root = Path(path)
    index_path = root / "rdata.index.json"
    if not index_path.is_file():
        raise ArtifactError(f"Rdata index is missing: {index_path}.")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Cannot read rdata index {index_path}: {error}.") from error
    if not isinstance(index, dict):
        raise ArtifactError("Rdata index must be a JSON object.")
    if index.get("schema") != RDATA_INDEX_SCHEMA:
        raise ArtifactError(f"Unsupported rdata index schema {index.get('schema')!r}.")
    image_name = str(index.get("image", ""))
    if not image_name or Path(image_name).is_absolute() or ".." in Path(image_name).parts:
        raise ArtifactError(f"Rdata image must be a safe relative path, got {image_name!r}.")
    image = root / image_name
    if not image.is_file():
        raise ArtifactError(f"Rdata image is missing: {image}.")
    image_nbytes = image.stat().st_size
    if image_nbytes != int(index.get("nbytes", -1)):
        raise ArtifactError("Rdata image size does not match its index.")
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
        if start < 0 or end > image_nbytes:
            raise ArtifactError(f"Rdata entry {weight_key!r} is out of bounds.")
        ranges.append((start, end, weight_key, str(entry.get("sha256", ""))))
    ranges.sort()
    for lhs, rhs in zip(ranges, ranges[1:]):
        if lhs[1] > rhs[0]:
            raise ArtifactError(f"Rdata entries {lhs[2]!r} and {rhs[2]!r} overlap.")
    _verify_image_hashes(image, str(index.get("sha256", "")), ranges)
    if module is not None:
        _verify_module_contract(module, index, entries)
    return index


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
    if int(index.get("nbytes", -1)) != plan.rdata_bytes:
        raise ArtifactError("Rdata image size does not match the module buffer plan.")
    if int(index.get("alignment", -1)) != plan.alignment:
        raise ArtifactError("Rdata alignment does not match the module buffer plan.")
    actual = {str(entry["buffer"]): entry for entry in entries if isinstance(entry, Mapping)}
    expected = {buffer.id: buffer for buffer in plan.buffers if buffer.storage == "rdata"}
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
