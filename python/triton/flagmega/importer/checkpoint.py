# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Lazy Hugging Face config/index/safetensors checkpoint access."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from triton.flagmega.errors import ImporterError
from triton.flagmega.ir import DType


_SAFETENSORS_DTYPES = {
    "BOOL": DType.BOOL,
    "I32": DType.INT32,
    "I64": DType.INT64,
    "BF16": DType.BFLOAT16,
    "F32": DType.FLOAT32,
    "F8_E4M3": DType.FLOAT8_E4M3FN,
    "F8_E4M3FN": DType.FLOAT8_E4M3FN,
}


@dataclass(frozen=True)
class TensorInfo:
    key: str
    dtype: DType
    shape: tuple[int, ...]
    source: str
    data_offsets: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        if not self.key or not self.source:
            raise ImporterError("TensorInfo requires non-empty key and source.")
        if not isinstance(self.dtype, DType):
            raise ImporterError(f"Tensor {self.key!r} has an invalid dtype descriptor.")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.shape
        ):
            raise ImporterError(f"Tensor {self.key!r} has invalid shape {self.shape!r}.")
        if self.data_offsets is not None:
            object.__setattr__(self, "data_offsets", tuple(self.data_offsets))
            if (
                len(self.data_offsets) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in self.data_offsets
                )
                or not 0 <= self.data_offsets[0] <= self.data_offsets[1]
            ):
                raise ImporterError(
                    f"Tensor {self.key!r} has invalid data offsets {self.data_offsets!r}."
                )


@dataclass(frozen=True)
class TensorByteRange:
    """Verified contiguous tensor payload inside a local checkpoint shard.

    Artifact packing can copy representation-preserving constant recipes from
    this range without importing PyTorch or constructing a model-sized tensor.
    The range is an optional checkpoint capability; semantic transforms still
    use ``load_tensor``.
    """

    path: Path
    offset: int
    nbytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve())
        if (
            isinstance(self.offset, bool)
            or not isinstance(self.offset, int)
            or self.offset < 0
            or isinstance(self.nbytes, bool)
            or not isinstance(self.nbytes, int)
            or self.nbytes < 0
        ):
            raise ImporterError("TensorByteRange offset/nbytes must be non-negative integers.")


class Checkpoint(Protocol):
    @property
    def config(self) -> Mapping[str, Any]: ...

    @property
    def keys(self) -> tuple[str, ...]: ...

    def tensor_info(self, key: str) -> TensorInfo: ...

    def load_tensor(self, key: str, *, device: str = "cpu"): ...

    def tensor_byte_range(self, key: str) -> TensorByteRange | None: ...

    def cache_identity(self, keys: Iterable[str]) -> Mapping[str, Any] | None: ...


class DirectoryCheckpoint:
    """A metadata-first checkpoint reader that never instantiates a model."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        config_path = self.root / "config.json"
        if not config_path.is_file():
            raise ImporterError(f"Checkpoint config.json is missing under {self.root}.")
        try:
            self._config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ImporterError(f"Failed to read checkpoint config {config_path}: {error}") from error
        self._headers: dict[str, dict[str, Any]] = {}
        self._payload_offsets: dict[str, int] = {}
        self._weight_map = self._load_weight_map()

    @property
    def config(self) -> Mapping[str, Any]:
        return self._config

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._weight_map))

    def tensor_info(self, key: str) -> TensorInfo:
        try:
            source = self._weight_map[key]
        except KeyError as error:
            raise ImporterError(f"Checkpoint tensor {key!r} is missing.") from error
        header = self._header(source)
        try:
            metadata = header[key]
        except KeyError as error:
            raise ImporterError(f"Tensor {key!r} is absent from its indexed shard {source!r}.") from error
        raw_dtype = str(metadata.get("dtype"))
        try:
            dtype = _SAFETENSORS_DTYPES[raw_dtype]
        except KeyError as error:
            raise ImporterError(f"Tensor {key!r} uses unsupported safetensors dtype {raw_dtype!r}.") from error
        shape = tuple(int(value) for value in metadata.get("shape", ()))
        offsets = metadata.get("data_offsets")
        return TensorInfo(
            key,
            dtype,
            shape,
            source,
            None if offsets is None else (int(offsets[0]), int(offsets[1])),
        )

    def load_tensor(self, key: str, *, device: str = "cpu"):
        info = self.tensor_info(key)
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise ImporterError("Loading checkpoint values requires the optional 'safetensors' package.") from error
        path = self._resolve_shard(info.source)
        with safe_open(path, framework="pt", device=device) as handle:
            return handle.get_tensor(key)

    def tensor_byte_range(self, key: str) -> TensorByteRange:
        """Return the already-validated raw safetensors payload range."""

        info = self.tensor_info(key)
        if info.data_offsets is None:  # pragma: no cover - safetensors requires it.
            raise ImporterError(f"Tensor {key!r} has no safetensors data offsets.")
        path = self._resolve_shard(info.source)
        payload_offset = self._payload_offsets.get(info.source)
        if payload_offset is None:
            # ``tensor_info`` has already parsed and validated the complete
            # header. Reading its fixed prefix again avoids retaining a second
            # model-sized metadata structure solely for one scalar offset.
            try:
                with path.open("rb") as stream:
                    prefix = stream.read(8)
                if len(prefix) != 8:
                    raise ImporterError(
                        f"Safetensors shard {path} is shorter than its header prefix."
                    )
                payload_offset = 8 + struct.unpack("<Q", prefix)[0]
            except ImporterError:
                raise
            except (OSError, struct.error) as error:
                raise ImporterError(
                    f"Failed to locate safetensors payload in {path}: {error}"
                ) from error
            self._payload_offsets[info.source] = payload_offset
        start, end = info.data_offsets
        return TensorByteRange(path, payload_offset + start, end - start)

    def cache_identity(self, keys: Iterable[str]) -> Mapping[str, Any]:
        """Return a cheap identity for build-cache invalidation.

        The identity deliberately includes both the validated tensor metadata
        and the backing shard's file identity. Replacing or editing a shard
        therefore invalidates cached materializations without reading all
        model payload bytes on every compiler iteration.
        """

        tensors = []
        sources: dict[str, dict[str, object]] = {}
        for key in sorted(set(str(value) for value in keys)):
            info = self.tensor_info(key)
            path = self._resolve_shard(info.source)
            stat = path.stat()
            sources.setdefault(info.source, {
                "source": info.source,
                "path": str(path),
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
            tensors.append({
                "key": info.key,
                "dtype": info.dtype.value,
                "shape": list(info.shape),
                "source": info.source,
                "data_offsets": (
                    None
                    if info.data_offsets is None
                    else list(info.data_offsets)
                ),
            })
        return {
            "schema": "flagmega.directory-checkpoint-cache-identity/v1",
            "tensors": tensors,
            "sources": [sources[key] for key in sorted(sources)],
        }

    def _load_weight_map(self) -> dict[str, str]:
        index_candidates = sorted(self.root.glob("*.safetensors.index.json"))
        if index_candidates:
            if len(index_candidates) != 1:
                raise ImporterError(
                    f"Checkpoint {self.root} contains multiple safetensors indices: "
                    f"{[value.name for value in index_candidates]}.")
            try:
                data = json.loads(index_candidates[0].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ImporterError(
                    f"Failed to read safetensors index {index_candidates[0]}: {error}"
                ) from error
            weight_map = data.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ImporterError(f"Safetensors index {index_candidates[0]} has no non-empty weight_map.")
            result = {}
            for key, value in weight_map.items():
                tensor_key = str(key)
                source = str(value)
                if not tensor_key or tensor_key == "__metadata__":
                    raise ImporterError(
                        f"Safetensors index {index_candidates[0]} contains invalid tensor key {tensor_key!r}."
                    )
                self._resolve_shard(source, require_exists=False)
                result[tensor_key] = source
            return result

        shards = sorted(self.root.glob("*.safetensors"))
        if not shards:
            raise ImporterError(f"Checkpoint {self.root} has no .safetensors files or index.")
        weight_map: dict[str, str] = {}
        for shard in shards:
            # ``glob`` follows the directory entry but does not establish that
            # a symlink target belongs to this checkpoint.  Route unindexed
            # shards through the same resolver as indexed shards.
            path = self._resolve_shard(shard.name)
            header = _read_safetensors_header(path)
            for key in header:
                if key == "__metadata__":
                    continue
                if key in weight_map:
                    raise ImporterError(f"Tensor {key!r} appears in both {weight_map[key]!r} and {shard.name!r}.")
                weight_map[key] = shard.name
            self._headers[shard.name] = header
        return weight_map

    def _header(self, source: str) -> dict[str, Any]:
        if source not in self._headers:
            path = self._resolve_shard(source)
            self._headers[source] = _read_safetensors_header(path)
        return self._headers[source]

    def _resolve_shard(self, source: str, *, require_exists: bool = True) -> Path:
        candidate = Path(source)
        if (
            not source
            or candidate.is_absolute()
            or candidate.suffix != ".safetensors"
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise ImporterError(
                f"Checkpoint shard path {source!r} must be a safe relative .safetensors path."
            )
        path = (self.root / candidate).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            if not self._is_huggingface_blob(path):
                raise ImporterError(
                    f"Checkpoint shard path {source!r} escapes root {self.root}."
                ) from error
        if require_exists and not path.is_file():
            raise ImporterError(
                f"Checkpoint shard {source!r} does not exist under {self.root}."
            )
        return path

    def _is_huggingface_blob(self, path: Path) -> bool:
        """Accept the canonical HF snapshot-to-blob symlink, and only it.

        Hugging Face stores immutable objects at
        ``models--ORG--NAME/blobs`` and exposes a revision through symlinks in
        ``snapshots/REV``.  A generic escaping symlink remains rejected.
        """

        snapshots = self.root.parent
        model_root = snapshots.parent
        if snapshots.name != "snapshots" or not model_root.name.startswith("models--"):
            return False
        blob_root = (model_root / "blobs").resolve()
        try:
            relative = path.relative_to(blob_root)
        except ValueError:
            return False
        return bool(relative.parts) and all(part not in {"", ".", ".."} for part in relative.parts)


class MemoryCheckpoint:
    """Metadata/tensor checkpoint used by tests and programmatic importers."""

    def __init__(
        self,
        config: Mapping[str, Any],
        tensors: Mapping[str, TensorInfo],
        values: Mapping[str, Any] | None = None,
    ) -> None:
        self._config = dict(config)
        self._tensors = dict(tensors)
        self._values = dict(values or {})

    @property
    def config(self) -> Mapping[str, Any]:
        return self._config

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._tensors))

    def tensor_info(self, key: str) -> TensorInfo:
        try:
            return self._tensors[key]
        except KeyError as error:
            raise ImporterError(f"Checkpoint tensor {key!r} is missing.") from error

    def load_tensor(self, key: str, *, device: str = "cpu"):
        try:
            value = self._values[key]
        except KeyError as error:
            raise ImporterError(f"Memory checkpoint has no tensor value for {key!r}.") from error
        return value.to(device=device) if hasattr(value, "to") else value

    def tensor_byte_range(self, key: str) -> None:
        # In-memory values have no stable external byte range. Artifact writers
        # deliberately fall back to the ordinary tensor evaluator.
        self.tensor_info(key)
        return None

    def cache_identity(self, keys: Iterable[str]) -> None:
        # Mutable in-memory tensor objects do not have a stable cheap identity.
        for key in keys:
            self.tensor_info(str(key))
        return None


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    try:
        file_bytes = path.stat().st_size
        with path.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise ImporterError(f"Safetensors shard {path} is shorter than its header prefix.")
            header_length = struct.unpack("<Q", length_bytes)[0]
            if header_length <= 0 or header_length > 256 * 1024 * 1024:
                raise ImporterError(f"Safetensors shard {path} has invalid header length {header_length}.")
            payload = stream.read(header_length)
            if len(payload) != header_length:
                raise ImporterError(f"Safetensors shard {path} has a truncated header.")
        header = json.loads(payload)
    except ImporterError:
        raise
    except (OSError, json.JSONDecodeError, struct.error) as error:
        raise ImporterError(f"Failed to read safetensors header {path}: {error}") from error
    if not isinstance(header, dict):
        raise ImporterError(f"Safetensors header {path} must be a JSON object.")
    data_bytes = file_bytes - 8 - header_length
    intervals = []
    for key, metadata in header.items():
        if key == "__metadata__":
            if not isinstance(metadata, dict):
                raise ImporterError(f"Safetensors __metadata__ in {path} must be an object.")
            continue
        if not isinstance(key, str) or not key or not isinstance(metadata, dict):
            raise ImporterError(f"Safetensors header {path} contains an invalid tensor entry.")
        raw_dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if raw_dtype not in _SAFETENSORS_DTYPES:
            raise ImporterError(
                f"Tensor {key!r} uses unsupported safetensors dtype {raw_dtype!r}."
            )
        if not isinstance(shape, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in shape
        ):
            raise ImporterError(f"Tensor {key!r} has invalid safetensors shape {shape!r}.")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= data_bytes
        ):
            raise ImporterError(f"Tensor {key!r} has invalid safetensors offsets {offsets!r}.")
        expected = prod(shape, start=1) * _SAFETENSORS_DTYPES[raw_dtype].itemsize
        if offsets[1] - offsets[0] != expected:
            raise ImporterError(
                f"Tensor {key!r} byte range has size {offsets[1] - offsets[0]}, expected {expected}."
            )
        intervals.append((offsets[0], offsets[1], key))
    intervals.sort()
    for lhs, rhs in zip(intervals, intervals[1:]):
        if lhs[1] > rhs[0]:
            raise ImporterError(
                f"Safetensors tensors {lhs[2]!r} and {rhs[2]!r} overlap in {path}."
            )
    return header
