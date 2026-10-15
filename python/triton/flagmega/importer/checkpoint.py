# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Lazy safetensors and GGUF checkpoint access.

The GGUF reader deliberately exposes quantized tensors as their packed byte
range.  Quantized weights are never dequantized on the host; consumers can
use :meth:`quantization_of` to select a device-side quantized kernel.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import struct
import hashlib
from dataclasses import dataclass
from math import prod
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping, Protocol

from triton.flagmega.errors import ImporterError
from triton.flagmega.ir import DType, tensor_type

_SAFETENSORS_DTYPES = {
    "BOOL": DType.BOOL,
    "I32": DType.INT32,
    "I64": DType.INT64,
    "BF16": DType.BFLOAT16,
    "F16": DType.FLOAT16,
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
    quant_type: str | None = None
    physical_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        if self.physical_shape is not None:
            object.__setattr__(self, "physical_shape", tuple(self.physical_shape))
        if not self.key or not self.source:
            raise ImporterError("TensorInfo requires non-empty key and source.")
        if not isinstance(self.dtype, DType):
            raise ImporterError(f"Tensor {self.key!r} has an invalid dtype descriptor.")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in self.shape):
            raise ImporterError(f"Tensor {self.key!r} has invalid shape {self.shape!r}.")
        if self.data_offsets is not None:
            object.__setattr__(self, "data_offsets", tuple(self.data_offsets))
            if (len(self.data_offsets) != 2
                    or any(not isinstance(value, int) or isinstance(value, bool) for value in self.data_offsets)
                    or not 0 <= self.data_offsets[0] <= self.data_offsets[1]):
                raise ImporterError(f"Tensor {self.key!r} has invalid data offsets {self.data_offsets!r}.")
        if self.quant_type is not None and not str(self.quant_type):
            raise ImporterError(f"Tensor {self.key!r} has an empty quantization type.")
        if self.physical_shape is not None and any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                                                   for value in self.physical_shape):
            raise ImporterError(f"Tensor {self.key!r} has invalid physical shape {self.physical_shape!r}.")

    @property
    def storage_type(self):
        """Exact packed storage type; logical dimensions remain in TensorInfo."""
        if self.quant_type is not None:
            if self.physical_shape is None:
                raise ImporterError(f"Quantized tensor {self.key!r} requires a physical shape.")
            return tensor_type(DType.UINT8, self.physical_shape)
        return tensor_type(self.dtype, self.shape)


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
    quant_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve())
        self._validate_limits()

    def _validate_limits(self) -> None:
        if (
            isinstance(self.offset, bool)
            or not isinstance(self.offset, int)
            or self.offset < 0
            or isinstance(self.nbytes, bool)
            or not isinstance(self.nbytes, int)
            or self.nbytes < 0
        ):
            raise ImporterError("TensorByteRange offset/nbytes must be non-negative integers.")

    @classmethod
    def _from_resolved_shard(cls, path: Path, offset: int, nbytes: int):
        """Use DirectoryCheckpoint's already canonical, verified shard path.

        Shard resolution enforces containment/existence and pins one path for
        the checkpoint handle. Every tensor still checks its range limits;
        re-resolving that same file per tensor adds no new validation.
        """
        result = object.__new__(cls)
        object.__setattr__(result, "path", path)
        object.__setattr__(result, "offset", offset)
        object.__setattr__(result, "nbytes", nbytes)
        result._validate_limits()
        return result


class Checkpoint(Protocol):

    @property
    def config(self) -> Mapping[str, Any]: ...

    @property
    def keys(self) -> tuple[str, ...]: ...

    def tensor_info(self, key: str) -> TensorInfo: ...

    def load_tensor(self, key: str, *, device: str = "cpu"): ...

    def tensor_byte_range(self, key: str) -> TensorByteRange | None: ...

    def cache_identity(self, keys: Iterable[str]) -> Mapping[str, Any] | None: ...

    def quantization_of(self, key: str) -> str | None:
        ...


class DirectoryCheckpoint:
    """A metadata-first checkpoint reader that never instantiates a model."""

    def __init__(self, root: str | Path, *, integrity_report: str | Path | None = None) -> None:
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
        self._tensor_ranges: dict[str, TensorByteRange] = {}
        self._resolved_shards: dict[str, Path] = {}
        self._weight_map = self._load_weight_map()
        self._source_digests: dict[Path, tuple[tuple[int, ...], str]] = {}
        self._tensor_handles = OrderedDict()
        self._tensor_handles_lock = RLock()
        if integrity_report is not None:
            self.attach_integrity_report(integrity_report)

    def attach_integrity_report(self, path: str | Path) -> None:
        """Bind a previously completed local SHA-256 audit to unchanged inputs.

        Reports are trusted producer evidence, like the private materialization
        cache. Both mtime and ctime must precede the report. Reuse is bound to
        the file's device, inode, size and both timestamps.
        """
        report_path = Path(path)
        report = json.loads(report_path.read_text())
        if (report.get("ok") is not True or Path(report.get("checkpoint", "")).resolve() != self.root
                or not isinstance(report.get("shards"), list)):
            raise ImporterError("Checkpoint integrity report does not describe this verified input.")
        audited_at = report_path.stat().st_mtime_ns
        bound = {}
        for record in report["shards"]:
            source = self._resolve_shard(str(record.get("file", "")))
            stat = source.stat()
            digest = record.get("sha256")
            if (record.get("ok") is not True or record.get("size") != stat.st_size
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                    or record.get("expected_sha256", digest) != digest
                    or max(stat.st_mtime_ns, stat.st_ctime_ns) > audited_at):
                raise ImporterError(f"Checkpoint integrity report is stale or invalid for {source}.")
            bound[source] = (self._source_identity(stat), digest)
        self._source_digests.update(bound)

    @staticmethod
    def _source_identity(stat) -> tuple[int, ...]:
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def source_file_digest(self, path: str | Path) -> str:
        """Digest one source file once, or reuse an unchanged audited identity."""
        source = Path(path).resolve()
        if source not in set(self._resolved_shards.values()):
            raise ImporterError(f"Digest source {source} is not a resolved checkpoint shard.")
        before = self._source_identity(source.stat())
        cached = self._source_digests.get(source)
        if cached is not None and cached[0] == before:
            return cached[1]
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        if self._source_identity(source.stat()) != before:
            raise ImporterError(f"Checkpoint source {source} changed during its digest.")
        value = digest.hexdigest()
        self._source_digests[source] = (before, value)
        return value

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
        # Each returned tensor retains its source storage. Opening one file
        # per tensor therefore retains one whole-shard mapping per weight,
        # exhausting the process's mapping budget on large expert banks.
        # Share a bounded set of readers; returned tensor views keep their
        # storage alive independently when a reader is evicted.
        cache_key = (path, str(device))
        with self._tensor_handles_lock:
            identity = self._source_identity(path.stat())
            cached = self._tensor_handles.pop(cache_key, None)
            if cached is not None and cached[0] != identity:
                cached[1].__exit__(None, None, None)
                cached = None
            if cached is None:
                handle = safe_open(path, framework="pt", device=device)
                handle.__enter__()
                cached = (identity, handle)
            self._tensor_handles[cache_key] = cached
            if len(self._tensor_handles) > 128:
                _, (_, evicted) = self._tensor_handles.popitem(last=False)
                evicted.__exit__(None, None, None)
            handle = cached[1]
            return handle.get_tensor(key)

    def tensor_byte_range(self, key: str) -> TensorByteRange:
        """Return the already-validated raw safetensors payload range."""

        cached = self._tensor_ranges.get(key)
        if cached is not None:
            return cached
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
        result = TensorByteRange._from_resolved_shard(path, payload_offset + start, end - start)
        self._tensor_ranges[key] = result
        return result

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
            if info.source not in sources:
                path = self._resolve_shard(info.source)
                stat = path.stat()
                sources[info.source] = {
                    "source": info.source,
                    "path": str(path),
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
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

    def quantization_of(self, key: str) -> str | None:
        """Return the device-side encoding for a tensor, if any.

        Safetensors checkpoints are represented by ordinary scalar dtypes and
        therefore do not need a quantized kernel selector.
        """

        self.tensor_info(key)
        return None

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
            validated_sources = set()
            for key, value in weight_map.items():
                tensor_key = str(key)
                source = str(value)
                if not tensor_key or tensor_key == "__metadata__":
                    raise ImporterError(
                        f"Safetensors index {index_candidates[0]} contains invalid tensor key {tensor_key!r}."
                    )
                if source not in validated_sources:
                    self._resolve_shard(source, require_exists=False)
                    validated_sources.add(source)
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
        if require_exists and source in self._resolved_shards:
            return self._resolved_shards[source]
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
        if require_exists:
            # Retain the canonical path, including an accepted HF blob target.
            # A later symlink change cannot redirect a cached reader outside
            # the validated tree. cache_identity still stats this backing file.
            self._resolved_shards[source] = path
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

    def quantization_of(self, key: str) -> str | None:
        self.tensor_info(key)
        return None


class GGUFCheckpoint:
    """Metadata-first GGUF checkpoint with packed quantized tensor ranges.

    ``gguf`` is an optional dependency because ordinary FlagMega users only
    need safetensors.  The package's ``GGUFReader`` uses a read-only memmap;
    quantized tensor values are never copied or dequantized by this class. Quantized
    tensor names are also exposed under the Qwen3.8 Hugging Face namespace so
    the existing layer importer can consume the same semantic graph.
    """

    _QWEN_PREFIX = "model.language_model."
    _QUANTIZED = frozenset({
        "Q4_0",
        "Q4_1",
        "Q5_0",
        "Q5_1",
        "Q8_0",
        "Q8_1",
        "Q2_K",
        "Q3_K",
        "Q4_K",
        "Q5_K",
        "Q6_K",
        "Q8_K",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ3_XXS",
        "IQ1_S",
        "IQ4_NL",
        "IQ3_S",
        "IQ2_S",
        "IQ4_XS",
        "IQ1_M",
        "TQ1_0",
        "TQ2_0",
        "MXFP4",
        "NVFP4",
        "Q1_0",
    })

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file() or self.path.suffix.lower() != ".gguf":
            raise ImporterError(f"GGUF checkpoint must be an existing .gguf file: {self.path}")
        try:
            from gguf import GGUFEndian, GGUFReader
        except ImportError as error:
            raise ImporterError("GGUF checkpoints require the optional 'gguf' package; "
                                "install it with `python -m pip install gguf`.") from error
        try:
            self._reader = GGUFReader(self.path)
        except Exception as error:
            raise ImporterError(f"Failed to read GGUF checkpoint {self.path}: {error}") from error
        if self._reader.endianess != GGUFEndian.LITTLE:
            raise ImporterError("Native GGUF kernels require a little-endian checkpoint.")
        self._raw = {tensor.name: tensor for tensor in self._reader.tensors}
        if not self._raw:
            raise ImporterError(f"GGUF checkpoint {self.path} contains no tensors.")
        self._aliases = self._build_aliases()
        self._virtual_scales: dict[str, str] = {}
        for canonical, raw in self._aliases.items():
            if not canonical.endswith(".weight") or canonical.endswith("embed_tokens.weight"):
                continue
            raw_tensor = self._raw.get(raw)
            if raw_tensor is not None and str(raw_tensor.tensor_type.name) in self._QUANTIZED:
                self._virtual_scales[canonical[:-len("weight")] + "weight_scale_inv"] = canonical
        self._infos: dict[str, TensorInfo] = {}
        self._config = self._build_config()

    @property
    def config(self) -> Mapping[str, Any]:
        return self._config

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted((*self._raw.keys(), *self._aliases.keys(), *self._virtual_scales.keys())))

    def tensor_info(self, key: str) -> TensorInfo:
        if key in self._infos:
            return self._infos[key]
        virtual_base = self._virtual_scales.get(key)
        if virtual_base is not None:
            base = self.tensor_info(virtual_base)
            if len(base.shape) != 2:
                raise ImporterError(f"GGUF virtual scale source {virtual_base!r} is not a matrix.")
            shape = tuple((int(value) + 127) // 128 for value in base.shape)
            info = TensorInfo(key, DType.FLOAT32, shape, self.path.name)
            self._infos[key] = info
            return info
        raw_key = self._aliases.get(key, key)
        tensor = self._raw.get(raw_key)
        if tensor is None:
            raise ImporterError(f"GGUF tensor {key!r} is missing.")
        raw_type = str(tensor.tensor_type.name)
        shape = tuple(int(value) for value in reversed(tuple(tensor.shape)))
        if key.endswith("linear_attn.conv1d.weight") and len(shape) == 2:
            # GGUF writes this [out_channels, kernel] tensor with reversed
            # dimensions.  The importer contract is HF's [out, 1, kernel].
            shape = (shape[0], 1, shape[1])
        physical_shape = tuple(int(value) for value in tensor.data.shape)
        if raw_type in self._QUANTIZED:
            # Logical dimensions describe decoded matrix elements; storage
            # is explicitly packed UINT8, never an FP8 or dense weight tensor.
            dtype = DType.UINT8
        else:
            dtype = {
                "F32": DType.FLOAT32,
                "F16": DType.FLOAT16,
                "BF16": DType.BFLOAT16,
                "F64": DType.FLOAT32,
                "I8": DType.INT32,
                "I16": DType.INT32,
                "I32": DType.INT32,
                "I64": DType.INT64,
            }.get(raw_type)
            if dtype is None:
                raise ImporterError(f"GGUF tensor {key!r} uses unsupported storage type {raw_type!r}.")
        info = TensorInfo(
            key,
            dtype,
            shape,
            self.path.name,
            quant_type=raw_type if raw_type in self._QUANTIZED else None,
            physical_shape=physical_shape,
        )
        self._infos[key] = info
        return info

    def quantization_of(self, key: str) -> str | None:
        return self.tensor_info(key).quant_type

    def load_tensor(self, key: str, *, device: str = "cpu"):
        info = self.tensor_info(key)
        if key in self._virtual_scales:
            try:
                import torch
            except ImportError as error:
                raise ImporterError("Loading GGUF virtual scales requires PyTorch.") from error
            return torch.ones(info.shape, dtype=torch.float32, device=device)
        if info.quant_type is not None:
            raise ImporterError(f"GGUF tensor {key!r} is packed as {info.quant_type}; "
                                "use a native quantized FlagMega kernel instead of loading a dequantized tensor.")
        tensor = self._raw[self._aliases.get(key, key)]
        try:
            import torch
        except ImportError as error:
            raise ImporterError("Loading scalar GGUF tensors requires PyTorch.") from error
        # gguf stores dimensions in reverse order.  Scalar tensors are the
        # only values currently materialized by the compiler-side loader.
        import numpy as np
        values = np.asarray(tensor.data)
        if str(tensor.tensor_type.name) == "BF16":
            values = (values.astype(np.uint32) << 16).view(np.float32)
        if key in self._aliases:
            # GGML stores the actual decay coefficient and RMS multiplier;
            # HF's semantic graph expects A_log and a zero-centred norm weight.
            # These tiny F32 parameters are unrelated to quantized matrices.
            if key.endswith("linear_attn.A_log"):
                if not np.all(values < 0):
                    raise ImporterError(f"GGUF SSM decay {key!r} must be strictly negative.")
                values = np.log(-values)
            elif key.endswith("norm.weight") and not key.endswith("linear_attn.norm.weight"):
                values = values - 1.0
        # Scalar metadata is the only GGUF payload materialized by the loader;
        # copy the read-only memmap before handing it to PyTorch so callers do
        # not receive a tensor whose writes have undefined behavior.
        result = torch.from_numpy(np.array(values, copy=True, order="C")).reshape(info.shape)
        dtype = {
            DType.FLOAT32: torch.float32,
            DType.FLOAT16: torch.float16,
            DType.BFLOAT16: torch.bfloat16,
            DType.INT32: torch.int32,
            DType.INT64: torch.int64,
        }[info.dtype]
        return result.to(device=device, dtype=dtype)

    def tensor_byte_range(self, key: str) -> TensorByteRange | None:
        info = self.tensor_info(key)
        if info.quant_type is None:
            return None
        tensor = self._raw[self._aliases.get(key, key)]
        return TensorByteRange(self.path, int(tensor.data_offset), int(tensor.n_bytes), info.quant_type)

    def cache_identity(self, keys: Iterable[str]) -> Mapping[str, Any]:
        stat = self.path.stat()
        tensors = []
        for key in sorted(set(str(value) for value in keys)):
            info = self.tensor_info(key)
            tensors.append({
                "key": key,
                "dtype": info.dtype.value,
                "shape": list(info.shape),
                "quant_type": info.quant_type,
                "physical_shape": None if info.physical_shape is None else list(info.physical_shape),
            })
        return {
            "schema": "flagmega.gguf-checkpoint-cache-identity/v2",
            "path": str(self.path),
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "tensors": tensors,
        }

    def _metadata(self, key: str, default: Any = None) -> Any:
        field = self._reader.get_field(key)
        if field is None:
            return default
        try:
            return field.contents()
        except Exception as error:
            raise ImporterError(f"Failed to decode GGUF metadata {key!r}: {error}") from error

    def _build_config(self) -> dict[str, Any]:
        arch = str(self._metadata("general.architecture", ""))
        if arch not in {"qwen35", "qwen3next"}:
            raise ImporterError(f"GGUF checkpoint architecture {arch!r} is not Qwen3.8/qwen35.")
        hidden = _gguf_positive_int(self._metadata(f"{arch}.embedding_length"), "embedding_length")
        intermediate = _gguf_positive_int(self._metadata(f"{arch}.feed_forward_length"), "feed_forward_length")
        layers = _gguf_positive_int(self._metadata(f"{arch}.block_count"), "block_count")
        inner = _gguf_positive_int(self._metadata(f"{arch}.ssm.inner_size"), "ssm.inner_size")
        key_heads = _gguf_positive_int(self._metadata(f"{arch}.ssm.group_count"), "ssm.group_count")
        value_dim = inner
        # Qwen3.8 stores [2 * key_dim + value_dim, hidden] in GGUF's logical
        # (reversed) shape.  Derive the remaining head geometry from layer 0
        # instead of hard-coding a model revision.
        qkv = self._raw.get("blk.0.attn_qkv.weight")
        if qkv is None:
            raise ImporterError("GGUF checkpoint is missing blk.0.attn_qkv.weight.")
        qkv_out = int(qkv.shape[1])
        if (qkv_out - value_dim) % 2:
            raise ImporterError("GGUF QKV output must contain equally sized query and key projections.")
        key_dim = (qkv_out - value_dim) // 2
        if key_dim <= 0 or key_dim % key_heads:
            raise ImporterError("GGUF Qwen3.8 QKV shape is incompatible with ssm.group_count.")
        key_head_dim = key_dim // key_heads
        value_heads = _gguf_positive_int(
            self._metadata(f"{arch}.ssm.time_step_rank", value_dim // key_head_dim),
            "ssm.time_step_rank",
        )
        if value_dim % value_heads:
            raise ImporterError("GGUF SSM inner size must divide into complete value heads.")
        layers -= int(self._metadata(f"{arch}.nextn_predict_layers", 0))
        if layers <= 0:
            raise ImporterError("GGUF checkpoint must contain at least one decoder layer beyond MTP.")
        interval = _gguf_positive_int(self._metadata(f"{arch}.full_attention_interval", 4), "full_attention_interval")
        vocab = self._metadata("tokenizer.ggml.tokens")
        vocab_size = len(vocab) if isinstance(vocab, list) else int(self._raw["token_embd.weight"].shape[1])
        text = {
            "layer_types": ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(layers)],
            "vocab_size":
            vocab_size,
            "num_hidden_layers":
            layers,
            "hidden_size":
            hidden,
            "intermediate_size":
            intermediate,
            "linear_num_key_heads":
            key_heads,
            "linear_num_value_heads":
            value_heads,
            "linear_key_head_dim":
            key_head_dim,
            "linear_value_head_dim":
            value_dim // value_heads,
            "linear_conv_kernel_dim":
            _gguf_positive_int(self._metadata(f"{arch}.ssm.conv_kernel"), "ssm.conv_kernel"),
            "rms_norm_eps":
            float(self._metadata(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)),
            "pad_token_id":
            self._metadata("tokenizer.ggml.padding_token_id"),
            "num_attention_heads":
            self._metadata(f"{arch}.attention.head_count"),
            "num_key_value_heads":
            self._metadata(f"{arch}.attention.head_count_kv"),
            "head_dim":
            self._metadata(f"{arch}.attention.key_length"),
            "rope_theta":
            self._metadata(f"{arch}.rope.freq_base", 10000000.0),
            "partial_rotary_factor":
            float(self._metadata(f"{arch}.rope.dimension_count", 64)) /
            int(self._metadata(f"{arch}.attention.key_length", 256)),
        }
        return {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": text,
            "quantization_config": {
                "quant_method": "gguf",
                "activation_scheme": "native",
                "fmt": "IQ4_XS/Q4_K",
                "weight_block_size": [128, 128],
            },
            "gguf": {
                "path": self.path.name, "architecture": arch, "value_head_order":
                "tiled" if arch == "qwen35" else "grouped"
            },
        }

    def _build_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for raw in self._raw:
            if raw == "output.weight":
                aliases["lm_head.weight"] = raw
                continue
            if raw == "output_norm.weight":
                aliases[self._QWEN_PREFIX + "norm.weight"] = raw
                continue
            if raw == "token_embd.weight":
                aliases[self._QWEN_PREFIX + "embed_tokens.weight"] = raw
                continue
            if not raw.startswith("blk."):
                continue
            parts = raw.split(".")
            if len(parts) < 3:
                continue
            try:
                layer = int(parts[1])
            except ValueError:
                continue
            tail = ".".join(parts[2:])
            names = {
                "attn_norm.weight": "input_layernorm.weight",
                "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
                "attn_gate.weight": "linear_attn.in_proj_z.weight",
                "attn_q.weight": "self_attn.q_proj.weight",
                "attn_k.weight": "self_attn.k_proj.weight",
                "attn_v.weight": "self_attn.v_proj.weight",
                "attn_q_norm.weight": "self_attn.q_norm.weight",
                "attn_k_norm.weight": "self_attn.k_norm.weight",
                "attn_output.weight": "self_attn.o_proj.weight",
                "ssm_beta.weight": "linear_attn.in_proj_b.weight",
                "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
                "ssm_conv1d.weight": "linear_attn.conv1d.weight",
                "ssm_a": "linear_attn.A_log",
                "ssm_dt.bias": "linear_attn.dt_bias",
                "ssm_norm.weight": "linear_attn.norm.weight",
                "ssm_out.weight": "linear_attn.out_proj.weight",
                "post_attention_norm.weight": "post_attention_layernorm.weight",
                "ffn_gate.weight": "mlp.gate_proj.weight",
                "ffn_up.weight": "mlp.up_proj.weight",
                "ffn_down.weight": "mlp.down_proj.weight",
            }
            mapped = names.get(tail)
            if mapped is not None:
                aliases[f"{self._QWEN_PREFIX}layers.{layer}.{mapped}"] = raw
        return aliases


def checkpoint_from_path(path: str | Path) -> Checkpoint:
    """Open either a safetensors directory or a single GGUF file."""

    value = Path(path)
    if value.is_file() and value.suffix.lower() == ".gguf":
        return GGUFCheckpoint(value)
    return DirectoryCheckpoint(value)


def _gguf_positive_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ImporterError(f"GGUF metadata {name!r} must be a positive integer.") from error
    if result <= 0:
        raise ImporterError(f"GGUF metadata {name!r} must be positive, got {result}.")
    return result


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
