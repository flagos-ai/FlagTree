# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Hierarchical, flag-controlled compiler dumps inspired by nncase.

Python checkpoints are the authoritative editable representation.  Every IR
dump also gets the dialect-appropriate compact ``.il`` or ``.script`` view.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
from collections import OrderedDict
from contextlib import AbstractContextManager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path, PurePath
from typing import IO, Iterator
from contextlib import contextmanager

from triton.flagmega.diagnostics.paths import encode_file_component
from triton.flagmega.ir.print_weights import WeightPrintAnalysis

from triton.flagmega.ir import (
    FunctionDumpInfo,
    IRModule,
    companion_suffix,
    function_module,
    module_source,
    text_source,
)


class DumpFlags(IntFlag):
    """Categories compatible with the nncase dump flag model."""

    NONE = 0
    IMPORT_OPS = 1 << 1
    PASS_IR = 1 << 2
    EGRAPH_COST = 1 << 3
    REWRITE = 1 << 4
    CALIBRATION = 1 << 5
    EVALUATOR = 1 << 6
    COMPILE = 1 << 7
    TILING = 1 << 8
    SCHEDULE = 1 << 9
    CODEGEN = 1 << 10

    # Familiar spellings for code ported from Nncase.Diagnostics.DumpFlags.
    Nothing = NONE
    ImportOps = IMPORT_OPS
    PassIR = PASS_IR
    EGraphCost = EGRAPH_COST
    Rewrite = REWRITE
    Calibration = CALIBRATION
    Evaluator = EVALUATOR
    Compile = COMPILE
    Tiling = TILING
    Schedule = SCHEDULE
    CodeGen = CODEGEN


_FLAG_NAMES = {
    "none": DumpFlags.NONE,
    "nothing": DumpFlags.NONE,
    "import-ops": DumpFlags.IMPORT_OPS,
    "pass-ir": DumpFlags.PASS_IR,
    "egraph-cost": DumpFlags.EGRAPH_COST,
    "rewrite": DumpFlags.REWRITE,
    "calibration": DumpFlags.CALIBRATION,
    "evaluator": DumpFlags.EVALUATOR,
    "compile": DumpFlags.COMPILE,
    "tiling": DumpFlags.TILING,
    "schedule": DumpFlags.SCHEDULE,
    "codegen": DumpFlags.CODEGEN,
}

_FLAG_ALIASES = {
    "importops": "import-ops",
    "passir": "pass-ir",
    "egraphcost": "egraph-cost",
}


def parse_dump_flags(value: str | DumpFlags | int | None) -> DumpFlags:
    """Parse a comma/pipe separated CLI spelling into :class:`DumpFlags`."""

    if value is None:
        return DumpFlags.NONE
    if isinstance(value, DumpFlags):
        return value
    if isinstance(value, int):
        return DumpFlags(value)
    result = DumpFlags.NONE
    names = [item.strip().lower().replace("_", "-") for item in re.split(r"[,|+]", value)]
    for name in names:
        name = _FLAG_ALIASES.get(name, name)
        if not name:
            continue
        if name == "all":
            for flag in _FLAG_NAMES.values():
                result |= flag
            continue
        try:
            result |= _FLAG_NAMES[name]
        except KeyError as error:
            available = ", ".join(sorted((*_FLAG_NAMES, "all")))
            raise ValueError(f"Unknown dump flag {name!r}; available: {available}.") from error
    return result


def dump_flags_text(flags: DumpFlags) -> str:
    if flags == DumpFlags.NONE:
        return "none"
    return ",".join(name for name, flag in _FLAG_NAMES.items() if name != "nothing" and flag and flags & flag)


@dataclass(frozen=True)
class DumpRecord:
    category: str
    function: str
    relative_path: str
    companion_path: str
    dialect: str
    stage: str
    semantic_hash: str
    module_semantic_hash: str

    def to_data(self) -> dict[str, str]:
        return {
            "category": self.category,
            "function": self.function,
            "relative_path": self.relative_path,
            "companion_path": self.companion_path,
            "dialect": self.dialect,
            "stage": self.stage,
            "semantic_hash": self.semantic_hash,
            "module_semantic_hash": self.module_semantic_hash,
        }


@dataclass(frozen=True)
class ArtifactRecord:
    """Indexed non-IR diagnostic output (dot/cost/value/schedule/codegen)."""

    category: str
    kind: str
    producer: str
    relative_path: str
    size: int
    sha256: str
    source_semantic_hash: str | None = None

    def to_data(self) -> dict[str, object]:
        return {
            "category": self.category,
            "kind": self.kind,
            "producer": self.producer,
            "relative_path": self.relative_path,
            "size": self.size,
            "sha256": self.sha256,
            "source_semantic_hash": self.source_semantic_hash,
        }


@dataclass(frozen=True)
class FunctionDump:
    function: str
    checkpoint: Path
    text_dump: Path
    semantic_hash: str

    def to_data(self) -> dict[str, str]:
        return {
            "function": self.function,
            "checkpoint": str(self.checkpoint),
            "text_dump": str(self.text_dump),
            "semantic_hash": self.semantic_hash,
        }


@dataclass(frozen=True)
class ModuleDump:
    directory: Path
    functions: tuple[FunctionDump, ...]

    def function(self, name: str) -> FunctionDump:
        try:
            return next(item for item in self.functions if item.function == name)
        except StopIteration as error:
            raise KeyError(name) from error


@dataclass(frozen=True)
class _RenderedFunctionDump:
    owner: IRModule
    source: str
    companion: str
    suffix: str
    semantic_hash: str


class DumpManager:
    """Own the dump tree, inherited flags, and machine-readable index."""

    def __init__(self, directory: str | Path | None, flags: DumpFlags = DumpFlags.NONE) -> None:
        self.directory = None if directory is None else Path(directory)
        self.flags = DumpFlags(flags)
        self._records: list[DumpRecord] = []
        self._artifacts: list[ArtifactRecord] = []
        # Manager/pass/compile boundaries often dump the identical immutable
        # module object two or three times. Keep a bounded render cache so the
        # files remain independent while verification and Python/IL rendering
        # are performed only once per function view.
        self._render_cache: OrderedDict[
            tuple[int, str, bool], _RenderedFunctionDump
        ] = OrderedDict()
        self._weight_analysis: WeightPrintAnalysis | None = None
        if self.directory is not None and self.flags != DumpFlags.NONE:
            self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Dumper:
        return Dumper(self, Path(), self.flags)

    @property
    def records(self) -> tuple[DumpRecord, ...]:
        return tuple(self._records)

    @property
    def artifacts(self) -> tuple[ArtifactRecord, ...]:
        return tuple(self._artifacts)

    def _record(self, record: DumpRecord) -> None:
        self._records.append(record)
        if self.directory is None:
            return
        destination = self.directory / "dumps.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, delete=False, prefix=f".{destination.name}."
        ) as stream:
            json.dump([item.to_data() for item in self._records], stream, indent=2, sort_keys=True)
            stream.write("\n")
            temporary = Path(stream.name)
        os.replace(temporary, destination)

    def _record_artifact(self, record: ArtifactRecord) -> None:
        if any(value.relative_path == record.relative_path for value in self._artifacts):
            raise RuntimeError(f"Diagnostic artifact {record.relative_path!r} was written twice.")
        self._artifacts.append(record)
        if self.directory is None:
            return
        _atomic_json(
            self.directory / "artifacts.json",
            {
                "schema": "flagmega.diagnostic-artifacts/v1",
                "artifacts": [item.to_data() for item in self._artifacts],
            },
        )

    def _render_function(
        self,
        module: IRModule,
        function_name: str,
        function_index: int,
    ) -> _RenderedFunctionDump:
        include_unreferenced = function_name == module.entry
        key = (id(module), function_name, include_unreferenced)
        cached = self._render_cache.get(key)
        if cached is not None and cached.owner is module:
            self._render_cache.move_to_end(key)
            return cached
        view = function_module(
            module,
            function_name,
            include_unreferenced=include_unreferenced,
        )
        if self._weight_analysis is None or self._weight_analysis.owner is not module:
            self._weight_analysis = WeightPrintAnalysis.analyze(module)
        rendered = _RenderedFunctionDump(
            owner=module,
            source=module_source(
                view,
                dump_info=FunctionDumpInfo(
                    module_entry=module.entry,
                    function_name=function_name,
                    function_index=function_index,
                    function_count=len(module.functions),
                    module_semantic_hash=module.semantic_hash,
                    node_order=tuple(node.id for node in module.nodes),
                    selection_point_order=tuple(
                        point.id for point in module.selection_points
                    ),
                    selection_order=tuple(
                        record.point_id for record in module.selections
                    ),
                ),
            ),
            companion=text_source(view, weight_analysis=self._weight_analysis),
            suffix=companion_suffix(view),
            semantic_hash=view.semantic_hash,
        )
        self._render_cache[key] = rendered
        self._render_cache.move_to_end(key)
        while len(self._render_cache) > 16:
            self._render_cache.popitem(last=False)
        return rendered


@dataclass(frozen=True)
class Dumper:
    """A path-scoped dump writer; child flags can only restrict the parent."""

    manager: DumpManager
    relative_directory: Path
    flags: DumpFlags

    @property
    def directory(self) -> Path | None:
        if self.manager.directory is None:
            return None
        return self.manager.directory / self.relative_directory

    def is_enabled(self, flags: DumpFlags) -> bool:
        return bool(self.flags & flags == flags)

    def create_sub_dumper(self, subdirectory: str | Path, flags: DumpFlags | None = None) -> Dumper:
        relative = _safe_relative(subdirectory)
        child_flags = self.flags if flags is None else self.flags & DumpFlags(flags)
        return Dumper(self.manager, self.relative_directory / relative, child_flags)

    # Alias the corrected Python spelling and the historical nncase spelling.
    create_sub_dummper = create_sub_dumper

    def dump_module(
        self,
        module: IRModule,
        prefix: str,
        *,
        category: DumpFlags,
    ) -> ModuleDump | None:
        if not self.is_enabled(category) or self.directory is None:
            return None
        directory = self.directory / _safe_relative(prefix)
        function_dumps: list[FunctionDump] = []
        root = self.manager.directory
        assert root is not None
        for function_index, function in enumerate(module.functions):
            rendered = self.manager._render_function(
                module, function.name, function_index
            )
            path = directory / f"{_file_name(function.name)}.py"
            _atomic_text(path, rendered.source)
            companion = path.with_suffix(rendered.suffix)
            _atomic_text(companion, rendered.companion)
            function_dump = FunctionDump(
                function.name, path, companion, rendered.semantic_hash
            )
            function_dumps.append(function_dump)
            self.manager._record(DumpRecord(
                category=_flag_name(category),
                function=function.name,
                relative_path=path.relative_to(root).as_posix(),
                companion_path=companion.relative_to(root).as_posix(),
                dialect=module.dialect,
                stage=module.stage,
                semantic_hash=rendered.semantic_hash,
                module_semantic_hash=module.semantic_hash,
            ))
        return ModuleDump(directory, tuple(function_dumps))

    def open_file(self, relative_path: str | Path, mode: str = "w", **kwargs: object) -> IO[object]:
        if self.directory is None:
            raise RuntimeError("Dump output is disabled because no dump directory was configured.")
        path = self.directory / _safe_relative(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open(mode, **kwargs)  # type: ignore[return-value]

    @contextmanager
    def open_artifact(
        self,
        relative_path: str | Path,
        mode: str = "w",
        *,
        category: DumpFlags,
        kind: str,
        producer: str,
        source_semantic_hash: str | None = None,
        **kwargs: object,
    ) -> Iterator[IO[object]]:
        """Write and checksum one flag-gated non-IR diagnostic artifact."""

        if not self.is_enabled(category):
            raise RuntimeError(
                f"Diagnostic producer {producer!r} tried to write disabled category "
                f"{_flag_name(category)!r}."
            )
        if not kind or not producer:
            raise ValueError("Diagnostic artifacts require non-empty kind and producer.")
        relative = _safe_relative(relative_path)
        stream = self.open_file(relative, mode, **kwargs)
        try:
            yield stream
        finally:
            stream.close()
        path = self.directory / relative  # type: ignore[operator]
        root = self.manager.directory
        assert root is not None
        payload = path.read_bytes()
        self.manager._record_artifact(ArtifactRecord(
            category=_flag_name(category),
            kind=kind,
            producer=producer,
            relative_path=path.relative_to(root).as_posix(),
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            source_semantic_hash=source_semantic_hash,
        ))


_CURRENT_DUMPER: ContextVar[Dumper | None] = ContextVar("flagmega_current_dumper", default=None)


class DumpScope(AbstractContextManager[Dumper]):
    """Context-local hierarchical dumper, matching nncase's DumpScope model."""

    def __init__(
        self,
        dumper_or_subdirectory: Dumper | str | Path,
        flags: DumpFlags | None = None,
    ) -> None:
        current = self.current()
        self.dumper = (
            dumper_or_subdirectory
            if isinstance(dumper_or_subdirectory, Dumper)
            else current.create_sub_dumper(dumper_or_subdirectory, flags)
        )
        self._token = None

    @staticmethod
    def current() -> Dumper:
        current = _CURRENT_DUMPER.get()
        if current is None:
            return DumpManager(None).root
        return current

    def __enter__(self) -> Dumper:
        self._token = _CURRENT_DUMPER.set(self.dumper)
        return self.dumper

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._token is not None:
            _CURRENT_DUMPER.reset(self._token)


def _safe_relative(path: str | Path) -> Path:
    candidate = PurePath(path)
    if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"Dump path must be a non-empty safe relative path, got {str(path)!r}.")
    return Path(*candidate.parts)


def _flag_name(flag: DumpFlags) -> str:
    for name, candidate in _FLAG_NAMES.items():
        if candidate == flag and name != "nothing":
            return name
    return str(int(flag))


def _file_name(name: str) -> str:
    if not name:
        raise ValueError(f"Function name {name!r} cannot be represented as a dump filename.")
    return encode_file_component(name)


def _atomic_text(destination: Path, source: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(destination: Path, value: object) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False,
        prefix=f".{destination.name}.",
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, destination)


__all__ = [
    "ArtifactRecord",
    "DumpFlags",
    "FunctionDump",
    "DumpManager",
    "DumpRecord",
    "DumpScope",
    "Dumper",
    "ModuleDump",
    "dump_flags_text",
    "parse_dump_flags",
]
