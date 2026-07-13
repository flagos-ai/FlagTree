from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from typing import Any


@dataclass
class DebugLaunchContext:
    debug_kernel_id: int
    hidden_arg: int
    handle: Any | None = None
    metadata: Any | None = None


@dataclass
class DebugExportedRun:
    debug_kernel_id: int
    raw_buffer: bytes
    meta: dict[str, Any] | None = None
    runtime_metadata: dict[str, Any] | None = None
    decoded: dict[str, Any] | None = None
    report: str = ""


class DebugCollectRuntime:
    """Debugger runtime bridge for post-kernel export and host decode."""

    def __init__(self) -> None:
        self._exported_runs: list[DebugExportedRun] = []

    @staticmethod
    def _binding():
        return importlib.import_module("triton._C.libtriton").debugger

    @staticmethod
    def _metadata_to_dict(metadata: Any) -> dict[str, Any]:
        if metadata is None:
            return {}
        if isinstance(metadata, dict):
            result = dict(metadata)
        else:
            result = {}
            asdict = getattr(metadata, "_asdict", None)
            if callable(asdict):
                result.update(dict(asdict()))
            elif hasattr(metadata, "__dict__"):
                result.update(dict(vars(metadata)))

            for key in (
                    "debug_enabled",
                    "debug_kernel_id",
                    "debug_protocol_version",
                    "debug_record_level",
                    "debug_addr_level",
                    "debug_export_mode",
                    "debug_record_capacity",
                    "debug_record_size",
                    "debug_records_per_instance",
                    "debug_record_layout",
                    "debug_record_plan",
                    "debug_full_dump_payload_bytes_per_instance",
                    "debug_full_dump_plan",
                    "debug_backend_name",
                    "debug_target_name",
                    "debug_metadata_json",
                    "name",
                    "backend_name",
                    "target_name",
                    "target",
            ):
                try:
                    value = getattr(metadata, key)
                except AttributeError:
                    continue
                if not key.startswith("debug_") and key in result:
                    continue
                result[key] = value
        return result

    @staticmethod
    def _target_to_name(target: Any) -> str:
        if isinstance(target, dict):
            arch = target.get("arch")
            if arch is not None:
                return str(arch)
            backend = target.get("backend")
            if backend is not None:
                return str(backend)
            return ""
        arch = getattr(target, "arch", None)
        if arch is not None:
            return str(arch)
        backend = getattr(target, "backend", None)
        if backend is not None:
            return str(backend)
        return ""

    @staticmethod
    def _target_backend(target: Any) -> str:
        if isinstance(target, dict):
            backend = target.get("backend")
            return "" if backend is None else str(backend)
        backend = getattr(target, "backend", None)
        return "" if backend is None else str(backend)

    @staticmethod
    def _normalize_backend_name(backend_name: Any) -> str:
        name = "" if backend_name is None else str(backend_name)
        return name

    @staticmethod
    def _normalize_launch_metadata(metadata_dict: dict[str, Any]) -> dict[str, Any]:
        target = metadata_dict.get("target")
        target_name = (metadata_dict.get("debug_target_name") or metadata_dict.get("target_name")
                       or DebugCollectRuntime._target_to_name(target))
        backend_name = (metadata_dict.get("debug_backend_name") or os.environ.get("FLAGTREE_BACKEND")
                        or metadata_dict.get("backend_name") or DebugCollectRuntime._target_backend(target))
        metadata_dict["debug_backend_name"] = DebugCollectRuntime._normalize_backend_name(backend_name)
        metadata_dict["debug_target_name"] = str(target_name)
        return metadata_dict

    def prepare(self, metadata: Any, stream, runtime_metadata: Any | None = None) -> DebugLaunchContext:
        metadata_dict = self._normalize_launch_metadata(self._metadata_to_dict(metadata))
        runtime_metadata_dict = dict(runtime_metadata or {})
        if metadata_dict.get("debug_record_layout"):
            runtime_metadata_dict.setdefault("record_layout", metadata_dict["debug_record_layout"])
        if metadata_dict.get("debug_record_plan") is not None:
            runtime_metadata_dict.setdefault("record_plan", metadata_dict["debug_record_plan"])
        if metadata_dict.get("debug_records_per_instance") is not None:
            runtime_metadata_dict.setdefault("records_per_instance", int(metadata_dict["debug_records_per_instance"]))
        if metadata_dict.get("debug_full_dump_plan") is not None:
            runtime_metadata_dict.setdefault("full_dump_plan", metadata_dict["debug_full_dump_plan"])
        if (int(metadata_dict.get("debug_record_level", 1)) == 2
                and int(metadata_dict.get("debug_full_dump_payload_bytes_per_instance", 0)) > 0
                and metadata_dict.get("debug_full_dump_plan")):
            from triton.runtime import debugger as process_debugger

            if process_debugger.get_output_dir() is None:
                raise RuntimeError("level-2 debugger full dump requires debugger output_dir")
        handle = self._binding().prepare_launch(metadata_dict, int(stream or 0), runtime_metadata_dict)
        return DebugLaunchContext(
            debug_kernel_id=int(metadata_dict.get("debug_kernel_id", 0)),
            hidden_arg=int(handle.hidden_arg_value),
            handle=handle,
            metadata=metadata,
        )

    def hidden_arg(self, ctx: DebugLaunchContext) -> int:
        return int(ctx.hidden_arg)

    def export(self, ctx: DebugLaunchContext, stream) -> DebugExportedRun:
        del stream
        if ctx.handle is None:
            return DebugExportedRun(debug_kernel_id=ctx.debug_kernel_id, raw_buffer=b"")

        exported = ctx.handle.finish()
        ctx.handle = None
        metadata_dict = self._normalize_launch_metadata(self._metadata_to_dict(ctx.metadata))
        if (int(metadata_dict.get("debug_record_level", 1)) == 2
                and int(metadata_dict.get("debug_full_dump_payload_bytes_per_instance", 0)) > 0
                and metadata_dict.get("debug_full_dump_plan")):
            from triton.runtime import debugger as process_debugger

            finalized = process_debugger._finalize_exported_run(  # noqa: SLF001
                dict(exported), metadata_dict)
            run = DebugExportedRun(
                debug_kernel_id=ctx.debug_kernel_id,
                raw_buffer=bytes(finalized.get("raw_buffer", b"")),
                meta=dict(finalized.get("meta", {})),
                runtime_metadata=dict(finalized.get("runtime_metadata", {})),
                decoded=dict(finalized.get("decoded", {})),
                report=str(finalized.get("report", "")),
            )
            self._exported_runs.append(run)
            return run

        raw_buffer = bytes(exported.get("raw_buffer", b""))
        decoded = self._binding().decode_exported_run(exported)

        report = ""
        metadata_json = getattr(ctx.metadata, "debug_metadata_json", None)
        if metadata_json:
            report = self._binding().render_text_report(exported, str(metadata_json))

        run = DebugExportedRun(
            debug_kernel_id=ctx.debug_kernel_id,
            raw_buffer=raw_buffer,
            meta=dict(exported.get("meta", {})),
            runtime_metadata=dict(exported.get("runtime_metadata", {})),
            decoded=dict(decoded),
            report=report,
        )
        self._exported_runs.append(run)
        return run

    def release(self, ctx: DebugLaunchContext) -> None:
        if ctx.handle is not None:
            ctx.handle.release()
            ctx.handle = None

    def clear_exported_runs(self) -> None:
        self._exported_runs.clear()

    def peek_exported_runs(self) -> list[DebugExportedRun]:
        return list(self._exported_runs)

    def take_exported_runs(self) -> list[DebugExportedRun]:
        runs = list(self._exported_runs)
        self._exported_runs.clear()
        return runs


default_debug_collect_runtime = DebugCollectRuntime()
