from dataclasses import dataclass
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

_ir_import_error = None
_passes_import_error = None
_rpu_import_error = None
try:
    from triton._C.libtriton import ir
except ImportError as error:
    ir = None
    _ir_import_error = error
try:
    from triton._C.libtriton import passes
except ImportError as error:
    passes = None
    _passes_import_error = error
try:
    from triton._C.libtriton import rpu
except ImportError as error:
    rpu = None
    _rpu_import_error = error

from typing import Dict
from types import ModuleType

from triton.backends.compiler import BaseBackend, GPUTarget, Language


@dataclass(frozen=True)
class RPUOptions:
    num_warps: int = 1
    num_ctas: int = 1
    num_stages: int = 1
    cluster_dims: tuple = (1, 1, 1)
    debug: bool = False
    llvm_root: str = "/opt/rpu/llvm"
    rpu_asm_path: str = "/opt/rpu/llvm/bin/rpuasm"
    allow_fp8e4nv: bool = False
    allow_fp8e4b15: bool = False
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: tuple = ("ieee", )
    max_num_imprecise_acc_default: int = 0
    sanitize_overflow: bool = True
    arch: str = "rpu-v1"

    def hash(self):
        key = "_".join(f"{name}-{value}" for name, value in sorted(self.__dict__.items()))
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


# The compile pipeline (triton.compiler.compiler) serializes a stage's return
# value with str() for cache/dump artifacts and passes the same value to the
# next stage. RPU's rpuplan/rpuexec stages pass a rich bundle to the next stage
# but want a different on-disk artifact (the recognized rpu.plan.json, or a
# fallback summary, rather than the bundle's repr). Each bundle therefore
# defines __str__ to return that artifact text — defaulting to the module text —
# so the stock loop needs no split-stage hook to tell value and artifact apart.
@dataclass(frozen=True)
class RPUPlanStageBundle:
    module: object
    kernel_name: str
    pattern: str
    artifact_text: str | None = None

    def __str__(self):
        return self.artifact_text if self.artifact_text is not None else str(self.module)


@dataclass(frozen=True)
class RPUExecutableStageBundle:
    module: object
    kernel_name: str
    pattern: str
    source_kind: str
    fallback_reason: str | None = None
    artifact_text: str | None = None

    def __str__(self):
        return self.artifact_text if self.artifact_text is not None else str(self.module)


LEGACY_DIRECT_FALLBACK_REASONS = (
    "env_opt_out",
    "missing_executable_hooks",
    "unsupported_executable_shape",
)


class RPUBackend(BaseBackend):
    _module_contexts = {}
    supports_native_tensor_specialization = False

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "rpu"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "rpubin"

    def hash(self) -> str:
        key = f"rpu-{self.target.arch}-{self.target.warp_size}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def parse_options(self, options: dict) -> RPUOptions:
        accepted = {name: options[name] for name in RPUOptions.__dataclass_fields__ if name in options}
        accepted.setdefault("llvm_root", os.getenv("RPU_LLVM_ROOT", RPUOptions.llvm_root))
        if "rpu_asm_path" not in accepted:
            # The assembler ships inside the LLVM toolchain as
            # <llvm_root>/bin/rpuasm; RPU_ASM_PATH overrides it only when the
            # assembler lives outside the toolchain.
            accepted["rpu_asm_path"] = os.getenv("RPU_ASM_PATH") or str(Path(accepted["llvm_root"]) / "bin" / "rpuasm")
        return RPUOptions(**accepted)

    def add_stages(self, stages: dict, options: RPUOptions, language) -> None:
        if language != Language.TRITON:
            raise NotImplementedError(f"RPU backend does not yet support language={language}")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata)
        stages["rpuplan"] = lambda src, metadata: self.make_post_ttir(src, metadata)
        stages["rpuexec"] = lambda src, metadata: self.make_rpuexec(src, metadata)
        stages["rpurc"] = lambda src, metadata: self.make_rpurc(src, metadata)
        stages["rpuinstr"] = lambda src, metadata: self.make_rpuinstr(src, metadata, options)
        if os.getenv("RPU_OUTPUT_ELF") == "1":
            stages["rpuelf"] = lambda src, metadata: self.make_rpuelf(src, metadata)
        stages["rpubin"] = lambda src, metadata: self.make_rpubin(src, metadata)

    def load_dialects(self, context):
        if rpu is not None and hasattr(rpu, "load_dialects"):
            return rpu.load_dialects(context)
        return None

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

    def get_codegen_implementation(self, options):
        # min_dot_size must accept (lhs_type, rhs_type) and return a 3-tuple
        # of minimum dot shape (M, N, K). RPU has no special min — return 1s
        # so tl.dot accepts any shape; the actual dot tile lowering is enforced
        # later by the RPU executable dialect/emitter, not at the Triton front-end.
        return {"min_dot_size": lambda lhs_type, rhs_type: (1, 1, 1)}

    @staticmethod
    def _record_pipeline_event(metadata, stage: str, event: str, detail: dict | None = None):
        events = metadata.get("rpu_pipeline_events")
        if events is None:
            events = []
            metadata["rpu_pipeline_events"] = events
        events.append({"stage": stage, "event": event, "detail": detail or {}})

    @staticmethod
    def _record_direct_executable_lower_event(metadata, event, detail=None):
        RPUBackend._record_pipeline_event(
            metadata,
            "direct_executable_lower",
            event,
            detail,
        )

    @staticmethod
    def _record_rpurc_failure(metadata, failure_kind: str, error, summary: dict | None = None,
                              detail: dict | None = None):
        event_detail = {
            "kernel_name": summary.get("kernel_name") if summary is not None else metadata.get("name"),
            "pattern": summary.get("pattern") if summary is not None else metadata.get("rpu_pattern"),
            "source": "rpu_plan.kernel",
            "failure_kind": failure_kind,
            "error": str(error),
        }
        if detail:
            event_detail.update(detail)
        RPUBackend._record_pipeline_event(metadata, "rpurc_emit", "fail", event_detail)
        RPUBackend._write_debug_failure_manifest(metadata)

    @staticmethod
    def _debug_bundle_root():
        root = os.getenv("RPU_DEBUG_ARTIFACT_DIR", "")
        if not root:
            return None
        return Path(root).expanduser().resolve()

    @staticmethod
    def _debug_bundle_error(path, error):
        raise RuntimeError(f"RPU debug bundle failed for {path}: {error}") from error

    @staticmethod
    def _get_debug_bundle(metadata):
        root = RPUBackend._debug_bundle_root()
        if root is None:
            return None
        bundle = metadata.get("rpu_debug_bundle")
        if bundle is not None:
            return bundle
        try:
            root.mkdir(parents=True, exist_ok=True)
            bundle_dir = Path(tempfile.mkdtemp(prefix="rpu_debug_", dir=str(root))).resolve()
        except Exception as error:
            RPUBackend._debug_bundle_error(root, error)
        bundle = {
            "dir": str(bundle_dir),
            "manifest": str((bundle_dir / "manifest.json").resolve()),
            "files": {},
        }
        metadata["rpu_debug_bundle"] = bundle
        RPUBackend._write_debug_manifest(metadata)
        return bundle

    @staticmethod
    def _record_debug_bytes_artifact(metadata, logical_name, filename, data):
        bundle = RPUBackend._get_debug_bundle(metadata)
        if bundle is None:
            return
        path = Path(bundle["dir"]) / filename
        try:
            path.write_bytes(data)
        except Exception as error:
            RPUBackend._debug_bundle_error(path, error)
        bundle["files"][logical_name] = {
            "path": str(path.resolve()),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        RPUBackend._write_debug_manifest(metadata)

    @staticmethod
    def _record_debug_text_artifact(metadata, logical_name, filename, text):
        RPUBackend._record_debug_bytes_artifact(
            metadata,
            logical_name,
            filename,
            str(text).encode("utf-8"),
        )

    @staticmethod
    def _write_debug_manifest(metadata):
        bundle = metadata.get("rpu_debug_bundle")
        if bundle is None:
            return
        manifest_path = Path(bundle["manifest"])
        manifest = {
            "version": 1,
            "kernel_name": metadata.get("name"),
            "pattern": metadata.get("rpu_pattern"),
            "files": bundle.get("files", {}),
            "pipeline_events": metadata.get("rpu_pipeline_events", []),
        }
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        except Exception as error:
            RPUBackend._debug_bundle_error(manifest_path, error)

    @staticmethod
    def _write_debug_failure_manifest(metadata):
        if RPUBackend._debug_bundle_root() is None:
            return
        RPUBackend._get_debug_bundle(metadata)
        RPUBackend._write_debug_manifest(metadata)

    @staticmethod
    def _repro_clang_invocation(cmd):
        argv = list(cmd)
        if len(argv) >= 2 and argv[-2] == "-c":
            argv[-1] = "rpurc.rc"
        return {"cwd": ".", "argv": argv}

    @staticmethod
    def make_ttir(src, metadata):
        RPUBackend._record_pipeline_event(metadata, "ttir_cleanup", "start")
        metadata["shared"] = 0
        try:
            cleanup_passes = RPUBackend._run_ttir_cleanup_passes(src)
        except Exception as error:
            RPUBackend._record_pipeline_event(
                metadata,
                "ttir_cleanup",
                "fail",
                {"error": str(error)},
            )
            RPUBackend._record_debug_text_artifact(metadata, "ttir", "ttir.mlir", str(src))
            RPUBackend._write_debug_manifest(metadata)
            raise
        RPUBackend._record_pipeline_event(
            metadata,
            "ttir_cleanup",
            "end",
            {"passes": cleanup_passes},
        )
        RPUBackend._record_debug_text_artifact(metadata, "ttir", "ttir.mlir", str(src))
        return src

    @staticmethod
    def _run_ttir_cleanup_passes(mod):
        if ir is None:
            raise RuntimeError("RPU MLIR plan recognizer requires triton._C.libtriton.ir") from _ir_import_error
        if passes is None:
            raise RuntimeError("RPU MLIR plan recognizer requires triton._C.libtriton.passes") from _passes_import_error
        if not hasattr(passes, "common") or not hasattr(passes, "ttir"):
            return []

        pm = ir.pass_manager(mod.context)
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "rpu_ttir_cleanup")
        return [
            "common.inliner",
            "ttir.rewrite_tensor_pointer",
            "ttir.combine",
            "common.canonicalizer",
            "ttir.reorder_broadcast",
            "common.cse",
            "common.licm",
            "common.symbol_dce",
        ]

    @staticmethod
    def make_post_ttir(src, metadata):
        if isinstance(src, str) or not hasattr(src, "context"):
            raise RuntimeError("RPU post-ttir stage expects an MLIR module; TTIR text fallback is disabled")
        return RPUBackend._make_post_ttir_stage_from_mlir_module(src, metadata)

    @staticmethod
    def make_rpuplan(src, metadata):
        return RPUBackend.make_post_ttir(src, metadata)

    @staticmethod
    def _record_optional_plan_trace(mod, metadata):
        try:
            trace_json = RPUBackend._get_optional_module_str_attr(mod, "rpu.plan.trace")
            if trace_json is None:
                return
            trace = json.loads(trace_json)
        except Exception:
            return
        metadata["rpu_plan_trace"] = trace
        RPUBackend._record_debug_text_artifact(
            metadata,
            "rpuplan_trace",
            "rpu.plan.trace.json",
            trace_json,
        )

    @staticmethod
    def _record_rpuplan_mlir_artifact(mod, metadata):
        if RPUBackend._debug_bundle_root() is None:
            return
        RPUBackend._record_debug_text_artifact(
            metadata,
            "rpuplan_mlir",
            "rpu.plan.mlir",
            str(mod),
        )

    @staticmethod
    def _record_rpuexec_mlir_artifact(mod, metadata):
        if RPUBackend._debug_bundle_root() is None:
            return
        RPUBackend._record_debug_text_artifact(
            metadata,
            "rpuexec_mlir",
            "rpu.exec.mlir",
            str(mod),
        )

    @staticmethod
    def _make_post_ttir_stage_from_mlir_module(mod, metadata):
        if ir is None:
            raise RuntimeError("RPU MLIR plan recognizer requires triton._C.libtriton.ir") from _ir_import_error
        if rpu is None:
            raise RuntimeError("RPU MLIR plan recognizer requires triton._C.libtriton.rpu") from _rpu_import_error
        stage = RPUBackend._try_make_direct_executable_stage_from_ttir(mod, metadata)
        if stage is not None:
            return stage
        return RPUBackend._make_rpuplan_stage_from_mlir_module_via_recognizer(mod, metadata)

    @staticmethod
    def _make_rpuplan_from_mlir_module(mod, metadata):
        return RPUBackend._make_post_ttir_stage_from_mlir_module(mod, metadata)

    @staticmethod
    def _make_rpuexec_stage_bundle_from_module(mod):
        exec_summary = RPUBackend._get_rpuexec_kernel_summary(mod)
        # artifact_text defaults to None, so the bundle serializes as str(mod).
        bundle = RPUExecutableStageBundle(
            module=mod,
            kernel_name=exec_summary["kernel_name"],
            pattern=exec_summary["pattern"],
            source_kind="rpu_executable",
            fallback_reason=None,
        )
        return bundle, exec_summary

    @staticmethod
    def _try_make_direct_executable_stage_from_ttir(mod, metadata):
        if os.environ.get("RPU_USE_EXECUTABLE_DIALECT", "1") == "0":
            return None
        if not hasattr(rpu, "passes") or not hasattr(rpu.passes, "add_lower_supported_ttir_to_executable"):
            return None

        direct_executable_patterns = []
        if hasattr(rpu, "_supported_executable_kernel_kinds"):
            direct_executable_patterns = list(RPUBackend._supported_executable_patterns())

        RPUBackend._record_direct_executable_lower_event(
            metadata,
            "start",
            {"patterns": direct_executable_patterns},
        )
        try:
            pm = ir.pass_manager(mod.context)
            rpu.passes.add_lower_supported_ttir_to_executable(pm)
            pm.run(mod, "rpu_lower_supported_ttir_to_executable")
            direct_executable_module = mod
            metadata["rpu_direct_executable_candidate"] = True
            RPUBackend._record_direct_executable_lower_event(
                metadata,
                "candidate",
                {
                    "status": "available",
                    "patterns": direct_executable_patterns,
                },
            )
        except Exception as error:
            metadata["rpu_direct_executable_candidate"] = False
            RPUBackend._record_direct_executable_lower_event(
                metadata,
                "skip",
                {
                    "reason": "unsupported_ttir",
                    "error": str(error),
                },
            )
            return None

        try:
            direct_bundle, exec_summary = RPUBackend._make_rpuexec_stage_bundle_from_module(direct_executable_module)
        except Exception as error:
            RPUBackend._record_direct_executable_lower_event(
                metadata,
                "fail",
                {"error": str(error)},
            )
            RPUBackend._write_debug_failure_manifest(metadata)
            raise RuntimeError(
                "RPU direct executable stage summary failed while creating executable-first stage value") from error

        metadata["name"] = exec_summary["kernel_name"]
        metadata["shared"] = 0
        metadata["rpu_pattern"] = exec_summary["pattern"]
        RPUBackend._record_rpuexec_mlir_artifact(direct_executable_module, metadata)
        RPUBackend._record_direct_executable_lower_event(
            metadata,
            "artifact",
            {
                "status": "available",
                "source_boundary": "rpu_executable",
                "artifact_kind": "rpu_exec_mlir",
            },
        )
        RPUBackend._record_direct_executable_lower_event(
            metadata,
            "end",
            {
                "kernel_name": exec_summary["kernel_name"],
                "pattern": exec_summary["pattern"],
                "artifact_source": "rpu_exec_mlir",
            },
        )
        RPUBackend._write_debug_manifest(metadata)
        return direct_bundle

    @staticmethod
    def _make_rpuplan_stage_from_mlir_module_via_recognizer(mod, metadata):
        RPUBackend._record_pipeline_event(
            metadata,
            "rpuplan_recognize",
            "start",
        )
        pm = ir.pass_manager(mod.context)
        rpu.passes.add_recognize_plan(pm)
        try:
            pm.run(mod, "rpu_recognize_plan")
        except Exception as error:
            RPUBackend._record_pipeline_event(
                metadata,
                "rpuplan_recognize",
                "fail",
                {"error": str(error)},
            )
            RPUBackend._record_optional_plan_trace(mod, metadata)
            RPUBackend._write_debug_manifest(metadata)
            raise
        try:
            plan_json = rpu.get_module_str_attr(mod, "rpu.plan.json")
        except Exception as error:
            RPUBackend._record_pipeline_event(
                metadata,
                "rpuplan_recognize",
                "fail",
                {"error": str(error)},
            )
            RPUBackend._write_debug_failure_manifest(metadata)
            raise RuntimeError("RPU MLIR plan recognizer failed to produce rpu.plan.json artifact") from error
        try:
            summary = RPUBackend._get_rpuplan_kernel_summary(mod)
        except Exception as error:
            RPUBackend._record_pipeline_event(
                metadata,
                "rpuplan_recognize",
                "fail",
                {"error": str(error)},
            )
            RPUBackend._write_debug_failure_manifest(metadata)
            raise
        metadata["name"] = summary["kernel_name"]
        metadata["shared"] = 0
        metadata["rpu_pattern"] = summary["pattern"]
        RPUBackend._record_debug_text_artifact(metadata, "rpuplan", "rpu.plan.json", plan_json)
        RPUBackend._record_rpuplan_mlir_artifact(mod, metadata)
        RPUBackend._record_optional_plan_trace(mod, metadata)
        RPUBackend._record_pipeline_event(
            metadata,
            "rpuplan_recognize",
            "end",
            {"kernel_name": summary["kernel_name"], "pattern": summary["pattern"]},
        )
        RPUBackend._write_debug_manifest(metadata)
        # artifact_text=plan_json so the .rpuplan cache/dump artifact stays the
        # recognized plan JSON while the live module is passed to make_rpuexec.
        return RPUPlanStageBundle(
            module=mod,
            kernel_name=summary["kernel_name"],
            pattern=summary["pattern"],
            artifact_text=plan_json,
        )

    @staticmethod
    def _make_rpuplan_from_mlir_module_via_recognizer(mod, metadata):
        return RPUBackend._make_rpuplan_stage_from_mlir_module_via_recognizer(mod, metadata)

    @staticmethod
    def _get_optional_module_str_attr(mod, name):
        try:
            return rpu.get_module_str_attr(mod, name)
        except Exception:
            return None

    @staticmethod
    def _get_rpuplan_kernel_summary(mod):
        try:
            summary = rpu._get_rpuplan_kernel_summary(mod)
        except Exception as error:
            raise RuntimeError("RPU MLIR plan recognizer failed to produce a valid rpu_plan.kernel") from error
        return {
            "kernel_name": summary["kernel_name"],
            "pattern": summary["pattern"],
        }

    @staticmethod
    def _get_rpuexec_kernel_summary(mod):
        try:
            summary = rpu._get_rpuexec_kernel_summary(mod)
        except Exception as error:
            raise RuntimeError("RPU executable stage summary failed") from error
        return {
            "kernel_name": summary["kernel_name"],
            "pattern": summary["pattern"],
        }

    @staticmethod
    def _remember_module_context(mod, context):
        RPUBackend._module_contexts[id(mod)] = context
        try:
            mod.context = context
        except Exception:
            pass

    @staticmethod
    def _get_module_context(mod):
        context = getattr(mod, "context", None)
        if context is not None:
            return context
        context = RPUBackend._module_contexts.get(id(mod))
        if context is not None:
            return context
        raise RuntimeError("RPU MLIR module context is unavailable")

    @staticmethod
    def _supported_executable_patterns():
        return tuple(rpu._supported_executable_kernel_kinds())

    @staticmethod
    def _format_rpuexec_fallback_artifact(summary, reason, detail=None):
        return ("# rpu rpuexec direct fallback\n"
                f"kernel_name: {summary['kernel_name']}\n"
                f"pattern: {summary['pattern']}\n"
                "source_kind: rpu_plan_kernel\n"
                f"reason: {reason}\n" + (f"detail: {detail}\n" if detail else ""))

    @staticmethod
    def make_rpuexec(src, metadata):
        if isinstance(src, RPUExecutableStageBundle):
            summary = {"kernel_name": src.kernel_name, "pattern": src.pattern}
            metadata["name"] = summary["kernel_name"]
            metadata["shared"] = 0
            metadata["rpu_pattern"] = summary["pattern"]
            metadata["rpu_rpuexec_source_kind"] = src.source_kind
            RPUBackend._record_pipeline_event(
                metadata,
                "rpuexec_lower",
                "rpu_executable",
                {
                    "kernel_name": summary["kernel_name"],
                    "pattern": summary["pattern"],
                    "source_boundary": "executable_stage_value",
                },
            )
            RPUBackend._record_rpuexec_mlir_artifact(src.module, metadata)
            # Pass the executable bundle through unchanged; it serializes as
            # str(src.module) (artifact_text is None for executable bundles).
            return src

        if not isinstance(src, RPUPlanStageBundle):
            error = ("RPU make_rpuexec requires RPUPlanStageBundle or "
                     "RPUExecutableStageBundle from rpuplan stage")
            RPUBackend._record_pipeline_event(
                metadata,
                "rpuexec_lower",
                "fail",
                {"error": error},
            )
            RPUBackend._write_debug_failure_manifest(metadata)
            raise RuntimeError(error)

        summary = {"kernel_name": src.kernel_name, "pattern": src.pattern}
        metadata["name"] = summary["kernel_name"]
        metadata["shared"] = 0
        metadata["rpu_pattern"] = summary["pattern"]

        use_exec = os.environ.get("RPU_USE_EXECUTABLE_DIALECT", "1") != "0"
        has_executable_hooks = (hasattr(rpu, "_rpuplan_supports_executable_lowering")
                                and hasattr(rpu, "_lower_rpuplan_to_executable_module")
                                and hasattr(rpu, "get_module_str_attr") and hasattr(rpu, "passes")
                                and hasattr(rpu.passes, "add_emit_executable_to_rpurc")
                                and hasattr(rpu, "_get_rpuexec_kernel_summary")
                                and hasattr(rpu, "_supported_executable_kernel_kinds"))
        executable_patterns = tuple()
        if has_executable_hooks:
            executable_patterns = RPUBackend._supported_executable_patterns()
            metadata["rpu_executable_patterns"] = list(executable_patterns)
        reason = None
        fallback_detail = None
        if not use_exec:
            reason = "env_opt_out"
        elif not has_executable_hooks:
            reason = "missing_executable_hooks"
        elif summary["pattern"] not in executable_patterns:
            reason = "unsupported_executable_pattern"
        elif not rpu._rpuplan_supports_executable_lowering(src.module):
            reason = "unsupported_executable_shape"
            if hasattr(rpu, "_rpuplan_executable_lowering_failure_reason"):
                fallback_detail = rpu._rpuplan_executable_lowering_failure_reason(src.module)

        if reason is None:
            try:
                exec_module = rpu._lower_rpuplan_to_executable_module(src.module)
                try:
                    RPUBackend._remember_module_context(
                        exec_module,
                        RPUBackend._get_module_context(src.module),
                    )
                except RuntimeError:
                    pass
                exec_bundle, exec_summary = RPUBackend._make_rpuexec_stage_bundle_from_module(exec_module)
            except Exception as error:
                RPUBackend._record_pipeline_event(
                    metadata,
                    "rpuexec_lower",
                    "fail",
                    {"pattern": src.pattern, "error": str(error)},
                )
                RPUBackend._write_debug_failure_manifest(metadata)
                raise RuntimeError(f"RPU executable stage lowering failed for pattern {src.pattern}") from error

            metadata["rpu_rpuexec_source_kind"] = "rpu_executable"
            metadata["name"] = exec_summary["kernel_name"]
            metadata["rpu_pattern"] = exec_summary["pattern"]
            RPUBackend._record_pipeline_event(
                metadata,
                "rpuexec_lower",
                "rpu_executable",
                {
                    "kernel_name": exec_summary["kernel_name"],
                    "pattern": exec_summary["pattern"],
                },
            )
            RPUBackend._record_rpuexec_mlir_artifact(exec_module, metadata)
            return exec_bundle

        if reason not in LEGACY_DIRECT_FALLBACK_REASONS:
            error = ("RPU executable stage does not allow direct fallback for reason "
                     f"{reason} and pattern {summary['pattern']}")
            RPUBackend._record_pipeline_event(
                metadata,
                "rpuexec_lower",
                "fail",
                {
                    "error": error,
                    "pattern": summary["pattern"],
                    "fallback_reason": reason,
                    "supported_executable_patterns": list(executable_patterns),
                },
            )
            RPUBackend._write_debug_failure_manifest(metadata)
            raise RuntimeError(error)

        if not hasattr(rpu, "_direct_rpurc_supported_patterns"):
            error = ("RPU direct fallback requires _direct_rpurc_supported_patterns "
                     f"for pattern {summary['pattern']}")
            RPUBackend._record_pipeline_event(
                metadata,
                "rpuexec_lower",
                "fail",
                {
                    "error": error,
                    "pattern": summary["pattern"],
                    "fallback_reason": reason,
                },
            )
            RPUBackend._write_debug_failure_manifest(metadata)
            raise RuntimeError(error)

        direct_patterns = tuple(rpu._direct_rpurc_supported_patterns())
        metadata["rpu_direct_rpurc_patterns"] = list(direct_patterns)
        if summary["pattern"] not in direct_patterns:
            error = (f"RPU direct fallback for pattern {summary['pattern']} does not "
                     "have a direct rpurc stage emitter")
            RPUBackend._record_pipeline_event(
                metadata,
                "rpuexec_lower",
                "fail",
                {
                    "error": error,
                    "pattern": summary["pattern"],
                    "fallback_reason": reason,
                    "supported_patterns": list(direct_patterns),
                },
            )
            RPUBackend._write_debug_failure_manifest(metadata)
            raise RuntimeError(error)

        metadata["rpu_rpuexec_source_kind"] = "rpu_plan_kernel"
        fallback_event_detail = {
            "kernel_name": summary["kernel_name"],
            "pattern": summary["pattern"],
            "reason": reason,
        }
        if fallback_detail:
            fallback_event_detail["executable_builder_failure"] = fallback_detail
        RPUBackend._record_pipeline_event(
            metadata,
            "rpuexec_lower",
            "rpu_plan_kernel",
            fallback_event_detail,
        )
        # artifact_text is the human-readable fallback summary; the live module
        # is still passed to make_rpurc via the bundle.
        return RPUExecutableStageBundle(
            module=src.module,
            kernel_name=summary["kernel_name"],
            pattern=summary["pattern"],
            source_kind="rpu_plan_kernel",
            fallback_reason=reason,
            artifact_text=RPUBackend._format_rpuexec_fallback_artifact(summary, reason, fallback_detail),
        )

    @staticmethod
    def _emit_rpurc_from_executable_via_pass(module):
        context = RPUBackend._get_module_context(module)
        pm = ir.pass_manager(context)
        rpu.passes.add_emit_executable_to_rpurc(pm)
        pm.run(module, "rpu_emit_rpurc_from_executable")
        return {
            "kernel_name": rpu.get_module_str_attr(module, "rpu.rpurc.kernel_name"),
            "source_kind": rpu.get_module_str_attr(module, "rpu.rpurc.source_kind"),
            "source": rpu.get_module_str_attr(module, "rpu.rpurc.source"),
        }

    @staticmethod
    def make_rpurc(src, metadata):
        if not isinstance(src, RPUExecutableStageBundle):
            error = "RPU make_rpurc requires RPUExecutableStageBundle from rpuexec stage"
            RPUBackend._record_rpurc_failure(metadata, "missing_rpuexec_bundle", error)
            raise RuntimeError(error)

        summary = {"kernel_name": src.kernel_name, "pattern": src.pattern}
        metadata["name"] = summary["kernel_name"]
        metadata["shared"] = 0
        metadata["rpu_pattern"] = summary["pattern"]

        if src.source_kind == "rpu_executable":
            try:
                exec_result = RPUBackend._emit_rpurc_from_executable_via_pass(src.module)
            except Exception as error:
                RPUBackend._record_rpurc_failure(
                    metadata,
                    "executable_emitter_failure",
                    error,
                    summary,
                )
                raise RuntimeError(f"RPU executable rpurc emission failed for pattern {summary['pattern']}") from error

            metadata["name"] = exec_result["kernel_name"]
            metadata["shared"] = 0
            metadata["rpu_pattern"] = src.pattern
            metadata["rpu_rpurc_source_kind"] = exec_result["source_kind"]
            source = exec_result["source"]
            RPUBackend._record_pipeline_event(
                metadata,
                "rpurc_emit",
                "rpu_executable",
                {
                    "kernel_name": exec_result["kernel_name"],
                    "pattern": src.pattern,
                    "source": "rpu.executable",
                },
            )
            RPUBackend._record_debug_text_artifact(metadata, "rpurc", "rpurc.rc", source)
            return source

        if src.source_kind != "rpu_plan_kernel":
            error = f"RPU make_rpurc received unsupported source kind {src.source_kind}"
            RPUBackend._record_rpurc_failure(
                metadata,
                "unsupported_rpurc_source_kind",
                error,
                summary,
            )
            raise RuntimeError(error)

        try:
            direct_result = rpu._emit_rpurc_from_plan(src.module)
        except Exception as error:
            RPUBackend._record_rpurc_failure(
                metadata,
                "direct_emitter_failure",
                error,
                summary,
            )
            raise RuntimeError(f"RPU rpurc stage emission failed for pattern {summary['pattern']}") from error

        metadata["name"] = direct_result["kernel_name"]
        metadata["shared"] = 0
        metadata["rpu_pattern"] = direct_result["pattern"]
        metadata["rpu_rpurc_source_kind"] = "rpu_plan_kernel"
        source = direct_result["source"]
        RPUBackend._record_pipeline_event(
            metadata,
            "rpurc_emit",
            "rpu_plan_kernel",
            {
                "kernel_name": direct_result["kernel_name"],
                "pattern": direct_result["pattern"],
                "source": "rpu_plan.kernel",
            },
        )
        RPUBackend._record_debug_text_artifact(metadata, "rpurc", "rpurc.rc", source)
        return source

    @staticmethod
    def _resolve_clang(options: RPUOptions) -> Path:
        explicit = os.getenv("RPU_CLANG")
        if explicit:
            path = Path(explicit)
            if not path.exists():
                raise RuntimeError(f"RPU clang from RPU_CLANG not found: {path}")
            return path
        root = Path(options.llvm_root)
        # Hxcc release package layout: ${root}/bin/clang
        # Legacy x86 source-tree layout: ${root}/build/bin/clang
        for candidate in (root / "bin" / "clang", root / "build" / "bin" / "clang"):
            if candidate.exists():
                return candidate
        raise RuntimeError(f"RPU clang not found under {root}; tried bin/clang and build/bin/clang")

    @staticmethod
    def _clang_command(options: RPUOptions, emit_elf: bool = False):
        clang = RPUBackend._resolve_clang(options)
        return [
            str(clang),
            "-O2",
            "-std=c++17",
            "-fno-exceptions",
            "--rpu-elf=1" if emit_elf else "--rpu-elf=0",
            "--target=rpu-rhino-rpuhsa",
            "--rpu-device-only",
            f"--rpuas-path={options.rpu_asm_path}",
            "-mllvm",
            "-disable-lsr",
            "-mllvm",
            "-enable-int2short=true",
            "-mllvm",
            "-dse-memoryssa-scanlimit=5000",
            "-mllvm",
            "-structurizecfg-reorder-region",
            "-mllvm",
            "-machine-sink-split=0",
            "-mllvm",
            "-hoist-cheap-insts=true",
            "-mllvm",
            "-rpu-convert-loopstep-to-repeat",
            "-mllvm",
            "-simplifycfg-max-small-block-size=0",
        ]

    @staticmethod
    def _rpuinstr_text_from_rpubin(bin_bytes: bytes) -> str:
        """Derive the rpuinstr listing from the .rpubin payload.

        The rpuinstr text artifact is just the .rpubin payload re-printed as a
        hex listing, one instruction word per line. Deriving it here keeps
        make_rpuinstr independent
        of whether the assembler also wrote a separate ``{kernel}.o`` text file:
        the standalone ``rpuasm.bin`` emits only the ``-*.bin`` payload, whereas
        the legacy ``rpuasm`` additionally wrote a ``.o`` whose contents are
        byte-for-byte this derivation.
        """
        if len(bin_bytes) % 8 != 0:
            raise RuntimeError(f"rpubin length {len(bin_bytes)} is not a multiple of 8; "
                               "cannot derive rpuinstr listing")
        return "".join(bin_bytes[offset:offset + 8][::-1].hex() + "\n" for offset in range(0, len(bin_bytes), 8))

    @staticmethod
    def make_rpuinstr(src, metadata, options: RPUOptions):
        kernel_name = metadata.get("name", "rpu_kernel")
        RPUBackend._record_debug_text_artifact(metadata, "rpurc", "rpurc.rc", str(src))
        RPUBackend._record_pipeline_event(
            metadata,
            "rpuinstr_compile",
            "start",
            {"kernel_name": kernel_name},
        )
        with tempfile.TemporaryDirectory(prefix="triton_rpu_") as temp_dir:
            temp_path = Path(temp_dir)
            src_path = temp_path / f"{kernel_name}.rc"
            src_path.write_text(str(src))
            env = os.environ.copy()
            env["TMPDIR"] = temp_dir
            try:
                cmd = RPUBackend._clang_command(options) + ["-c", str(src_path)]
            except Exception as error:
                RPUBackend._record_pipeline_event(
                    metadata,
                    "rpuinstr_compile",
                    "fail",
                    {"kernel_name": kernel_name, "error": str(error)},
                )
                RPUBackend._write_debug_failure_manifest(metadata)
                raise
            metadata["rpu_clang_command"] = cmd
            RPUBackend._record_debug_text_artifact(
                metadata,
                "clang_command",
                "clang_command.json",
                json.dumps(RPUBackend._repro_clang_invocation(cmd), indent=2, sort_keys=True) + "\n",
            )
            result = subprocess.run(cmd, cwd=temp_dir, env=env, text=True, capture_output=True)
            metadata["rpu_clang_output"] = {
                "stdout": result.stdout.rstrip(),
                "stderr": result.stderr.rstrip(),
            }
            RPUBackend._record_debug_text_artifact(
                metadata,
                "clang_stdout",
                "clang_stdout.txt",
                result.stdout,
            )
            RPUBackend._record_debug_text_artifact(
                metadata,
                "clang_stderr",
                "clang_stderr.txt",
                result.stderr,
            )
            if result.returncode != 0:
                RPUBackend._record_pipeline_event(
                    metadata,
                    "rpuinstr_compile",
                    "fail",
                    {
                        "kernel_name": kernel_name,
                        "returncode": result.returncode,
                        "stdout": result.stdout.rstrip(),
                        "stderr": result.stderr.rstrip(),
                    },
                )
                RPUBackend._write_debug_manifest(metadata)
                raise RuntimeError("RPU clang failed\n"
                                   f"command: {' '.join(cmd)}\n"
                                   f"stdout:\n{result.stdout.rstrip()}\n"
                                   f"stderr:\n{result.stderr.rstrip()}")

            bin_paths = sorted(temp_path.glob(f"{kernel_name}-*.bin"), key=lambda path: path.stat().st_mtime)
            if not bin_paths:
                RPUBackend._record_pipeline_event(
                    metadata,
                    "rpuinstr_compile",
                    "fail",
                    {"kernel_name": kernel_name, "error": "missing compiler outputs"},
                )
                RPUBackend._write_debug_manifest(metadata)
                raise RuntimeError(f"RPU clang did not emit expected outputs for {kernel_name}")
            bin_bytes = bin_paths[-1].read_bytes()
            # Derive the rpuinstr listing from the .rpubin instead of reading a
            # separate {kernel}.o: the standalone rpuasm.bin emits only the
            # -*.bin payload (no .o). The derivation is byte-identical to the
            # legacy .o (verified for add/mul).
            instr_bytes = RPUBackend._rpuinstr_text_from_rpubin(bin_bytes).encode()
            metadata["rpu_artifacts"] = {
                "rpubin": {
                    "bytes": len(bin_bytes),
                    "sha256": hashlib.sha256(bin_bytes).hexdigest(),
                },
                "rpuinstr": {
                    "bytes": len(instr_bytes),
                    "sha256": hashlib.sha256(instr_bytes).hexdigest(),
                },
            }
            metadata["_rpubin_hex"] = bin_bytes.hex()
            RPUBackend._record_debug_text_artifact(
                metadata,
                "rpuinstr",
                "rpuinstr.txt",
                instr_bytes.decode(),
            )
            RPUBackend._record_debug_bytes_artifact(
                metadata,
                "rpubin",
                "rpubin.bin",
                bin_bytes,
            )
            if os.getenv("RPU_OUTPUT_ELF") == "1":
                RPUBackend._emit_rpuelf(kernel_name, src_path, temp_path, temp_dir, env, metadata, options)
            RPUBackend._record_pipeline_event(
                metadata,
                "rpuinstr_compile",
                "end",
                {"kernel_name": kernel_name, "artifacts": metadata["rpu_artifacts"]},
            )
            RPUBackend._write_debug_manifest(metadata)
            return instr_bytes.decode()

    @staticmethod
    def _emit_rpuelf(kernel_name, src_path, temp_path, temp_dir, env, metadata, options: RPUOptions):
        """Second clang invocation with --rpu-elf=1 to emit the device ELF.

        Runs only when RPU_OUTPUT_ELF=1. The first invocation emits the legacy
        .o text / -*.bin payload that make_rpubin still consumes; the ELF here
        is the launch_kernel-ready artifact and is recorded alongside, not in
        place of, the existing rpubin.
        """
        cmd = RPUBackend._clang_command(options, emit_elf=True) + ["-c", str(src_path)]
        result = subprocess.run(cmd, cwd=temp_dir, env=env, text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError("RPU clang ELF emission failed\n"
                               f"command: {' '.join(cmd)}\n"
                               f"stdout:\n{result.stdout.rstrip()}\n"
                               f"stderr:\n{result.stderr.rstrip()}")
        elf_path = temp_path / f"{kernel_name}.ref"
        if not elf_path.exists():
            raise RuntimeError(f"RPU clang did not emit expected ELF for {kernel_name}")
        elf_bytes = elf_path.read_bytes()
        metadata["rpu_artifacts"]["rpuelf"] = {
            "bytes": len(elf_bytes),
            "sha256": hashlib.sha256(elf_bytes).hexdigest(),
        }
        metadata["_rpuelf_hex"] = elf_bytes.hex()
        RPUBackend._record_debug_bytes_artifact(metadata, "rpuelf", "rpuelf.ref", elf_bytes)

    @staticmethod
    def make_rpuelf(src, metadata):
        elf_hex = metadata.pop("_rpuelf_hex", None)
        if elf_hex is None:
            raise RuntimeError("RPU ELF payload missing from previous compile stage")
        # Surface the device ELF as hex text so the stock compile loop serializes
        # and reads kernel.asm["rpuelf"] as text (no special binary handling).
        # Consumers recover the bytes via bytes.fromhex(kernel.asm["rpuelf"]).
        return elf_hex

    @staticmethod
    def make_rpubin(src, metadata):
        bin_hex = metadata.pop("_rpubin_hex", None)
        if bin_hex is None:
            raise RuntimeError("RPU binary payload missing from previous compile stage")
        return bytes.fromhex(bin_hex)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )
