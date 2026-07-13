"""FlagTree debugger: TTIR pass orchestration and compile-time metadata keys."""
from __future__ import annotations

import json
import os

from triton._C.libtriton import ir, passes

_instrumentation_mode = ""
_DISABLED_BUILD_MESSAGE = ("FlagTree debugger support is not available in this build; rebuild with "
                           "-DFLAGTREE_ENABLE_DEBUGGER=ON")


def set_instrumentation_mode(mode: str) -> None:
    global _instrumentation_mode
    _instrumentation_mode = str(mode or "")


def get_instrumentation_mode() -> str:
    return _instrumentation_mode


def _get_debug_passes():
    return getattr(passes, "flagtree_debug", None)


def _run_pass_manager(pm, mod, description: str) -> None:
    try:
        pm.run(mod, description)
    except TypeError:
        pm.run(mod)


def _debug_launch_hidden_arg_enabled() -> bool:
    # Keep the environment variable as a compatibility hook for subprocesses
    # and older scripts, but make the Python debugger API the normal path.
    if os.environ.get("TRITON_FLAGTREE_DEBUG_LAUNCH_PTR", "") == "1":
        return True
    try:
        from triton.runtime import debugger

        return debugger.is_active()
    except Exception:
        return False


def _kernel_internal_timeline_supported() -> bool:
    try:
        import triton

        backend = str(triton.runtime.driver.active.get_current_target().backend).lower()
    except Exception:
        return False
    return backend in {"ascend", "npu", "cann"}


def _finish_metadata_only_tensor_pointer_debug(fd, mod, metadata: dict) -> bool:
    has_tensor_pointer = getattr(fd, "has_triton_tensor_pointer_types", None)
    if has_tensor_pointer is None or not has_tensor_pointer(mod):
        return False

    if not fd.assign_debug_collect_scope_ids_without_erase(mod):
        raise RuntimeError("failed to resolve debug collect scopes")
    assign_metadata = getattr(fd, "assign_debug_op_ids_and_metadata_without_pass_manager", None)
    erase_markers = getattr(fd, "erase_debug_collect_markers", None)
    if assign_metadata is None or erase_markers is None:
        raise RuntimeError("debug tensor-pointer metadata fallback is unavailable")
    if not assign_metadata(mod):
        raise RuntimeError("failed to assign debug op ids")
    erase_markers(mod)

    tracked_table_json = fd.get_debug_tracked_op_table_json(mod)
    kernel_metadata_json = fd.get_debug_kernel_metadata_json(mod)
    metadata["debug_kernel_id"] = fd.get_debug_kernel_id(mod)
    metadata["debug_metadata_json"] = kernel_metadata_json
    metadata["debug_tracked_table"] = json.loads(tracked_table_json)
    metadata["debug_records_per_instance"] = 0
    metadata["debug_record_size"] = 64
    metadata["debug_record_layout"] = ""
    metadata["debug_record_plan"] = []
    metadata["debug_launch_hidden_arg"] = False
    metadata["debug_metadata_only_reason"] = "triton_tensor_pointer"
    return True


def run_ttir_debug_passes_if_needed(mod, metadata: dict) -> None:
    """If the module contains debug collect markers, run debug passes and set metadata."""
    fd = _get_debug_passes()
    if fd is None:
        if get_instrumentation_mode().startswith("debugger"):
            raise RuntimeError(_DISABLED_BUILD_MESSAGE)
        return
    has_markers = fd.has_debug_collect_markers(mod)
    auto_collect = get_instrumentation_mode() == "debugger_auto"

    if auto_collect:
        try:
            from triton.runtime import debugger

            debug_config = debugger.current_compile_config()
        except Exception:
            debug_config = {}
        level = int(debug_config.get("debug_record_level", 1))
        addr_level = int(debug_config.get("debug_addr_level", 0))
        has_markers = bool(fd.insert_default_debug_collect_markers(mod, level, addr_level))

    if not has_markers:
        metadata["debug_enabled"] = False
        metadata["debug_launch_hidden_arg"] = False
        return
    try:
        from triton.runtime import debugger

        debug_config = debugger.current_compile_config()
    except Exception:
        debug_config = {}

    metadata["debug_enabled"] = True
    metadata["debug_protocol_version"] = 2
    metadata["debug_record_level"] = int(debug_config.get("debug_record_level", 1))
    metadata["debug_addr_level"] = int(debug_config.get("debug_addr_level", 0))
    metadata["debug_export_mode"] = debug_config.get("debug_export_mode", "POST_KERNEL_EXPORT")
    if "debug_record_capacity" in debug_config:
        metadata["debug_record_capacity"] = int(debug_config["debug_record_capacity"])
    metadata["debug_launch_hidden_arg"] = _debug_launch_hidden_arg_enabled()
    fd.set_debug_kernel_id_seed(mod, str(metadata.get("hash") or ""))
    fd.set_debug_hidden_arg_abi_enabled(mod, bool(metadata["debug_launch_hidden_arg"]))
    fd.set_debug_addr_level(mod, int(metadata["debug_addr_level"]))
    fd.set_debug_timeline_enabled(mod, bool(auto_collect and _kernel_internal_timeline_supported()))
    fd.set_debug_timeline_only(mod, bool(auto_collect))

    if _finish_metadata_only_tensor_pointer_debug(fd, mod, metadata):
        return

    metadata_pm = ir.pass_manager(mod.context)
    metadata_pm.enable_debug()
    fd.add_resolve_debug_scope(metadata_pm)
    fd.add_assign_debug_op_id(metadata_pm)
    # Debug collect markers in called helper functions are ignored by the
    # metadata pass to avoid call-graph id propagation.  After those markers are
    # erased, run the regular inliner so helper calls such as FlagGems
    # triton_lang_extension.program_id return to the call-free TTIR shape that
    # Ascend's ttadapter path expects.
    passes.common.add_inliner(metadata_pm)
    passes.common.add_symbol_dce(metadata_pm)
    _run_pass_manager(metadata_pm, mod, "flagtree_debug_collect_metadata")

    tracked_table_json = fd.get_debug_tracked_op_table_json(mod)
    kernel_metadata_json = fd.get_debug_kernel_metadata_json(mod)
    metadata["debug_kernel_id"] = fd.get_debug_kernel_id(mod)
    metadata["debug_metadata_json"] = kernel_metadata_json
    metadata["debug_tracked_table"] = json.loads(tracked_table_json)

    instrumentation_pm = ir.pass_manager(mod.context)
    instrumentation_pm.enable_debug()
    fd.add_insert_instrumentation(instrumentation_pm)
    passes.common.add_cse(instrumentation_pm)
    passes.common.add_canonicalizer(instrumentation_pm)
    _run_pass_manager(instrumentation_pm, mod, "flagtree_debug_collect")
    metadata["debug_records_per_instance"] = int(fd.get_debug_records_per_instance(mod))
    metadata["debug_record_size"] = int(fd.get_debug_record_size(mod))
    metadata["debug_record_layout"] = fd.get_debug_record_layout(mod)
    metadata["debug_record_plan"] = json.loads(fd.get_debug_record_plan_json(mod))
    metadata["debug_full_dump_payload_bytes_per_instance"] = int(fd.get_debug_full_dump_payload_bytes_per_instance(mod))
    metadata["debug_full_dump_plan"] = json.loads(fd.get_debug_full_dump_plan_json(mod))
    if metadata["debug_records_per_instance"] <= 0:
        # The user may request dynamic debugger collection, but the IR pass is
        # the source of truth for whether a hidden-arg ABI was actually added.
        # Metadata-only regions must not make the launcher append
        # __debug_ctrl_ptr, otherwise Ascend kernel arguments become misaligned.
        metadata["debug_launch_hidden_arg"] = False


def run_ttadapter_debug_passes_if_needed(mod, metadata: dict) -> None:
    """Simplify debugger-only TTAdapter writes after hidden arg becomes memref."""
    if not metadata.get("debug_enabled", False):
        return
    if not metadata.get("debug_launch_hidden_arg", False):
        return
    if int(metadata.get("debug_records_per_instance", 0)) <= 0:
        return

    fd = _get_debug_passes()
    if fd is None:
        raise RuntimeError(_DISABLED_BUILD_MESSAGE)

    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    fd.add_simplify_record_memref_writes(pm)
    _run_pass_manager(pm, mod, "flagtree_debug_ttadapter")


def prepare_launch_debug_ctrl(compiled_kernel, stream) -> None:
    """Set CudaLauncher.debug_ctrl_ptr before launch through the transfer engine."""
    del stream  # reserved for async engine / stream-ordered alloc
    if compiled_kernel._run is None:
        compiled_kernel._init_handles()
    launcher = compiled_kernel._run
    if getattr(launcher, "debug_launch_hidden_arg", False):
        launcher.debug_ctrl_ptr = int(getattr(compiled_kernel, "_debug_ctrl_ptr", 0))
