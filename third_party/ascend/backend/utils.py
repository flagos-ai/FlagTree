# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import functools
import hashlib
import os
import re
import shutil
import subprocess
import sysconfig
import warnings
from pathlib import Path
import logging
import platform
from triton.backends.ascend.backend_register import backend_strategy_registry

import pybind11

_is_compile_on_910_95 = None


def _multibuffer_mode_to_tuple(value):
    """Freeze a mode dictionary, or restore its pairs from a JSON cache.

    Sort keys for stable hashing regardless of dictionary insertion order.
    NPU IR owns level names, count semantics and legacy-option handling.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        value = tuple(value.items())
    if not isinstance(value, (list, tuple)) or any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                                                   or not isinstance(pair[0], str) or type(pair[1]) is not int
                                                   for pair in value):
        raise TypeError("multibuffer_mode must be a dict[str, int], cached (str, int) pairs, or None")
    return tuple(sorted((level, count) for level, count in value))


# Compatibility boundary for compile-option cleanup.  Public dictionaries
# route renamed options and discard backend-managed options before community
# JIT validates the remaining keys.
_DEPRECATED_NPU_OPTIONS = frozenset({
    "add_auto_scheduling",
    "allow_fp8e4nv",
    "arch",
    "auto_blockify_size",
    "auto_tile_and_bind_subblock",
    "bisheng_options",
    "code_motion",
    "compile_on_910_95",
    "enable_auto_blockify",
    "enable_bishengir_simt_optimization",
    "enable_buffer_insert_optimization",
    "enable_cce_vf_auto_sync",
    "enable_cce_vf_remove_membar",
    "enable_cross_if_fusion",
    "enable_drop_unit_dims",
    "enable_linearize",
    "enable_mask_fallback_conversion",
    "enable_nd2nz_on_vector",
    "enable_select_analysis",
    "enable_simt_reorder_instruction",
    "enable_sync_block_lock",
    "enable_ub_refine_opt",
    "enable_vf_fusion",
    "force_simt_only",
    "force_simt_template",
    "graph_optimize_emit_remarks",
    "graph_optimize_max_rewrites_per_function",
    "graph_optimize_rule_mask",
    "graph_optimize_ub_capacity_bytes",
    "grid_num_tiles",
    "has_auto_blockify_blacklist_op",
    "inter_cache_num",
    "intra_cache_num",
    "kernel_name",
    "load_cache_num",
    "llvm_version",
    "mix_mode",
    "ops_reorder",
    "optimize_dynamic_offset",
    "parallel_mode",
    "simt_reorder_instruction",
    "storage_align",
    "stream",
    "use_bytecode",
    "vf_merge_level",
    "warp_size",
})

# These names remain reserved while the compatibility layer accepts them as
# deprecated compile options.  Kernel parameters, including tl.constexpr
# parameters supplied through autotune configs, must not reuse them.
_RESERVED_NPU_OPTION_NAMES = _DEPRECATED_NPU_OPTIONS
_WARNED_DEPRECATED_NPU_OPTIONS = set()

# Boolean compatibility switches route only their enabled state.  An enabled
# legacy force switch overrides compile_mode to preserve its historical force
# semantics; False keeps the canonical mode or backend default unchanged.
_DEPRECATED_NPU_OPTION_ROUTES = {
    "force_simt_only": ("compile_mode", "simt_only"),
    "force_simt_template": ("compile_mode", "simd_simt_template"),
}

# Apply weaker selectors first so pure SIMT wins if both legacy force switches
# are enabled together.
_DEPRECATED_NPU_OPTION_ROUTE_PRECEDENCE = (
    "force_simt_template",
    "force_simt_only",
)

# Renamed value options preserve the complete user value.  A simultaneously
# supplied canonical option wins via setdefault below.
_DEPRECATED_NPU_OPTION_ALIASES = {
    "intra_cache_num": "buf_slot_num_of_veccore",
    "inter_cache_num": "buf_slot_num_of_crosscore",
    "load_cache_num": "buf_slot_num_of_gm",
    "enable_bishengir_simt_optimization": "simt_optimization_mode",
}

# Removed no-op options keep using the compatibility path above.  This table
# only specializes their migration guidance; route and alias options retain
# the behavior and messages defined by PR #1729.
_DEPRECATED_NPU_OPTION_DETAILS = {
    "add_auto_scheduling": "it is ignored; the removed DAG scheduling switch has no replacement.",
    "allow_fp8e4nv": "it is ignored; this option has no replacement because it had no effective consumer.",
    "arch": "it is ignored; the target architecture is injected from GPUTarget.arch.",
    "auto_blockify_size": "it is ignored; this option has no replacement because it had no effective consumer.",
    "auto_tile_and_bind_subblock":
    "it is ignored; tiling and subblock binding are derived from Linalg IR and lock semantics.",
    "bisheng_options": "it is ignored; BiSheng compiler arguments and libdevice linkage are backend-managed.",
    "code_motion": "it is ignored; the removed vendor compiler control has no replacement.",
    "enable_auto_blockify": "it is ignored; automatic block mapping and its safety blacklist are backend-managed.",
    "enable_buffer_insert_optimization":
    "it is ignored; DynamicCV keeps buffer insertion optimization enabled internally.",
    "enable_cce_vf_auto_sync": "it is ignored; the removed vendor compiler control has no replacement.",
    "enable_cce_vf_remove_membar": "it is ignored; the removed vendor compiler control has no replacement.",
    "enable_cross_if_fusion": "it is ignored; the removed vendor compiler control has no replacement.",
    "enable_drop_unit_dims": "it is ignored; consider 'enable_flatten' when flattening is intended.",
    "enable_linearize": "it is ignored; this option has no replacement because it had no effective consumer.",
    "enable_mask_fallback_conversion": "it is ignored; the backend fixes mask fallback conversion to False.",
    "enable_nd2nz_on_vector": "it is ignored; the backend fixes vector ND2NZ conversion to False.",
    "enable_select_analysis": "it is ignored; the backend fixes select analysis to True.",
    "enable_simt_reorder_instruction": "it is ignored; instruction reordering is controlled by simt_optimization_mode.",
    "enable_sync_block_lock": "it is ignored; this option has no replacement because it had no effective consumer.",
    "enable_ub_refine_opt": "it is ignored; the backend keeps UB refine optimization disabled.",
    "enable_vf_fusion": "it is ignored; this switch is no longer forwarded to the NPU compiler.",
    "graph_optimize_emit_remarks": "it is ignored; the backend fixes graph-optimization remarks to False.",
    "graph_optimize_max_rewrites_per_function":
    "it is ignored; the backend fixes the maximum rewrites per function to 64.",
    "graph_optimize_rule_mask":
    "it is ignored; the backend fixes the graph-optimization rule mask to 3071 (legacy + IAT + PTSM).",
    "graph_optimize_ub_capacity_bytes":
    "it is ignored; the backend derives an 80%-of-raw-UB graph budget from the target architecture.",
    "has_auto_blockify_blacklist_op": "it is ignored; the safety flag is derived by scanning TTIR.",
    "kernel_name": "it is ignored; the kernel name is derived from TTIR.",
    "llvm_version": "it is ignored; this option has no replacement because it had no effective consumer.",
    "mix_mode": "it is ignored; mix mode is derived from Linalg IR as internal metadata.",
    "ops_reorder": "it is ignored; the removed vendor compiler control has no replacement.",
    "optimize_dynamic_offset": "it is ignored; the backend fixes dynamic-offset optimization to False.",
    "parallel_mode": "it is ignored; parallel mode is derived from compile_mode and Linalg IR.",
    "simt_reorder_instruction":
    "it is ignored; this option has been moved into the first digit of simt_optimization_mode.",
    "storage_align": "it is ignored; the removed vendor compiler control has no replacement.",
    "stream": "it is ignored; launch streams are managed by the runtime and driver.",
    "use_bytecode": "it is ignored; the bytecode pipeline is always enabled.",
    "warp_size": "it is ignored; the Ascend backend fixes warp_size to 32.",
}

_DEPRECATED_ASCEND_ENV_VARS = frozenset({
    "LLVM_ROOT",
    "MLIR_ROOT",
    "TRITON_ALL_BLOCKS_PARALLEL",
    "TRITON_ASCEND_ARCH",
    "TRITON_ASCEND_COMPILE_SPEED_OPT",
    "TRITON_BACKEND",
    "TRITON_DISABLE_FFTS",
    "TRITON_REGISTER_TENSOR_MSPROF",
})
_DEPRECATED_ASCEND_ENV_VAR_DETAILS = {
    "LLVM_ROOT":
    "it is ignored; set CC to select the CPU launcher compiler.",
    "MLIR_ROOT":
    "it is ignored; packaged or PATH-discovered MLIR tools are used instead.",
    "TRITON_ALL_BLOCKS_PARALLEL":
    "it is ignored; automatic block mapping is managed by backend policy.",
    "TRITON_ASCEND_ARCH":
    "use an explicit GPUTarget.arch instead; host environment overrides are ignored.",
    "TRITON_ASCEND_COMPILE_SPEED_OPT":
    "it is ignored; this variable has no replacement because it had no effective consumer.",
    "TRITON_BACKEND":
    "it is ignored; the backend policy is fixed to 'torch_npu'.",
    "TRITON_DISABLE_FFTS":
    "it is ignored; FFTS policy is derived from the explicit target architecture.",
    "TRITON_REGISTER_TENSOR_MSPROF":
    "it is ignored; tensor-shape msprof registration is no longer controlled by an environment variable.",
}
_WARNED_DEPRECATED_ASCEND_ENV_VARS = set()


class _InternalNPUOptionInt(int):
    """Marks an integer option value produced by the Ascend backend itself."""


def _is_internal_npu_option_value(value) -> bool:
    return isinstance(value, _InternalNPUOptionInt)


def _get_deprecated_npu_options(options) -> set[str]:
    return {
        name
        for name in options.keys() & _DEPRECATED_NPU_OPTIONS
        if not _is_internal_npu_option_value(options[name])
    }


def _warn_deprecated_npu_option(name: str) -> None:
    if name in _WARNED_DEPRECATED_NPU_OPTIONS:
        return
    _WARNED_DEPRECATED_NPU_OPTIONS.add(name)
    route = _DEPRECATED_NPU_OPTION_ROUTES.get(name)
    alias = _DEPRECATED_NPU_OPTION_ALIASES.get(name)
    detail = _DEPRECATED_NPU_OPTION_DETAILS.get(name)
    if route is not None:
        replacement_name, replacement_value = route
        message = (f"Ascend compile option '{name}' is deprecated; "
                   f"use {replacement_name}={replacement_value!r} instead.")
    elif alias is not None:
        message = f"Ascend compile option '{name}' is deprecated; use '{alias}' instead."
    elif detail is not None:
        message = (f"Ascend compile option '{name}' is deprecated and will be removed in a future release; "
                   f"{detail}")
    else:
        message = (f"Ascend compile option '{name}' is deprecated and ignored; "
                   "the backend-managed/default behavior is used instead.")
    warnings.warn(
        message,
        FutureWarning,
        stacklevel=3,
    )


def _remove_deprecated_npu_options(options, *, in_place=False):
    """Normalize reserved legacy NPU options, copying by default."""
    normalized = options if in_place else dict(options)
    deprecated = _get_deprecated_npu_options(normalized)
    active_routes = [
        _DEPRECATED_NPU_OPTION_ROUTES[name]
        for name in _DEPRECATED_NPU_OPTION_ROUTE_PRECEDENCE
        if name in deprecated and normalized[name]
    ]
    for name in sorted(deprecated):
        _warn_deprecated_npu_option(name)
        alias = _DEPRECATED_NPU_OPTION_ALIASES.get(name)
        if alias is not None:
            normalized.setdefault(alias, normalized[name])
        normalized.pop(name)
    for replacement_name, replacement_value in active_routes:
        normalized[replacement_name] = replacement_value
    return normalized


def _warn_deprecated_ascend_env_var(name: str) -> None:
    if (name not in _DEPRECATED_ASCEND_ENV_VARS or name not in os.environ
            or name in _WARNED_DEPRECATED_ASCEND_ENV_VARS):
        return
    _WARNED_DEPRECATED_ASCEND_ENV_VARS.add(name)
    detail = _DEPRECATED_ASCEND_ENV_VAR_DETAILS.get(name)
    if detail is None:
        message = (f"Ascend environment variable '{name}' is deprecated and ignored; "
                   "the default behavior is used instead.")
    else:
        message = (f"Ascend environment variable '{name}' is deprecated and will be removed in a future release; "
                   f"{detail}")
    warnings.warn(
        message,
        FutureWarning,
        stacklevel=3,
    )


def _warn_deprecated_ascend_env_vars() -> None:
    for name in sorted(_DEPRECATED_ASCEND_ENV_VARS & os.environ.keys()):
        _warn_deprecated_ascend_env_var(name)


def is_compile_on_910_95(arch: str = None) -> bool:
    """Return whether the compilation target belongs to the Ascend 950 generation."""
    if arch is not None:
        return isinstance(arch, str) and arch.startswith(("Ascend910_95", "Ascend950"))

    global _is_compile_on_910_95
    if _is_compile_on_910_95 is None:
        try:
            import acl
            name_lower = acl.get_soc_name().lower()
            _is_compile_on_910_95 = ("ascend910_95" in name_lower or "ascend950" in name_lower
                                     or "910_958b" in name_lower)
        except (ImportError, AttributeError):
            _is_compile_on_910_95 = False
    return _is_compile_on_910_95


AUTO_BLOCKIFY_BLACKLIST_RULES = (
    (re.compile(r"\btt\.atomic_(?:rmw|cas)\b"), "atomic operations"),
    (re.compile(r"\btt\.elementwise_inline_asm\b"), "inline elementwise assembly"),
    (
        re.compile(r"\btt\.load\b[^\n]*\bisVolatile\s*=\s*true\b"),
        "loads with volatile",
    ),
    (
        re.compile(r"\btt\.(?:load|store)\b[^\n]*\bcacheModifier\s*="),
        "loads or stores with cache modifiers",
    ),
)

backend_policy = None


def get_backend_func(name, *args, **kwargs):
    global backend_policy
    if backend_policy is None:
        _warn_deprecated_ascend_env_var("TRITON_BACKEND")
        backend_policy = "torch_npu"
    return backend_strategy_registry.execute_func(backend_policy, name, *args, **kwargs)


def get_logger(logger_name, logger_level_str):
    '''
    '''
    logging_level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.setLevel(logging_level_mapping.get(logger_level_str.upper(), "INFO"))
    logger.addHandler(console_handler)
    return logger


def downgrade_llir(llir):
    llir = _downgrade_mem_attrs(llir)
    llir = _downgrade_stacksaverestore_intrinsics(llir)
    return llir


def _downgrade_mem_attrs(llir: str):
    memory_pattern = r"memory\([^()]*\)"

    def replace_mem_attr(m):
        attrs = m[0][7:-1].split(",")
        if len(attrs) == 0:
            return "readnone"
        loc_map = {"argmem": 1, "inaccessiblemem": 2, "other": 4}
        loc_attr = 0
        rw_map = {"readwrite": 3, "write": 2, "read": 1, "none": 0}
        rw_attr = 0
        for attr_pair in attrs:
            pair = attr_pair.split(":")
            assert len(pair) <= 2
            if len(pair) == 1:
                rw = rw_map[pair[0].strip()]
                loc = loc_map["other"]  # all location
            else:
                rw = rw_map[pair[1].strip()]
                loc_str = pair[0].strip()
                if loc_str == "argmem" or loc_str == "inaccessiblemem":
                    loc = loc_map[loc_str]
                else:
                    loc = loc_map["other"]
            if rw > 0:
                loc_attr = loc_attr | loc
                rw_attr = rw_attr | rw
        rev_rw_map = {0: "readnone", 1: "readonly", 2: "writeonly"}
        if rw_attr in rev_rw_map:
            rw_attr_str = rev_rw_map[rw_attr]
        else:
            rw_attr_str = ""
        rev_loc_map = {
            1: "argmemonly",
            2: "inaccessiblememonly",
            3: "inaccessiblemem_or_argmemonly",
        }
        if loc_attr in rev_loc_map:
            loc_attr_str = rev_loc_map[loc_attr]
        else:
            loc_attr_str = ""
        return rw_attr_str + " " + loc_attr_str

    return re.sub(memory_pattern, replace_mem_attr, llir)


def _downgrade_stacksaverestore_intrinsics(llir: str):
    llir = re.sub(r"llvm\.stacksave\.\w+", "llvm.stacksave", llir)
    llir = re.sub(r"llvm\.stackrestore\.\w+", "llvm.stackrestore", llir)
    return llir


def _get_triton_adapter_opt_path() -> str:
    path = os.path.dirname(__file__)
    path = os.path.join(path, "triton-adapter-opt")
    return path


def _get_default_tool_path(path: str, *paths) -> str:
    relative_path = os.path.join(path, *paths)
    try:
        import triton._C.libtriton as libtriton
        tool_path = os.path.join(os.path.dirname(libtriton.__file__), relative_path)
        if os.path.exists(tool_path) and os.access(tool_path, os.X_OK):
            return tool_path
    except (ImportError, AttributeError):
        pass
    return _get_tool_path(os.path.basename(relative_path))


def _get_mlir_path(path: str, *paths) -> str:
    _warn_deprecated_ascend_env_var("MLIR_ROOT")
    return _get_default_tool_path(path, *paths)


def _get_llvm_path(path: str, *paths) -> str:
    _warn_deprecated_ascend_env_var("LLVM_ROOT")
    return _get_default_tool_path(path, *paths)


def _get_tool_path(tool_name: str) -> str:
    """
    Get the path to a Triton tool binary.
    Search order:
      1. Installed package location (triton/_C/)
      2. TRITON_BUILD_DIR environment variable
      3. System PATH
    """
    try:
        import triton._C.libtriton as libtriton
        tool_path = os.path.join(os.path.dirname(libtriton.__file__), tool_name)
        if os.path.exists(tool_path) and os.access(tool_path, os.X_OK):
            return tool_path
    except (ImportError, AttributeError):
        pass

    build_path = os.getenv("TRITON_BUILD_DIR", "")
    if build_path:
        tool_path = os.path.join(build_path, "bin", tool_name)
        if os.path.exists(tool_path) and os.access(tool_path, os.X_OK):
            return tool_path

    tool_path = shutil.which(tool_name)
    if tool_path:
        return tool_path

    raise EnvironmentError(f"Could not find {tool_name} tool. "
                           f"It should be installed in triton/_C/ directory or available in PATH.")


def _get_triton_opt_path() -> str:
    """
    Get the path to triton-opt tool.
    This tool is used to convert ttir to ttadapter.
    """
    return _get_tool_path("triton-opt")


def _get_triton_mlir_opt_path() -> str:
    """
    Get the path to triton-mlir-opt tool.
    This tool is used to convert MLIR to Bytecode format, supporting both
    MLIR native ops and AscendNPU-IR custom ops.
    """
    return _get_tool_path("triton-mlir-opt")


def _get_bishengir_payload_tool(tool: str):
    """Use bundled tools as a pair, or leave resolution to PATH/CANN."""
    root = Path(__file__).resolve().parent / "bishengir"
    if not os.path.lexists(root):
        return None
    for name in ("bishengir-opt", "bishengir-compile"):
        executable = root / "bin" / name
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise EnvironmentError(f"Incomplete BiShengIR payload {root}: missing executable {executable}")
    env = os.environ.copy()
    bin_dir = str(root / "bin")
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", os.defpath)
    return str(root / "bin" / tool), env


def _get_bishengir_opt_path() -> str:
    payload_tool = _get_bishengir_payload_tool("bishengir-opt")
    if payload_tool is not None:
        return payload_tool
    env = os.environ.copy()
    path = shutil.which("bishengir-opt")
    if path is None:
        root = os.getenv("TRITON_NPU_COMPILER_PATH")
        if not root:
            raise EnvironmentError("Couldn't find executable bishengir-opt or TRITON_NPU_COMPILER_PATH")
        path = os.path.join(root, "bishengir-opt")
    return path, env


def _get_npucompiler_path() -> str:
    payload_tool = _get_bishengir_payload_tool("bishengir-compile")
    if payload_tool is not None:
        return payload_tool
    env = os.environ.copy()
    path = shutil.which("bishengir-compile")
    if path is None:
        root = os.getenv("TRITON_NPU_COMPILER_PATH")
        if not root:
            raise EnvironmentError("Couldn't find executable bishengir-compile or TRITON_NPU_COMPILER_PATH.")
        path = os.path.join(root, "npuc")
    return path, env


def _get_bisheng_path() -> str:
    bisheng_path = shutil.which("bisheng")
    if bisheng_path is None:
        npu_compiler_root = os.getenv("TRITON_NPU_COMPILER_PATH", None)
        if npu_compiler_root is None:
            raise EnvironmentError("Couldn't find executable bisheng or TRITON_NPU_COMPILER_PATH")
        bisheng_path = os.path.join(npu_compiler_root, "ccec")
    return bisheng_path


def _is_valid_bishengir_path(path: str) -> bool:
    if not path or not isinstance(path, str):
        return False
    if os.path.basename(path) != "bishengir-compile":
        return False
    if not os.path.isfile(path) or not os.access(path, os.X_OK):
        return False
    return True


# grep bishengir-compile's option limit-auto-multi-buffer-buffer to check
# if bishengir-compile is a newer version which does not generate kernel_reloc.o
# any more.
def _check_bishengir_api_change() -> bool:
    bishengir_path, _ = _get_npucompiler_path()
    if not _is_valid_bishengir_path(bishengir_path):
        print(f"ERROR: Invalid bishengir path format: {bishengir_path}")
        return False
    try:
        result = subprocess.run(
            [bishengir_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and 'limit-auto-multi-buffer-buffer' in result.stdout:
            # bishengir-compile is newer version
            return True
        else:
            # bishengir-compile is older version
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def _check_bishengir_is_regbased() -> bool:
    bishengir_path, _ = _get_npucompiler_path()
    if not _is_valid_bishengir_path(bishengir_path):
        print(f"ERROR: Invalid bishengir path format: {bishengir_path}")
        return False
    try:
        result = subprocess.run(
            [bishengir_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and 'reg-based' in result.stdout:
            # bishengir-compile is regbased version
            return True
        else:
            # bishengir-compile is membased version
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


@functools.lru_cache(None)
def _get_ascend_path() -> Path:
    path = os.getenv("ASCEND_HOME_PATH", "")
    if path == "":
        raise EnvironmentError("ASCEND_HOME_PATH is not set, source <ascend-toolkit>/set_env.sh first")
    return Path(path)


def _is_ascend_sanitizer_enabled() -> bool:
    return os.getenv("TRITON_ENABLE_SANITIZER", "false").lower() in ("true", "1")


def _is_debug_line_info_disabled() -> bool:
    return os.getenv("TRITON_DISABLE_LINE_INFO", "true").lower() in ("true", "1")


def _is_auto_map_parallel_blocks_enabled() -> bool:
    """Return the fixed backend policy for automatic block mapping."""
    _warn_deprecated_ascend_env_var("TRITON_ALL_BLOCKS_PARALLEL")
    return True


def _get_auto_blockify_blacklist_reasons(ir_text: str):
    return [description for pattern, description in AUTO_BLOCKIFY_BLACKLIST_RULES if pattern.search(ir_text)]


def _warn_auto_blockify_disabled(kernel_name: str, blacklist_reasons) -> None:
    if not blacklist_reasons:
        return
    reasons = ", ".join(blacklist_reasons)
    print(f"[WARNING] AutoBlockify disabled for kernel '{kernel_name}'. "
          f"Unsafe ops: {reasons}. Enabling may cause correctness issues. "
          "The compiler keeps AutoBlockify disabled for this kernel.")


def _enable_print_ub_bits() -> bool:
    return os.getenv("ENABLE_PRINT_UB_BITS", "false").lower() in ("true", "1")


def _enable_dump_memory_info() -> bool:
    return os.getenv("TRITON_MEMORY_DISPLAY", "false").lower() in ("true", "1")


def _enable_msdebug() -> bool:
    return os.getenv("LLVM_EXTRACT_DI_LOCAL_VARIABLES", "false").lower() in ("true", "1")


def _get_cxx():
    cxx = os.environ.get("CC")
    if cxx is None:
        clangxx = shutil.which("clang++")
        gxx = shutil.which("g++")
        cxx = clangxx if clangxx is not None else gxx
        if cxx is None:
            raise RuntimeError("Failed to find C++ compiler")
    return cxx


def _build_npu_ext(obj_name: str, header_or_src_path, src_path=None, *, kernel_launcher="torch",
                   precompile=False) -> str:
    header_path = None
    if src_path is None:
        src_path = header_or_src_path
    else:
        header_path = header_or_src_path

    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    src_dir = os.path.dirname(src_path)
    so_path = os.path.join(src_dir, f"{obj_name}{suffix}")
    cxx = _get_cxx()
    cc_cmd = [cxx, src_path]
    cc_cmd += [f"-w"]
    if hasattr(sysconfig, "get_default_scheme"):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    if scheme == "posix_local":
        scheme = "posix_prefix"
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    cc_cmd += [f"-I{py_include_dir}"]
    cc_cmd += [f"-I{os.path.dirname(os.path.realpath(__file__))}"]
    asc_path = _get_ascend_path()
    if header_path is not None:
        cc_cmd += [f"-I{os.path.dirname(header_path)}"]

    rt_path = os.path.join(asc_path, "include/experiment/runtime/runtime/rt.h")
    if not os.path.exists(rt_path):
        cc_cmd += [
            f"-I{os.path.join(asc_path, 'pkg_inc')}",
            f"-I{os.path.join(asc_path, 'pkg_inc/profiling')}",
        ]

    cc_cmd += [
        f"-I{os.path.join(asc_path, 'include')}",
        f"-I{os.path.join(asc_path, 'include/experiment')}",
        f"-I{os.path.join(asc_path, 'include/experiment/msprof')}",
        f"-I{pybind11.get_include()}",
        f"-L{os.path.join(asc_path, 'lib64')}",
        "-lruntime",
        "-lascendcl",
    ]
    if kernel_launcher == "torch":
        if obj_name == "npu_utils":
            cc_cmd += get_backend_func("get_cc_cmd_npu_utils")
        else:
            cc_cmd += get_backend_func("get_cc_cmd")

    cc_cmd += cann_version_compile_args()
    cc_cmd += ["-std=c++17", "-shared", "-fPIC", "-o", so_path]

    result = subprocess.run(cc_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        return so_path
    else:
        raise RuntimeError(f"Failed to compile {src_path}, error: {result.stderr},cmd={cc_cmd}")


def _get_kernel_target(metadata: dict):
    if "target" not in metadata:
        raise Exception("No target provided!")
    sub_target = metadata["target"].arch
    assert isinstance(sub_target, str)
    if sub_target.startswith("Ascend910B"):
        mix_mode = metadata["mix_mode"]
        if mix_mode.lower().strip("_").startswith("aiv"):
            return "ascend_910b_vec", "c220-vec", "aiv"
        elif mix_mode.lower().strip("_").startswith("aic"):
            return "ascend_910b_cube", "c220-cube", "aic"
        else:
            return "ascend_910b", "c220", "mix"
    elif sub_target.startswith("Ascend910"):
        return "ascend_910", "c100", "mix"
    else:
        raise NotImplementedError(f"NPU subtarget {sub_target} not supported yet")


def _check_cxx11_abi():
    return get_backend_func("cxx_abi")


def convert_sigtype_to_int(sigty: str):
    MAP_SIGTYPE_TO_INT = {
        # Boolean
        "i1": 12,  # BOOL
        # Integer types
        "i4": 29,  # INT4
        "i8": 2,  # INT8
        "i16": 6,  # INT16
        "i32": 3,  # INT32
        "i64": 9,  # INT64
        # Unsigned integer types
        "u1": 30,  # UINT1
        "u8": 4,  # UINT8
        "u16": 7,  # UINT16
        "u32": 8,  # UINT32
        "u64": 10,  # UINT64
        # Floating point types
        "fp16": 1,  # FLOAT16
        "bf16": 27,  # DT_BF16
        "fp32": 0,  # FLOAT
        "fp64": 11,  # DOUBLE
        "fp8e5": 35,  # FLOAT8_E5M2
        "fp8e4nv": 36,  # FLOAT8_E4M3FN
    }
    if sigty not in MAP_SIGTYPE_TO_INT:
        raise ValueError(f"Unsupported data type: {sigty}")

    return MAP_SIGTYPE_TO_INT[sigty]


def convert_dtype_to_numpy(dtype):
    return get_backend_func("type_convert")[dtype]


def _check_bishengir_able_save_ir() -> bool:
    bishengir_path, _ = _get_npucompiler_path()
    if not _is_valid_bishengir_path(bishengir_path):
        print(f"ERROR: Invalid bishengir path format: {bishengir_path}")
        return False
    try:
        result = subprocess.run(
            [bishengir_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and 'save-linked-ir' in result.stdout:
            return True
        else:
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def get_ascend_arch_from_env():
    _warn_deprecated_ascend_env_var("TRITON_ASCEND_ARCH")
    return ""


def ub_size_in_kbytes_for_arch(arch: str) -> int:
    """Return the raw UB capacity for a supported Ascend target architecture.

    This resolver is deliberately independent of the active runtime device so
    compiler options can be normalized for an explicit compilation target.
    Unknown and empty targets fail closed instead of guessing a UB capacity.
    """
    if not isinstance(arch, str) or not arch:
        return 0
    if arch.startswith(("Ascend910_95", "Ascend950")):
        return 256
    if arch.startswith(("Ascend910A", "Ascend910B", "Ascend910D", "Ascend910_93", "Ascend310B")):
        return 192
    return 0


def graph_ub_budget_bytes_for_arch(arch: str) -> int:
    """Return the conservative graph-optimization UB budget in bytes.

    StoreCoalescing does not model all live local buffers, so its budget is
    capped at 80 percent of the target's raw UB capacity.
    """
    return ub_size_in_kbytes_for_arch(arch) * 1024 * 80 // 100


def is_ffts_supported(arch: str):
    """Return whether the specified compilation target supports FFTS."""
    if not isinstance(arch, str) or not arch:
        return False
    if is_compile_on_910_95(arch) or arch in ["Ascend910A", "Ascend310B4"]:
        return False
    return True


def force_disable_ffts(arch: str) -> bool:
    """Return whether the selected target requires FFTS to be disabled."""
    _warn_deprecated_ascend_env_var("TRITON_DISABLE_FFTS")
    return is_compile_on_910_95(arch)


def _parse_cann_version(line: str):
    m = re.search(r'(\d+)\.(\d+)(?:\.(\d+))?', line)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2))
        patch = int(m.group(3)) if m.group(3) is not None else 0
        return (major, minor, patch)
    return None


def _find_cann_version_file():
    ascend_path = str(_get_ascend_path())
    arch = get_machine_arch()
    candidates = [
        os.path.join(ascend_path, arch + "-linux", "ascend_toolkit_install.info"),
        os.path.join(ascend_path, arch + "-linux", "ascend_all_cann_install.info"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


@functools.lru_cache(None)
def get_cann_version():
    _cann_version = None
    try:
        version_file = _find_cann_version_file()
        if version_file is None:
            _cann_version = None
            return None
        with open(version_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if "version" in line.lower():
                    parsed = _parse_cann_version(line)
                    if parsed is not None:
                        _cann_version = parsed
                        return _cann_version
        _cann_version = None
    except Exception:
        raise EnvironmentError("Could not parse CANN version file")


def is_cann_version_at_least(major: int, minor: int = 0, patch: int = 0) -> bool:
    v = get_cann_version()
    if v is None:
        return False
    return v >= (major, minor, patch)


def cann_version_compile_args():
    if is_cann_version_at_least(9, 1, 0):
        return ["-DTRITON_CANN_910"]
    return []


def triton_enable_libdevice_simt(arch: str = None) -> bool:
    """Return whether the environment switch selects SIMT libdevice."""
    return bool(os.getenv("TRITON_ENABLE_LIBDEVICE_SIMT", False)) and is_compile_on_910_95(arch)


def get_cann_version_file_hash():
    cann_version_file_path = _find_cann_version_file()
    return get_file_hash256(cann_version_file_path)


def get_file_hash256(file_path):
    if file_path is None:
        raise ValueError("file_path is None")
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256.update(byte_block)
        return sha256.hexdigest()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Can't find file {os.path.basename(file_path)}") from e


def get_machine_arch():
    ARCHITECTURE_ALIASES = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "i386": "x86_64",
        "i686": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
        "armv7l": "aarch64",
        "armv8l": "aarch64",
        "arm": "aarch64",
    }
    system_arch = platform.machine()
    return ARCHITECTURE_ALIASES[system_arch]
