import os
import platform
import shutil
import re
from ctypes import CDLL, RTLD_GLOBAL
import triton
import glob

SPINE_MLIR_BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def get_spine_mlir_cc_debug() -> bool:
    debug_or_not = int(os.getenv("SPINE_MLIR_DEBUG_MODE", "0"))
    return debug_or_not == 1


def get_cpu_arch() -> str:
    return platform.machine()


def get_target_arch() -> str:
    cpu_arch = get_cpu_arch()
    return os.getenv("SPINE_TRITON_TARGET_ARCH", cpu_arch)


def get_cross_toolchain() -> str:
    return os.getenv("SPINE_TRITON_CROSS_TOOLCHAIN", "")


def get_spine_mlir_opt_path() -> str:
    path = os.getenv("SPINE_MLIR_OPT_PATH", "")
    if path == "":
        return os.path.join(SPINE_MLIR_BASE_PATH, "bin", "spine-opt")
    return path


def get_spine_mlir_opt_options() -> str:
    opts = os.getenv("SPINE_TRITON_PIPELINE_OPTION", "")
    return opts


def get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        return os.path.join(SPINE_MLIR_BASE_PATH, "bin", bin_name)
    return os.path.join(path, bin_name)


def get_spine_triton_opt_path() -> str:
    spine_triton_opt_path = os.path.join(SPINE_MLIR_BASE_PATH, "bin", "spine-triton-opt")
    if os.path.isfile(spine_triton_opt_path):
        path = spine_triton_opt_path
    else:
        print(
            "Warning: spine-triton-opt not found in the Triton installation path, getting SPINE_TRITON_OPT_PATH environment variable"
        )
        path = os.getenv("SPINE_TRITON_OPT_PATH", "")
        if path == "":
            raise Exception("SPINE_TRITON_OPT_PATH is not set.")
    return path


def dump_ir_if_needed(files, kernel_name=None):
    path = os.getenv("SPINE_TRITON_DUMP_PATH", "")
    if not path:
        return
    for f in files:
        if kernel_name is not None:
            shutil.copy(f, os.path.join(path, kernel_name + "_" + os.path.basename(f)))
        else:
            shutil.copy(f, os.path.join(path) + os.path.basename(f))


def extract_kernel_name(pattern, ir):
    matches = re.findall(pattern, ir)
    assert len(matches) == 1
    kernel_name = matches[0]
    return kernel_name


def get_cpu_name_from_arch_id(arch_id: str) -> str:
    target_arch_id_to_cpu_arch = {
        "0x503C": "spacemit-x60",
        "0x5064": "spacemit-x100",
        "0x50C8": "spacemit-x200",
        "0xA03C": "spacemit-a60",
        "0xA064": "spacemit-a100",
        "0xA0C8": "spacemit-a200",
        "0xF000": "spacemit-a200m",
    }
    cpu_name = target_arch_id_to_cpu_arch.get(arch_id, None)
    if cpu_name is None:
        raise ValueError(f"Unknown arch_id: {arch_id}")
    return cpu_name


# Load libspert into the global namespace (RTLD_GLOBAL) so that dlopen'd kernel
# .so files resolve their undefined spine_* symbols (spine_grid,
# spine_parallel_dispatch_Nd, spine_require_stream, spine_thread_tcm_malloc, ...)
# against it. libSpeIRRuntimeLibs is no longer used; libspert provides the full
# spert C-ABI on its own.
rpc_host = os.environ.get("SPINE_TRITON_RPC_HOST", "")
if not rpc_host:
    try:
        spine_mlir_lib_dir = os.path.join(SPINE_MLIR_BASE_PATH, "lib")
        libspert_pattern = os.path.join(spine_mlir_lib_dir, "libspert.so*")
        libspert_candidates = glob.glob(libspert_pattern)
        libspert_candidates = [f for f in libspert_candidates if os.path.isfile(f)]
        if not libspert_candidates:
            raise FileNotFoundError(f"Could not find libspert in {spine_mlir_lib_dir}. "
                                    f"Searched pattern: {libspert_pattern}")
        # Pick the longest-named file (the versioned real object, e.g. libspert.so.0.6.0)
        libspert_path = max(libspert_candidates, key=lambda f: len(os.path.basename(f)))
        libspert = CDLL(libspert_path, mode=RTLD_GLOBAL)
    except Exception as e:
        raise ImportError("can not find libspert. {}".format(e))

try:
    triton_path = os.path.dirname(triton.__file__)
    libtritonruntime_path = os.path.join(triton_path, "_C", "libSpineTritonRuntime.so")
    if os.path.isfile(libtritonruntime_path):
        libtritonruntime = CDLL(libtritonruntime_path, mode=RTLD_GLOBAL)
    else:
        spine_triton_opt_path = get_spine_triton_opt_path()
        if os.path.isfile(spine_triton_opt_path):
            spine_triton_lib_dir = os.path.join(os.path.dirname(spine_triton_opt_path), "triton/_C")
            libtritonruntime_path = os.path.join(spine_triton_lib_dir, "libSpineTritonRuntime.so")
            libtritonruntime = CDLL(libtritonruntime_path, mode=RTLD_GLOBAL)
except Exception as e:
    raise ImportError("can not find libtritonruntime. {}".format(e))
