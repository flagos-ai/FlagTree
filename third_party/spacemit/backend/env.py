import os
import platform
import shutil
import re
from ctypes import CDLL, RTLD_GLOBAL, c_int16, c_void_p
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


def alloc_mbarrier(expect_count: int, flag: int = 0, arrive_count: int = 0, transaction_count: int = 0) -> int:
    if expect_count <= 0:
        raise ValueError(f"expect_count must be positive, got {expect_count}")
    if expect_count > 32767:
        raise ValueError(f"expect_count must fit int16_t, got {expect_count}")
    if "libspeirruntime" not in globals():
        raise RuntimeError("libSpeIRRuntimeLibs is not loaded")
    alloc_fn = libspeirruntime.spine_mbarrier_alloc
    alloc_fn.argtypes = [c_int16, c_int16, c_int16, c_int16]
    alloc_fn.restype = c_void_p
    handle = alloc_fn(flag, arrive_count, transaction_count, expect_count)
    if not handle:
        raise RuntimeError("spine_mbarrier_alloc returned null")
    return int(handle)


def release_mbarrier(handle: int) -> None:
    if not handle:
        return
    if "libspeirruntime" not in globals():
        raise RuntimeError("libSpeIRRuntimeLibs is not loaded")
    release_fn = libspeirruntime.spine_mbarrier_release
    release_fn.argtypes = [c_void_p]
    release_fn.restype = None
    release_fn(c_void_p(handle))


try:
    spine_mlir_opt_path = get_spine_mlir_opt_path()
    if os.path.isfile(spine_mlir_opt_path):
        spine_mlir_lib_dir = os.path.join(SPINE_MLIR_BASE_PATH, "lib")
        lib_pattern = os.path.join(spine_mlir_lib_dir, "libSpeIRRuntimeLibs.so*")
        lib_candidates = glob.glob(lib_pattern)
        lib_candidates = [f for f in lib_candidates if os.path.isfile(f)]

        if not lib_candidates:
            raise FileNotFoundError(f"Could not find libSpeIRRuntimeLibs in {spine_mlir_lib_dir}. "
                                    f"Searched pattern: {lib_pattern}")

        libspeirruntime_path = lib_candidates[0]

        libspeirruntime = CDLL(libspeirruntime_path, mode=RTLD_GLOBAL)
except Exception as e:
    raise ImportError("can not find libspeirruntime. {}".format(e))

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
