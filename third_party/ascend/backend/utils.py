# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import functools
import os
import re
import shutil
import subprocess
import sysconfig
from pathlib import Path

import pybind11


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


def _get_mlir_path(path: str, *paths) -> str:
    root_path = os.getenv("MLIR_ROOT", "")
    if root_path == "":
        raise EnvironmentError("MLIR_ROOT is not set.")
    return os.path.join(root_path, path, *paths)


def _get_llvm_path(path: str, *paths) -> str:
    root_path = os.getenv("LLVM_ROOT", "")
    if root_path == "":
        raise EnvironmentError("LLVM_ROOT is not set.")
    return os.path.join(root_path, path, *paths)


def _get_npucompiler_path() -> str:
    npu_compiler_path = shutil.which("bishengir-compile")
    if npu_compiler_path is None:
        npu_compiler_root = os.getenv("TRITON_NPU_COMPILER_PATH", "")
        if npu_compiler_root is None:
            raise EnvironmentError(
                "Couldn't find executable bishengir-compile or TRITON_NPU_COMPILER_PATH."
            )
        npu_compiler_path = os.path.join(npu_compiler_root, "npuc")
    return npu_compiler_path


def _get_bisheng_path() -> str:
    bisheng_path = shutil.which("bisheng")
    if bisheng_path is None:
        npu_compiler_root = os.getenv("TRITON_NPU_COMPILER_PATH", "")
        if npu_compiler_root is None:
            raise EnvironmentError(
                "Couldn't find executable bisheng or TRITON_NPU_COMPILER_PATH"
            )
        bisheng_path = os.path.join(npu_compiler_root, "ccec")
    return bisheng_path


# grep bishengir-compile's option limit-auto-multi-buffer-buffer to check
# if bishengir-compile is a newer version which does not generate kernel_reloc.o
# any more.
def _check_bishengir_api_change() -> bool:
    bishengir_path = _get_npucompiler_path()
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
    bishengir_path = _get_npucompiler_path()
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
def _get_ascend_path() -> str:
    path = os.getenv("ASCEND_HOME_PATH", "")
    if path == "":
        raise EnvironmentError(
            "ASCEND_HOME_PATH is not set, source <ascend-toolkit>/set_env.sh first"
        )
    return Path(path)


def _is_ascend_sanitizer_enabled() -> bool:
    return os.getenv("TRITON_ENABLE_SANITIZER", "false").lower() in ("true", "1")


def _is_debug_line_info_disabled() -> bool:
    return os.getenv("TRITON_DISABLE_LINE_INFO", "true").lower() in ("true", "1")


def _is_auto_map_parallel_blocks_enabled() -> bool:
    if not _enable_unpublished_feature():
        return False
    return os.getenv("TRITON_ALL_BLOCKS_PARALLEL", "false").lower() in ("true", "1")


def _enable_unpublished_feature() -> bool:
    return os.getenv("ENABLE_UNPUBLISHED_FEATURE", "false").lower() in ("true", "1")


def _build_npu_ext(obj_name: str, src_path, src_dir, *, kernel_launcher=None) -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    so_path = os.path.join(src_dir, f"{obj_name}{suffix}")

    cxx = os.environ.get("CC")
    if cxx is None:
        clangxx = shutil.which("clang++")
        gxx = shutil.which("g++")
        cxx = clangxx if clangxx is not None else gxx
        if cxx is None:
            raise RuntimeError("Failed to find C++ compiler")
    cc_cmd = [cxx, src_path]
    # disable all warnings
    cc_cmd += [f"-w"]
    # find the python library
    if hasattr(sysconfig, "get_default_scheme"):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == "posix_local":
        scheme = "posix_prefix"
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    cc_cmd += [f"-I{py_include_dir}"]
    # device_print.h
    cc_cmd += [f"-I{os.path.dirname(os.path.realpath(__file__))}"]
    asc_path = _get_ascend_path()
    cc_cmd += [
        f"-I{os.path.join(asc_path, 'include')}",
        f"-I{os.path.join(asc_path, 'include/experiment')}",
        f"-I{os.path.join(asc_path, 'include/experiment/msprof')}",
        f"-I{pybind11.get_include()}",
        f"-L{os.path.join(asc_path, 'lib64')}",
        "-lruntime",
        "-lascendcl",
    ]

    import torch
    import torch_npu

    torch_path = os.path.dirname(os.path.realpath(torch.__file__))
    torch_npu_path = os.path.dirname(os.path.realpath(torch_npu.__file__))
    use_cxx11_abi = _check_cxx11_abi()
    cc_cmd += [
        f"-I{os.path.join(torch_path, 'include')}",
        f"-I{os.path.join(torch_npu_path, 'include')}",
        f"-L{os.path.join(torch_npu_path, 'lib')}",
        "-ltorch_npu",
        f"-D_GLIBCXX_USE_CXX11_ABI={use_cxx11_abi}",
    ]

    cc_cmd += ["-std=c++17", "-shared", "-fPIC", "-o", so_path]

    ret = subprocess.check_call(cc_cmd)

    if ret == 0:
        return so_path
    else:
        raise RuntimeError("Failed to compile " + src_path)


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
    import torch

    return 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0


def convert_sigtype_to_int(sigty: str):
    MAP_SIGTYPE_TO_INT = {
        # Boolean
        "i1": 12,  # BOOL
        # Integer types
        "i8": 2,  # INT8
        "i16": 6,  # INT16
        "i32": 3,  # INT32
        "i64": 9,  # INT64
        # Unsigned integer types
        "u32": 8,  # UINT32
        "u64": 10,  # UINT64
        # Floating point types
        "fp16": 1,  # FLOAT16
        "bf16": 27,  # DT_BF16
        "fp32": 0,  # FLOAT
        "fp64": 11,  # DOUBLE
    }
    if sigty not in MAP_SIGTYPE_TO_INT:
        raise ValueError(f"Unsupported data type: {sigty}")

    return MAP_SIGTYPE_TO_INT[sigty]
