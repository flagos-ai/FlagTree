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

import importlib.metadata
from pathlib import Path
import tempfile
import os
import os.path
import re
import subprocess
import sysconfig
from typing import Optional
import functools
import hashlib
from triton.runtime.cache import get_cache_manager, get_dump_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
from triton.backends.ascend.utils import (_build_npu_ext, _check_cxx11_abi, convert_sigtype_to_int, is_ffts_supported,
                                          force_disable_ffts, get_backend_func, get_cann_version)
from triton.backends.ascend.program_grid import (
    PROGRAM_GRID_TRANSFORMS_VERSION,
    ProgramGridContractError,
    get_persistent_transform,
    normalize_program_grid_transforms,
)
# Bind the already-imported utils module once so the launch hot path can write
# TRITON_PROFILER_REGISTERED without a per-launch `import triton` + attribute walk.
import triton.backends.ascend.utils as _ascend_utils


class NPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(NPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        # NPUUtils is a singleton, but __init__ must refresh the cached shared
        # object path on every construction. PyTorch Inductor may set
        # TRITON_CACHE_DIR after the driver first initializes, so keeping the
        # first path would make the launcher look for npu_utils.so in a newer
        # cache root where it was never built.
        self._cache_path = self._build_or_get_cached_so()
        if not hasattr(self, "npu_utils_mod"):
            self.npu_utils_mod = None

    def get_so_path(self):
        if self._cache_path is None:
            self._cache_path = self._build_or_get_cached_so()
        return self._cache_path

    def _build_or_get_cached_so(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src_path = os.path.join(dirname, "npu_utils.cpp")
        src = Path(src_path).read_text()
        cann_version = get_cann_version()
        cann_version_str = ".".join(map(str, cann_version)) if cann_version else ""
        torch_npu_version = importlib.metadata.version("torch_npu")
        key_parts = [cann_version_str, torch_npu_version, src]
        key = hashlib.md5("\0".join(key_parts).encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "npu_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is not None and os.path.exists(cache_path):
            return cache_path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_src_path = os.path.join(tmpdir, "npu_utils.cpp")
            with open(tmp_src_path, "w") as f:
                f.write(src)
            so = _build_npu_ext("npu_utils", tmp_src_path)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), fname, binary=True)
        return cache_path

    def _load_mod(self):
        if self.npu_utils_mod is not None:
            return self.npu_utils_mod

        import importlib.util
        spec = importlib.util.spec_from_file_location("npu_utils", self.get_so_path())
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.npu_utils_mod = mod
        return self.npu_utils_mod

    def load_binary(self, name, kernel, shared, device, mix_mode):
        return self._load_mod().load_kernel_binary(name, kernel, shared, device, mix_mode)

    def _get_npu_device_limit_form_env(self) -> tuple[int, int]:
        """Read and validate the NPU_DEVICE_LIMIT env var, return the capped AICore and AIVector counts.

        The env var format is ``cube_core_num,vector_core_num`` (e.g. ``"14,28"``),
        used to reduce the core count visible to Triton in multi-tenant sharding,
        performance tuning, and resource isolation scenarios.
        When unset, the hardware actual values are returned (AIVector = AICore x 2).

        Validation rules (any failure raises ValueError):
        1. Format must match ``^\\d+(,\\d+)$``; leading/trailing whitespace allowed,
           but no space after the comma;
        2. Both values must be positive;
        3. Neither value may exceed the hardware limit (AICore cap = device actual,
           AIVector cap = AICore x 2).

        Returns:
            tuple[int, int]: (num_aic, num_aiv) the capped AICore and AIVector counts.

        Raises:
            ValueError: when the env var is malformed, contains non-positive values,
                or exceeds the hardware limit; the error message includes the raw
                input and the hardware actual caps.
        """
        npu_device_limit_str = os.getenv("NPU_DEVICE_LIMIT")
        num_aic, num_aiv = self.get_device_core()
        if npu_device_limit_str is None:
            return num_aic, num_aiv

        is_valid = re.match(r'^\d+ *, *\d+$', npu_device_limit_str.strip())
        if is_valid:
            parts = [part.strip() for part in npu_device_limit_str.split(",")]
            num_aic_env = int(parts[0])
            num_aiv_env = int(parts[1])

            if num_aic_env <= 0 or num_aiv_env <= 0:
                raise ValueError(f"[ERROR]NPU_DEVICE_LIMIT={npu_device_limit_str}, which has non-positive value,"
                                 f"both cube_core_num and vector_core_num must be positive.")
            elif num_aic_env > num_aic or num_aiv_env > num_aiv:
                raise ValueError(
                    f"[ERROR]NPU_DEVICE_LIMIT={npu_device_limit_str}, both cube_core_num and vector_core_num "
                    f"must be less than or equal to device properties ({num_aic},{num_aiv}).")
            elif num_aic_env * (num_aiv / num_aic) != num_aiv_env:
                env_quotient = num_aiv_env / num_aic_env
                env_quotient_decimal = round(env_quotient, 1)
                quotient = num_aiv / num_aic
                quotient_decimal = round(quotient, 1)
                raise ValueError(
                    f"[ERROR]NPU_DEVICE_LIMIT={npu_device_limit_str}; expected ratio is consistent, actual, "
                    f"the ratio of vector_core_num/cube_core_num({num_aiv_env}/{num_aic_env}={env_quotient_decimal}) does "
                    f"not equal device properties vector_core_num/cube_core_num({num_aiv}/{num_aic}={quotient_decimal}) ratio."
                )
            else:
                debug = os.getenv("TRITON_DEBUG", 'false').lower() in ('true', '1')
                if debug:
                    print(
                        f"[DEBUG]NPU_DEVICE_LIMIT from env: cube_core_num={num_aic_env},vector_core_num={num_aiv_env})."
                    )
                return num_aic_env, num_aiv_env
        else:
            raise ValueError(f"[ERROR]NPU_DEVICE_LIMIT={npu_device_limit_str}, which has invalid format: "
                             f"It should be like '14,28' (cube_core_num,vector_core_num) "
                             f"and it must be a positive number.")

    @functools.lru_cache()
    def get_device_core(self):
        import torch
        device = torch.npu.current_device()
        prop = torch.npu.get_device_properties(device)
        cube_core_num, vector_core_num = prop.cube_core_num, prop.vector_core_num
        return cube_core_num, vector_core_num

    def has_device_limit(self):
        num_aic, num_aiv = self.get_device_core()
        try:
            return num_aic != self.get_aicore_num() or num_aiv != self.get_aivector_core_num()
        except ValueError:
            return False

    def get_device_properties(self, device):
        # temperoarily added "max_shared_mem" properties to avoid triton-compiler complain
        # fetch available memory at runtime
        num_aic, num_aiv = self._get_npu_device_limit_form_env()
        return {"max_shared_mem": 1, "num_aicore": num_aic, "num_vectorcore": num_aiv}

    def get_arch(self):
        # temporarily return empty arch descriptor
        return self._load_mod().get_arch()

    def get_aicore_num(self):
        # temporarily return empty arch descriptor
        return self.get_device_properties("npu")["num_aicore"]

    def get_aivector_core_num(self):
        return self.get_device_properties("npu")["num_vectorcore"]


class NPULauncher(object):

    def __init__(self, src, metadata):
        self.compile_only = os.getenv("TRITON_COMPILE_ONLY", 'false').lower() in ('true', '1')
        self.src = src
        self.metadata = metadata
        self.so_launcher_path = self._make_launcher_stub_path()
        # setup for remote run
        # TODO: use a var to pack all vars required to run on a remote machine
        self.mix_mode = metadata.mix_mode
        self.shared = metadata.shared
        import importlib.util
        spec = importlib.util.spec_from_file_location("__triton_launcher", self.so_launcher_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cst_key = lambda i: self.src.fn.arg_names.index(i) if isinstance(i, str) else i
        signature = {cst_key(key): value for key, value in self.src.signature.items()}
        self.launch = wrap_handle_tensordesc(getattr(mod, "launch"), signature)

    def _make_launcher_stub_path(self):
        header_src = generate_npu_header_src()
        constants = self.src.constants if hasattr(self.src, "constants") else dict()
        cst_key = lambda i: self.src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in self.src.signature.items()}
        wrapper_src = make_launcher(constants, signature, self.metadata)
        return make_npu_launcher_stub(header_src, wrapper_src, self.metadata.debug)

    def get_launcher_so_path(self):
        return self.so_launcher_path

    def __call__(self, *args, **kwargs):
        _ascend_utils._warn_deprecated_ascend_env_var("TRITON_REGISTER_TENSOR_MSPROF")
        if self.compile_only:
            cache_manager = get_cache_manager(args[5]['hash'])
            print("[INFO]: skip running kernel")
            print(f"[INFO]: The compiled kernel cache is in {cache_manager.cache_dir}")
            return
        profiler_registered = self.launch(*args, **kwargs)
        _ascend_utils.TRITON_PROFILER_REGISTERED = (profiler_registered == 1)


class NPUDriver(DriverBase):

    def __init__(self):
        self.utils = NPUUtils()
        self.launcher_cls = NPULauncher
        super().__init__()

    @classmethod
    def is_active(cls):

        def test_npucompiler():
            from triton.backends.ascend.utils import _get_bisheng_path
            npucompiler = _get_bisheng_path()
            targets = subprocess.check_output([npucompiler, "-print-targets"]).decode().strip().split()
            return "hiipu64" in targets

        try:
            return test_npucompiler()
        except Exception as e_npucompiler:
            import warnings
            red = "\x1b[31;20m"
            reset = "\x1b[0m"
            warnings.warn(red + str(e_npucompiler) + reset)
            return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_current_target(self):
        backend = "npu"
        arch = self.utils.get_arch()
        warp_size = 0
        return GPUTarget(backend, arch, warp_size)

    def get_current_device(self):
        """
        Get current device
        """
        import torch
        return torch.npu.current_device()

    def get_active_torch_device(self):
        import torch
        return torch.device("npu", self.get_current_device())

    def set_current_device(self, device):
        """
        Set current device as the given device
        """
        import torch
        return torch.npu.set_device(device)

    def get_current_stream(self, device: Optional[int] = None) -> int:
        """
        Get stream for current device
        """
        import torch
        import torch_npu
        if device is None:
            device = torch.npu.current_device()
        if hasattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait"):
            from torch_npu._C import _npu_getCurrentRawStreamNoWait
            return _npu_getCurrentRawStreamNoWait(device)
        else:
            from torch_npu._C import _npu_getCurrentRawStream
            return _npu_getCurrentRawStream(device)

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_device_interface(self):
        return get_backend_func("get_device_interface")

    def get_empty_cache_for_benchmark(self):
        cache_size = 192 * 1024 * 1024
        return get_backend_func("get_empty_tensor", cache_size // 4)

    def clear_cache(self, cache):
        cache.zero_()


def make_npu_launcher_stub(header_src, wrapper_src, debug=False):
    """
    Generate the launcher stub to launch the kernel
    """
    so_cache_key = hashlib.sha256((header_src + "\0" + wrapper_src).encode("utf-8")).hexdigest()
    so_cache_manager = get_cache_manager(so_cache_key)
    use_cxx11_abi = _check_cxx11_abi()
    name = f"launcher_cxx11abi{use_cxx11_abi}"
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so_name = f"{name}{suffix}"

    if debug:
        dump_manager = get_dump_manager(so_cache_key)
        print(f"Dumping precompiled.h to {dump_manager.cache_dir}")
        dump_manager.put(header_src, "precompiled.h", binary=False)
        print(f"Dumping {name}.cxx to {dump_manager.cache_dir}")
        dump_manager.put(wrapper_src, f"{name}.cxx", binary=False)

    cache_path = so_cache_manager.get_file(so_name)
    if cache_path is not None:
        return cache_path

    kernel_launcher_type = "torch"
    import fcntl
    cache_dir = os.getenv("TRITON_CACHE_DIR", os.path.join(os.path.expanduser("~/.triton"), "cache"))
    lock_dir = os.path.join(cache_dir, "lock")
    if not (os.path.exists(lock_dir)):
        os.makedirs(lock_dir, exist_ok=True)
    with open(os.path.join(lock_dir, f"{so_cache_key}.lock"), "w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            cache_path = so_cache_manager.get_file(so_name)
            if cache_path is not None:
                return cache_path
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, f"{name}.cxx")
                with open(src_path, "w") as f:
                    f.write(wrapper_src)
                so_path = _build_npu_ext(name, src_path, kernel_launcher=kernel_launcher_type)
                if debug:
                    with open(so_path, "rb") as f:
                        dump_manager.put(f.read(), so_name, binary=True)
                with open(so_path, "rb") as f:
                    so_cache_path = so_cache_manager.put(f.read(), so_name, binary=True)
            return so_cache_path
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def extract_device_print_code_from_cann():
    from triton.backends.ascend.utils import _get_bisheng_path
    ccec_compiler_bin_folder, _ = os.path.split(os.path.realpath(_get_bisheng_path()))
    ccec_compiler_folder, _ = os.path.split(ccec_compiler_bin_folder)
    clang_version = os.listdir(os.path.join(ccec_compiler_folder, "lib/clang/"))[0]
    ccelib_path = os.path.join(ccec_compiler_folder, f"lib/clang/{clang_version}/include/ccelib")

    def read_header(header_path):
        with open(os.path.join(ccelib_path, header_path), 'r') as f:
            code = f.read()

        # remove all #include "..."
        lines = code.splitlines()
        purged_lines = []
        for line in lines:
            normalized_line = ' '.join(line.split())
            if not normalized_line.startswith('#include "'):
                purged_lines.append(line)
        code = '\n'.join(purged_lines)

        # remove [aicore] functions
        aicore_positions = []
        for m in re.finditer(r'\[aicore\]', code):
            aicore_positions.append(m.start())

        def find_aicore_function_span(src, pos):
            for i in range(pos - 1, -1, -1):
                if src[i] == '}':  # this relies on that all [aicore] functions come after normal functions
                    left = i + 1
                    break
            n = len(src)
            brace_nest = 0
            for j in range(pos, n, 1):
                if src[j] == '{':
                    brace_nest += 1
                elif src[j] == '}':
                    brace_nest -= 1
                    if brace_nest == 0:
                        right = j
                        break
            return left, right

        new_code = ''
        segment_start = 0
        for pos in aicore_positions:
            left, right = find_aicore_function_span(code, pos)
            new_code += code[segment_start:left]
            segment_start = right + 1
        new_code += code[segment_start:]

        # remove __gm__ and rename macros
        new_code = new_code.replace('__gm__', ' ')
        new_code = new_code.replace('__CCELIB_RT_ERROR_NONE', 'RT_ERROR_NONE')
        new_code = new_code.replace('__CCELIB_RT_MEMORY_HBM', 'RT_MEMORY_HBM')
        new_code = new_code.replace('__CCELIB_RT_MEMCPY_HOST_TO_DEVICE', 'RT_MEMCPY_HOST_TO_DEVICE')
        new_code = new_code.replace('__CCELIB_RT_MEMCPY_DEVICE_TO_HOST', 'RT_MEMCPY_DEVICE_TO_HOST')
        return new_code

    # the following headers should be included in this order
    return '\n'.join([
        read_header('common/common_impl.h'),
        read_header('internal/debug_tunnel/payload.h'),
        read_header('internal/debug_tunnel/payload_impl.h'),
        read_header('internal/debug_tunnel/tunnel.h'),
        read_header('internal/debug_tunnel/tunnel_impl.h')
    ])


def generate_npu_header_src():
    enable_taskqueue = os.getenv("TRITON_ENABLE_TASKQUEUE", 'true').lower() in ('true', '1')
    return f"""
#ifndef TRITON_NPU_HEADERS
#define TRITON_NPU_HEADERS
#include <assert.h>
#include <stdbool.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <memory>
#include <sys/syscall.h>
#include <vector>
#include <map>
#include <mutex>
#include <tuple>
#include <Python.h>
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
{get_backend_func("header_file", enable_taskqueue)}
#endif

// Compatibility shim for CANN runtime API transition (rt -> aclrt in 9.0.0).
#ifdef TRITON_CANN_910
using cann_error = aclError;
using cann_stream = aclrtStream;
using cann_func_handle = aclrtFuncHandle;
using cann_memcpy_kind = aclrtMemcpyKind;
static constexpr cann_error CANN_SUCCESS = ACL_SUCCESS;
static constexpr cann_memcpy_kind CANN_MEMCPY_HOST_TO_DEVICE = ACL_MEMCPY_HOST_TO_DEVICE;
static inline cann_error cann_malloc_host(void **ptr, size_t size) {{ return aclrtMallocHost(ptr, size); }}
static inline cann_error cann_free_host(void *ptr) {{ return aclrtFreeHost(ptr); }}
static inline cann_error cann_memcpy_async(void *dst, size_t destMax, const void *src, size_t count, cann_memcpy_kind kind, cann_stream stream) {{ return aclrtMemcpyAsync(dst, destMax, src, count, kind, stream); }}
static inline cann_error cann_memset_async(void *dst, size_t destMax, int32_t value, size_t count, cann_stream stream) {{ return aclrtMemsetAsync(dst, destMax, value, count, stream); }}
static inline cann_error cann_synchronize_stream(cann_stream stream) {{ return aclrtSynchronizeStream(stream); }}
static inline cann_error cann_get_hardware_sync_addr(void **addr) {{ return aclrtGetHardwareSyncAddr(addr); }}
static inline cann_error cann_launch_kernel(cann_func_handle func, uint32_t block_dim, cann_stream stream, void *cfg, void *args, size_t arg_size) {{
  return aclrtLaunchKernelWithHostArgs(func, block_dim, stream, static_cast<aclrtLaunchKernelCfg *>(cfg), args, arg_size, nullptr, 0);
}}
static inline void* cann_get_launch_kernel_cfg(uint32_t shared_mem_dynamic_size) {{
  // thread_local storage: launch is synchronous on the launcher thread, so the
  // returned pointer remains valid until cann_launch_kernel returns.
  static thread_local aclrtLaunchKernelAttr attrInfo;
  static thread_local aclrtLaunchKernelCfg cfgCfgInfo;
  attrInfo.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
  aclrtLaunchKernelAttrValue attrValue;
  attrValue.localMemorySize = shared_mem_dynamic_size;
  attrInfo.value = attrValue;
  cfgCfgInfo.attrs = &attrInfo;
  cfgCfgInfo.numAttrs = 1;
  return &cfgCfgInfo;
}}
#else
using cann_error = rtError_t;
using cann_stream = rtStream_t;
using cann_func_handle = const void*;
using cann_memcpy_kind = rtMemcpyKind_t;
static constexpr cann_error CANN_SUCCESS = RT_ERROR_NONE;
static constexpr cann_memcpy_kind CANN_MEMCPY_HOST_TO_DEVICE = RT_MEMCPY_HOST_TO_DEVICE;
static inline cann_error cann_malloc_host(void **ptr, size_t size) {{ return rtMallocHost(ptr, size, RT_MEMORY_HOST); }}
static inline cann_error cann_free_host(void *ptr) {{ return rtFreeHost(ptr); }}
static inline cann_error cann_memcpy_async(void *dst, size_t destMax, const void *src, size_t count, cann_memcpy_kind kind, cann_stream stream) {{ return rtMemcpyAsync(dst, destMax, src, count, kind, stream); }}
static inline cann_error cann_memset_async(void *dst, size_t destMax, int32_t value, size_t count, cann_stream stream) {{ return rtMemsetAsync(dst, destMax, value, count, stream); }}
static inline cann_error cann_synchronize_stream(cann_stream stream) {{ return rtStreamSynchronize(stream); }}
static inline cann_error cann_get_hardware_sync_addr(void **addr) {{ uint32_t len = 0; return rtGetC2cCtrlAddr(reinterpret_cast<uint64_t *>(addr), &len); }}
static inline cann_error cann_launch_kernel(cann_func_handle func, uint32_t block_dim, cann_stream stream, void *cfg, void *args, size_t arg_size) {{
  if (cfg != nullptr) {{
    rtArgsEx_t argsInfo = {{}};
    argsInfo.args = args;
    argsInfo.argsSize = arg_size;
    return rtKernelLaunchWithFlagV2(func, block_dim, &argsInfo, NULL, stream, 0, static_cast<rtTaskCfgInfo_t *>(cfg));
  }}
  return rtKernelLaunch(func, block_dim, args, arg_size, NULL, stream);
}}
static inline void* cann_get_launch_kernel_cfg(uint32_t shared_mem_dynamic_size) {{
  // thread_local storage: launch is synchronous on the launcher thread, so the
  // returned pointer remains valid until cann_launch_kernel returns.
  static thread_local rtTaskCfgInfo_t cfgInfo;
  cfgInfo.localMemorySize = shared_mem_dynamic_size;
  return &cfgInfo;
}}
#endif

// Initialize the reusable lock workspace asynchronously on the compute stream.
// Synchronous cann_memcpy does not synchronize with the stream, so a prior
// kernel still consuming the lock buffer could be clobbered. Cache pinned
// host templates for the lifetime of the process: queued copies and captured
// graphs may outlive the launch that first used them.
static inline cann_error initializeLocks(uint64_t orderedLocks,
                                         uint64_t unorderedLocks,
                                         uint64_t participants,
                                         int64_t lockInitValue,
                                         uint64_t bytes,
                                         uint64_t elements,
                                         uint64_t orderedElements,
                                         uint64_t unorderedStride,
                                         void *dst, cann_stream stream) {{
  using Key = std::tuple<uint64_t, uint64_t, uint64_t, int64_t>;
  static std::mutex mutex;
  static std::map<Key, void *> templates;
  const void *src;
  {{
    std::lock_guard<std::mutex> guard(mutex);
    auto &cached = templates[Key(orderedLocks, unorderedLocks, participants,
                                 lockInitValue)];
    if (!cached) {{
      void *data = nullptr;
      cann_error ret = cann_malloc_host(&data, bytes);
      if (ret != CANN_SUCCESS)
        return ret;
      auto *init = static_cast<int64_t *>(data);
      for (uint64_t i = 0; i < elements; ++i)
        init[i] = unorderedLocks ? 0 : lockInitValue;
      for (uint64_t i = 0; i < unorderedLocks; ++i)
        init[orderedElements + i * unorderedStride] =
            static_cast<int64_t>(participants);
      cached = data;
    }}
    src = cached;
  }}
  return cann_memcpy_async(dst, bytes, src, bytes, CANN_MEMCPY_HOST_TO_DEVICE,
                           stream);
}}
"""


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    if ty.startswith("tensordesc"):
        return "void*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


_BASE_ARGS_FORMAT = "iiiKKOOOO"
_BASE_ARGS_FORMAT_LEN = len(_BASE_ARGS_FORMAT)


def make_tensordesc_arg(arg):
    return [arg.base, *arg.shape, *arg.strides, arg.padding == "nan", *arg.shape, *arg.strides]


def wrap_handle_tensordesc(launcher, signature):
    has_tensor_desc_arg = any(isinstance(sig, str) and sig.startswith("tensordesc") for sig in signature.values())
    if not has_tensor_desc_arg:
        return launcher

    tensordesc_indices = set(
        [i for i, sig in enumerate(signature.values()) if isinstance(sig, str) and sig.startswith("tensordesc")])

    def inner(*args):
        final_args = list(args[:_BASE_ARGS_FORMAT_LEN])
        for i, arg in enumerate(args[_BASE_ARGS_FORMAT_LEN:]):
            if i in tensordesc_indices:
                final_args.extend(make_tensordesc_arg(arg))
            else:
                final_args.append(arg)
        return launcher(*final_args)

    return inner


_CPP_DEVICE_POINTER = r"""
typedef struct _DevicePtrInfo {
  void* dev_ptr;
  bool valid;
} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject* obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = nullptr;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = reinterpret_cast<void*>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }
  if (obj == Py_None) {
    return ptr_info;
  }
  // Cache the interned "data_ptr" key once instead of rebuilding a temporary
  // PyUnicode on every call. Function-local static init is thread-safe in C++11
  // and the GIL is held here, so the one-time init is safe.
  static PyObject* data_ptr_str = PyUnicode_InternFromString("data_ptr");
  // PyObject_CallMethodNoArgs avoids creating a temporary tuple and a temporary
  // method-name PyUnicode on every call (Python 3.9+).
  PyObject* ret = PyObject_CallMethodNoArgs(obj, data_ptr_str);
  if (ret) {
    if (!PyLong_Check(ret)) {
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      Py_DECREF(ret);
      return ptr_info;
    }
    ptr_info.dev_ptr = reinterpret_cast<void*>(PyLong_AsUnsignedLongLong(ret));
    Py_DECREF(ret);
    if (!ptr_info.dev_ptr) {
      return ptr_info;
    }
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}
"""

_CPP_MSPROF_EXTERN = r"""
extern "C" {
typedef int (*callback)(unsigned int type, void* data, unsigned int len);
extern int MsprofReportApi(unsigned int agingFlag, const MsprofApi* api);
extern unsigned long int MsprofSysCycleTime();
extern int MsprofRegisterCallback(unsigned int moduleId, callback handle);
static unsigned int __MsprofFlagL0 = 0;
static unsigned int __MsprofFlagL1 = 0;
static std::vector<int> tensorKinds;

int ProfCtrlHandle(unsigned int CtrlType, void* CtrlData, unsigned int DataLen) {
  if ((CtrlData == nullptr) || (DataLen == 0U)) {
    return 1;
  }
  if (CtrlType == 1) {
    MsprofCommandHandle* handle = (MsprofCommandHandle*)(CtrlData);
    if (handle->type >= 6) {
      return 1;
    }
    if (handle->type == 1) {
      __MsprofFlagL0 = ((0x00000800ULL & handle->profSwitch) == 0x00000800ULL) ? 1 : 0;
      __MsprofFlagL1 = ((0x00000002ULL & handle->profSwitch) == 0x00000002ULL) ? 1 : 0;
    }
  }
  return 0;
}
}
"""

_CPP_MSPROF_CALLBACK = r"""
    MsprofRegisterCallback(8, ProfCtrlHandle);
"""

_CPP_MSPROF_BEFORE_LAUNCH = r"""
    unsigned long int beginTime = 0;
    unsigned long int endTime = 0;
    unsigned long int opNameHashID = 0;
    unsigned int threadId = 0;
    char* _kernelName = const_cast<char*>(kernelName);
    size_t length = kernelName ? strlen(kernelName) : 0;
    if (__MsprofFlagL0 || __MsprofFlagL1) {
      beginTime = MsprofSysCycleTime();
    }
"""

_CPP_ALIGN_LAUNCH_OFFSET = r"""
static inline size_t _align_launch_offset(size_t offset, size_t alignment) {
  return (offset + alignment - 1) & ~(alignment - 1);
}

// cann_get_hardware_sync_addr returns a per-process per-stream constant address;
// re-querying it on every kernel launch is pure overhead. Cache the most
// recently observed (stream, ffts_addr) pair on the calling thread.
// Thread-safety: launch_call is invoked synchronously from the launcher thread
// by cann_async_launch (see npu_utils.cpp), so thread_local is safe.
static thread_local cann_stream g_last_ffts_stream = nullptr;
static thread_local void* g_last_ffts_addr = nullptr;
static inline cann_error get_ffts_addr(cann_stream stream, void** out_addr) {
  if (stream == g_last_ffts_stream && g_last_ffts_addr) {
    *out_addr = g_last_ffts_addr;
    return CANN_SUCCESS;
  }
  void* ffts_addr = nullptr;
  cann_error ret = cann_get_hardware_sync_addr(&ffts_addr);
  if (ret == CANN_SUCCESS) {
    g_last_ffts_stream = stream;
    g_last_ffts_addr = ffts_addr;
    *out_addr = ffts_addr;
  }
  return ret;
}
"""

_CPP_GET_TENSOR_SHAPE = r"""
static std::vector<int64_t> _get_tensor_shape(PyObject* tensor) {
  std::vector<int64_t> shape;
  if (!tensor || tensor == Py_None) {
    return shape;
  }
  // Cache the interned "size" key once; avoid temporary PyUnicode/tuple per call.
  static PyObject* size_str = PyUnicode_InternFromString("size");
  // PyObject_CallMethodNoArgs avoids building "size" PyUnicode and an empty
  // tuple on every launch (Python 3.9+).
  PyObject* size_result = PyObject_CallMethodNoArgs(tensor, size_str);
  if (!size_result) {
    // Defensive: profiling-only path; swallow attribute errors so subsequent
    // PyErr_Occurred() checks in launch() are not poisoned.
    PyErr_Clear();
    return shape;
  }
  PyObject* seq = PySequence_Fast(size_result, "Expected a sequence from tensor.size()");
  if (seq) {
    Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < len; ++i) {
      PyObject* dim = items[i];
      if (PyLong_Check(dim)) {
        shape.push_back(PyLong_AsLong(dim));
      }
    }
  }
  Py_DECREF(seq);
  Py_DECREF(size_result);
  return shape;
}
"""


# the template is from triton-adapter HEAD. Wrapping the generated kernel binary into a python module
def make_launcher(constants, signature, metadata):
    import os
    workspace_size = int(metadata.workspace_size) \
                          if hasattr(metadata, 'workspace_size') else -1
    lock_init_value = int(metadata.lock_init_value if hasattr(metadata, 'lock_init_value') else metadata.
                          lock_init_val if hasattr(metadata, 'lock_init_val') else 0)
    sync_block_lock_layout = int(getattr(metadata, "sync_block_lock_layout", 0))
    ordered_sync_block_lock_count = sync_block_lock_layout & 0xFFFFFFFF
    unordered_sync_block_lock_count = (sync_block_lock_layout >> 32) & 0xFFFFFFFF
    has_sync_block_lock = (ordered_sync_block_lock_count + unordered_sync_block_lock_count) > 0
    unordered_sync_block_participant_factor = (2 if unordered_sync_block_lock_count > 0 and metadata.mix_mode == "mix"
                                               and getattr(metadata, "auto_tile_and_bind_subblock", False) else 1)
    sync_block_lock_layout_stmt = f"""
    constexpr uint64_t syncBlockLockCacheLineI64 = 8;
    constexpr uint64_t syncBlockLockOrderedCount = {ordered_sync_block_lock_count};
    constexpr uint64_t syncBlockLockUnorderedCount = {unordered_sync_block_lock_count};
    constexpr uint64_t syncBlockLockParticipantFactor = {unordered_sync_block_participant_factor};
    const uint64_t syncBlockLockParticipantNum =
        blockNum * syncBlockLockParticipantFactor;
    const uint64_t syncBlockLockUnorderedStrideI64 =
        (1 + 2 * syncBlockLockParticipantNum) * syncBlockLockCacheLineI64;
    const uint64_t syncBlockLockOrderedI64Count =
        syncBlockLockOrderedCount * syncBlockLockCacheLineI64;
    const uint64_t syncBlockLockI64Count = syncBlockLockOrderedI64Count +
        syncBlockLockUnorderedCount * syncBlockLockUnorderedStrideI64;
    const uint64_t syncBlockLockSize = syncBlockLockI64Count * sizeof(int64_t);"""
    # Initialize on the compute stream. Synchronous memcpy is not ordered
    # against the stream and could overwrite lock state still consumed by a
    # previous kernel, so non-zero patterns go through initializeLocks which
    # copies a cached pinned host template asynchronously on the stream.
    if not unordered_sync_block_lock_count and not lock_init_value:
        lock_init_stmt = ("ret = cann_memset_async(syncBlockLock_ptr, syncBlockLockSize, 0, "
                          "syncBlockLockSize, stream);")
    else:
        lock_init_stmt = ("ret = initializeLocks("
                          "syncBlockLockOrderedCount, syncBlockLockUnorderedCount, "
                          "syncBlockLockParticipantNum, "
                          f"{lock_init_value}, "
                          "syncBlockLockSize, syncBlockLockI64Count, "
                          "syncBlockLockOrderedI64Count, syncBlockLockUnorderedStrideI64, "
                          "syncBlockLock_ptr, stream);")
    bs_task_type = metadata.bs_task_type if hasattr(metadata, 'bs_task_type') else 0
    mix_mode = metadata.mix_mode
    compile_on_910_95 = metadata.compile_on_910_95
    parallel_mode = metadata.parallel_mode
    enable_simt = ("simt" in parallel_mode) or metadata.is_pure_simt

    def _expand_signature(signature):
        output = []
        # Expand tensor descriptor arguments into base pointer, shape and
        # strides. Ascend always rewrites tensordesc to pointer (no TMA).
        for sig in signature:
            if isinstance(sig, str) and sig.startswith("tensordesc"):
                match = re.match("tensordesc<([^[>]*)\\[([^]]*)\\]", sig)
                dtype = match.group(1)
                shape = match.group(2)
                ndim = shape.count(",") + 1

                output.append("*" + dtype)
                # Currently the host side tensor descriptors get passed in as a
                # tensor desc, shape, and strides. We have no way to use these
                # shape and strides when processing tensor descriptors which is
                # why we provide our own decomposition above. Sadly this means
                # we have to pass the shape and strides twice.
                for _ in range(2 * ndim):
                    output.append("i64")
                output.append("i1")

                for _ in range(ndim):
                    output.append("i32")
                for _ in range(ndim):
                    output.append("i64")
            else:
                output.append(sig)

        return output

    def _flatten_signature(sig, output):
        # Flatten tuples
        if isinstance(sig, tuple):
            for x in sig:
                _flatten_signature(x, output)
        else:
            output.append(sig)

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty.startswith("tensordesc"):
            return "PyObject*"
        if ty in ("constexpr"):
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty.startswith("tensordesc"):
            return "O"
        if ty in ("constexpr"):
            return "O"
        if ty == "void*":
            return "O"
        return {
            "float": "f",
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty_to_cpp(ty)]

    def _format_to_fastcall_stmt(ty, var, idx):
        """Generate C statement to parse argument at index idx from METH_FASTCALL args array."""
        fmt = format_of(ty)
        if fmt == "O":
            return f"{var} = args[{idx}];"
        elif fmt == "i":
            return f"{var} = (int)PyLong_AsLong(args[{idx}]);"
        elif fmt == "L":
            return f"{var} = (int64_t)PyLong_AsLongLong(args[{idx}]);"
        elif fmt == "K":
            return f"{var} = (uint64_t)PyLong_AsUnsignedLongLong(args[{idx}]);"
        elif fmt == "I":
            return f"{var} = (uint32_t)PyLong_AsUnsignedLong(args[{idx}]);"
        elif fmt == "H":
            return f"{var} = (uint16_t)PyLong_AsUnsignedLong(args[{idx}]);"
        elif fmt == "B":
            return f"{var} = (uint8_t)PyLong_AsUnsignedLong(args[{idx}]);"
        elif fmt == "h":
            return f"{var} = (int16_t)PyLong_AsLong(args[{idx}]);"
        elif fmt == "b":
            return f"{var} = (int8_t)PyLong_AsLong(args[{idx}]);"
        elif fmt == "l":
            return f"{var} = (long)PyLong_AsLong(args[{idx}]);"
        elif fmt == "f":
            return f"{var} = (float)PyFloat_AsDouble(args[{idx}]);"
        elif fmt == "d":
            return f"{var} = PyFloat_AsDouble(args[{idx}]);"
        else:
            raise ValueError(f"Unsupported format: {fmt} for type {ty}")

    def _format_of_msprof_task_type_ratio(bs_task_type, mix_mode):
        # Default fallback based on mix_mode
        default_task_type = "MSPROF_GE_TASK_TYPE_AIV" if mix_mode == "aiv" else "MSPROF_GE_TASK_TYPE_AI_CORE"

        if not bs_task_type:
            return default_task_type, 0

        task_type_num, mix_block_dim_ratio = divmod(int(bs_task_type), 10)
        task_type_map = {
            1: "MSPROF_GE_TASK_TYPE_AIV",
            2: "MSPROF_GE_TASK_TYPE_AI_CORE",
            3: "MSPROF_GE_TASK_TYPE_MIX_AIC",
            4: "MSPROF_GE_TASK_TYPE_MIX_AIV",
        }

        task_type = task_type_map.get(task_type_num, default_task_type)
        return task_type, mix_block_dim_ratio

    """
    args:
        int gridX, gridY, gridZ;
        aclrtStream stream;
        aclrtFuncHandle functon;
        PyObject* packed_metadata, *launch_metadata;
        PyObject* launch_enter_hook, *launch_exit_hook;
        *args_expand
    """

    expand_signature = _expand_signature(signature.values())
    signature = {i: s for i, s in enumerate(expand_signature)}

    args_format = ''.join([format_of(ty) for ty in signature.values()])
    format = _BASE_ARGS_FORMAT + args_format

    flat_signature = []
    for sig in signature.values():
        _flatten_signature(sig, flat_signature)
    signature = {i: s for i, s in enumerate(flat_signature)}
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''
    # Total expected argument count for METH_FASTCALL arity check.
    total_nargs = _BASE_ARGS_FORMAT_LEN + len(signature)
    # Generate manual parsing statements for signature args (indices 9..) used by
    # the METH_FASTCALL fast path in launch().
    fastcall_sig_parse_stmts = '\n  '.join(
        _format_to_fastcall_stmt(ty, f"_arg{i}", _BASE_ARGS_FORMAT_LEN + i) for i, ty in signature.items())
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors.
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items() if ty != "constexpr")
    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"ptr_info{i}.dev_ptr")
        elif ty != "constexpr":
            internal_args_list.append(f"_arg{i}")

    # generate glue code
    newline = '\n  '
    ptr_decls = [
        f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return nullptr;"
        for i, ty in signature.items()
        if ty[0] == "*"
    ]
    grid_info = {'X': 'i32', 'Y': 'i32', 'Z': 'i32'}
    # TODO: automatically check if gather load ops are used.

    arch = metadata.target.arch
    target_support_ffts = is_ffts_supported(arch) and (not force_disable_ffts(arch))
    enable_device_print = os.getenv("TRITON_DEVICE_PRINT", 'false').lower() in ('true', '1')
    enable_taskqueue = os.getenv("TRITON_ENABLE_TASKQUEUE", 'true').lower() in ('true', '1')
    enable_grid_warn_print = os.getenv("TRITON_GRID_WARN_PRINT", 'false').lower() in ('true', '1')
    npu_utils = NPUUtils()
    num_physical_blocks = npu_utils.get_aivector_core_num() if mix_mode == "aiv" else npu_utils.get_aicore_num()
    task_type, mix_block_dim_ratio = _format_of_msprof_task_type_ratio(bs_task_type, mix_mode)
    is_mix_task_type = "true" if ("MIX" in task_type) else "false"
    LINE_CHANGE_CHAR = '\n'
    alloc_success_code = 'return 1;'
    sync_lock_fail_code = 'fprintf(stderr, "Error: syncBlockLock allocation failed\\n"); return;'
    workspace_fail_code = 'fprintf(stderr, "Error: workspace allocation failed\\n"); return;'
    npu_utils_so_path = NPUUtils().get_so_path()
    # The generated launcher source is part of its cache key. Preserve only the
    # deterministic cache-key directory so the launcher can be reused after the
    # cache root changes.
    npu_utils_cache_relative = os.path.join(
        os.path.basename(os.path.dirname(npu_utils_so_path)),
        os.path.basename(npu_utils_so_path),
    )
    cpp_npu_utils_dlopen = f"""
typedef void* (*triton_allocate_workspace_t)(uint64_t, void**);
typedef void* (*triton_allocate_sync_block_lock_t)(uint64_t, void*, void**);
typedef void  (*triton_async_launch_t)(void*, const char*);
typedef void  (*triton_release_retained_tensor_t)(void*);

static triton_allocate_workspace_t g_allocate_workspace = nullptr;
static triton_allocate_sync_block_lock_t g_allocate_sync_block_lock = nullptr;
static triton_async_launch_t g_async_launch = nullptr;
static triton_release_retained_tensor_t g_release_retained_tensor = nullptr;

static bool npu_utils_ready() {{
    return g_allocate_workspace &&
           g_allocate_sync_block_lock &&
           g_async_launch &&
           g_release_retained_tensor;
}}

static void init_npu_utils() {{
    if (npu_utils_ready()) return;
    const char* cache_root = std::getenv("TRITON_CACHE_DIR");
    std::string npu_utils_path;
    if (cache_root && cache_root[0] != '\\0') {{
        npu_utils_path = std::string(cache_root) + "/{npu_utils_cache_relative}";
    }} else {{
        const char* triton_home = std::getenv("TRITON_HOME");
        const char* home = std::getenv("HOME");
        const char* base = triton_home && triton_home[0] != '\\0' ? triton_home : home;
        if (!base || base[0] == '\\0') {{
            fprintf(stderr, "Error: neither TRITON_CACHE_DIR nor TRITON_HOME/HOME is set\\n");
            return;
        }}
        npu_utils_path = std::string(base) + "/.triton/cache/{npu_utils_cache_relative}";
    }}
    void* handle = dlopen(npu_utils_path.c_str(), RTLD_LAZY);
    if (!handle) {{
        fprintf(stderr, "Error: dlopen %s failed: %s\\n", npu_utils_path.c_str(), dlerror());
        return;
    }}
    g_allocate_workspace = (triton_allocate_workspace_t)dlsym(handle, "triton_allocate_workspace");
    g_allocate_sync_block_lock = (triton_allocate_sync_block_lock_t)dlsym(handle, "triton_allocate_sync_block_lock");
    g_async_launch = (triton_async_launch_t)dlsym(handle, "triton_async_launch");
    g_release_retained_tensor = (triton_release_retained_tensor_t)dlsym(handle, "triton_release_retained_tensor");
}}

static void release_npu_tensor_handle(void* handle) {{
    if (!handle) return;
    if (!g_release_retained_tensor) {{
        fprintf(stderr, "Error: triton_release_retained_tensor is unavailable\\n");
        return;
    }}
    g_release_retained_tensor(handle);
}}
"""

    mapping_applied = getattr(metadata, "program_grid_mapping_applied", None)
    row_coalescing_applied = getattr(metadata, "row_coalescing_applied", None)
    auto_blockify_enabled = getattr(metadata, "auto_blockify_enabled", None)
    ptsm_cap_authorized = getattr(metadata, "ptsm_cap_authorized", None)
    if not isinstance(mapping_applied, bool):
        raise RuntimeError("compiler metadata missing program_grid_mapping_applied")
    if not isinstance(row_coalescing_applied, bool):
        raise RuntimeError("compiler metadata missing row_coalescing_applied")
    if not isinstance(auto_blockify_enabled, bool):
        raise RuntimeError("compiler metadata missing auto_blockify_enabled")
    if not isinstance(ptsm_cap_authorized, bool):
        raise RuntimeError("compiler metadata missing ptsm_cap_authorized")
    raw_program_grid_transforms = getattr(metadata, "program_grid_transforms", None)
    if not mapping_applied:
        if raw_program_grid_transforms is not None:
            raise RuntimeError("program_grid_mapping_applied disagrees with program_grid_transforms")
        program_grid_transforms = None
    else:
        if raw_program_grid_transforms is None:
            raise RuntimeError("program_grid_mapping_applied requires program_grid_transforms")
        try:
            program_grid_transforms = normalize_program_grid_transforms(raw_program_grid_transforms)
        except ProgramGridContractError as error:
            raise RuntimeError(f"invalid program_grid_transforms launcher metadata: {error}") from error

    persistent_transform = (get_persistent_transform(program_grid_transforms)
                            if program_grid_transforms is not None else None)
    if program_grid_transforms is not None and row_coalescing_applied:
        raise RuntimeError("program-grid transforms conflict with legacy RowCoalescing")
    if auto_blockify_enabled and ptsm_cap_authorized:
        raise RuntimeError("auto_blockify_enabled and ptsm_cap_authorized cannot both be true")
    if persistent_transform is None:
        if ptsm_cap_authorized:
            raise RuntimeError("ptsm_cap_authorized requires a persistent transform")
    elif not ptsm_cap_authorized:
        raise RuntimeError("persistent program-grid transform lacks PTSM cap authorization")
    elif mix_mode != "aiv":
        raise RuntimeError("persistent program-grid transform requires final mix_mode=aiv")

    launcher_cap_enabled = auto_blockify_enabled

    program_grid_finalization = ""
    if program_grid_transforms is not None:
        finalization_lines = [
            f"// hacc.program_grid_transforms v{PROGRAM_GRID_TRANSFORMS_VERSION}: dynamic original-grid ABI.",
            "uint32_t originalGridX = static_cast<uint32_t>(gridX);",
            "uint32_t originalGridY = static_cast<uint32_t>(gridY);",
        ]
        axis_names = {0: ("gridX", "originalGridX"), 1: ("gridY", "originalGridY")}
        for transform in program_grid_transforms["transforms"]:
            grid_name, original_name = axis_names[transform["axis"]]
            factor = transform["factor"]
            finalization_lines.append(
                f"{grid_name} = (uint32_t)(((uint64_t){original_name} + {factor - 1}u) / {factor}u);")
    if persistent_transform is not None:
        axis = persistent_transform["axis"]
        grid_name = {0: "gridX", 1: "gridY", 2: "gridZ"}[axis]
        other_grid_names = tuple(name for index, name in enumerate(("gridX", "gridY", "gridZ")) if index != axis)
        finalization_lines.extend((
            f"uint64_t otherPrograms = (uint64_t){other_grid_names[0]} * (uint64_t){other_grid_names[1]};",
            "uint32_t axisCap = std::max((uint32_t)1,",
            f"    (uint32_t)((uint64_t){num_physical_blocks} / std::max((uint64_t)1, otherPrograms)));",
            f"{grid_name} = std::min({grid_name}, static_cast<int>(axisCap));",
        ))
    if program_grid_transforms is not None:
        program_grid_finalization = "\n  ".join(finalization_lines)

    # Full-TA tile/strided coalescing: the compiler recorded a coalesce factor H
    # and the program-id/grid axis it applies to. Each program now covers H tiles
    # along that axis, so the host shrinks the matching grid dim by H here (the
    # equivalent of what bishengir AutoBlockify used to do via hacc.coalesce_factor;
    # bishengir no longer touches it). RowCoalescing can request ceil-div because
    # its generated row mask handles tail rows.
    coalesce_factor = int(getattr(metadata, "coalesce_factor", 1) or 1)
    coalesce_axis = int(getattr(metadata, "coalesce_axis", -1))
    coalesce_grid_ceil_div = bool(getattr(metadata, "coalesce_grid_ceil_div", False))
    if coalesce_factor > 1 and coalesce_axis in (0, 1, 2):
        _coalesce_grid_var = {0: "gridX", 1: "gridY", 2: "gridZ"}[coalesce_axis]
        _coalesce_grid_expr = (f"({_coalesce_grid_var} + {coalesce_factor} - 1) / {coalesce_factor}"
                               if coalesce_grid_ceil_div else f"{_coalesce_grid_var} / {coalesce_factor}")
        coalesce_grid_div = (
            f"// coalescing: each program covers {coalesce_factor} tiles along "
            f"axis {coalesce_axis}; shrink that grid dim.\n" +
            ("" if coalesce_grid_ceil_div else f"  assert({_coalesce_grid_var} % {coalesce_factor} == 0 && "
             f"\"ChunkCoalescing: grid[{coalesce_axis}] not divisible by coalesce_factor {coalesce_factor}\");\n") +
            f"  {_coalesce_grid_var} = {_coalesce_grid_expr};")
    else:
        coalesce_grid_div = ""
    if program_grid_transforms is not None and coalesce_factor > 1:
        raise RuntimeError("program-grid transforms conflict with legacy coalesce metadata")
    if row_coalescing_applied:
        if coalesce_factor <= 1 or coalesce_axis not in (0, 1, 2):
            raise RuntimeError("row_coalescing_applied requires valid coalesce metadata")
    elif coalesce_factor != 1 or coalesce_axis != -1 or coalesce_grid_ceil_div:
        raise RuntimeError("unmatched RowCoalescing must retain default coalesce metadata")
    has_dynamic_program_grid_abi = program_grid_transforms is not None

    cpp_device_pointer = _CPP_DEVICE_POINTER
    cpp_msprof_extern = _CPP_MSPROF_EXTERN
    cpp_msprof_callback = _CPP_MSPROF_CALLBACK
    cpp_msprof_call_before_launch = _CPP_MSPROF_BEFORE_LAUNCH

    cpp_msprof_call_after_launch = f"""
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {{
      endTime = MsprofSysCycleTime();
      opNameHashID = MsprofGetHashId(_kernelName, length);
      threadId = (unsigned int)(syscall(SYS_gettid));
      MsprofApi info;
      info.level = MSPROF_REPORT_NODE_LEVEL;
      info.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      info.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
      info.threadId = threadId;
      info.reserve = 0;
      info.beginTime = beginTime;
      info.endTime = endTime;
      info.itemId = opNameHashID;
      MsprofReportApi(false, &info);
    }}
    if (__MsprofFlagL1)
    {{
      MsprofCompactInfo nodeBasicInfo;
      nodeBasicInfo.level = MSPROF_REPORT_NODE_LEVEL;
      nodeBasicInfo.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
      nodeBasicInfo.threadId = threadId;
      nodeBasicInfo.timeStamp = endTime;
      nodeBasicInfo.data.nodeBasicInfo.opName = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.opType = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.taskType = {task_type};
      nodeBasicInfo.data.nodeBasicInfo.blockDim = nodeBasicBlockDim;
      MsprofReportCompactInfo(0, static_cast<void *>(&nodeBasicInfo), sizeof(MsprofCompactInfo));

      // 'mix' kernel need to report the ctxID
      if ({is_mix_task_type} > 0) {{
        MsprofAdditionalInfo info;
        info.level = MSPROF_REPORT_NODE_LEVEL;
        info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
        info.threadId = threadId;
        info.timeStamp = endTime;
        MsprofContextIdInfo ctxId;
        ctxId.opName = opNameHashID;
        ctxId.ctxIdNum = 1;
        for (uint32_t i = 0; i < ctxId.ctxIdNum; i++) {{
          ctxId.ctxIds[i] = i;
        }}
        size_t copyLen = sizeof(MsprofContextIdInfo);
        if (copyLen > MSPROF_ADDTIONAL_INFO_DATA_LENGTH) {{
          copyLen = MSPROF_ADDTIONAL_INFO_DATA_LENGTH;
        }}
        memcpy(info.data, &ctxId, copyLen);
        MsprofReportAdditionalInfo(false, static_cast<void *>(&info), sizeof(MsprofAdditionalInfo));
      }}

      // Report tensor info
      int max_tensors_num = tensorShapes.size() < MSPROF_GE_TENSOR_DATA_NUM ? tensorShapes.size() : MSPROF_GE_TENSOR_DATA_NUM;
      MsprofAdditionalInfo tensorInfo;
      tensorInfo.level = MSPROF_REPORT_NODE_LEVEL;
      tensorInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
      tensorInfo.threadId = threadId;
      tensorInfo.timeStamp = endTime;
      auto profTensorData = reinterpret_cast<MsprofTensorInfo *>(tensorInfo.data);
      profTensorData->opName = opNameHashID;
      int tensorCount = 0;
      int dataTypes[MSPROF_GE_TENSOR_DATA_NUM];
      if (tensorShapes.size() > 0) {{
        {LINE_CHANGE_CHAR.join(
          f'dataTypes[{idx}] = {convert_sigtype_to_int(ty[1:])};'
          for idx, (_, ty) in enumerate(
            (k, v) for k, v in signature.items() if v.startswith("*")
          )
          if idx < 5
        )}
      }}
      for (int i = 0; i < tensorShapes.size() && tensorCount < MSPROF_GE_TENSOR_DATA_NUM; i++) {{
        auto fillTensorData = [&](int index, int tensorType) {{
          profTensorData->tensorData[index].tensorType = tensorType;
          profTensorData->tensorData[index].format = 2; // GeDataFormat: ND = 2
          profTensorData->tensorData[index].dataType = dataTypes[i];
          int nDim = tensorShapes[i].size();
          nDim = nDim < MSPROF_GE_TENSOR_DATA_SHAPE_LEN ? nDim : MSPROF_GE_TENSOR_DATA_SHAPE_LEN;
          for (int j = 0; j < nDim; j++) {{
            profTensorData->tensorData[index].shape[j] = tensorShapes[i][j];
          }}
          for (int j = nDim; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {{
            profTensorData->tensorData[index].shape[j] = 0;
          }}
        }};
        int tensorType = (i < tensorKinds.size()) ? tensorKinds[i] : 0;  // DeFault tensor type is input
        if (tensorType == TENSOR_KIND_INPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) {{
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_INPUT);
          tensorCount++;
        }}
        if ((tensorType == TENSOR_KIND_OUTPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) && tensorCount < MSPROF_GE_TENSOR_DATA_NUM){{
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_OUTPUT);
          tensorCount++;
        }}
      }}
      profTensorData->tensorNum = tensorCount;
      MsprofReportAdditionalInfo(false, static_cast<void *>(&tensorInfo), sizeof(MsprofAdditionalInfo));
    }}
"""

    def _make_kernel_launch(args_ptr, args_size, indent="    "):
        cfg_setup = ""
        if compile_on_910_95 and enable_simt:
            cfg_setup = f"{indent}void *kernel_cfg = cann_get_launch_kernel_cfg({metadata.shared_mem_dynamic_size});\n"
        else:
            cfg_setup = f"{indent}void *kernel_cfg = nullptr;\n"
        return f"""{cfg_setup}{indent}ret = cann_launch_kernel(func, blockNum, stream, kernel_cfg, {args_ptr}, {args_size});
"""

    cpp_kernel_launch = _make_kernel_launch("static_cast<void*>(launch_args.data())", "launch_args.size()")
    cpp_kernel_launch_local = _make_kernel_launch("&args", "sizeof(args)", indent="        ")

    npu_headers = generate_npu_header_src()

    if not mapping_applied and not row_coalescing_applied:
        _launch_preamble = f"""
  void* workspace_addr_ptr = nullptr;
  void* workspace_handle = nullptr;
  {coalesce_grid_div}
  uint32_t blockNum4Workspace = gridX * gridY * gridZ;
  {get_backend_func("pre_launch", True)}
  {f'''
  uint64_t totalWorkSpaceSize = (uint64_t){workspace_size} * blockNum4Workspace;
  {get_backend_func("allocate_memory", "totalWorkSpaceSize", "stream")}
  std::shared_ptr<void> workspace_handle_guard(workspace_handle, release_npu_tensor_handle);
  if (!workspace_addr_ptr) {{
    {workspace_fail_code}
  }}
  ''' if workspace_size > 0 else ''}"""
        _python_launch_preamble = _launch_preamble
    else:
        _program_grid_launch_preamble = f"""
  {program_grid_finalization}
  {coalesce_grid_div if program_grid_transforms is None else ''}"""
        _launch_preamble = f"""
  void* workspace_addr_ptr = nullptr;
  void* workspace_handle = nullptr;
{_program_grid_launch_preamble}
  uint32_t blockNum4Workspace = gridX * gridY * gridZ;
  {get_backend_func("pre_launch", True)}
  {f'''
  uint64_t totalWorkSpaceSize = (uint64_t){workspace_size} * blockNum4Workspace;
  {get_backend_func("allocate_memory", "totalWorkSpaceSize", "stream")}
  std::shared_ptr<void> workspace_handle_guard(workspace_handle, release_npu_tensor_handle);
  if (!workspace_addr_ptr) {{
    {workspace_fail_code}
  }}
  ''' if workspace_size > 0 else ''}"""
        _python_launch_preamble = f"""
  void* workspace_addr_ptr = nullptr;
  void* workspace_handle = nullptr;
{_program_grid_launch_preamble}
  uint32_t blockNum4Workspace = gridX * gridY * gridZ;
  {get_backend_func("pre_launch", True)}
  {f'''
  uint64_t totalWorkSpaceSize = (uint64_t){workspace_size} * blockNum4Workspace;
  {get_backend_func("allocate_memory", "totalWorkSpaceSize", "stream")}
  std::shared_ptr<void> workspace_handle_guard(workspace_handle, release_npu_tensor_handle);
  if (!workspace_addr_ptr) {{
    {workspace_fail_code}
  }}
  ''' if workspace_size > 0 else ''}"""

    preserve_legacy_launcher_layout = mapping_applied or row_coalescing_applied
    original_grid_offset_decls = ("    size_t original_grid_x_offset = reserve_slot(sizeof(uint32_t), 4); "
                                  "size_t original_grid_y_offset = reserve_slot(sizeof(uint32_t), 4);\n" if
                                  has_dynamic_program_grid_abi else "    \n" if preserve_legacy_launcher_layout else "")
    original_grid_copy = ("    memcpy(launch_args.data() + original_grid_x_offset, &originalGridX, sizeof(uint32_t)); "
                          "memcpy(launch_args.data() + original_grid_y_offset, &originalGridY, sizeof(uint32_t));\n"
                          if has_dynamic_program_grid_abi else "    \n" if preserve_legacy_launcher_layout else "")
    original_grid_struct_fields = ("      uint32_t originalGridX __attribute__((aligned(4))); "
                                   "uint32_t originalGridY __attribute__((aligned(4)));\n"
                                   if has_dynamic_program_grid_abi else
                                   "      \n" if preserve_legacy_launcher_layout else "")
    original_grid_struct_values = ("      static_cast<uint32_t>(originalGridX), static_cast<uint32_t>(originalGridY),\n"
                                   if has_dynamic_program_grid_abi else
                                   "      \n" if preserve_legacy_launcher_layout else "")

    _launch_lambda_pre = f"""  {'std::function<cann_error()> launch_call = [=]() -> cann_error' if enable_taskqueue else ''} {{
    {get_backend_func("pre_launch", False)}
    uint32_t blockNum = gridX * gridY * gridZ;

    #ifdef ENABLE_GRID_WARN_PRINT
      static bool warned = false;
      if (!warned && blockNum > (uint32_t){num_physical_blocks}) {{
        printf("WARNING: Grid %u > physical limit {num_physical_blocks}, performance maybe reduced.\\n",blockNum);
        warned = true;
    }}
    #endif
    {'blockNum = std::min(blockNum, (uint32_t)' + str(num_physical_blocks) + ');' if launcher_cap_enabled else ''}
    // set mixBlockNumRation for nodeBasicBlockDim for msprof report
    uint32_t mixBlockNumRation = {mix_block_dim_ratio};
    uint32_t nodeBasicBlockDim = (mixBlockNumRation << 16) + blockNum;

    {'cce::internal::DebugTunnelData *DTData = cce::internal::DebugTunnel::Open(blockNum);' if enable_device_print else ''}
    cann_error ret = CANN_SUCCESS;
    {'void *ffts_addr = nullptr; ret = get_ffts_addr(stream, &ffts_addr);' if target_support_ffts else ''}
    {'if (ret != CANN_SUCCESS) return ret;' if (target_support_ffts and enable_taskqueue) else 'if (ret != CANN_SUCCESS) return;' if (target_support_ffts and (not enable_taskqueue)) else ''}
    // stub argument for workspace
    void *syncBlockLock_ptr = nullptr;
    void *syncBlockLock_handle = nullptr;
    uint16_t ModuleId = 0;
    {f'''
    {sync_block_lock_layout_stmt}
    {get_backend_func("allocate_sync_block_lock", "syncBlockLockSize", "stream")}
    std::shared_ptr<void> syncBlockLock_handle_guard(syncBlockLock_handle, release_npu_tensor_handle);
    if (!syncBlockLock_ptr) {{
      {alloc_success_code if enable_taskqueue else sync_lock_fail_code}
    }}
    {lock_init_stmt}
    if (ret != CANN_SUCCESS) {{
      return {'ret' if enable_taskqueue else ''};
    }}
    ''' if has_sync_block_lock else ''}
    {'if (ret != CANN_SUCCESS) {{ return ret; }}' if (workspace_size > 0 and enable_taskqueue) else 'if (ret != CANN_SUCCESS) {{ return; }}' if (workspace_size > 0 and not enable_taskqueue) else ''}"""

    _launch_lambda_post = f"""
    {cpp_msprof_call_before_launch}
    __KERNEL_LAUNCH_CALL__
    {'void*& stream_ref = const_cast<void*&>(stream);' if enable_device_print else ''}
    {'cce::internal::DebugTunnel::Close(DTData, stream_ref);' if enable_device_print else ''}
    {cpp_msprof_call_after_launch}
    {'return ret;' if enable_taskqueue else 'ret = cann_synchronize_stream(stream);'}
  }};
  {f'''{get_backend_func("async_launch", "launch_call") if enable_taskqueue else ''}'''}
  return;
}}"""

    return f"""
{npu_headers}
{'#define __CCE_ENABLE_PRINT__' if enable_device_print else ''}
{extract_device_print_code_from_cann() if enable_device_print else ''}
#define PY_SSIZE_T_CLEAN
{'#define ENABLE_GRID_WARN_PRINT' if enable_grid_warn_print else ''}
#define TENSOR_KIND_INPUT 0
#define TENSOR_KIND_OUTPUT 1
#define TENSOR_KIND_INPUT_OUTPUT 2

{cpp_msprof_extern}

{cpp_npu_utils_dlopen}

{cpp_device_pointer}

{_CPP_ALIGN_LAUNCH_OFFSET}

extern "C" {{
void triton_launch_kernel(const char* kernelName, cann_func_handle func, cann_stream stream,
    int gridX, int gridY, int gridZ,
    const int64_t* shapes_data, const int* shape_dims, int num_tensors,
    const int* tensor_kinds,
    const void* const* kernel_args, const size_t* arg_sizes, int num_args) {{
  if (gridX <= 0 || gridY <= 0 || gridZ <= 0) {{
    printf("WARNING: Skipping launch for kernel '%s' due to empty grid (gridX=%d, gridY=%d, gridZ=%d).\\n",
           kernelName, gridX, gridY, gridZ);
    return;
  }}
  std::vector<std::vector<int64_t>> tensorShapes;
  if (shapes_data != nullptr && shape_dims != nullptr) {{
    int shapes_idx = 0;
    for (int tensor_idx = 0; tensor_idx < num_tensors; ++tensor_idx) {{
      std::vector<int64_t> tensorShape;
      for (int dim_idx = 0; dim_idx < shape_dims[tensor_idx]; ++dim_idx) {{
        tensorShape.push_back(shapes_data[shapes_idx++]);
      }}
      tensorShapes.push_back(tensorShape);
    }}
  }}
  std::vector<int> tensorKinds;
  if (tensor_kinds != nullptr && num_tensors > 0) {{
    tensorKinds.assign(tensor_kinds, tensor_kinds + num_tensors);
  }}
  if (num_args > 0 && (kernel_args == nullptr || arg_sizes == nullptr)) {{
    return;
  }}
  std::vector<size_t> launch_arg_sizes;
  launch_arg_sizes.reserve(num_args);
  std::vector<std::vector<char>> copied_kernel_args;
  copied_kernel_args.reserve(num_args);
  for (int arg_idx = 0; arg_idx < num_args; ++arg_idx) {{
    launch_arg_sizes.push_back(arg_sizes[arg_idx]);
    copied_kernel_args.emplace_back(arg_sizes[arg_idx]);
    memcpy(copied_kernel_args.back().data(), kernel_args[arg_idx], arg_sizes[arg_idx]);
  }}

  // Only 1D parallelization is supported for NPU.
  // Pointer type becomes flattened 1-D Memref tuple: base_ptr, data_ptr,
  // offset, shape, stride. base_ptr offset shape and stride are not used,
  // arbitrarily set for now.
{_launch_preamble}
{_launch_lambda_pre}

    size_t args_offset = 0;
    auto reserve_slot = [&](size_t size, size_t alignment) -> size_t {{
      args_offset = _align_launch_offset(args_offset, alignment);
      size_t current_offset = args_offset;
      args_offset += size;
      return current_offset;
    }};
    {'size_t ffts_offset = reserve_slot(sizeof(void*), 8);' if target_support_ffts else ''}
    {'size_t sync_block_lock_offset = reserve_slot(sizeof(void*), 8);' if not metadata.is_pure_simt else ''}
    {'size_t workspace_offset = reserve_slot(sizeof(void*), 8);' if not metadata.is_pure_simt else ''}
    size_t kernel_args_offset = args_offset;
    for (int arg_idx = 0; arg_idx < num_args; ++arg_idx) {{
      size_t alignment = launch_arg_sizes[arg_idx] >= 8 ? 8 : (launch_arg_sizes[arg_idx] >= 4 ? 4 : 1);
      args_offset = _align_launch_offset(args_offset, alignment);
      args_offset += launch_arg_sizes[arg_idx];
    }}
{original_grid_offset_decls}    size_t grid_offset = reserve_slot(sizeof(int32_t), 4);
    reserve_slot(sizeof(int32_t), 4);
    reserve_slot(sizeof(int32_t), 4);
    {'reserve_slot(sizeof(void*), 8);' if metadata.is_pure_simt else ''}
    {'reserve_slot(sizeof(void*), 8);' if metadata.is_pure_simt else ''}
    {'size_t dtdata_offset = reserve_slot(sizeof(void*), 8);' if enable_device_print else 'reserve_slot(sizeof(void*), 8);'}
    size_t total_size = args_offset;

    std::vector<char> launch_args(total_size, 0);
    {'memcpy(launch_args.data() + ffts_offset, &ffts_addr, sizeof(void*));' if target_support_ffts else ''}
    {f'memcpy(launch_args.data() + sync_block_lock_offset, &syncBlockLock_ptr, sizeof(void*));' if not metadata.is_pure_simt else ''}
    {f'memcpy(launch_args.data() + workspace_offset, &workspace_addr_ptr, sizeof(void*));' if not metadata.is_pure_simt else ''}
    size_t kernel_arg_offset = kernel_args_offset;
    for (int arg_idx = 0; arg_idx < num_args; ++arg_idx) {{
      size_t alignment = launch_arg_sizes[arg_idx] >= 8 ? 8 : (launch_arg_sizes[arg_idx] >= 4 ? 4 : 1);
      kernel_arg_offset = _align_launch_offset(kernel_arg_offset, alignment);
      memcpy(launch_args.data() + kernel_arg_offset, copied_kernel_args[arg_idx].data(), launch_arg_sizes[arg_idx]);
      kernel_arg_offset += launch_arg_sizes[arg_idx];
    }}
{original_grid_copy}    memcpy(launch_args.data() + grid_offset, &gridX, sizeof(int32_t));
    memcpy(launch_args.data() + grid_offset + sizeof(int32_t), &gridY, sizeof(int32_t));
    memcpy(launch_args.data() + grid_offset + 2 * sizeof(int32_t), &gridZ, sizeof(int32_t));
    {'memcpy(launch_args.data() + dtdata_offset, &DTData, sizeof(void*));' if enable_device_print else ''}

{_launch_lambda_post.replace('__KERNEL_LAUNCH_CALL__', cpp_kernel_launch)}
}} // extern "C"

static void _launch(const char* kernelName, cann_func_handle func, cann_stream stream,
    int gridX, int gridY, int gridZ,
    std::vector<std::vector<int64_t>> &tensorShapes, std::vector<int> &tensorKinds{(', ' + arg_decls) if len(arg_decls) > 0 else ''}) {{
  // Keep Python launcher on the stable local packing path.
  if (gridX <=0 || gridY <=0 || gridZ <=0) {{
    printf("WARNING: Skipping launch for kernel '%s' due to empty grid (gridX=%d, gridY=%d, gridZ=%d).\\n", kernelName, gridX, gridY, gridZ);
    return;
  }}
{_python_launch_preamble}
{_launch_lambda_pre}
    struct __attribute__((packed)) {{
      {'void* ffts_addr __attribute__((aligned(8)));' if target_support_ffts else ''}
      {'void* syncBlockLock __attribute__((aligned(8)));' if not metadata.is_pure_simt else ''}
      {'void* workspace_addr __attribute__((aligned(8)));' if not metadata.is_pure_simt else ''}
      {' '.join(f'{ty_to_cpp(ty)} arg{i} __attribute__((aligned({4 if ty[0] != "*" and ty[-2:] != "64" else 8})));' for i, ty in signature.items() if ty != "constexpr")}
{original_grid_struct_fields}      {' '.join(f'{ty_to_cpp(ty)} grid{mark} __attribute__((aligned(4)));' for mark, ty in grid_info.items())}
      {'void* global_scratch __attribute__((aligned(8)));' if metadata.is_pure_simt else ''}
      {'void* profile_scratch __attribute__((aligned(8)));' if metadata.is_pure_simt else ''}
      {'void* DTData __attribute__((aligned(8)));'}
    }} args = {{
      {'static_cast<void*>(ffts_addr),' if target_support_ffts else ''}
      {('static_cast<void*>(syncBlockLock_ptr),' if has_sync_block_lock else 'nullptr,') if not metadata.is_pure_simt else ''}
      {('static_cast<void*>(workspace_addr_ptr),' if workspace_size > 0 else 'nullptr,') if not metadata.is_pure_simt else ''}
      {(lambda _rt: (', '.join(_rt) + ',') if _rt else '')(
        [f'static_cast<{ty_to_cpp(ty)}>(arg{i})' for i, ty in signature.items() if ty != "constexpr"]
      )}
{original_grid_struct_values}      {', '.join(f'static_cast<{ty_to_cpp(ty)}>(grid{mark})' for mark, ty in grid_info.items())}
      {', static_cast<void*>(nullptr)' if metadata.is_pure_simt else ''}
      {', static_cast<void*>(nullptr)' if metadata.is_pure_simt else ''}
      {', static_cast<void*>(DTData)' if enable_device_print else ', static_cast<void*>(nullptr)'}
    }};
{_launch_lambda_post.replace('__KERNEL_LAUNCH_CALL__', cpp_kernel_launch_local)}

{_CPP_GET_TENSOR_SHAPE}

static PyObject* launch(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {{
  int gridX, gridY, gridZ;
  cann_stream stream;
  cann_func_handle function;
  PyObject *packedMetadata = nullptr;
  PyObject *launch_metadata = nullptr;
  PyObject *launch_enter_hook = nullptr;
  PyObject *launch_exit_hook = nullptr;
  std::vector<std::vector<int64_t>> tensorShapes;

  {newline.join([f"{_extracted_type(ty)} _arg{i};" for i, ty in signature.items()])}
  // METH_FASTCALL fast path: avoid per-call tuple allocation (METH_VARARGS) and
  // skip PyArg_ParseTuple's format-string interpreter by parsing manually.
  // Borrowed-reference semantics match PyArg_ParseTuple("O").
  if (nargs != {total_nargs}) {{
    PyErr_Format(PyExc_TypeError, "launch expects %d arguments, got %zd", {total_nargs}, nargs);
    return nullptr;
  }}
  gridX = (int)PyLong_AsLong(args[0]);
  gridY = (int)PyLong_AsLong(args[1]);
  gridZ = (int)PyLong_AsLong(args[2]);
  stream = reinterpret_cast<cann_stream>(PyLong_AsUnsignedLongLong(args[3]));
  function = reinterpret_cast<cann_func_handle>(PyLong_AsUnsignedLongLong(args[4]));
  packedMetadata = args[5];
  launch_metadata = args[6];
  launch_enter_hook = args[7];
  launch_exit_hook = args[8];
  {fastcall_sig_parse_stmts}
  if (PyErr_Occurred()) {{
    return nullptr;
  }}
  if (__MsprofFlagL1) {{
    {
      LINE_CHANGE_CHAR.join(
        f"{{ auto tmp = _get_tensor_shape(_arg{i}); if (!tmp.empty()) tensorShapes.push_back(tmp); }}"
        for i, ty in signature.items() if ty[0] == "*"
      )
    }
  }}

  if (launch_enter_hook != Py_None){{
    PyObject* hook_args = Py_BuildValue("(O)", launch_metadata);
    PyObject* hook_ret = PyObject_CallObject(launch_enter_hook, hook_args);
    Py_DECREF(hook_args);
    if (!hook_ret)
      return nullptr;
  }}

  // get kernel_name (use interned key to avoid temporary PyUnicode per call)
  static PyObject* key_kernel_name = PyUnicode_InternFromString("kernel_name");
  PyObject* kernelNameObj = PyDict_GetItemWithError(packedMetadata, key_kernel_name);
  if (!kernelNameObj) {{
    PyErr_SetString(PyExc_KeyError, "packedMetadata missing 'kernel_name'");
    return nullptr;
  }}
  const char* kernelName = PyUnicode_AsUTF8(kernelNameObj);
  // get tensor_kinds (use interned key, cache result in tensorKinds)
  if (tensorKinds.empty()) {{
    static PyObject* key_tensor_kinds = PyUnicode_InternFromString("tensor_kinds");
    PyObject* tensorKindList = PyDict_GetItemWithError(packedMetadata, key_tensor_kinds);
    if (tensorKindList) {{
      Py_ssize_t size = PySequence_Size(tensorKindList);
      for (Py_ssize_t i = 0; i < size; ++i) {{
        PyObject* kind = PySequence_GetItem(tensorKindList, i);
        tensorKinds.push_back(PyLong_AsLong(kind));
        Py_DECREF(kind);
      }}
    }}
  }}

  // raise exception asap
  {newline.join(ptr_decls)}
  _launch(kernelName, function, stream,
          gridX, gridY, gridZ,
          tensorShapes, tensorKinds
          {', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});
  if (PyErr_Occurred()) {{
    return nullptr;
  }}
  if(launch_exit_hook != Py_None){{
    PyObject* hook_args = Py_BuildValue("(O)", launch_metadata);
    PyObject* hook_ret = PyObject_CallObject(launch_exit_hook, hook_args);
    Py_DECREF(hook_args);
    if (!hook_ret)
      return nullptr;
  }}
  Py_RETURN_NONE;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", (PyCFunction)launch, METH_FASTCALL, "Entry point for all kernels with this signature"}},
  {{nullptr, nullptr, 0, nullptr}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_launcher\",
  nullptr, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == nullptr) {{
    return nullptr;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  {cpp_msprof_callback}
  // One-time initialization of NPU utils (dlsym lookup for g_async_launch etc.)
  // Moved here from the per-call async_launch path to avoid repeated dlsym work.
  init_npu_utils();
  return m;
}}
"""
