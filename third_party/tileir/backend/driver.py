import functools
import sys
import os
import subprocess
from dataclasses import dataclass
from typing import Any
import shutil
from pathlib import Path
import tempfile
import threading
# flagtree: do NOT import torch at module level. Backend discovery imports this module
# even when tileir is inactive (e.g. AMD-only systems, CUDA-less environments). Top-level
# `import torch` would cause `import triton` to fail with ModuleNotFoundError on those
# systems. All functions that need torch already do their own lazy `import torch`.
from triton.backends.nvidia.driver import (
    library_dirs,
    include_dirs,
    libraries,
    ty_to_cpp,
)

from triton import knobs
from triton.runtime.build import compile_module_from_src
from triton.runtime.cache import get_cache_manager
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver
from triton.backends.tileir.conf import TileIREnvConf
from triton.tools.tensor_descriptor import TensorDescriptor

# ------------------------
# Utils
# ------------------------


class TileIRUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(TileIRUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        tile_mod_path = dirname
        nvidia_mod_path = os.path.join(os.path.dirname(dirname), "nvidia")
        tile_mod = compile_module_from_src(
            Path(os.path.join(tile_mod_path, "driver.c")).read_text(),
            "tileir_utils",
            library_dirs(),
            include_dirs,
            libraries,
        )
        nvidia_mod = compile_module_from_src(
            Path(os.path.join(nvidia_mod_path, "driver.c")).read_text(),
            "cuda_utils",
            library_dirs(),
            include_dirs,
            libraries,
        )
        self.init_nvidia_function(nvidia_mod)
        self.init_tileir_function(tile_mod)

    def init_tileir_function(self, mod):
        self._load_binary = mod.load_tileir_binary

    def load_binary(self, name, kernel, shared, device):
        module, function, n_regs, n_spills, _static_smem, n_max_threads = self._load_binary(name, kernel, device)
        return module, function, n_regs, n_spills, n_max_threads

    def init_nvidia_function(self, mod):
        self.get_device_properties = mod.get_device_properties
        self.cuOccupancyMaxActiveClusters = mod.cuOccupancyMaxActiveClusters
        self.set_printf_fifo_size = mod.set_printf_fifo_size


# ------------------------
# Launcher
# ------------------------

dirname = os.path.dirname(__file__)

FLOAT_STORAGE_TYPE = {
    "fp16": "uint16_t",
    "bf16": "uint16_t",
    "fp32": "uint32_t",
    "f32": "uint32_t",
    "fp64": "uint64_t",
}
FLOAT_PACK_FUNCTION = {
    "fp16": "pack_fp16",
    "bf16": "pack_bf16",
    "fp32": "pack_fp32",
    "f32": "pack_fp32",
    "fp64": "pack_fp64",
}

_BASE_ARGS_FORMAT = "iiiKKpOOOO"
_BASE_ARGS_FORMAT_LEN = len(_BASE_ARGS_FORMAT)


def make_launcher(constants, signature):

    def _flatten_signature(sig, output):
        # Flatten tuples
        if isinstance(sig, tuple):
            for x in sig:
                _flatten_signature(x, output)
        else:
            output.append(sig)

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ",".join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == "*":
            return "PyObject*"
        if ty in ("constexpr", "nvTmaDesc"):
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = "".join(map(format_of, ty))
            return f"({val})"
        if ty[0] == "*":
            return "O"
        if ty in ("constexpr", "nvTmaDesc"):
            return "O"
        return {
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

    args_format = "".join([format_of(ty) for ty in signature.values()])
    format = _BASE_ARGS_FORMAT + args_format

    flat_signature = []
    for sig in signature.values():
        _flatten_signature(sig, flat_signature)
    signature = {i: s for i, s in enumerate(flat_signature)}
    args_list = (", " + ", ".join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else "")
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors, such as tensor descriptors for CUDA.
    arg_decl_list = []
    for i, ty in signature.items():
        if ty == "constexpr":
            continue
        if ty in FLOAT_STORAGE_TYPE:
            arg_decl_list.append(f"{FLOAT_STORAGE_TYPE[ty]} arg{i}")
        else:
            arg_decl_list.append(f"{ty_to_cpp(ty)} arg{i}")
    arg_decls = ", ".join(arg_decl_list)
    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"ptr_info{i}.dev_ptr")
        elif ty in FLOAT_STORAGE_TYPE:
            internal_args_list.append(f"_arg{i}_storage")
        elif ty == "nvTmaDesc":
            # Note: we have to dereference the pointer
            internal_args_list.append(f"*tma_ptr{i}")
        elif ty != "constexpr":
            internal_args_list.append(f"_arg{i}")

    import torch

    device_id = torch.cuda.current_device()
    # generate glue code
    newline = "\n  "
    float_storage_decls = [
        f"{FLOAT_STORAGE_TYPE[ty]} _arg{i}_storage = {FLOAT_PACK_FUNCTION[ty]}(_arg{i});"
        for i, ty in signature.items()
        if ty in FLOAT_STORAGE_TYPE
    ]
    params = [f"&arg{i}" for i, ty in signature.items() if ty != "constexpr"]
    src = f"""
#include \"cuda.h\"
#include <stdbool.h>
#include <Python.h>
#include <dlfcn.h>

static inline void gpuAssert(CUresult code, const char *file, int line)
{{
   if (code != CUDA_SUCCESS)
   {{
      const char* prefix = "Triton Error [TileIR]: ";
      const char* str;
      cuGetErrorString(code, &str);
      char err[1024] = {{0}};
      strcat(err, prefix);
      strcat(err, str);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }}
}}

#define CUDA_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

typedef CUresult (*cuLaunchKernelEx_t)(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra);

static cuLaunchKernelEx_t getLaunchKernelExHandle() {{
  // Open the shared library
  void* handle = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!handle) {{
    PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so.1");
    return NULL;
  }}
  // Clear any existing error
  dlerror();
  cuLaunchKernelEx_t cuLaunchKernelExHandle = (cuLaunchKernelEx_t)dlsym(handle, "cuLaunchKernelEx");
  // Check for errors
  const char *dlsym_error = dlerror();
  if (dlsym_error) {{
    PyErr_SetString(PyExc_RuntimeError, "Failed to retrieve cuLaunchKernelEx from libcuda.so.1");
    return NULL;
  }}
  return cuLaunchKernelExHandle;
}}

static void _launch(int numTilesX, int numTilesY, int numTilesZ, int launch_pdl, CUstream stream, CUfunction function{", " + arg_decls if len(arg_decls) > 0 else ""}) {{
  void *params[] = {{ {", ".join(f"{i}" for i in params)} }};
  if (numTilesX*numTilesY*numTilesZ > 0) {{
    int numAttrs = 1;
    CUlaunchAttribute launchAttr[2];
    launchAttr[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
    launchAttr[0].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    if (launch_pdl != 0) {{
        CUlaunchAttribute pdlAttr = {{ .id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, .value = 1}};
        launchAttr[numAttrs++] = pdlAttr;
    }}
    CUlaunchConfig config;
    config.gridDimX = numTilesX;
    config.gridDimY = numTilesY;
    config.gridDimZ = numTilesZ;
    config.blockDimX = 1;
    config.blockDimY = 1;
    config.blockDimZ = 1;
    config.sharedMemBytes = 0;
    config.hStream = stream;
    config.attrs = launchAttr;
    config.numAttrs = numAttrs;
    static cuLaunchKernelEx_t cuLaunchKernelExHandle = NULL;
    if (cuLaunchKernelExHandle == NULL) {{
    cuLaunchKernelExHandle = getLaunchKernelExHandle();
    }}
    CUDA_CHECK(cuLaunchKernelExHandle(&config, function, params, 0));
  }}
}}

typedef struct _DevicePtrInfo {{
    CUdeviceptr dev_ptr;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    uint64_t dev_ptr;
    int status = cuPointerGetAttribute(&dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == CUDA_ERROR_INVALID_VALUE) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }}
    ptr_info.dev_ptr = dev_ptr;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}

static uint16_t pack_fp16(double f) {{
    uint16_t result;
    // from https://github.com/python/pythoncapi-compat
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 && !defined(PYPY_VERSION)
    _PyFloat_Pack2(f, (void*)&result, 1);
#else
    PyFloat_Pack2(f, (void*)&result, 1);
#endif
    return result;
}}

static uint16_t pack_bf16(double f) {{
    float f32 = (float)f;
    uint32_t u32 = *(uint32_t*)&f32;
    return (uint16_t)(u32 >> 16);
}}

static uint32_t pack_fp32(double f) {{
    float f32 = (float)f;
    return *(uint32_t*)&f32;
}}

static uint64_t pack_fp64(double f) {{
    return *(uint64_t*)&f;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int numTilesX, numTilesY, numTilesZ;
  uint64_t _stream;
  uint64_t _function;
  int launch_pdl;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {" ".join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &numTilesX, &numTilesY, &numTilesZ,
                                           &_stream, &_function, &launch_pdl,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // todo: triton doesn't need this fix, do we make something wrong?
  CUcontext ctx = NULL;
  cuCtxGetCurrent(&ctx);
  if (!ctx) {{
    CUdevice device;
    cuDeviceGet(&device, /*ordinal=*/{device_id});
    cuDevicePrimaryCtxRetain(&ctx, /*device=*/device);
    cuCtxSetCurrent(ctx);
  }}
  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  {newline.join(float_storage_decls)}
  Py_BEGIN_ALLOW_THREADS;

  _launch(numTilesX, numTilesY, numTilesZ, launch_pdl, (CUstream)_stream, (CUfunction)_function{", " + ", ".join(internal_args_list) if len(internal_args_list) > 0 else ""});
  Py_END_ALLOW_THREADS;
  if (PyErr_Occurred()) {{
    return NULL;
  }}

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;

  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  "__triton_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


# This function unpacks a tensordesc object into its components:
# - data pointer
# - shape dimensions
# - stride values
def make_tensordesc_arg(arg):
    assert isinstance(arg, TensorDescriptor)
    data_ptr = arg.base.data_ptr()
    shape = arg.shape
    strides = arg.strides
    # Currently only contiguous tensors are supported
    assert strides[-1] == 1
    # The 0 is a placeholder that replaces the tensordesc type when passing to kernel.
    # nvidia oss backend passes tensordesc directly, but tileir needs to decompose it.
    result = [0, data_ptr, *shape, *strides[:-1]]
    return result


def wrap_handle_tensordesc(launcher):

    def inner(*args):
        # flagtree: upstream used `args[:9]`, but TileIRLauncher.__call__ unconditionally
        # inserts `launch_pdl` at position 5 before invoking self.launch — making metadata
        # 10 args, not 9. Slicing at 9 mis-routed launch_exit_hook as the first kernel
        # argument and shifted every TensorDescriptor argument by one slot.
        meta_args = args[:10]
        raw_kernel_args = args[10:]
        final_args = []
        for i, arg in enumerate(raw_kernel_args):
            if isinstance(arg, TensorDescriptor):
                final_args.extend(make_tensordesc_arg(arg))
            else:
                final_args.append(arg)
        return launcher(*meta_args, *final_args)

    return inner


class TileIRLauncher(object):

    def __init__(self, src, metadata):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}

        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        has_tensordesc = any("tensordesc" in value for value in signature.values())
        self.ori_signature_len = len(signature)
        if has_tensordesc:
            # convert one tensordesc type to [placeholder, ptr, shape and stride] type
            post_signature = {}
            for key, value in src.signature.items():
                key = arg_idx(key)
                if "tensordesc" in value:
                    shape_str = value.split("[")[1].split("]")[0]
                    shape = [int(s) for s in shape_str.split(",")]
                    dtype = value.split("<")[1].split("[")[0]
                    post_signature[key] = "i32"
                    post_signature[f"{key}_ptr"] = f"*{dtype}"
                    # add shape and dynamic stride args to signature; the last
                    # stride is statically known to be 1.
                    for idx in range(len(shape)):
                        post_signature[f"{key}_shape_{idx}"] = "i32"
                    for idx in range(len(shape) - 1):
                        post_signature[f"{key}_stride_{idx}"] = "i64"
                else:
                    post_signature[key] = value
            self.signature = post_signature
        else:
            self.signature = signature
        if os.environ.get("FLAGTREE_TILEIR_DUMP_SIGNATURE", "0") == "1":
            with open("/tmp/flagtree_tileir_signature.txt", "a") as f:
                f.write(f"src_signature={signature}\n")
                f.write(f"launcher_signature={self.signature}\n")
        self.constants = constants
        src = make_launcher(self.constants, self.signature)
        mod = compile_module_from_src(src, "__triton_launcher", library_dirs(), include_dirs, libraries)
        if has_tensordesc:
            self.launch = wrap_handle_tensordesc(mod.launch)
        else:
            self.launch = mod.launch
        self.launch_pdl = metadata.launch_pdl

    def __call__(self, *args, **kwargs):
        # TODO: below if branch is for torch 2.8.0a0+5228986c39.nvinternal commit
        # where constexpr arguments are not passed to the launch function by inductor
        # remove this after torch
        # 9 is the number of metadata arguments in `src` defined in `make_launcher`
        num_launch_args = 9
        num_params = len(args) - num_launch_args
        if num_params < self.ori_signature_len:
            extra_args = [self.constants[(i, )] for i in range(num_params, self.ori_signature_len)]
            model_args = args + tuple(extra_args)
        else:
            model_args = args
        model_args = model_args[:5] + (self.launch_pdl, ) + model_args[5:]

        self.launch(*model_args, **kwargs)


class TileIRDriver(GPUDriver):

    def __init__(self):
        self.utils = TileIRUtils()  # TODO: make static
        self.launcher_cls = TileIRLauncher
        super().__init__()

    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        warp_size = 32
        return GPUTarget("tileir", capability, warp_size)

    def get_active_torch_device(self):
        import torch

        return torch.device("cuda", self.get_current_device())

    def get_device_interface(self):
        import torch

        return torch.cuda

    @staticmethod
    def is_active():
        # TileIR is selected per kernel through the backend routing hook. The
        # global active driver remains CudaDriver for CUDA device/stream queries;
        # Triton's target-driver lookup instantiates TileIRDriver only for a
        # TileIR-compiled kernel.
        #
        # The upstream triton-to-tile-ir fork returned True when ENABLE_TILE=1 to
        # make the driver replace CudaDriver wholesale. In flagtree that would
        # produce a "2 active drivers" runtime error (since CudaDriver also reports
        # active on CUDA hosts), and is unnecessary — our per-kernel routing already
        # handles backend selection without changing the global active driver.
        return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device="cuda")

    def clear_cache(self, cache):
        cache.zero_()


# The upstream fork eagerly constructs `GlobalTileIRDriver = TileIRDriver()` here.
# FlagTree resolves target drivers lazily, avoiding CUDA probing and extension builds
# during a plain `import triton` on AMD-only or CUDA-less systems.
