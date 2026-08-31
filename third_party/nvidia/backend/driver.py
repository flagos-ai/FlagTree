import functools
import os
import subprocess
import triton
import re
from pathlib import Path
from triton import knobs
from triton.runtime.build import compile_module_from_src
from triton.runtime import _allocation
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver

dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [os.path.join(dirname, "include")]
libdevice_dir = os.path.join(dirname, "lib")
libraries = ['libcuda.so.1']
PyCUtensorMap = None


@functools.lru_cache()
def libcuda_dirs():
    if env_libcuda_path := knobs.nvidia.libcuda_path:
        return [env_libcuda_path]

    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
    # each line looks like the following:
    # libcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1
    locs = [line.split()[-1] for line in libs.splitlines() if "libcuda.so.1" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path and not dirs:
        dirs = [dir for dir in env_ld_library_path.split(":") if os.path.exists(os.path.join(dir, "libcuda.so.1"))]
    msg = 'libcuda.so cannot found!\n'
    if locs:
        msg += 'Possible files are located at %s.' % str(locs)
        msg += 'Please create a symlink of libcuda.so to any of the files.'
    else:
        msg += 'Please make sure GPU is set up and then run "/sbin/ldconfig"'
        msg += ' (requires sudo) to refresh the linker cache.'
    assert any(os.path.exists(os.path.join(path, 'libcuda.so.1')) for path in dirs), msg
    return dirs


@functools.lru_cache()
def library_dirs():
    return [libdevice_dir, *libcuda_dirs()]


# ------------------------
# Utils
# ------------------------


class CudaUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CudaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        mod = compile_module_from_src(
            src=Path(os.path.join(dirname, "driver.c")).read_text(),
            name="cuda_utils",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )
        global PyCUtensorMap
        PyCUtensorMap = mod.PyCUtensorMap
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.cuOccupancyMaxActiveClusters = mod.cuOccupancyMaxActiveClusters
        self.set_printf_fifo_size = mod.set_printf_fifo_size
        self.fill_tma_descriptor = mod.fill_tma_descriptor
        self.encode_tma_descriptor = mod.encode_tma_descriptor


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "CUdeviceptr"
    if ty.startswith("tensordesc"):
        return "CUtensorMap"
    return {
        "i1": "int8_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint8_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "double",
        "bf16": "double",
        "fp32": "double",
        "f32": "double",
        "fp64": "double",
        "nvTmaDesc": "CUtensorMap",
    }[ty]


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

_BASE_ARGS_FORMAT = "iiiKKppOOOOOO"
_BASE_ARGS_FORMAT_LEN = len(_BASE_ARGS_FORMAT)


def _expand_launcher_signature(signature, tensordesc_meta):
    output = []
    tensordesc_idx = 0
    for sig in signature:
        if isinstance(sig, str) and sig.startswith("tensordesc"):
            meta = tensordesc_meta[tensordesc_idx] if tensordesc_meta else None
            tensordesc_idx += 1

            match = re.match("tensordesc<([^[>]*)\\[([^]]*)\\]", sig)
            if match is None:
                raise ValueError(f"Invalid tensor descriptor signature: {sig}")
            dtype = match.group(1)
            shape = match.group(2)
            ndim = shape.count(",") + 1

            if meta is None:
                output.append("*" + dtype)
                output.extend(["i64"] * (2 * ndim))
                output.append("i1")
            else:
                output.append("nvTmaDesc")

            output.extend(["i32"] * ndim)
            output.extend(["i64"] * ndim)
        else:
            output.append(sig)

    if tensordesc_meta and tensordesc_idx != len(tensordesc_meta):
        raise ValueError("Launcher signature did not consume all tensor descriptors")
    return output


def _flatten_launcher_signature(signature):
    output = []

    def flatten(sig):
        if isinstance(sig, tuple):
            for element in sig:
                flatten(element)
        else:
            output.append(sig)

    for sig in signature:
        flatten(sig)
    return output


def make_launcher(
    constants,
    signature,
    tensordesc_meta,
    global_scratch_size,
    global_scratch_reset_per_launch,
):

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty in ("constexpr", "nvTmaDesc"):
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty in ("constexpr", "nvTmaDesc"):
            return "O"
        if ty.startswith("tensordesc"):
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

    expand_signature = _expand_launcher_signature(signature.values(), tensordesc_meta)
    signature = {i: s for i, s in enumerate(expand_signature)}

    args_format = ''.join([format_of(ty) for ty in signature.values()])
    format = _BASE_ARGS_FORMAT + args_format

    flat_signature = _flatten_launcher_signature(signature.values())
    signature = {i: s for i, s in enumerate(flat_signature)}
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''
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
    arg_decls = ', '.join(arg_decl_list)
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
    params = range(len(signature))

    # generate glue code
    newline = '\n  '
    ptr_decls = [
        f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;"
        for i, ty in signature.items()
        if ty[0] == "*"
    ]
    tma_decls = [
        f"CUtensorMap* tma_ptr{i} = getTmaDesc(_arg{i}); if (!tma_ptr{i}) return NULL;" for i, ty in signature.items()
        if ty == "nvTmaDesc"
    ]
    float_storage_decls = [
        f"{FLOAT_STORAGE_TYPE[ty]} _arg{i}_storage = {FLOAT_PACK_FUNCTION[ty]}(_arg{i});"
        for i, ty in signature.items()
        if ty in FLOAT_STORAGE_TYPE
    ]
    params = [f"&arg{i}" for i, ty in signature.items() if ty != "constexpr"]
    params.append("&global_scratch")
    params.append("&profile_scratch")
    prepared_arg_fields = []
    prepared_set_cases = []
    prepared_call_args = []
    prepared_runtime_flags = []
    signed_integer_limits = {
        "i1": ("INT8_MIN", "INT8_MAX"),
        "i8": ("INT8_MIN", "INT8_MAX"),
        "i16": ("INT16_MIN", "INT16_MAX"),
        "i32": ("INT32_MIN", "INT32_MAX"),
        "i64": ("INT64_MIN", "INT64_MAX"),
    }
    unsigned_integer_limits = {
        "u1": "UINT8_MAX",
        "u8": "UINT8_MAX",
        "u16": "UINT16_MAX",
        "u32": "UINT32_MAX",
        "u64": "UINT64_MAX",
    }
    for i, ty in signature.items():
        is_runtime = ty != "constexpr"
        prepared_runtime_flags.append("1" if is_runtime else "0")
        if not is_runtime:
            continue

        storage_type = FLOAT_STORAGE_TYPE.get(ty, ty_to_cpp(ty))
        prepared_arg_fields.append(f"{storage_type} arg{i};")
        prepared_call_args.append(f"launch_arguments.arg{i}")
        if isinstance(ty, str) and ty.startswith("*"):
            assignment = f"""
      DevicePtrInfo pointer = getPreparedPointer(obj, {i}, trusted_pointer_arguments);
      if (!pointer.valid) return 0;
      storage->arg{i} = pointer.dev_ptr;
"""
        elif ty == "nvTmaDesc":
            assignment = f"""
      CUtensorMap *tensor_map = getTmaDesc(obj);
      if (tensor_map == NULL) return 0;
      storage->arg{i} = *tensor_map;
"""
        elif ty in FLOAT_STORAGE_TYPE:
            assignment = f"""
      double value = PyFloat_AsDouble(obj);
      if (value == -1.0 && PyErr_Occurred()) return 0;
      storage->arg{i} = {FLOAT_PACK_FUNCTION[ty]}(value);
"""
        elif ty in signed_integer_limits:
            minimum, maximum = signed_integer_limits[ty]
            assignment = f"""
      long long value = PyLong_AsLongLong(obj);
      if (value == -1 && PyErr_Occurred()) return 0;
      if (value < {minimum} || value > {maximum}) {{
        PyErr_Format(PyExc_OverflowError, "kernel argument {i} is outside the {ty} range");
        return 0;
      }}
      storage->arg{i} = ({storage_type})value;
"""
        elif ty in unsigned_integer_limits:
            maximum = unsigned_integer_limits[ty]
            assignment = f"""
      unsigned long long value = PyLong_AsUnsignedLongLong(obj);
      if (value == (unsigned long long)-1 && PyErr_Occurred()) return 0;
      if (value > {maximum}) {{
        PyErr_Format(PyExc_OverflowError, "kernel argument {i} is outside the {ty} range");
        return 0;
      }}
      storage->arg{i} = ({storage_type})value;
"""
        else:
            raise ValueError(f"Unsupported prepared CUDA launcher type: {ty}")
        prepared_set_cases.append(f"""    case {i}: {{{assignment}      return 1;
    }}""")

    prepared_arg_count = len(signature)
    prepared_array_size = max(1, prepared_arg_count)
    prepared_arg_fields = "\n  ".join(prepared_arg_fields) or "char unused;"
    prepared_set_cases = "\n".join(prepared_set_cases)
    prepared_call_args = ", ".join(prepared_call_args)
    prepared_call_suffix = (f", {prepared_call_args}" if prepared_call_args else "")
    prepared_static_initializers = "\n".join(f"""  if (!state->is_dynamic[{i}] &&
      !setPreparedArgument(&state->arguments, {i},
                           PyTuple_GET_ITEM(flat_arguments, {i}),
                           trusted_pointer_arguments)) {{
    goto fail;
  }}""" for i, ty in signature.items() if ty != "constexpr")
    prepared_runtime_flags = ", ".join(prepared_runtime_flags) or "0"
    reset_global_scratch_each_launch = "1" if global_scratch_reset_per_launch else "0"
    tle_cpp_define = "#define __TLE__ 1"
    src = f"""
#include \"cuda.h\"
#include <dlfcn.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
{tle_cpp_define}

typedef struct {{
  PyObject_HEAD;
  _Alignas(128) CUtensorMap tensorMap;
}} PyCUtensorMapObject;

static inline void gpuAssert(CUresult code, const char *file, int line)
{{
   if (code != CUDA_SUCCESS)
   {{
      const char* prefix = "Triton Error [CUDA]: ";
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

#ifdef __TLE__
static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int launch_cooperative_grid, int launch_pdl, int reset_global_scratch, int shared_memory, CUstream stream, CUfunction function, CUdeviceptr global_scratch, CUdeviceptr profile_scratch{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  void *params[] = {{ {', '.join(params)} }};
  if (gridX*gridY*gridZ > 0) {{
    if (launch_cooperative_grid != 0 && global_scratch != 0 && {global_scratch_size} > 0 &&
        ({reset_global_scratch_each_launch} || reset_global_scratch != 0)) {{
      size_t global_scratch_bytes =
          (size_t)gridX * gridY * gridZ * num_ctas * {global_scratch_size};
      CUDA_CHECK(cuMemsetD8Async(global_scratch, 0, global_scratch_bytes, stream));
    }}
    int cluster_size = clusterDimX * clusterDimY * clusterDimZ;
    bool use_cluster_launch = cluster_size != 1;
    // 4 attributes that we can currently pass maximum
    CUlaunchAttribute launchAttr[4];
    static cuLaunchKernelEx_t cuLaunchKernelExHandle = NULL;
    if (cuLaunchKernelExHandle == NULL) {{
      cuLaunchKernelExHandle = getLaunchKernelExHandle();
    }}
    CUlaunchConfig config;
    config.gridDimX = gridX * num_ctas;
    config.gridDimY = gridY;
    config.gridDimZ = gridZ;
    if (use_cluster_launch) {{
      config.gridDimX *= clusterDimX;
      config.gridDimY *= clusterDimY;
      config.gridDimZ *= clusterDimZ;
    }}
    config.blockDimX = 32 * num_warps;
    config.blockDimY = 1;
    config.blockDimZ = 1;
    config.sharedMemBytes = shared_memory;
    config.hStream = stream;
    config.attrs = launchAttr;
    int num_attrs = 0;

    if (launch_pdl != 0) {{
      CUlaunchAttribute pdlAttr = {{ .id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, .value = 1}};
      launchAttr[num_attrs] = pdlAttr;
      ++num_attrs;
    }}

    if (launch_cooperative_grid != 0) {{
      CUlaunchAttribute coopAttr = {{ .id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE, .value = 1}};
      launchAttr[num_attrs] = coopAttr;
      ++num_attrs;
    }}

    if (use_cluster_launch) {{
      CUlaunchAttribute clusterAttr = {{}};
      clusterAttr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      clusterAttr.value.clusterDim.x = clusterDimX;
      clusterAttr.value.clusterDim.y = clusterDimY;
      clusterAttr.value.clusterDim.z = clusterDimZ;
      launchAttr[num_attrs] = clusterAttr;
      ++num_attrs;

      CUlaunchAttribute clusterSchedulingAttr = {{}};
      clusterSchedulingAttr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      clusterSchedulingAttr.value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
      launchAttr[num_attrs] = clusterSchedulingAttr;
      ++num_attrs;
    }}

    // Cluster size 16 is non-portable. Does work for H100 and B200 tho.
    config.numAttrs = num_attrs;
    if (num_ctas == 16 || cluster_size == 16) {{
      CUDA_CHECK(cuFuncSetAttribute(
          function,
          CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
          1
      ));
    }}

    CUDA_CHECK(cuLaunchKernelExHandle(&config, function, params, 0));
  }}
}}
#else
static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int launch_cooperative_grid, int launch_pdl, int reset_global_scratch, int shared_memory, CUstream stream, CUfunction function, CUdeviceptr global_scratch, CUdeviceptr profile_scratch{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  void *params[] = {{ {', '.join(params)} }};
  if (gridX*gridY*gridZ > 0) {{
    if (launch_cooperative_grid != 0 && global_scratch != 0 && {global_scratch_size} > 0 &&
        ({reset_global_scratch_each_launch} || reset_global_scratch != 0)) {{
      size_t global_scratch_bytes =
          (size_t)gridX * gridY * gridZ * num_ctas * {global_scratch_size};
      CUDA_CHECK(cuMemsetD8Async(global_scratch, 0, global_scratch_bytes, stream));
    }}
    // 4 attributes that we can currently pass maximum
    CUlaunchAttribute launchAttr[4];
    static cuLaunchKernelEx_t cuLaunchKernelExHandle = NULL;
    if (cuLaunchKernelExHandle == NULL) {{
      cuLaunchKernelExHandle = getLaunchKernelExHandle();
    }}
    CUlaunchConfig config;
    config.gridDimX = gridX * num_ctas;
    config.gridDimY = gridY;
    config.gridDimZ = gridZ;

    config.blockDimX = 32 * num_warps;
    config.blockDimY = 1;
    config.blockDimZ = 1;
    config.sharedMemBytes = shared_memory;
    config.hStream = stream;
    config.attrs = launchAttr;
    int num_attrs = 0;

    if (launch_pdl != 0) {{
      CUlaunchAttribute pdlAttr = {{ .id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, .value = 1}};
      launchAttr[num_attrs] = pdlAttr;
      ++num_attrs;
    }}

    if (launch_cooperative_grid != 0) {{
      CUlaunchAttribute coopAttr = {{ .id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE, .value = 1}};
      launchAttr[num_attrs] = coopAttr;
      ++num_attrs;
    }}

    if (num_ctas != 1) {{
      CUlaunchAttribute clusterAttr = {{}};
      clusterAttr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      clusterAttr.value.clusterDim.x = num_ctas;
      clusterAttr.value.clusterDim.y = 1;
      clusterAttr.value.clusterDim.z = 1;
      launchAttr[num_attrs] = clusterAttr;
      ++num_attrs;

      CUlaunchAttribute clusterSchedulingAttr = {{}};
      clusterSchedulingAttr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      clusterSchedulingAttr.value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
      launchAttr[num_attrs] = clusterSchedulingAttr;
      ++num_attrs;
    }}

    // num_ctas == 16 is non-portable. Does work for H100 and B200 tho
    config.numAttrs = num_attrs;
    if (num_ctas == 16) {{
      CUDA_CHECK(cuFuncSetAttribute(
          function,
          CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
          1
      ));
    }}

    CUDA_CHECK(cuLaunchKernelExHandle(&config, function, params, 0));
  }}
}}
#endif

typedef struct _DevicePtrInfo {{
    CUdeviceptr dev_ptr;
    bool valid;
}} DevicePtrInfo;

static PyObject* data_ptr_str = NULL;
static PyObject* py_tensor_map_type = NULL;

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
  PyObject *ret = PyObject_CallMethodNoArgs(obj, data_ptr_str);
  if (!ret) {{
    PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
    ptr_info.valid = false;
    goto cleanup;
  }}
  if (!PyLong_Check(ret)) {{
    PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
    ptr_info.valid = false;
    goto cleanup;
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
  }} else if (status != CUDA_SUCCESS) {{
      CUDA_CHECK(status);  // Catch any other cuda API errors
      ptr_info.valid = false;
  }}
  ptr_info.dev_ptr = dev_ptr;
cleanup:
  Py_XDECREF(ret);
  return ptr_info;

}}

static inline CUtensorMap* getTmaDesc(PyObject *obj) {{
  if (sizeof(CUtensorMap*) != 8) {{
    PyErr_SetString(PyExc_SystemError, "getTmaDesc() requires 64-bit compilation");
    return NULL;
  }}

if (Py_TYPE(obj) != (PyTypeObject*)py_tensor_map_type) {{
    PyErr_Format(PyExc_TypeError, "object must be of type PyCUtensorMap, got %s", Py_TYPE(obj)->tp_name);
    return NULL;
}}

  CUtensorMap* map = &((PyCUtensorMapObject*)obj)->tensorMap;
  uintptr_t align_128 = (uintptr_t)map & (128 - 1);
  if (align_128 != 0) {{
    PyErr_Format(PyExc_ValueError, "CUtensorMap must be aligned to 128B, but got (&map) mod 128 = %ld", align_128);
    return NULL;
  }}
  return map;
}}

static void ensureCudaContext() {{
  CUcontext pctx;
  CUDA_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {{
    // Ensure device context.
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&pctx, device));
    CUDA_CHECK(cuCtxSetCurrent(pctx));
  }}
}}

static uint16_t pack_fp16(double f) {{
    uint16_t result;
    // from https://github.com/python/pythoncapi-compat
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 && !defined(PYPY_VERSION)
    _PyFloat_Pack2(f, (unsigned char*)&result, 1);
#else
    PyFloat_Pack2(f, (unsigned char*)&result, 1);
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

#define PREPARED_CUDA_LAUNCHER_CAPSULE "triton.cuda.PreparedLauncher"
#define PREPARED_ARGUMENT_COUNT {prepared_arg_count}
#define PREPARED_ARGUMENT_ARRAY_SIZE {prepared_array_size}
#define PREPARED_LAUNCH_BASE_ARGUMENT_COUNT 10

typedef struct {{
  {prepared_arg_fields}
}} PreparedArgumentStorage;

typedef struct {{
  CUfunction function;
  int num_warps;
  int num_ctas;
  int shared_memory;
  int cluster_dim_x;
  int cluster_dim_y;
  int cluster_dim_z;
  int launch_cooperative_grid;
  int launch_pdl;
  int trusted_pointer_arguments;
  CUdeviceptr initialized_global_scratch;
  Py_ssize_t dynamic_count;
  int dynamic_indices[PREPARED_ARGUMENT_ARRAY_SIZE];
  unsigned char is_dynamic[PREPARED_ARGUMENT_ARRAY_SIZE];
  PreparedArgumentStorage arguments;
}} PreparedLaunchState;

static const unsigned char prepared_runtime_arguments[PREPARED_ARGUMENT_ARRAY_SIZE] = {{ {prepared_runtime_flags} }};

static inline DevicePtrInfo getPreparedPointer(PyObject *obj, int idx, int trusted) {{
  if (!trusted) {{
    return getPointer(obj, idx);
  }}

  DevicePtrInfo ptr_info = {{0, true}};
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(obj);
    if (ptr_info.dev_ptr == (CUdeviceptr)-1 && PyErr_Occurred()) {{
      ptr_info.valid = false;
    }}
    return ptr_info;
  }}
  if (obj == Py_None) {{
    return ptr_info;
  }}

  PyObject *ret = PyObject_CallMethodNoArgs(obj, data_ptr_str);
  if (ret == NULL) {{
    PyErr_Format(PyExc_TypeError,
                 "Prepared pointer argument %d must be uint64, None, or expose data_ptr()",
                 idx);
    ptr_info.valid = false;
    return ptr_info;
  }}
  if (!PyLong_Check(ret)) {{
    PyErr_Format(PyExc_TypeError,
                 "data_ptr() for prepared pointer argument %d must return uint64",
                 idx);
    ptr_info.valid = false;
    Py_DECREF(ret);
    return ptr_info;
  }}
  ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(ret);
  if (ptr_info.dev_ptr == (CUdeviceptr)-1 && PyErr_Occurred()) {{
    ptr_info.valid = false;
  }}
  Py_DECREF(ret);
  return ptr_info;
}}

static int setPreparedArgument(PreparedArgumentStorage *storage,
                               int index,
                               PyObject *obj,
                               int trusted_pointer_arguments) {{
  switch (index) {{
{prepared_set_cases}
    default:
      PyErr_Format(PyExc_IndexError,
                   "Prepared CUDA argument index %d is invalid or constexpr",
                   index);
      return 0;
  }}
}}

static int parsePreparedInt32(PyObject *obj, const char *name, int *output) {{
  long value = PyLong_AsLong(obj);
  if (value == -1 && PyErr_Occurred()) {{
    return 0;
  }}
  if (value < INT_MIN || value > INT_MAX) {{
    PyErr_Format(PyExc_OverflowError, "%s is outside the int32 range", name);
    return 0;
  }}
  *output = (int)value;
  return 1;
}}

static int parsePreparedKernelMetadata(PyObject *kernel_metadata,
                                       PreparedLaunchState *state) {{
#ifdef __TLE__
  if (!PyTuple_Check(kernel_metadata)) {{
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
    return 0;
  }}
  Py_ssize_t kernel_metadata_size = PyTuple_Size(kernel_metadata);
  if (kernel_metadata_size == 3) {{
    if (!PyArg_ParseTuple(kernel_metadata, "iii", &state->num_warps,
                          &state->num_ctas, &state->shared_memory)) {{
      return 0;
    }}
  }} else if (kernel_metadata_size == 6) {{
    if (!PyArg_ParseTuple(kernel_metadata, "iiiiii", &state->num_warps,
                          &state->num_ctas, &state->shared_memory,
                          &state->cluster_dim_x, &state->cluster_dim_y,
                          &state->cluster_dim_z)) {{
      return 0;
    }}
  }} else {{
    PyErr_SetString(PyExc_TypeError,
                    "kernel_metadata must contain 3 or 6 integers");
    return 0;
  }}
  if (state->cluster_dim_x <= 0 || state->cluster_dim_y <= 0 ||
      state->cluster_dim_z <= 0) {{
    PyErr_SetString(PyExc_ValueError, "cluster dims must be positive");
    return 0;
  }}
#else
  if (!PyArg_ParseTuple(kernel_metadata, "iii", &state->num_warps,
                        &state->num_ctas, &state->shared_memory)) {{
    PyErr_SetString(PyExc_TypeError,
                    "kernel_metadata must be (num_warps, num_ctas, shared_memory)");
    return 0;
  }}
#endif
  return 1;
}}

static void destroyPreparedLaunchState(PyObject *capsule) {{
  PreparedLaunchState *state = (PreparedLaunchState *)PyCapsule_GetPointer(
      capsule, PREPARED_CUDA_LAUNCHER_CAPSULE);
  if (state != NULL) {{
    PyMem_Free(state);
  }} else {{
    PyErr_Clear();
  }}
}}

static PyObject* prepare_launcher(PyObject* self, PyObject* args) {{
  uint64_t function;
  PyObject *kernel_metadata = NULL;
  PyObject *dynamic_indices = NULL;
  PyObject *flat_arguments = NULL;
  int launch_cooperative_grid;
  int launch_pdl;
  int trusted_pointer_arguments;
  if (!PyArg_ParseTuple(args, "KOpppOO", &function, &kernel_metadata,
                        &launch_cooperative_grid, &launch_pdl,
                        &trusted_pointer_arguments, &dynamic_indices,
                        &flat_arguments)) {{
    return NULL;
  }}
  if (!PyTuple_Check(dynamic_indices)) {{
    PyErr_SetString(PyExc_TypeError,
                    "dynamic_indices must be a tuple of flattened argument indices");
    return NULL;
  }}
  if (!PyTuple_Check(flat_arguments) ||
      PyTuple_GET_SIZE(flat_arguments) != PREPARED_ARGUMENT_COUNT) {{
    PyErr_Format(PyExc_TypeError,
                 "flat_arguments must be a tuple with %d entries",
                 PREPARED_ARGUMENT_COUNT);
    return NULL;
  }}
  Py_ssize_t dynamic_count = PyTuple_GET_SIZE(dynamic_indices);
  if (dynamic_count > PREPARED_ARGUMENT_COUNT) {{
    PyErr_SetString(PyExc_ValueError,
                    "Prepared CUDA launcher has too many dynamic arguments");
    return NULL;
  }}

  PreparedLaunchState *state = (PreparedLaunchState *)PyMem_Calloc(
      1, sizeof(PreparedLaunchState));
  if (state == NULL) {{
    return PyErr_NoMemory();
  }}
  state->function = (CUfunction)function;
  state->cluster_dim_x = 1;
  state->cluster_dim_y = 1;
  state->cluster_dim_z = 1;
  state->launch_cooperative_grid = launch_cooperative_grid;
  state->launch_pdl = launch_pdl;
  state->trusted_pointer_arguments = trusted_pointer_arguments;
  state->dynamic_count = dynamic_count;
  if (!parsePreparedKernelMetadata(kernel_metadata, state)) {{
    goto fail;
  }}

  for (Py_ssize_t i = 0; i < dynamic_count; ++i) {{
    PyObject *index_object = PyTuple_GET_ITEM(dynamic_indices, i);
    long index = PyLong_AsLong(index_object);
    if (index == -1 && PyErr_Occurred()) {{
      goto fail;
    }}
    if (index < 0 || index >= PREPARED_ARGUMENT_COUNT ||
        !prepared_runtime_arguments[index]) {{
      PyErr_Format(PyExc_IndexError,
                   "Prepared CUDA dynamic argument index %ld is invalid or constexpr",
                   index);
      goto fail;
    }}
    if (state->is_dynamic[index]) {{
      PyErr_Format(PyExc_ValueError,
                   "Prepared CUDA dynamic argument index %ld is duplicated",
                   index);
      goto fail;
    }}
    state->is_dynamic[index] = 1;
    state->dynamic_indices[i] = (int)index;
  }}

{prepared_static_initializers}

  return PyCapsule_New(state, PREPARED_CUDA_LAUNCHER_CAPSULE,
                       destroyPreparedLaunchState);

fail:
  PyMem_Free(state);
  return NULL;
}}

static PyObject* launch_prepared(PyObject* self, PyObject* args) {{
  if (!PyTuple_Check(args) || PyTuple_GET_SIZE(args) <
      PREPARED_LAUNCH_BASE_ARGUMENT_COUNT) {{
    PyErr_SetString(PyExc_TypeError,
                    "Prepared CUDA launch received an invalid argument tuple");
    return NULL;
  }}
  PreparedLaunchState *state = (PreparedLaunchState *)PyCapsule_GetPointer(
      PyTuple_GET_ITEM(args, 0), PREPARED_CUDA_LAUNCHER_CAPSULE);
  if (state == NULL) {{
    return NULL;
  }}
  Py_ssize_t expected_count =
      PREPARED_LAUNCH_BASE_ARGUMENT_COUNT + state->dynamic_count;
  if (PyTuple_GET_SIZE(args) != expected_count) {{
    PyErr_Format(PyExc_TypeError,
                 "Prepared CUDA launcher expects %zd dynamic arguments, got %zd",
                 state->dynamic_count,
                 PyTuple_GET_SIZE(args) - PREPARED_LAUNCH_BASE_ARGUMENT_COUNT);
    return NULL;
  }}

  int grid_x, grid_y, grid_z;
  if (!parsePreparedInt32(PyTuple_GET_ITEM(args, 1), "grid_x", &grid_x) ||
      !parsePreparedInt32(PyTuple_GET_ITEM(args, 2), "grid_y", &grid_y) ||
      !parsePreparedInt32(PyTuple_GET_ITEM(args, 3), "grid_z", &grid_z)) {{
    return NULL;
  }}
  uint64_t stream = PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(args, 4));
  if (stream == (uint64_t)-1 && PyErr_Occurred()) {{
    return NULL;
  }}
  PyObject *global_scratch_obj = PyTuple_GET_ITEM(args, 5);
  PyObject *profile_scratch_obj = PyTuple_GET_ITEM(args, 6);
  PyObject *launch_metadata = PyTuple_GET_ITEM(args, 7);
  PyObject *launch_enter_hook = PyTuple_GET_ITEM(args, 8);
  PyObject *launch_exit_hook = PyTuple_GET_ITEM(args, 9);

  ensureCudaContext();
  if (PyErr_Occurred()) {{
    return NULL;
  }}
  CUdeviceptr global_scratch = 0;
  if (global_scratch_obj != Py_None) {{
    DevicePtrInfo pointer = getPreparedPointer(global_scratch_obj, -1, 1);
    if (!pointer.valid) {{
      return NULL;
    }}
    global_scratch = pointer.dev_ptr;
  }}
  int reset_global_scratch =
      global_scratch != 0 &&
      state->initialized_global_scratch != global_scratch;
  CUdeviceptr profile_scratch = 0;
  if (profile_scratch_obj != Py_None) {{
    DevicePtrInfo pointer = getPointer(profile_scratch_obj, -1);
    if (!pointer.valid) {{
      return NULL;
    }}
    profile_scratch = pointer.dev_ptr;
  }}

  PreparedArgumentStorage launch_arguments = state->arguments;
  for (Py_ssize_t i = 0; i < state->dynamic_count; ++i) {{
    int index = state->dynamic_indices[i];
    PyObject *value = PyTuple_GET_ITEM(
        args, PREPARED_LAUNCH_BASE_ARGUMENT_COUNT + i);
    if (!setPreparedArgument(&launch_arguments, index, value,
                             state->trusted_pointer_arguments)) {{
      return NULL;
    }}
  }}

  if (launch_enter_hook != Py_None) {{
    PyObject *ret = PyObject_CallOneArg(launch_enter_hook, launch_metadata);
    if (ret == NULL) {{
      return NULL;
    }}
    Py_DECREF(ret);
  }}

  Py_BEGIN_ALLOW_THREADS;
#ifdef __TLE__
  _launch(grid_x, grid_y, grid_z, state->num_warps, state->num_ctas,
          state->cluster_dim_x, state->cluster_dim_y, state->cluster_dim_z,
          state->launch_cooperative_grid, state->launch_pdl,
          reset_global_scratch,
          state->shared_memory, (CUstream)stream, state->function,
          global_scratch, profile_scratch{prepared_call_suffix});
#else
  _launch(grid_x, grid_y, grid_z, state->num_warps, state->num_ctas,
          state->launch_cooperative_grid, state->launch_pdl,
          reset_global_scratch,
          state->shared_memory, (CUstream)stream, state->function,
          global_scratch, profile_scratch{prepared_call_suffix});
#endif
  Py_END_ALLOW_THREADS;
  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if (reset_global_scratch) {{
    state->initialized_global_scratch = global_scratch;
  }}

  if (launch_exit_hook != Py_None) {{
    PyObject *ret = PyObject_CallOneArg(launch_exit_hook, launch_metadata);
    if (ret == NULL) {{
      return NULL;
    }}
    Py_DECREF(ret);
  }}

  Py_RETURN_NONE;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  // ensure cuda context is valid before calling any CUDA APIs, e.g. before getPointer calls cuPointerGetAttributes
  ensureCudaContext();

  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int launch_cooperative_grid;
  int launch_pdl;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *global_scratch_obj = NULL;
  PyObject *profile_scratch_obj = NULL;
  {newline.join([f"{_extracted_type(ty)} _arg{i};" for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &_stream, &_function, &launch_cooperative_grid, &launch_pdl, &global_scratch_obj, &profile_scratch_obj,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook{args_list})) {{
    return NULL;
  }}

  int num_warps, num_ctas, shared_memory;
#ifdef __TLE__
  int clusterDimX = 1, clusterDimY = 1, clusterDimZ = 1;
  if (!PyTuple_Check(kernel_metadata)) {{
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
    return NULL;
  }}
  Py_ssize_t kernel_metadata_size = PyTuple_Size(kernel_metadata);
  if (kernel_metadata_size == 3) {{
    if (!PyArg_ParseTuple(kernel_metadata, \"iii\", &num_warps, &num_ctas, &shared_memory)) {{
      PyErr_SetString(PyExc_TypeError, "kernel_metadata must be (num_warps, num_ctas, shared_memory)");
      return NULL;
    }}
  }} else if (kernel_metadata_size == 6) {{
    if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory,
                          &clusterDimX, &clusterDimY, &clusterDimZ)) {{
      PyErr_SetString(PyExc_TypeError,
                      "kernel_metadata must be (num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ)");
      return NULL;
    }}
  }} else {{
    PyErr_SetString(PyExc_TypeError,
                    "kernel_metadata must contain 3 or 6 integers");
    return NULL;
  }}
  if (clusterDimX <= 0 || clusterDimY <= 0 || clusterDimZ <= 0) {{
    PyErr_SetString(PyExc_ValueError, "cluster dims must be positive");
    return NULL;
  }}
#else
  if (!PyArg_ParseTuple(kernel_metadata, \"iii\", &num_warps, &num_ctas, &shared_memory)) {{
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
    return NULL;
  }}
#endif

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* ret = PyObject_CallOneArg(launch_enter_hook, launch_metadata);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}

  CUdeviceptr global_scratch = 0;
  if (global_scratch_obj != Py_None) {{
    DevicePtrInfo global_scratch_info = getPointer(global_scratch_obj, -1);
    if (!global_scratch_info.valid) {{
      return NULL;
    }}
    global_scratch = global_scratch_info.dev_ptr;
  }}

  CUdeviceptr profile_scratch = 0;
  if (profile_scratch_obj != Py_None) {{
    DevicePtrInfo profile_scratch_info = getPointer(profile_scratch_obj, -1);
    if (!profile_scratch_info.valid) {{
      return NULL;
    }}
    profile_scratch = profile_scratch_info.dev_ptr;
  }}

  // raise exception asap
  {newline.join(ptr_decls)}
  {newline.join(tma_decls)}
  {newline.join(float_storage_decls)}
  Py_BEGIN_ALLOW_THREADS;
#ifdef __TLE__
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, launch_cooperative_grid, launch_pdl, 1, shared_memory, (CUstream)_stream, (CUfunction)_function, global_scratch, profile_scratch{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});
#else
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, launch_cooperative_grid, launch_pdl, 1, shared_memory, (CUstream)_stream, (CUfunction)_function, global_scratch, profile_scratch{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});
#endif
  Py_END_ALLOW_THREADS;
  if (PyErr_Occurred()) {{
    return NULL;
  }}

  if(launch_exit_hook != Py_None){{
    PyObject* ret = PyObject_CallOneArg(launch_exit_hook, launch_metadata);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}

  Py_RETURN_NONE;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{"prepare", prepare_launcher, METH_VARARGS, "Bind invariant CUDA launch arguments"}},
  {{"launch_prepared", launch_prepared, METH_VARARGS, "Launch with only dynamic CUDA arguments"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_launcher(void) {{
  data_ptr_str = PyUnicode_InternFromString("data_ptr");
  if(data_ptr_str == NULL) {{
    return NULL;
  }}
  PyObject* driver_mod = PyImport_ImportModule("triton.backends.nvidia.driver");
  if (driver_mod == NULL) {{
    return NULL;
  }}
  py_tensor_map_type = PyObject_GetAttrString(driver_mod, "PyCUtensorMap");
  if (py_tensor_map_type == NULL) {{
    return NULL;
  }}

  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


# The TMA dtype enum values are slightly different on host vs device...
TMA_DTYPE_DEVICE_TO_HOST = dict((i, i) for i in range(16))
TMA_DTYPE_DEVICE_TO_HOST[8] = 10
TMA_DTYPE_DEVICE_TO_HOST[9] = 8
TMA_DTYPE_DEVICE_TO_HOST[10] = 9


def make_tensordesc_arg(arg, metadata):
    if metadata is None:
        # Currently the host side tensor descriptors get decomposed in
        # the frontend to tensor desc, shape, and strides. We have no
        # way to use these shape and strides when processing tensor
        # descriptors which is why we provide our own decomposition
        # above. Sadly this means we have to pass the shape and strides
        # twice.
        return [arg.base, *arg.shape, *arg.strides, arg.padding == "nan", *arg.shape, *arg.strides]

    swizzle = metadata["swizzle"]
    elem_size = metadata["elem_size"]
    elem_type = metadata["elem_type"]
    block_size = metadata["block_size"]
    fp4_padded = metadata["fp4_padded"]

    shape = arg.shape
    strides = arg.strides
    assert strides[-1] == 1
    padding = 1 if arg.padding == "nan" else 0

    if fp4_padded:
        shape = list(shape)
        shape[-1] *= 2

    cu_tensor_map = triton.runtime.driver.active.utils.fill_tma_descriptor(
        arg.base.data_ptr(),
        swizzle,
        elem_size,
        TMA_DTYPE_DEVICE_TO_HOST[elem_type],
        block_size,
        shape,
        strides,
        padding,
    )

    return [cu_tensor_map, *shape, *strides]


def wrap_handle_tensordesc(launcher, signature, tensordesc_meta):
    has_tensor_desc_arg = any(isinstance(sig, str) and sig.startswith("tensordesc") for sig in signature.values())
    if not has_tensor_desc_arg:
        return launcher

    tensordesc_indices = set(
        [i for i, sig in enumerate(signature.values()) if isinstance(sig, str) and sig.startswith("tensordesc")])
    assert not tensordesc_meta or len(tensordesc_meta) == len(tensordesc_indices)
    if not tensordesc_meta:
        tensordesc_meta = [None] * len(tensordesc_indices)

    def inner(*args):
        final_args = list(args[:_BASE_ARGS_FORMAT_LEN])
        tensordesc_idx = 0
        for i, arg in enumerate(args[_BASE_ARGS_FORMAT_LEN:]):
            if i in tensordesc_indices:
                final_args.extend(make_tensordesc_arg(arg, tensordesc_meta[tensordesc_idx]))
                tensordesc_idx += 1
            else:
                final_args.append(arg)
        return launcher(*final_args)

    return inner


def _prepare_pointer_argument(arg, signature, trusted_pointer_arguments):
    if isinstance(signature, tuple):
        if not isinstance(arg, tuple) or len(arg) != len(signature):
            raise TypeError("Prepared Triton tuple argument does not match its signature")
        return tuple(
            _prepare_pointer_argument(value, element_signature, trusted_pointer_arguments)
            for value, element_signature in zip(arg, signature))
    if isinstance(signature, str) and signature.startswith("*"):
        if not trusted_pointer_arguments or arg is None or isinstance(arg, int):
            return arg
        data_ptr = getattr(arg, "data_ptr", None)
        if not callable(data_ptr):
            raise TypeError("Prepared Triton pointer arguments must be integers, None, "
                            "or objects with data_ptr()")
        return int(data_ptr())
    return arg


def _flatten_launcher_argument(arg, signature, tensordesc_meta):
    if isinstance(signature, str) and signature.startswith("tensordesc"):
        return make_tensordesc_arg(arg, tensordesc_meta)
    if isinstance(signature, tuple):
        if not isinstance(arg, tuple) or len(arg) != len(signature):
            raise TypeError("Prepared Triton tuple argument does not match its signature")
        flattened = []
        for value, element_signature in zip(arg, signature):
            flattened.extend(_flatten_launcher_argument(value, element_signature, None))
        return flattened
    return [arg]


class CudaPreparedLauncher:
    """A CUDA launcher with invariant kernel arguments expanded and retained."""

    def __init__(
        self,
        launcher,
        arguments,
        dynamic_arg_indices,
        trusted_pointer_arguments,
        function,
        packed_metadata,
    ):
        signature = launcher.signature
        if len(arguments) != len(signature):
            raise TypeError(f"Prepared CUDA launcher expects {len(signature)} arguments, "
                            f"got {len(arguments)}")
        dynamic_arg_indices = tuple(dynamic_arg_indices)
        dynamic_set = set(dynamic_arg_indices)
        if len(dynamic_set) != len(dynamic_arg_indices):
            raise ValueError("Prepared CUDA dynamic argument indices must be unique")

        descriptor_metas = iter(launcher.tensordesc_meta)
        flat_arguments = []
        dynamic_bindings = {}
        dynamic_flat_indices = []
        for index, (argument, arg_signature) in enumerate(zip(arguments, signature)):
            descriptor_meta = (next(descriptor_metas)
                               if isinstance(arg_signature, str) and arg_signature.startswith("tensordesc") else None)
            flattened = _flatten_launcher_argument(
                argument,
                arg_signature,
                descriptor_meta,
            )
            start = len(flat_arguments)
            flat_arguments.extend(flattened)
            if index in dynamic_set:
                dynamic_flat_indices.extend(range(start, start + len(flattened)))
                flat_arguments[start:start + len(flattened)] = [None] * len(flattened)
                dynamic_bindings[index] = (
                    arg_signature,
                    descriptor_meta,
                    len(flattened),
                )
        try:
            next(descriptor_metas)
        except StopIteration:
            pass
        else:
            raise ValueError("Prepared CUDA launcher did not consume all descriptor metadata")
        missing = sorted(dynamic_set - dynamic_bindings.keys())
        if missing:
            raise IndexError(f"Prepared CUDA dynamic argument indices are invalid: {missing}")
        if len(flat_arguments) != len(launcher.flat_signature):
            raise RuntimeError("Prepared CUDA argument expansion does not match the compiled signature")

        self.launcher = launcher
        self.dynamic_bindings = tuple((index, *dynamic_bindings[index]) for index in dynamic_arg_indices)
        self.dynamic_arguments_are_flat = all(
            length == 1 and not isinstance(arg_signature, tuple)
            and not (isinstance(arg_signature, str) and arg_signature.startswith("tensordesc"))
            for _, arg_signature, _, length in self.dynamic_bindings)
        self.native_state = launcher.raw_prepare(
            function,
            packed_metadata,
            launcher.launch_cooperative_grid,
            launcher.launch_pdl,
            bool(trusted_pointer_arguments),
            tuple(dynamic_flat_indices),
            tuple(flat_arguments),
        )
        self.scratch_by_launch = {}

    def _prepared_scratch(self, gridX, gridY, gridZ, stream):
        key = (gridX, gridY, gridZ, stream)
        cached = self.scratch_by_launch.get(key)
        if cached is None:
            global_backing = self.launcher._allocate_global_scratch(gridX, gridY, gridZ, stream)
            profile_backing = self.launcher._allocate_profile_scratch(gridX, gridY, gridZ, stream)
            global_pointer = _prepare_pointer_argument(global_backing, "*i8", True)
            profile_pointer = _prepare_pointer_argument(profile_backing, "*i8", True)
            cached = (
                global_backing,
                profile_backing,
                global_pointer,
                profile_pointer,
            )
            self.scratch_by_launch[key] = cached
        return cached[2], cached[3]

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        if len(args) < 4:
            raise TypeError("Prepared CUDA launcher is missing launch metadata")
        launch_args = args[:4]
        dynamic_args = args[4:]
        if len(dynamic_args) != len(self.dynamic_bindings):
            raise TypeError(f"Prepared CUDA launcher expects {len(self.dynamic_bindings)} "
                            f"dynamic arguments, got {len(dynamic_args)}")

        if self.dynamic_arguments_are_flat:
            flattened_dynamic_args = dynamic_args
        else:
            flattened_dynamic_args = []
            for value, (_, signature, descriptor_meta, length) in zip(dynamic_args, self.dynamic_bindings):
                flattened = _flatten_launcher_argument(
                    value,
                    signature,
                    descriptor_meta,
                )
                if len(flattened) != length:
                    raise RuntimeError("Prepared CUDA argument expansion changed arity")
                flattened_dynamic_args.extend(flattened)
        global_scratch, profile_scratch = self._prepared_scratch(gridX, gridY, gridZ, stream)
        self.launcher.raw_prepared_launch(
            self.native_state,
            gridX,
            gridY,
            gridZ,
            stream,
            global_scratch,
            profile_scratch,
            launch_args[1],
            launch_args[2],
            launch_args[3],
            *flattened_dynamic_args,
        )


class CudaLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)
        launcher_src = make_launcher(
            constants,
            signature,
            tensordesc_meta,
            metadata.global_scratch_size,
            getattr(metadata, "global_scratch_reset_per_launch", True),
        )
        mod = compile_module_from_src(
            src=launcher_src,
            name="__triton_launcher",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )

        self.num_ctas = getattr(metadata, "num_ctas", 1)
        self.signature = tuple(signature.values())
        descriptor_count = sum(
            isinstance(arg_signature, str) and arg_signature.startswith("tensordesc")
            for arg_signature in self.signature)
        self.tensordesc_meta = tuple(tensordesc_meta or (None, ) * descriptor_count)
        self.flat_signature = tuple(
            _flatten_launcher_signature(_expand_launcher_signature(
                self.signature,
                self.tensordesc_meta,
            )))
        self.raw_launch = mod.launch
        self.raw_prepare = mod.prepare
        self.raw_prepared_launch = mod.launch_prepared
        self.launch = wrap_handle_tensordesc(mod.launch, signature, tensordesc_meta)
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align
        self.profile_scratch_size = metadata.profile_scratch_size
        self.profile_scratch_align = metadata.profile_scratch_align
        self.launch_cooperative_grid = metadata.launch_cooperative_grid
        self.launch_pdl = metadata.launch_pdl

    def prepare(
        self,
        arguments,
        dynamic_arg_indices,
        *,
        trusted_pointer_arguments=False,
        function,
        packed_metadata,
    ):
        return CudaPreparedLauncher(
            self,
            arguments,
            dynamic_arg_indices,
            trusted_pointer_arguments,
            function,
            packed_metadata,
        )

    def _allocate_scratch_buffer(self, gridX, gridY, gridZ, stream, size, align, allocator):
        if size <= 0:
            return None
        grid_size = gridX * gridY * gridZ
        alloc_size = grid_size * self.num_ctas * size
        return allocator.get()(alloc_size, align, stream)

    def _allocate_global_scratch(self, gridX, gridY, gridZ, stream):
        return self._allocate_scratch_buffer(
            gridX,
            gridY,
            gridZ,
            stream,
            self.global_scratch_size,
            self.global_scratch_align,
            _allocation._allocator,
        )

    def _allocate_profile_scratch(self, gridX, gridY, gridZ, stream):
        return self._allocate_scratch_buffer(
            gridX,
            gridY,
            gridZ,
            stream,
            self.profile_scratch_size,
            self.profile_scratch_align,
            _allocation._profile_allocator,
        )

    def _allocate_scratch(self, gridX, gridY, gridZ, stream):
        global_scratch = self._allocate_global_scratch(gridX, gridY, gridZ, stream)
        profile_scratch = self._allocate_profile_scratch(gridX, gridY, gridZ, stream)
        return global_scratch, profile_scratch

    def _launch_expanded(
        self,
        gridX,
        gridY,
        gridZ,
        stream,
        function,
        launch_args,
        kernel_args,
        scratch=None,
    ):
        global_scratch, profile_scratch = (self._allocate_scratch(gridX, gridY, gridZ, stream)
                                           if scratch is None else scratch)
        self.raw_launch(
            gridX,
            gridY,
            gridZ,
            stream,
            function,
            self.launch_cooperative_grid,
            self.launch_pdl,
            global_scratch,
            profile_scratch,
            *launch_args,
            *kernel_args,
        )

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        global_scratch, profile_scratch = self._allocate_scratch(gridX, gridY, gridZ, stream)
        self.launch(gridX, gridY, gridZ, stream, function, self.launch_cooperative_grid, self.launch_pdl,
                    global_scratch, profile_scratch, *args)


class CudaDriver(GPUDriver):

    def __init__(self):
        self.utils = CudaUtils()  # TODO: make static
        self.launcher_cls = CudaLauncher
        super().__init__()

    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        warp_size = 32
        return GPUTarget("cuda", capability, warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("cuda", self.get_current_device())

    def get_device_interface(self):
        import torch
        return torch.cuda

    @staticmethod
    def is_active():
        try:
            import torch
            return torch.cuda.is_available() and (torch.version.hip is None)
        except ImportError:
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
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')

    def clear_cache(self, cache):
        cache.zero_()
