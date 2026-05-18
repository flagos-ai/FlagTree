import functools
import os
import subprocess
import weakref
from collections import OrderedDict
from pathlib import Path

from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase
from triton.runtime.build import compile_module_from_src

dirname = os.path.dirname(os.path.realpath(__file__))
_TENSORDESC_CACHE_LIMIT = 1024


def _split_paths(value: str):
    return [p for p in value.split(":") if p]


@functools.lru_cache()
def _musa_home_dirs():
    candidates = []
    for key in ("MUSA_HOME", "MUSA_ROOT"):
        if val := os.getenv(key):
            candidates.append(val)
    return candidates


@functools.lru_cache()
def _musa_include_dirs():
    include_dirs = [os.path.join(dirname, "include")]
    if env_inc := os.getenv("TRITON_MUSA_INCLUDE_PATH"):
        include_dirs.append(env_inc)
    for home in _musa_home_dirs():
        include_dirs.append(os.path.join(home, "include"))

    # Validate that musa.h exists in one of the include dirs.
    for inc in include_dirs:
        if os.path.exists(os.path.join(inc, "musa.h")):
            return include_dirs
    raise RuntimeError("Cannot find musa.h. Set TRITON_MUSA_INCLUDE_PATH or MUSA_HOME/MUSA_ROOT to a valid MUSA SDK.")


@functools.lru_cache()
def _libmusa_dirs():

    def has_libmusa(path: str) -> bool:
        return (os.path.exists(os.path.join(path, "libmusa.so")) or os.path.exists(os.path.join(path, "libmusa.so.1")))

    paths = []

    if env_lib := os.getenv("TRITON_LIBMUSA_PATH") or os.getenv("TRITON_MUSA_LIB_PATH"):
        if os.path.isfile(env_lib):
            paths.append(os.path.dirname(env_lib))
        else:
            paths.append(env_lib)

    for home in _musa_home_dirs():
        paths.append(os.path.join(home, "lib"))
        paths.append(os.path.join(home, "lib64"))

    env_ld = os.getenv("LD_LIBRARY_PATH")
    if env_ld:
        paths.extend(_split_paths(env_ld))

    # Try ldconfig cache
    try:
        libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
        locs = [line.split()[-1] for line in libs.splitlines() if "libmusa.so" in line]
        paths.extend([os.path.dirname(loc) for loc in locs])
    except Exception:
        pass

    # Filter to existing directories that contain libmusa.
    valid = [p for p in paths if has_libmusa(p)]
    if not valid:
        raise RuntimeError(
            "libmusa.so/libmusa.so.1 not found. Set TRITON_LIBMUSA_PATH/TRITON_MUSA_LIB_PATH or MUSA_HOME/MUSA_ROOT, "
            "or update LD_LIBRARY_PATH.")
    return valid


def _library_dirs():
    return [os.path.join(dirname, "lib"), *_libmusa_dirs()]


# ------------------------
# Utils
# ------------------------


class MusaUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(MusaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        src = Path(os.path.join(dirname, "driver.c")).read_text()
        mod = compile_module_from_src(
            src=src,
            name="musa_utils",
            include_dirs=_musa_include_dirs(),
            library_dirs=_library_dirs(),
            libraries=["musa"],
        )
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.set_printf_fifo_size = mod.set_printf_fifo_size
        for name in ("fill_1d_tma_descriptor", "fill_2d_tma_descriptor", "fill_3d_tma_descriptor",
                     "fill_4d_tma_descriptor", "fill_5d_tma_descriptor"):
            if hasattr(mod, name):
                setattr(self, name, getattr(mod, name))


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    # Align ABI mapping with NVIDIA/AMD for host-side signatures.
    if ty[0] == '*':
        return "MUdeviceptr"
    if ty.startswith("tensordesc<"):
        return "MUdeviceptr"
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
        "fp16": "double",
        "bf16": "double",
        "fp32": "double",
        "f32": "double",
        "fp64": "double",
        "constexpr": "int64_t",
    }[ty]


def ty_to_cpp_param(ty):
    if ty[0] == '*':
        return "MUdeviceptr"
    if ty.startswith("tensordesc<"):
        return "MUdeviceptr"
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
        "fp16": "uint16_t",
        "bf16": "uint16_t",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
        "constexpr": "int64_t",
    }[ty]


def _parse_tensordesc_type(ty: str):
    if not isinstance(ty, str) or not ty.startswith("tensordesc<") or not ty.endswith(">"):
        return None
    body = ty[len("tensordesc<"):-1]
    dtype, sep, shape = body.partition("[")
    if not sep or not shape.endswith("]"):
        return None
    dims = [dim.strip() for dim in shape[:-1].split(",") if dim.strip()]
    if not dtype or not dims:
        return None
    return dtype.strip(), len(dims)


def _get_tensordesc_abi_expanded_args(rank: int, metadata):
    full_abi_args = 1 + 2 * rank
    if metadata is None:
        return full_abi_args
    abi_args = int(metadata.get("abi_expanded_args", full_abi_args))
    if abi_args not in (1, full_abi_args):
        raise ValueError(
            f"unsupported MUSA tensor descriptor ABI expansion: expected 1 or {full_abi_args}, got {abi_args}")
    return abi_args


def _expand_tensordesc_signature(signature_types, tensordesc_meta=None):
    expanded_types = []
    expanded_index = {}
    tensordesc_idx = 0
    for i, ty in enumerate(signature_types):
        desc_info = _parse_tensordesc_type(ty)
        if desc_info is None:
            expanded_index[i] = [len(expanded_types)]
            expanded_types.append(ty)
            continue

        dtype, rank = desc_info
        desc_meta = None
        if tensordesc_meta is not None and tensordesc_idx < len(tensordesc_meta):
            desc_meta = tensordesc_meta[tensordesc_idx]
        abi_expanded_args = _get_tensordesc_abi_expanded_args(rank, desc_meta)
        mapped = []
        expanded_types.append(f"*{dtype}")
        mapped.append(len(expanded_types) - 1)
        if abi_expanded_args != 1:
            for _ in range(rank):
                expanded_types.append("i32")
                mapped.append(len(expanded_types) - 1)
            for _ in range(rank):
                expanded_types.append("i64")
                mapped.append(len(expanded_types) - 1)
        expanded_index[i] = mapped
        tensordesc_idx += 1

    return expanded_types, expanded_index


def _normalize_arg_path(key):
    if isinstance(key, int):
        return (key, )
    if isinstance(key, tuple):
        return key
    raise TypeError(f"unsupported signature path key: {key!r}")


def _expand_signature_tree(signature_types, tensordesc_meta=None):
    expanded_types = []
    expanded_index = {}
    tensordesc_idx = 0

    def visit(ty, path):
        nonlocal tensordesc_idx

        if isinstance(ty, tuple):
            mapped = []
            for child_idx, child_ty in enumerate(ty):
                mapped.extend(visit(child_ty, path + (child_idx, )))
            expanded_index[path] = mapped
            return mapped

        desc_info = _parse_tensordesc_type(ty)
        if desc_info is None:
            expanded_index[path] = [len(expanded_types)]
            expanded_types.append(ty)
            return expanded_index[path]

        _, rank = desc_info
        desc_meta = None
        if tensordesc_meta is not None and tensordesc_idx < len(tensordesc_meta):
            desc_meta = tensordesc_meta[tensordesc_idx]
        abi_expanded_args = _get_tensordesc_abi_expanded_args(rank, desc_meta)
        mapped = []
        dtype = desc_info[0]
        expanded_types.append(f"*{dtype}")
        mapped.append(len(expanded_types) - 1)
        if abi_expanded_args != 1:
            for _ in range(rank):
                expanded_types.append("i32")
                mapped.append(len(expanded_types) - 1)
            for _ in range(rank):
                expanded_types.append("i64")
                mapped.append(len(expanded_types) - 1)
        expanded_index[path] = mapped
        tensordesc_idx += 1
        return mapped

    for top_idx, ty in enumerate(signature_types):
        visit(ty, (top_idx, ))

    return expanded_types, expanded_index


def _expand_tensordesc_kernel_arg(arg, rank: int, metadata):
    if not (hasattr(arg, "base") and hasattr(arg, "shape") and hasattr(arg, "strides")):
        raise TypeError("tensor descriptor argument must provide base/shape/strides")
    shape = [int(v) for v in arg.shape]
    strides = [int(v) for v in arg.strides]
    if len(shape) != rank or len(strides) != rank:
        raise ValueError(
            f"tensor descriptor rank mismatch: expected {rank}, got shape={len(shape)} strides={len(strides)}")

    if metadata is not None and "block_size" in metadata:
        block_shape = [int(v) for v in metadata["block_size"]]
    else:
        block_shape = [int(v) for v in getattr(arg, "block_shape", ())]
    if len(block_shape) != rank:
        raise ValueError(f"tensor descriptor block rank mismatch: expected {rank}, got {len(block_shape)}")

    if metadata is not None and "elem_size" in metadata:
        elem_size = int(metadata["elem_size"])
    elif hasattr(arg.base, "element_size"):
        elem_size = int(arg.base.element_size())
    else:
        raise TypeError("cannot infer tensor descriptor element size")

    import torch
    import triton

    descriptor = torch.empty((64, ), dtype=torch.uint8, device=arg.base.device)
    fill_name = f"fill_{rank}d_tma_descriptor"
    fill_fn = getattr(triton.runtime.driver.active.utils, fill_name, None)
    if fill_fn is None:
        raise RuntimeError(f"musa driver utils missing {fill_name}")

    if rank > 5:
        raise RuntimeError(f"MUSA tensor descriptor rank {rank} is unsupported in launcher")
    fill_fn(arg.base.data_ptr(), *shape, *block_shape, elem_size, descriptor.data_ptr())

    if hasattr(torch, "musa"):
        torch.musa.synchronize()

    abi_expanded_args = _get_tensordesc_abi_expanded_args(rank, metadata)
    if abi_expanded_args == 1:
        return [descriptor], descriptor
    return [descriptor, *shape, *strides], descriptor


def _make_tensordesc_cache_key(arg, rank: int, metadata):
    base = getattr(arg, "base", None)
    if base is None or not hasattr(base, "data_ptr"):
        return None

    device = getattr(base, "device", None)
    device_type = getattr(device, "type", None)
    device_index = getattr(device, "index", None)

    try:
        shape = tuple(int(v) for v in arg.shape)
        strides = tuple(int(v) for v in arg.strides)
    except Exception:
        return None

    if metadata is not None and "block_size" in metadata:
        block_shape = tuple(int(v) for v in metadata["block_size"])
    else:
        try:
            block_shape = tuple(int(v) for v in getattr(arg, "block_shape", ()))
        except Exception:
            return None

    if metadata is not None and "elem_size" in metadata:
        elem_size = int(metadata["elem_size"])
    elif hasattr(base, "element_size"):
        elem_size = int(base.element_size())
    else:
        return None
    abi_expanded_args = _get_tensordesc_abi_expanded_args(rank, metadata)

    return (
        int(base.data_ptr()),
        device_type,
        device_index,
        shape,
        strides,
        block_shape,
        elem_size,
        int(rank),
        abi_expanded_args,
    )


def make_launcher(constants, signature, ids, warp_size):
    params = [i for i, ty in signature.items() if ty != "constexpr" and i not in constants]
    arg_decls = ', '.join(f"{ty_to_cpp_param(signature[i])} arg{i}" for i in params)

    def _parse_type(ty):
        if ty[0] == '*':
            return "PyObject*"
        if ty == "constexpr":
            # 3.5 runtime forwards constexpr Python objects in launch args.
            # They are compile-time only and should not be interpreted as C scalars.
            return "PyObject*"
        if ty in ("fp16", "bf16", "fp32", "f32", "fp64"):
            return "double"
        return ty_to_cpp_param(ty)

    def format_of(ty):
        return {
            "PyObject*": "O",
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
        }[ty]

    args_format = ''.join([format_of(_parse_type(ty)) for ty in signature.values()])
    format = "iiiKKOOOO" + args_format
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    packed_decls = []
    packed_inits = []
    launch_args = []
    for i in params:
        ty = signature[i]
        if ty[0] == "*":
            launch_args.append(f"ptr_info{i}.dev_ptr")
            continue
        if ty == "fp16":
            packed_decls.append(f"  uint16_t arg{i};")
            packed_inits.append(f"  arg{i} = pack_fp16(_arg{i});")
            launch_args.append(f"arg{i}")
        elif ty == "bf16":
            packed_decls.append(f"  uint16_t arg{i};")
            packed_inits.append(f"  arg{i} = pack_bf16(_arg{i});")
            launch_args.append(f"arg{i}")
        elif ty in ("fp32", "f32"):
            packed_decls.append(f"  float arg{i} = (float)_arg{i};")
            launch_args.append(f"arg{i}")
        else:
            launch_args.append(f"_arg{i}")

    packed_decls_src = "\n".join(packed_decls)
    packed_inits_src = "\n".join(packed_inits)

    src = f"""
#include \"musa.h\"
#include <stdbool.h>
#include <stdint.h>
#include <Python.h>
#include <dlfcn.h>

static inline uint16_t pack_fp16(double val) {{
  uint16_t result;
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 &&            \\
    !defined(PYPY_VERSION)
  _PyFloat_Pack2(val, (unsigned char *)&result, 1);
#else
  PyFloat_Pack2(val, (char *)&result, 1);
#endif
  return result;
}}

static inline uint16_t pack_bf16(double val) {{
  float f32 = (float)val;
  uint32_t u32 = *(uint32_t *)&f32;
  return (uint16_t)(u32 >> 16);
}}

static inline void gpuAssert(MUresult code, const char *file, int line)
{{
   if (code != MUSA_SUCCESS)
   {{
      const char* prefix = \"Triton Error [MUSA]: \";
      const char* str;
      muGetErrorString(code, &str);
      char err[1024] = {{0}};
      strcat(err, prefix);
      strcat(err, str);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }}
}}

#define MUSA_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

typedef MUresult (*muLaunchKernelEx_t)(const MUlaunchConfig *config, MUfunction f, void **kernelParams, void **extra);

static muLaunchKernelEx_t getLaunchKernelExHandle() {{
  void* handle = dlopen(\"libmusa.so\", RTLD_LAZY);
  if (!handle) {{
    handle = dlopen(\"libmusa.so.1\", RTLD_LAZY);
  }}
  if (!handle) {{
    PyErr_SetString(PyExc_RuntimeError, \"Failed to open libmusa.so or libmusa.so.1\");
    return NULL;
  }}
  dlerror();
  muLaunchKernelEx_t muLaunchKernelExHandle = (muLaunchKernelEx_t)dlsym(handle, \"muLaunchKernelEx\");
  const char *dlsym_error = dlerror();
  if (dlsym_error) {{
    PyErr_SetString(PyExc_RuntimeError, \"Failed to retrieve muLaunchKernelEx from libmusa\");
    return NULL;
  }}
  return muLaunchKernelExHandle;
}}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int shared_memory, MUstream stream, MUfunction function{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  MUdeviceptr global_scratch_ptr = 0;
  MUdeviceptr profile_scratch_ptr = 0;
  void *params[] = {{ {', '.join([*(f"&arg{i}" for i in params), "&global_scratch_ptr", "&profile_scratch_ptr"]) } }};
  if (gridX*gridY*gridZ > 0) {{
    if (num_ctas == 1) {{
      MUSA_CHECK(muLaunchKernel(function, gridX, gridY, gridZ, {warp_size}*num_warps, 1, 1, shared_memory, stream, params, 0));
    }} else {{
      MUlaunchAttribute launchAttr[2];
      launchAttr[0].id = MU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launchAttr[0].value.clusterDim.x = num_ctas;
      launchAttr[0].value.clusterDim.y = 1;
      launchAttr[0].value.clusterDim.z = 1;
      launchAttr[1].id = MU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      launchAttr[1].value.clusterSchedulingPolicyPreference = MU_CLUSTER_SCHEDULING_POLICY_SPREAD;
      MUlaunchConfig config;
      config.gridDimX = gridX * num_ctas;
      config.gridDimY = gridY;
      config.gridDimZ = gridZ;
      config.blockDimX = {warp_size} * num_warps;
      config.blockDimY = 1;
      config.blockDimZ = 1;
      config.sharedMemBytes = shared_memory;
      config.hStream = stream;
      config.attrs = launchAttr;
      config.numAttrs = 2;
      static muLaunchKernelEx_t muLaunchKernelExHandle = NULL;
      if (muLaunchKernelExHandle == NULL) {{
        muLaunchKernelExHandle = getLaunchKernelExHandle();
      }}
      MUSA_CHECK(muLaunchKernelExHandle(&config, function, params, 0));
    }}
  }}
}}

typedef struct _DevicePtrInfo {{
    MUdeviceptr dev_ptr;
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
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, \"data_ptr\");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, \"data_ptr method of Pointer object must return 64-bit int\");
      ptr_info.valid = false;
      Py_DECREF(ret);
      return ptr_info;
    }}
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    uint64_t dev_ptr;
    int status = muPointerGetAttribute(&dev_ptr, MU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == MUSA_ERROR_INVALID_VALUE) {{
        PyErr_Format(PyExc_ValueError,
                     \"Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)\", idx);
        ptr_info.valid = false;
    }} else if (status != MUSA_SUCCESS) {{
        MUSA_CHECK((MUresult)status);  // Catch any other musa API errors
        ptr_info.valid = false;
    }}
    ptr_info.dev_ptr = dev_ptr;
    Py_DECREF(ret);
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, \"Pointer argument must be either uint64 or have data_ptr method\");
  ptr_info.valid = false;
  return ptr_info;
}}

static void ensureMusaContext() {{
  MUcontext pctx;
  MUSA_CHECK(muCtxGetCurrent(&pctx));
  if (!pctx) {{
    // Ensure device context.
    MUdevice device;
    MUSA_CHECK(muDeviceGet(&device, 0));
    MUSA_CHECK(muDevicePrimaryCtxRetain(&pctx, device));
    MUSA_CHECK(muCtxSetCurrent(pctx));
  }}
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  // ensure musa context is valid before calling any MUSA APIs, e.g. before getPointer
  // calls muPointerGetAttributes
  ensureMusaContext();

  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{_parse_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &_stream, &_function,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  int num_warps, num_ctas, shared_memory;
  if (!PyArg_ParseTuple(kernel_metadata, \"iii\", &num_warps, &num_ctas, &shared_memory)) {{
    PyErr_SetString(PyExc_TypeError, \"kernel_metadata must be a tuple\");
    return NULL;
  }}

  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue(\"(O)\", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  {'; '.join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" for i in params if signature[i][0] == "*"])};
{packed_decls_src}
{packed_inits_src}
  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, shared_memory, (MUstream)_stream, (MUfunction)_function{', ' + ', '.join(launch_args) if len(launch_args) > 0 else ''});
  Py_END_ALLOW_THREADS;
  if (PyErr_Occurred()) {{
    return NULL;
  }}

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue(\"(O)\", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}}
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  "__triton_launcher",
  NULL,
  -1,
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


class MusaLauncher(object):

    def __init__(self, src, metadata):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()

        def cst_key(i):
            if isinstance(i, str):
                return src.fn.arg_names.index(i)
            if isinstance(i, tuple) and len(i) == 1:
                return i[0]
            return i

        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}

        ordered_sig_keys = sorted(signature.keys())
        self._signature_types = [signature[key] for key in ordered_sig_keys]
        self._has_structured_args = any(isinstance(ty, tuple) for ty in self._signature_types)
        self._has_tensordesc = any(
            _parse_tensordesc_type(ty) is not None for ty in self._walk_signature_types(self._signature_types))
        self._needs_runtime_expansion = self._has_structured_args or self._has_tensordesc
        self._tensordesc_meta = getattr(metadata, "tensordesc_meta", None)
        self._tensordesc_keepalive = []
        self._tensordesc_object_cache = OrderedDict()
        self._tensordesc_cache = OrderedDict()

        expanded_signature_types, expanded_index = _expand_signature_tree(self._signature_types, self._tensordesc_meta)
        expanded_signature = {idx: ty for idx, ty in enumerate(expanded_signature_types)}
        expanded_constants = {}
        for key, value in constants.items():
            path = _normalize_arg_path(key)
            if path not in expanded_index:
                continue
            for expanded_pos in expanded_index[path]:
                expanded_constants[expanded_pos] = value

        expanded_ids = {"ids_of_const_exprs": tuple()}
        if ids["ids_of_const_exprs"]:
            expanded_constexpr_ids = []
            for key in ids["ids_of_const_exprs"]:
                path = _normalize_arg_path(key)
                expanded_constexpr_ids.extend(expanded_index.get(path, ()))
            expanded_ids = {"ids_of_const_exprs": tuple(expanded_constexpr_ids)}

        target = getattr(metadata, "target", None)
        target_warp_size = getattr(target, "warp_size", None)
        warp_size = int(target_warp_size) if target_warp_size else 32
        src = make_launcher(expanded_constants, expanded_signature, expanded_ids, warp_size)
        mod = compile_module_from_src(
            src=src,
            name="__triton_launcher",
            include_dirs=_musa_include_dirs(),
            library_dirs=_library_dirs(),
            libraries=["musa"],
        )
        self.launch = mod.launch

    @staticmethod
    def _walk_signature_types(signature_types):
        for ty in signature_types:
            if isinstance(ty, tuple):
                yield from MusaLauncher._walk_signature_types(ty)
            else:
                yield ty

    def _expand_tensordesc_arg(self, arg, ty, tensordesc_idx):
        _, rank = _parse_tensordesc_type(ty)
        desc_meta = None
        if self._tensordesc_meta is not None and tensordesc_idx < len(self._tensordesc_meta):
            desc_meta = self._tensordesc_meta[tensordesc_idx]

        cached = None
        cache_key = _make_tensordesc_cache_key(arg, rank, desc_meta)
        object_cache_key = None
        object_ref = None
        try:
            object_ref = weakref.ref(arg)
            object_cache_key = (id(arg), cache_key)
        except TypeError:
            object_ref = None

        if object_cache_key is not None:
            cached = self._tensordesc_object_cache.get(object_cache_key)
            if cached is not None:
                cached_ref, cached_base_ptr, expanded_arg_values, keepalive = cached
                current_base = getattr(getattr(arg, "base", None), "data_ptr", None)
                current_base_ptr = int(current_base()) if current_base is not None else None
                if cached_ref() is arg and current_base_ptr == cached_base_ptr:
                    self._tensordesc_object_cache.move_to_end(object_cache_key)
                    cached = (expanded_arg_values, keepalive)
                else:
                    self._tensordesc_object_cache.pop(object_cache_key, None)
                    cached = None

        if cached is None:
            cached = self._tensordesc_cache.get(cache_key) if cache_key is not None else None
        if cached is None:
            expanded_arg_values, keepalive = _expand_tensordesc_kernel_arg(arg, rank, desc_meta)
            expanded_arg_values = tuple(expanded_arg_values)
            if object_cache_key is not None:
                self._tensordesc_object_cache[object_cache_key] = (
                    object_ref,
                    int(arg.base.data_ptr()),
                    expanded_arg_values,
                    keepalive,
                )
                self._tensordesc_object_cache.move_to_end(object_cache_key)
                if len(self._tensordesc_object_cache) > _TENSORDESC_CACHE_LIMIT:
                    self._tensordesc_object_cache.popitem(last=False)
            if cache_key is not None:
                # Reuse encoded descriptors across launches of the same tensor/view
                # so repeated TME kernels do not re-encode and synchronize every
                # descriptor argument on the host path.
                cached = (expanded_arg_values, keepalive)
                self._tensordesc_cache[cache_key] = cached
                self._tensordesc_cache.move_to_end(cache_key)
                if len(self._tensordesc_cache) > _TENSORDESC_CACHE_LIMIT:
                    self._tensordesc_cache.popitem(last=False)
        else:
            expanded_arg_values, keepalive = cached
            if cache_key is not None:
                self._tensordesc_cache.move_to_end(cache_key)
        return expanded_arg_values, keepalive

    def _expand_runtime_arg(self, arg, ty, expanded_kernel_args, launch_keepalive, tensordesc_state):
        if isinstance(ty, tuple):
            if not isinstance(arg, tuple):
                raise RuntimeError("launcher tuple argument does not match structured signature")
            if len(arg) != len(ty):
                raise RuntimeError("launcher tuple argument arity mismatch")
            for child_arg, child_ty in zip(arg, ty):
                self._expand_runtime_arg(child_arg, child_ty, expanded_kernel_args, launch_keepalive, tensordesc_state)
            return

        desc_info = _parse_tensordesc_type(ty)
        if desc_info is None:
            expanded_kernel_args.append(arg)
            return

        expanded_arg_values, keepalive = self._expand_tensordesc_arg(arg, ty, tensordesc_state[0])
        tensordesc_state[0] += 1
        expanded_kernel_args.extend(expanded_arg_values)
        launch_keepalive.append(keepalive)

    def __call__(self, *args, **kwargs):
        if not self._needs_runtime_expansion:
            self.launch(*args, **kwargs)
            return

        # launch(gridX, gridY, gridZ, stream, function, kernel_metadata,
        # launch_metadata, launch_enter_hook, launch_exit_hook, *kernel_args)
        launch_prefix = args[:9]
        kernel_args = args[9:]
        if len(kernel_args) != len(self._signature_types):
            raise RuntimeError("launcher argument count mismatch while expanding tensor descriptors")

        expanded_kernel_args = []
        launch_keepalive = []
        tensordesc_state = [0]
        for arg, ty in zip(kernel_args, self._signature_types):
            self._expand_runtime_arg(arg, ty, expanded_kernel_args, launch_keepalive, tensordesc_state)

        self._tensordesc_keepalive.extend(launch_keepalive)
        if len(self._tensordesc_keepalive) > 4096:
            self._tensordesc_keepalive = self._tensordesc_keepalive[-4096:]
        self.launch(*launch_prefix, *expanded_kernel_args, **kwargs)


class MusaDriver(DriverBase):

    def __init__(self):
        self.utils = MusaUtils()
        self.launcher_cls = MusaLauncher
        import torch
        if not hasattr(torch, "musa"):
            raise RuntimeError("torch.musa is not available")
        self._torch = torch

    @staticmethod
    def is_active():
        try:
            import torch
            return hasattr(torch, "musa") and torch.musa.is_available()
        except Exception:
            return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_current_target(self):
        arch = knobs.runtime.override_arch or os.getenv("TRITON_MUSA_ARCH") or "ph1"
        warp_size = 32
        return GPUTarget("musa", arch, warp_size)

    def get_active_torch_device(self):
        return self._torch.device("musa", self.get_current_device())

    def get_current_device(self):
        return self._torch.musa.current_device()

    def set_current_device(self, device):
        self._torch.musa.set_device(device)

    def get_current_stream(self, device):
        stream = self._torch.musa.current_stream(device)
        return getattr(stream, "musa_stream", getattr(stream, "cuda_stream", stream))

    def get_device_interface(self):
        return self._torch.musa

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        cache_size = 256 * 1024 * 1024
        return self._torch.empty(int(cache_size // 4), dtype=self._torch.int, device="musa")

    def clear_cache(self, cache):
        cache.zero_()
