#
# Copyright 2024 Enflame. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import re
import shutil
import sysconfig
import functools
import tempfile
from pathlib import Path
import hashlib
import subprocess

from triton.backends.enflame import toolkit
from triton.backends.enflame.toolkit import *
from triton.backends.enflame.filecache import get_cache_manager

from typing import Dict
from types import ModuleType

from triton.runtime.errors import OutOfResources


@functools.lru_cache()
def _version_key():
    target = os.path.join(os.path.dirname(__file__), "VERSION")
    if not os.path.isfile(target):
        target = __file__
    with open(target, "rb") as f:
        content = f.read()
    try:
        caps_version = subprocess.check_output(["topscc", "--version"], stderr=subprocess.STDOUT)
    except Exception:
        caps_version = b""
    return hashlib.sha256(content + caps_version).hexdigest()


def _make_so_cache_key(version_hash, signature, constants, **kwargs):
    # Get unique key for the compiled code
    signature = {k: 'ptr' if v[0] == '*' else v for k, v in signature.items()}
    key = f"{version_hash}-{''.join(signature.values())}-{constants}"
    for kw in kwargs:
        key = f"{key}-{kwargs.get(kw)}"
    key = hashlib.md5(key.encode("utf-8")).hexdigest()
    return key


#################################################################################
# below for gcu

# gcu kernel translation


def _get_topscc_root():
    return os.getenv("CAPS_PATH", "/opt/tops")


def _kernel_to_fatbin(kernel: str, arch: int, enable_transform: bool):
    # print(kernel)
    with tempfile.TemporaryDirectory() as tmpdir:
        bin = os.path.join(tmpdir, "kernel.fatbin")
        toolkit.compile(kernel, "--device-only", f"--arch=gcu{arch}", f"--output={bin}",
                        "--enable-transform" if enable_transform else "")
        with open(bin, "rb") as f:
            return f.read()


def build_gcu_ext(name, src, srcdir, extra_objects=[], extra_libraries=[]):
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    local_lib_path = os.path.join(toolkit.datadir, 'lib')

    cc = os.environ.get("CC")
    if cc is None:
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cc = gcc if gcc is not None else clang
        if cc is None:
            raise RuntimeError("Failed to find C compiler. Please specify via CC environment variable.")

    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]

    library_dirs = [os.path.join(_get_topscc_root(), "lib"), local_lib_path]
    include_dirs = [os.path.join(_get_topscc_root(), "include"), srcdir, py_include_dir]
    libraries = ["topsrt"] + extra_libraries

    cc_cmd = [cc, src, "-O3", "-shared", "-fPIC", "-w", "-o", so]
    cc_cmd += extra_objects
    cc_cmd += [f'-l{lib}' for lib in libraries]
    cc_cmd += [f"-L{dir}" for dir in library_dirs]
    cc_cmd += [f"-I{dir}" for dir in include_dirs]
    cc_cmd += [f"-Wl,-rpath={local_lib_path}"]
    subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)
    return so


#
# GCU
#
class GCUUtils(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(GCUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        utilsdir = os.path.join(toolkit.datadir, "utils")
        src = Path(os.path.join(utilsdir, "gcu.cpp")).read_text()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "gcu_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "main.cpp")
                with open(src_path, "w") as f:
                    f.write(src)
                so = build_gcu_ext("gcu_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location("gcu_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int32_t",
        "u1": "uint32_t",
        "i8": "int8_t",
        "u8": "uint8_t",
        "i16": "int16_t",
        "u16": "uint16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "double",
        "f16": "double",
        "bf16": "double",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
        "index": "int64_t",
    }[ty]


FLOAT_STORAGE_TYPE = {
    "f16": "uint16_t",
    "fp16": "uint16_t",
    "bf16": "uint16_t",
}

FLOAT_PACK_FUNCTION = {
    "f16": "pack_fp16",
    "fp16": "pack_fp16",
    "bf16": "pack_bf16",
}


def _extracted_type(ty):
    if ty[0] == '*':
        return "PyObject*"
    if ty[0] in ("constexpr"):
        return "PyObject*"
    return ty_to_cpp(ty)


def format_of(ty):
    return {
        "PyObject*": "O",
        "float": "f",
        "double": "d",
        "long": "l",
        "int8_t": "b",
        "int16_t": "h",
        "int32_t": "i",
        "int64_t": "l",
        "uint8_t": "B",
        "uint16_t": "H",
        "uint32_t": "I",
        "uint64_t": "K",
    }[ty]


_BASE_ARGS_FORMAT = "iiiKKOOOO"
_BASE_ARGS_FORMAT_LEN = len(_BASE_ARGS_FORMAT)


def _expand_signature(signature_values):
    """Expand tensordesc entries in the signature to match add_rewrite_tensor_descriptor_to_pointer.

    Each tensordesc<dtype[d0, d1, ...]> becomes:
      *dtype, shape(i64)*N, stride(i64)*N, padding(i1), shape(i32)*N, stride(i64)*N

    This mirrors the NVIDIA 3.6.0 meta=None path in _expand_signature."""
    output = []
    for sig in signature_values:
        if isinstance(sig, str) and sig.startswith("tensordesc"):
            match = re.match(r"tensordesc<([^[>]*)\[([^\]]*)\]", sig)
            if match:
                dtype = match.group(1)
                ndim = match.group(2).count(",") + 1
            else:
                dtype = "i8"
                ndim = 1
            output.append("*" + dtype)
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


def make_tensordesc_arg(arg):
    """Decompose a TensorDescriptor to the flat list expected by the launcher.
    Matches the NVIDIA 3.6.0 meta=None path in make_tensordesc_arg."""
    return [arg.base, *arg.shape, *arg.strides, arg.padding == "nan", *arg.shape, *arg.strides]


def wrap_handle_tensordesc(launcher, signature):
    """Wrap a launcher to decompose TensorDescriptor arguments at call time.
    Matches the NVIDIA 3.6.0 wrap_handle_tensordesc interface."""
    has_tensor_desc_arg = any(isinstance(sig, str) and sig.startswith("tensordesc") for sig in signature.values())
    if not has_tensor_desc_arg:
        return launcher

    tensordesc_indices = set(
        [i for i, sig in enumerate(signature.values()) if isinstance(sig, str) and sig.startswith("tensordesc")])

    def inner(*args, **kwargs):
        final_args = list(args[:_BASE_ARGS_FORMAT_LEN])
        for i, arg in enumerate(args[_BASE_ARGS_FORMAT_LEN:]):
            if i in tensordesc_indices:
                final_args.extend(make_tensordesc_arg(arg))
            else:
                final_args.append(arg)
        return launcher(*final_args, **kwargs)

    return inner


def generate_launcher(constants, signature, arch='gcu300', no_constant_args=False, redundant_sip=False):
    start_desc = len(signature)

    # Remap constants indices and expand tensordesc entries in the signature.
    old_keys = list(signature.keys())
    expanded = _expand_signature(signature.values())
    idx_map = {}
    new_idx = 0
    for orig_idx, ty in zip(old_keys, list(signature.values())):
        if isinstance(ty, str) and ty.startswith("tensordesc"):
            match = re.match(r"tensordesc<[^[>]*\[([^\]]*)\]", ty)
            ndim = (match.group(1).count(",") + 1) if match else 1
            new_idx += 1 + 2 * ndim + 1 + 2 * ndim
        else:
            idx_map[orig_idx] = new_idx
            new_idx += 1
    remapped_constants = {}
    for k, v in constants.items():
        if k in idx_map:
            remapped_constants[idx_map[k]] = v
    constants = remapped_constants
    signature = {i: s for i, s in enumerate(expanded)}

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
        elif ty != "constexpr":
            internal_args_list.append(f"_arg{i}")
    newline = '\n '
    ptr_decls = [
        f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;"
        for i, ty in signature.items()
        if ty[0] == "*"
    ]
    float_storage_decls = [
        f"{FLOAT_STORAGE_TYPE[ty]} _arg{i}_storage = {FLOAT_PACK_FUNCTION[ty]}(_arg{i});"
        for i, ty in signature.items()
        if ty in FLOAT_STORAGE_TYPE
    ]
    args_format = ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])
    format = _BASE_ARGS_FORMAT + args_format
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    params = range(len(signature))
    params = [f"&arg{i}" for i, ty in signature.items() if ty != "constexpr"]

    nonconst_args_size = 0
    for i, ty in signature.items():
        if ty != "constexpr":
            nonconst_args_size += 1

    # GCU 400/410: validate cluster dims at launch time.
    #   Supports 2D clusters: clusterDimX * clusterDimY <= 6, clusterDimZ == 1.
    #   cluster-internal CTA sharing (like NVIDIA DSMEM) is supported.
    #   Cooperative grid launch (grid-level barrier) is supported via
    #   topsLaunchAttributeCooperative on topsModuleLaunchKernelEx.
    #   (clusterDimX, clusterDimY, clusterDimZ) maps to config.blockDim.
    cluster_check_str = ''
    if 'gcu400' == arch or 'gcu410' == arch:
        cluster_check_str = """
      if (clusterDimZ != 1) {
        PyErr_Format(PyExc_ValueError,
          "GCU 400/410 does not support 3-D cluster: "
          "cluster_dims=(%d,%d,%d), clusterDimZ must be 1.",
          clusterDimX, clusterDimY, clusterDimZ);
        return NULL;
      }
      int cluster_total = clusterDimX * clusterDimY;
      if (cluster_total < 1 || cluster_total > 6) {
        PyErr_Format(PyExc_ValueError,
          "GCU 400/410 cluster size clusterDimX*clusterDimY = %d*%d = %d exceeds limit. "
          "Must be between 1 and 6 (max CTAs sharing local memory within a cluster).",
          clusterDimX, clusterDimY, cluster_total);
        return NULL;
      }"""

    # generate glue code
    launch_str = ''
    if 'gcu400' == arch or 'gcu410' == arch:
        if redundant_sip:
            launch_str += f"""topsLaunchConfig_t l_config;
      memset(&l_config, 0x0, sizeof(l_config));
      l_config.gridDim = dim3(gridX, gridY, gridZ);
      l_config.blockDim = dim3(1, 1, 1);
      struct topsLaunchAttribute att[2];
      att[0].id = topsLaunchAttributeThreadDimension;
      att[0].val.ThreadDim.x = num_warps;
      att[0].val.ThreadDim.y = 1;
      att[0].val.ThreadDim.z = 1;
      att[1].id = topsLaunchAttributeSchedulingPolicy;
      att[1].val.SchedulingPolicyPreference = topsSchedulingPolicyRedundant;
      l_config.attrs = att;
      l_config.numAttrs = 2;
      l_config.stream = stream;
      TOPS_CHECK(topsModuleLaunchKernelEx(&l_config, function, params, NULL));"""
        else:
            launch_str += f"""topsLaunchAttribute attrs[2];
      int num_attrs = 0;
      attrs[num_attrs].id = topsLaunchAttributeThreadDimension;
      attrs[num_attrs].val.ThreadDim = {{(unsigned int)num_warps, 1, 1}};
      ++num_attrs;

      if (launch_cooperative_grid != 0) {{
        attrs[num_attrs].id = topsLaunchAttributeCooperative;
        attrs[num_attrs].val.cooperative = 1;
        ++num_attrs;
      }}

      topsLaunchConfig_t config = {{}};
      config.gridDim = {{(unsigned int)gridX, (unsigned int)gridY, (unsigned int)gridZ}};
      config.blockDim = {{(unsigned int)clusterDimX, (unsigned int)clusterDimY, (unsigned int)clusterDimZ}};
      config.dynamicSmemBytes = shared_memory;
      config.attrs = attrs;
      config.numAttrs = num_attrs;
      config.stream = stream;
      TOPS_CHECK(topsModuleLaunchKernelEx(&config, function, params, NULL));"""
    elif 'gcu500' == arch:
        launch_str += 'TOPS_CHECK(topsModuleLaunchKernel(function, gridX, gridY, gridZ, num_warps * 128, 1, 1, shared_memory, stream, params, 0));'
    else:
        launch_str += 'TOPS_CHECK(topsModuleLaunchKernel(function, gridX, gridY, gridZ, num_warps, 1, 1, shared_memory, stream, params, 0));'
    src = f"""
#include <stdbool.h>
#include <Python.h>
#include <dlfcn.h>
#include <tops/tops_runtime.h>

static inline void gcuAssert(topsError_t code, const char *file, int line)
{{
   if (code != TOPS_SUCCESS)
   {{
      const char* prefix = "Kurama Error [TOPS]: ";
      const char* str = topsGetErrorString(code);
      char err[1024] = {{0}};
      strcat(err, prefix);
      strcat(err, str);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }}
}}

#define TOPS_CHECK(ans) {{ gcuAssert((ans), __FILE__, __LINE__); }}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int launch_cooperative_grid, int shared_memory, topsStream_t stream, topsFunction_t function{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  void *params[] = {{ {', '.join(params)} }};
  if (gridX*gridY*gridZ > 0) {{
      {launch_str}
  }}
}}

typedef struct _DevicePtrInfo {{
    topsDeviceptr_t dev_ptr;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (topsDeviceptr_t)PyLong_AsUnsignedLongLong(obj);
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
    ptr_info.dev_ptr = (topsDeviceptr_t)PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    uint64_t dev_ptr;
    int status = topsPointerGetAttribute(&dev_ptr, TOPS_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == TOPS_ERROR_INVALID_VALUE) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Kurama (cpu tensor?)", idx);
        ptr_info.valid = false;
    }}
    // ptr_info.dev_ptr = dev_ptr;
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
    // Python 3.6-3.11: _PyFloat_Pack2 expects unsigned char*
    // Python 3.12+: PyFloat_Pack2 expects char*
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 && !defined(PYPY_VERSION)
    _PyFloat_Pack2(f, (unsigned char*)&result, 1);
#else
    PyFloat_Pack2(f, (char*)&result, 1);
#endif
    return result;
}}

static uint16_t pack_bf16(double f) {{
    float f32 = (float)f;
    uint32_t u32 = *(uint32_t*)&f32;
    return (uint16_t)(u32 >> 16);
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {newline.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &_stream, &_function,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  // extract kernel metadata
  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  int launch_cooperative_grid = 0;
  if (!PyTuple_Check(kernel_metadata)) {{
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
    return NULL;
  }}
  Py_ssize_t kernel_metadata_size = PyTuple_Size(kernel_metadata);
  if (kernel_metadata_size == 7) {{
    if (!PyArg_ParseTuple(kernel_metadata, \"iiiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ, &launch_cooperative_grid)) {{
      return NULL;
    }}
  }} else if (kernel_metadata_size == 6) {{
    if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
      return NULL;
    }}
  }} else {{
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must contain 6 or 7 integers");
    return NULL;
  }}
  {cluster_check_str}
  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}


  // raise exception asap
  {newline.join(ptr_decls)}
  {newline.join(float_storage_decls)}
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, launch_cooperative_grid, shared_memory, (topsStream_t)_stream, (topsFunction_t)_function{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  if(PyErr_Occurred()) {{
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
  \"__triton_launcher\",
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


def compile_module_from_src(src, name):
    """Compile C++ source to a Python extension module with caching."""
    key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.cpp")
            with open(src_path, "w") as f:
                f.write(src)
            so = build_gcu_ext(name, src_path, tmpdir)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class GcuLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        redundant_sip = getattr(metadata, 'redundant_sip', False)
        src = generate_launcher(constants, signature, metadata.arch, redundant_sip=redundant_sip)
        mod = compile_module_from_src(src, "__triton_launcher")
        self.launch = wrap_handle_tensordesc(mod.launch, signature)

    def __call__(self, *args, **kwargs):
        grid_0 = args[0]
        grid_1 = args[1]
        grid_2 = args[2]
        if grid_0 > 65535:
            raise OutOfResources(grid_0, 65535, "grid.x")
        if grid_1 > 255:
            raise OutOfResources(grid_1, 255, "grid.y")
        if grid_2 > 255:
            raise OutOfResources(grid_2, 255, "grid.z")
        self.launch(*args, **kwargs)


class GCUDriver(object):

    def __init__(self):
        if os.getenv("COMPILE_ARCH"):
            self.arch = os.getenv("COMPILE_ARCH")
            self.utils = GCUUtils()
            self.get_current_stream = lambda idx=0: 0
            self.get_current_device = lambda idx=0: 0
            self.launcher_cls = GcuLauncher
            return
        else:
            self.arch = ""
        self.utils = GCUUtils()
        import torch
        self.get_current_stream = lambda idx: torch.gcu.current_stream(idx).gcu_stream
        self.get_current_device = lambda: torch.device(f"{device_name}:{torch.gcu.current_device()}").index
        self.launcher_cls = GcuLauncher

    def get_device_properties(self, device):
        if self.arch == "gcu300":
            return {
                'max_shared_mem': 67108864, 'multiprocessor_count': 2, 'max_threads_per_block': 12, 'sm_clock_rate':
                1416000, 'mem_clock_rate': 7000000, 'mem_bus_width': 384, 'version': 300
            }
        elif self.arch == "gcu400":
            return {
                'max_shared_mem': 67108864, 'multiprocessor_count': 2, 'max_threads_per_block': 12, 'sm_clock_rate':
                1416000, 'mem_clock_rate': 7000000, 'mem_bus_width': 384, 'version': 400
            }
        elif self.arch == "gcu500":
            return {
                'max_shared_mem': 67108864, 'multiprocessor_count': 4, 'max_threads_per_block': 1024, 'sm_clock_rate':
                1800000, 'mem_clock_rate': 9000000, 'mem_bus_width': 512, 'version': 500
            }
        props = self.utils.get_device_properties(device)
        props["version"] = int(props["arch_name"].split('-')[-1][3:])
        del props["arch_name"]
        return props

    def get_stream(self, idx=None):
        if self.arch in ("gcu300", "gcu400", "gcu500"):
            return 0
        if idx is None:
            idx = self.get_current_device()
        try:
            import torch
            return torch.gcu.current_stream(idx).gcu_stream
        except:
            return 0

    def get_arch(self):
        if self.arch == "gcu300":
            return "dtu-enflame-tops--gcu300"
        elif self.arch == "gcu400":
            return "dtu-enflame-tops--gcu400"
        elif self.arch == "gcu500":
            return "efgcu-enflame-tops--gcu500"
        device = self.get_current_device()
        device_properties = self.utils.get_device_properties(device)
        arch = device_properties['arch_name']
        return arch

    def get_warp_size(self):
        if self.arch == "gcu300":
            return 12
        elif self.arch == "gcu400":
            return 8
        elif self.arch == "gcu500":
            return 128
        device = self.get_current_device()
        device_properties = self.utils.get_device_properties(device)
        warp_size = device_properties['max_threads_per_block']
        return warp_size

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench


class GCUBackend(object):

    def __init__(self) -> None:
        self.driver = GCUDriver()

    def load_dialects(self, ctx):
        try:
            import triton._C.libtriton as _lt
            if hasattr(_lt, "distributed"):
                _lt.distributed.ir.load_dialects(ctx)
        except Exception:
            pass

    @functools.lru_cache()
    def hash(self):
        return f'{self.get_architecture_descriptor()}'

    def get_architecture_descriptor(self, **kwargs):
        device = self.driver.get_current_device()
        device_properties = self.driver.get_device_properties(device)
        capability = {
            "max_threads_per_block": device_properties["max_threads_per_block"], "multiprocessor_count":
            device_properties["multiprocessor_count"], "version": device_properties["version"], "max_shared_mem":
            device_properties["max_shared_mem"]
        }
        return capability

    def compile_kernel(self, name, kernel, enable_transform, signature, constants):
        arch = self.get_architecture_descriptor()
        kernel_key = f'{name}-{hashlib.md5(str(kernel).encode("utf-8")).hexdigest()}'
        cache_key = f"kernel-{arch['version']}-{_version_key()}"
        cache_manager = get_cache_manager(cache_key)
        cache_path = cache_manager.get_file(kernel_key)
        if cache_path is None:
            bin = _kernel_to_fatbin(kernel, arch["version"], enable_transform)
            return cache_manager.put(bin, kernel_key, binary=True)
        else:
            return cache_path

    def get_arch_name(self):
        return f"gcu{self.get_architecture_descriptor()['version']}"

    def get_num_clusters(self):
        return self.get_architecture_descriptor()['multiprocessor_count']

    def get_num_processors(self):
        return self.get_architecture_descriptor()['max_threads_per_block']

    def compile(self, name, kernel, enable_transform=False, signature={}, constants=[]):
        kernel_path = self.compile_kernel(name, kernel, enable_transform, signature, constants)
        with open(kernel_path, "rb") as binary:
            bin = binary.read()
            m, func, _, _ = self.get_load_binary_fn()(name, bin, 0, self.get_current_device())
            assert func != 0, "cannot find kenrel function"
            launcher_path = self.make_launcher_stub(name, signature, constants, True)
            import importlib.util
            spec = importlib.util.spec_from_file_location("__triton_launcher", launcher_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return Kernel(name, m, func, mod.launch, constants)

    def get_version_key(self):
        return _version_key()

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.gcu import libdevice
        return {"triton.language.extra.libdevice": libdevice}


""" never use

driver = GCUBackend()

class Kernel(object):

    def __init__(self, name, mod, func, launcher, constants):
        self.name = name
        self.mod = mod
        self.func = func
        self.launcher = launcher
        self.constants = constants

    def __call__(self, *args, gridX=1, gridY=1, gridZ=1, blockX=1):
        arch = driver.get_architecture_descriptor()
        self.launcher(
            gridX, gridY, gridZ, blockX,
            #0, 0, 0, 0,
            arch["max_shared_mem"],
            driver.get_stream(), self.func,
            None, None, None, *args)

    def __getitem__(self, dims):
        blockX, *gridXYZ = dims
        gridX = gridXYZ[0] if len(gridXYZ) >= 1 else 1
        gridY = gridXYZ[1] if len(gridXYZ) >= 2 else 1
        gridZ = gridXYZ[2] if len(gridXYZ) >= 3 else 1
        def launcher(*args):
            self.__call__(*args,
                          gridX=gridX, gridY=gridY, gridZ=gridZ,
                          blockX=blockX)
        return launcher

"""
