from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager

from pathlib import Path
import os
import tempfile
import hashlib


def _get_dfcart_include_path() -> str:
    return "/opt/thrive/include"


def _get_dfcart_lib_path() -> str:
    return "/opt/thrive/lib"


dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [os.path.join(dirname, "include"), _get_dfcart_include_path()]
library_dirs = [_get_dfcart_lib_path()]
libraries = ["stdc++", "dfcart"]
ccflags = []


def compile_module_from_src(src, name):
    key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    always_compile = os.environ.get("TRITON_ALWAYS_COMPILE", "0") == "1"
    if always_compile or cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.cpp")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, library_dirs, include_dirs, libraries, ccflags)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------ Launcher ------------------------


def ty_to_cpp(ty):
    if ty[0] == "*":
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
        "fp16": "double",
        "bf16": "double",
        "fp32": "double",
        "f32": "double",
        "fp64": "double",
    }[ty]


def make_launcher(constants, signature):

    def _flatten_signature(sig, output):
        if isinstance(sig, tuple):
            for x in sig:
                _flatten_signature(x, output)
        else:
            output.append(sig)

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty == "constexpr":
            return "PyObject*"
        if ty[0] == "*":
            return "PyObject*"
        return ty_to_cpp(ty)

    def _pack_arg(i, ty):
        if ty[0] == "*":
            return f"k_args.emplace_back(ptr_info{i}.dev_ptr, ptr_info{i}.nbytes, ptr_info{i}.element_size, VALUE_KIND_POINTER);"
        elif ty == "fp16":
            return f"uint16_t packed{i} = pack_fp16(arg{i}); k_args.emplace_back(&packed{i}, sizeof(packed{i}), VALUE_KIND_BYVALUE);"
        elif ty == "bf16":
            return f"uint16_t packed{i} = pack_bf16(arg{i}); k_args.emplace_back(&packed{i}, sizeof(packed{i}), VALUE_KIND_BYVALUE);"
        elif ty in ("fp32", "f32"):
            return f"uint32_t packed{i} = pack_fp32(arg{i}); k_args.emplace_back(&packed{i}, sizeof(packed{i}), VALUE_KIND_BYVALUE);"
        else:
            return f"k_args.emplace_back(&arg{i}, sizeof(arg{i}), VALUE_KIND_BYVALUE);"

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty == "constexpr":
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
            "char*": "s",
        }[ty_to_cpp(ty)]

    args_format = "".join([format_of(ty) for ty in signature.values()])
    format = "iiiKKOOOO" + args_format

    flat_signature = []
    for sig in signature.values():
        _flatten_signature(sig, flat_signature)
    signature = {i: s for i, s in enumerate(flat_signature)}

    args_list = ', ' + ', '.join(f"&arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''
    params = {i: ty for i, ty in signature.items() if ty != "constexpr"}

    from triton.runtime.driver import driver

    # Generate glue code
    newline = "\n  "
    src = f"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdio.h>
#include <string>
#include <vector>
#include <memory>
#include "driver.hpp"

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

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

typedef struct _DevicePtrInfo {{
  void* dev_ptr;
  size_t nbytes;
  size_t element_size;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;

  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(obj);
    return ptr_info;
  }}

  if (obj == Py_None) {{
    return ptr_info;
  }}

  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "Calling the get method must return a value of type int64");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(ret);
    if(!ptr_info.dev_ptr) {{
      return ptr_info;
    }}
    Py_DECREF(ret);
    return ptr_info;
  }}

  PyErr_SetString(PyExc_TypeError, "Failed to parse the pointer object");
  ptr_info.valid = false;
  return ptr_info;
}}

static void run_kernels(TTThriveStream stream, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                        TTThriveFunction kernel, uint32_t num_cores, int use_dshmem, int fw_preempt,
                        const std::vector<Arg_t>& args) {{
  size_t N = gridX * gridY * gridZ;
  unsigned pe_cap = {driver.active.get_device_capability()[-2]};
  unsigned num_blk = gridX*gridY*gridZ;
  if (fw_preempt == 0) {{
    num_blk = num_blk > pe_cap? pe_cap : num_blk;
  }}
  TTThriveDim3 cluster_dim = {{num_blk, 1, 1}};
  TTThriveDim3 block_dim = {{num_cores, 1, 1}};
  auto gridArgs = args;
  gridArgs.emplace_back(&gridX, 4, VALUE_KIND_BYVALUE);
  gridArgs.emplace_back(&gridY, 4, VALUE_KIND_BYVALUE);
  gridArgs.emplace_back(&gridZ, 4, VALUE_KIND_BYVALUE);
  launchAsync(kernel, cluster_dim, block_dim, gridArgs, 0, stream, use_dshmem != 0);
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  int64_t py_obj_stream;
  int64_t p_krnl;

  {' '.join([f"{_extracted_type(ty)} arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &py_obj_stream, &p_krnl,
                       &kernel_metadata, &launch_metadata, &launch_enter_hook,
                       &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  int num_cores, use_dshmem, fw_preempt;
  if (!PyArg_ParseTuple(kernel_metadata, \"iii\", &num_cores, &use_dshmem, &fw_preempt)) {{
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple of (num_cores, use_dshmem, fw_preempt)");
    return NULL;
  }}

  // Extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  TTThriveFunction ker = (TTThriveFunction)p_krnl;
  std::vector<Arg_t> k_args;
  {newline.join([f"DevicePtrInfo ptr_info{i} = getPointer(arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "// byvalue" for i, ty in signature.items()])};
  {newline.join([_pack_arg(i, ty) for i, ty in params.items()])};
  run_kernels((TTThriveStream)py_obj_stream, gridX, gridY, gridZ, ker, num_cores, use_dshmem, fw_preempt, k_args);

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  if (PyErr_Occurred()) {{
    return NULL;
  }}

  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_thrive_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_thrive_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


class ThriveLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        src = make_launcher(constants, signature)
        mod = compile_module_from_src(src, "__triton_thrive_launcher")
        self.launch = mod.launch

    @staticmethod
    def is_torch_tensor(obj):
        import torch
        return isinstance(obj, torch.Tensor)

    @staticmethod
    def check_args(*args):
        import torch
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor) and arg.device.type == 'cpu':
                raise RuntimeError(f"""Argument {i} is a CPU tensor,
                    but Thrive tensor is required""")

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        from triton.runtime.driver import driver
        if stream is None:
            device = driver.active.get_current_device()
            stream = driver.active.get_current_stream(device)
        self.check_args(*args)
        return self.launch(gridX, gridY, gridZ, stream, function, *args)


# ------------------------ Utils ------------------------


class ThriveUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(ThriveUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        mod = compile_module_from_src(Path(os.path.join(dirname, "driver.c")).read_text(), "thrive_utils")
        self._load_binary = mod.load_binary
        self.get_device_capability = mod.get_device_capability
        self.get_sram_used_bytes = mod.get_sram_used_bytes

    def load_binary(self, *arg):
        res = self._load_binary(*arg)
        return res

    @staticmethod
    def get_device_properties(device):
        # Get the properties by runtime api
        return {"max_shared_mem": 0}


# ------------------------ Driver ------------------------


class ThriveDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = ThriveUtils()
        self.launcher_cls = ThriveLauncher
        self.ctx = None
        self.capability = {}

    @staticmethod
    def is_active():
        try:
            import torch
            return torch.thrive.is_available()
        except ImportError:
            return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_device_capability(self):
        device = self.get_current_device()
        if device not in self.capability:
            import torch
            pe_cnt = 8  # FIXME: get pe_cnt from properties
            p = torch.thrive.get_device_properties(device)
            self.capability[device] = (0, 0, p.end_x() - p.start_x(), p.end_y() - p.start_y(), pe_cnt, 4)
        # cap_major/cap_minor/dies_x/dies_y/pes/cores
        return self.capability[device]

    def get_current_device(self):
        import torch
        return torch.thrive.current_device()

    def get_current_stream(self, device):
        import torch
        return torch.thrive.current_stream(device).raw_stream()

    def set_current_device(self, device):
        import torch
        return torch.thrive.set_device(device)

    def get_current_target(self):
        return GPUTarget("thrive", 0, 0)

    def get_active_torch_device(self):
        import torch
        return torch.device("thrive", self.get_current_device())

    def get_device_interface(self):
        import torch
        return torch.thrive

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch
        cache_size = 512 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device="thrive")

    def clear_cache(self, cache):
        cache.zero_()
