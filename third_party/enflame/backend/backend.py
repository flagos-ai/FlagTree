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
import sysconfig
import functools
import tempfile
from pathlib import Path
import hashlib

import setuptools
from setuptools import Extension

import torch
from triton.backends.enflame.toolkit import *
from triton.backends.enflame.filecache import get_cache_manager

import importlib.metadata
import site

from typing import Dict
from types import ModuleType


@functools.lru_cache()
def _version_key():
    contents = []
    # frontend
    with open(__file__, "rb") as f:
        contents += [hashlib.md5(f.read()).hexdigest()]
    return '-'.join(contents)


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
    print(kernel)
    with tempfile.TemporaryDirectory() as tmpdir:
        bin = os.path.join(tmpdir, "kernel.fatbin")
        toolkit.compile(kernel, "--device-only", f"--arch=gcu{arch}", f"--output={bin}",
                        "--enable-transform" if enable_transform else "")
        with open(bin, "rb") as f:
            return f.read()


def build_gcu_ext(name, src, srcdir, extra_objects=[], extra_libraries=[]):

    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    local_lib_path = os.path.join(datadir, 'lib')

    # fallback on setuptools
    extra_compile_args = ['-w']
    library_dirs = [os.path.join(_get_topscc_root(), "lib"), local_lib_path]
    include_dirs = [os.path.join(_get_topscc_root(), "include")]
    link_args = [f"-Wl,-rpath={local_lib_path}"]
    libraries = ["topsrt"] + extra_libraries
    define_macros = []

    # create extension module
    ext = Extension(name, [src], extra_objects=extra_objects, extra_compile_args=extra_compile_args,
                    include_dirs=include_dirs, library_dirs=library_dirs, libraries=libraries,
                    define_macros=define_macros, extra_link_args=link_args)

    args = ['build_ext']
    args.append('--build-temp=' + srcdir)
    args.append('--build-lib=' + srcdir)
    args = dict(
        name=name,
        ext_modules=[ext],
        script_args=args,
    )
    setuptools.setup(**args)
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
        utilsdir = os.path.join(datadir, "utils")
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
        "fp16": "float",
        "f16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
        "index": "int64_t",
    }[ty]


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


def generate_launcher(constants, signature, arch='gcu300', no_constant_args=False):
    start_desc = len(signature)
    #signature = generate_cu_signature(constants, signature, ids)
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items() if ty != "constexpr")
    args_format = ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])
    format = "iiiKKOOOO" + args_format
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    params = range(len(signature))
    params = [f"&arg{i}" for i, ty in signature.items() if ty != "constexpr"]

    nonconst_args_size = 0
    for i, ty in signature.items():
        if ty != "constexpr":
            nonconst_args_size += 1

    # generate glue code
    launch_str = ''
    if 'gcu400' == arch or 'gcu410' == arch:
        launch_str += f"""topsLaunchConfig_t config;
      memset(&config, 0x0, sizeof(config));
      config.gridDim = dim3(gridX, gridY, gridZ);
      config.blockDim = dim3(1, 1, 1);
      auto att = (struct topsLaunchAttribute *)malloc(
        sizeof(struct topsLaunchAttribute));
      att->id = topsLaunchAttributeThreadDimension;
      att->val.ThreadDim.x = num_warps;
      att->val.ThreadDim.y = 1;
      att->val.ThreadDim.z = 1;
      config.attrs = att;
      config.numAttrs = 1;
      config.stream = stream;
      TOPS_CHECK(topsModuleLaunchKernelEx(&config, function, params, NULL));"""
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

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, topsStream_t stream, topsFunction_t function{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  void *params[] = {{ {', '.join(params)} }};
  if (gridX*gridY*gridZ > 0) {{
      //printf("xxx %d %d %d\\n", gridX, gridY, gridZ);
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

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &_stream, &_function,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  // extract kernel metadata
  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
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


  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items() if ty != "constexpr"])};
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (topsStream_t)_stream, (topsFunction_t)_function{', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items() if ty != "constexpr") if nonconst_args_size > 0 else ''});

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
        src = generate_launcher(constants, signature, metadata.arch)
        mod = compile_module_from_src(src, "__triton_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class GCUDriver(object):

    def __init__(self):
        self.utils = GCUUtils()
        self.get_current_stream = lambda idx: torch.gcu.current_stream(idx).gcu_stream
        self.get_current_device = lambda: torch.device(f"{device_name}:{torch.gcu.current_device()}").index
        self.launcher_cls = GcuLauncher

    def get_device_properties(self, device):
        props = self.utils.get_device_properties(device)
        props["version"] = int(props["arch_name"].split('-')[-1][3:])
        del props["arch_name"]
        return props

    def get_stream(self, idx=None):
        if idx is None:
            idx = self.get_current_device()
        try:
            return torch.gcu.current_stream(idx).gcu_stream
        except:
            return 0

    def get_arch(self):
        device = self.get_current_device()
        device_properties = self.utils.get_device_properties(device)
        arch = device_properties['arch_name']
        return arch

    def get_warp_size(self):
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
        self.launcher(gridX, gridY, gridZ, blockX,
                      #0, 0, 0, 0,
                      arch["max_shared_mem"], driver.get_stream(), self.func, None, None, None, *args)

    def __getitem__(self, dims):
        blockX, *gridXYZ = dims
        gridX = gridXYZ[0] if len(gridXYZ) >= 1 else 1
        gridY = gridXYZ[1] if len(gridXYZ) >= 2 else 1
        gridZ = gridXYZ[2] if len(gridXYZ) >= 3 else 1

        def launcher(*args):
            self.__call__(*args, gridX=gridX, gridY=gridY, gridZ=gridZ, blockX=blockX)

        return launcher
