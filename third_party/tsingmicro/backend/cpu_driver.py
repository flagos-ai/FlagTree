import hashlib
import tempfile
import sysconfig

import os, subprocess, tempfile
import importlib.util
import sysconfig

from pathlib import Path

from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget


# The riscv compiler
def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_SYSPATH", "")
    if path == "":
        raise Exception("LLVM_SYSPATH is not set.")
    return os.path.join(path, "bin", bin_name)


# The riscv c header files and libraries path.
def _get_libc_root() -> str:
    path = os.getenv("LIB_C_ROOT", "")
    if path == "":
        raise Exception("LIB_C_ROOT is not set.")
    return path


# -------------------- Launcher ----------------------------
def _ty_to_cpp(ty):
    if ty[0] == '*':
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


def _extracted_type(ty):
    if ty[0] == '*':
        return "PyObject*"
    return _ty_to_cpp(ty)


def _format_of(ty):
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


def _generate_launcher(constants, signature, kernel_name):
    arg_decls = ', '.join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    args_format = ''.join([_format_of(_extracted_type(ty)) for ty in signature.values()])
    format = "iiiOOOO" + args_format
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    kernel_arg_decls = ', '.join(
        _ty_to_cpp(ty) if ty[0] != "*" else f"int64_t, void*" for i, ty in signature.items() if i not in constants)
    kernel_arg_decls += ', ' if kernel_arg_decls else ''

    kernel_parameters = ', '.join(f"static_cast<{_ty_to_cpp(ty)}>(arg{i})" if ty[0] != "*" else f"0, &ptr_arg{i}"
                                  for i, ty in signature.items()
                                  if i not in constants)
    kernel_parameters += ', ' if kernel_parameters else ''

    return f"""
#include <assert.h>
#include <stdbool.h>
#include <Python.h>
#include "ExecutionEngine/CRunnerUtils.h"
#include "ExecutionEngine/CRunnerUtils.cpp"

extern "C" {{
  // Pointer type (=Memref) becomes int64_t + MemRef struct
  // FIXME: understand what this int64_t is used for.
  void {kernel_name}({kernel_arg_decls}
                       int, int, int, int, int, int);
}}

static void _launch(int gridX, int gridY, int gridZ, {arg_decls}) {{
  if (gridX*gridY*gridZ > 0) {{
    // Cast "function" to the real function type.
    for(int x = 0; x < gridX; x++) {{
      for(int y = 0; y < gridY; y++) {{
        for(int z = 0; z < gridZ; z++) {{
          // Use some random type "char" here.
          {' '.join(f'StridedMemRefType<char, 0> ptr_arg{i} = {{static_cast<char *>(arg{i}), static_cast<char *>(arg{i}), 0}};' for i, ty in signature.items() if i not in constants and ty[0] == "*")}
          {kernel_name}({kernel_parameters}
                        gridX, gridY, gridZ, x, y, z);
        }}
      }}
    }}
  }}
}}

typedef struct _DevicePtrInfo {{
  void *dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
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
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  // [CPULauncher-specific]: We don't need the metadata below but just put them
  // here anyway to be consistent with others.
  // This will make updating the driver easier in the future.

  //  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  //  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
  //    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
  //    return NULL;
  //  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});

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
  \"__ztc_ref_cpu_kernel_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___ztc_ref_cpu_kernel_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""


def compile_module(launcher_src, kernel_placeholder_name):
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    py_lib_dir = sysconfig.get_config_var("LIBDIR")
    py_version = sysconfig.get_config_var("LDVERSION")
    py_lib = '{name}{py_version}'.format(name="python", py_version=py_version)
    cpu_backend_path = Path(__file__).resolve().parent
    clang = _get_llvm_bin_path("clang++")
    libc_inc = os.path.join(_get_libc_root(), "riscv64-unknown-elf", "include")
    libc_lib = os.path.join(_get_libc_root(), "riscv64-unknown-elf", "lib")
    include_dir = os.path.join(cpu_backend_path, "include")

    def launch(gridX, gridY, gridZ, stream, cu_function, kernel_metadata, launch_metadata, launch_enter_hook,
               launch_exit_hook, *args):
        # Unlike CUDA/HIP, we cannot easily pass function pointer across different pybind libraries.
        # Let's compile a kernel every time.
        # The cu_function parameter actually contains our assembly source code.
        # See CPUUtils.load_binary method.
        asm_src = cu_function
        kernel_name = kernel_metadata[6]  # see pack_metadata in compiler.py
        src = launcher_src.replace(kernel_placeholder_name, kernel_name)

        key = hashlib.md5(src.encode("utf-8") + asm_src).hexdigest()
        cache = get_cache_manager(key)
        name = "__ztc_ref_cpu_kernel_launcher"
        filename = f"{name}.so"
        cache_path = cache.get_file(filename)

        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                asm_src_path = os.path.join(tmpdir, "kernel.s")
                launcher_src_path = os.path.join(tmpdir, "main.cxx")
                so_path = os.path.join(tmpdir, "kernel.so")
                Path(asm_src_path).write_bytes(asm_src)
                Path(launcher_src_path).write_text(src)
                # Compile it together.
                subprocess.check_call([
                    clang, "-std=c++17", "--target=riscv64-unknown-elf", launcher_src_path, asm_src_path,
                    f"-I{libc_inc}", f"-I{py_include_dir}", f"-I{include_dir}", f"-I{libc_lib}", f"-L{py_lib_dir}",
                    "-shared", f"-l{py_lib}", "-fPIC", "-o", so_path
                ])

                with open(so_path, "rb") as f:
                    cache_path = cache.put(f.read(), filename, binary=True)

        # Load and launch the compiled kernel.
        spec = importlib.util.spec_from_file_location(name, cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.launch(gridX, gridY, gridZ, kernel_metadata, launch_metadata, launch_enter_hook, launch_exit_hook,
                          *args)

    return launch


class CPULauncher(object):

    def __init__(self, src, metadata):
        kernel_placeholder_name = "KERNEL_NAME_PLACEHOLDER"

        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        launcher_src = _generate_launcher(constants, signature, kernel_placeholder_name)
        # Later KERNEL_NAME_PLACEHOLDER will be used to assign the kernel name
        # in the following launch function.
        self.launch = compile_module(launcher_src, kernel_placeholder_name)

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class CPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    # Note:
    # nvidia and amd backends have their corresponding driver.c file that exposes
    # get_device_properties and load_binary using python bindings.
    # (see third_party/nvidia/backend/driver.c)
    # These methods are then used in compiler.py to initialize handles before running
    # the triton kernels.
    # Since we recompile the kernel every time (see compile_module above),
    # and the metadata generated by these functions aren't applicable to the cpu
    # backend, just define the same functions with dummy implementation.
    @staticmethod
    def get_device_properties(device):
        return {
            "max_shared_mem": 2**20, "multiprocessor_count": None, "sm_clock_rate": None, "mem_clock_rate": None,
            "mem_bus_width": None
        }

    # Important note:
    # Since we cannot easy pass function pointers around, we pass along the
    # assembly source code so that compile_module above can recompile the
    # module every time.
    @staticmethod
    def load_binary(name, kernel_asm, shared, device):
        return (None,  # module
                kernel_asm,  # function
                None,  # n_regs
                None  # n_spills
                )


class CPUDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        self.binary_ext = "cpuasm"

    # CPU driver won't be automatically chosen unless explicitly set through
    # triton.runtime.driver.set_active(CPUDriver())
    @staticmethod
    def is_active():
        return False

    def get_device_capability(self):
        return ("cpu", 0)

    def get_current_stream(self, device):
        return None

    def get_current_device(self):
        # CPU doesn't have a device to return. Return something.
        return "cpu"

    def set_current_device(self, device):
        # CPU doesn't have a device to set
        assert device == "cpu"
        return

    def get_current_target(self):
        return GPUTarget("cpu", 0, 0)

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args
