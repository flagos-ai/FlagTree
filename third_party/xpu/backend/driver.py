import os
import sys
import hashlib
import tempfile
import functools
from typing import List
from pathlib import Path

from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver

dirname = os.path.dirname(os.path.realpath(__file__))
arch = int(os.environ.get('TRITON_XPU_ARCH', '3'))
# 因为这里是还没有进入到compile阶段的一个判定，然后xpu4及以上不走这个target，所以就暂时仍使用xpu3的头文件和链接库，之后会在compile阶段再次判定
if arch >= 4:
    arch = 3
include_dir = [os.path.join(dirname, f"xpu{arch}", "include")]
libdevice_dir = os.path.join(dirname, f"xpu{arch}", "lib")
library_dir = os.path.join(dirname, f"xpu{arch}", "so")
if (os.path.exists(os.path.join(library_dir, "libLaunch_shared.a"))
        or os.path.exists(os.path.join(library_dir, "liblaunch_shared.so"))):
    libraries = ['launch', 'xpurt', 'launch_shared']
else:
    libraries = ['launch', 'xpurt']


def get_xpu_spec(xpu_arch, is_sdnn=False):
    """
    `is_sdnn=False`: return a tuple represents (num_clusters, num_cores)

    `is_sdnn=True`: return a tuple represents (num_sdnns, num_cores)
    """
    if xpu_arch == 2:
        return (8, 8) if is_sdnn else (8, 64)
    elif xpu_arch == 3:
        return (12, 8) if is_sdnn else (12, 64)
    elif xpu_arch == 4:
        return (6, 8) if is_sdnn else (12, 64)
    else:
        raise RuntimeError(f"Unknown XPU architecture: {xpu_arch}")


@functools.lru_cache()
def library_dirs():
    return [libdevice_dir, library_dir]


def compile_module_from_src(src, name):
    key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, library_dirs(), include_dir, libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)

    # check  compiled library cache
    if not os.path.exists(cache_path):
        raise RuntimeError(f"Compiled library not found: {cache_path}")

    if not os.access(cache_path, os.R_OK):
        raise RuntimeError(f"Compiled library not readable: {cache_path}")

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, cache_path)
        if spec is None:
            raise RuntimeError(f"Could not create module spec from {cache_path}")
        mod = importlib.util.module_from_spec(spec)

        # try import compiled module
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            # 检查共享库依赖
            import subprocess
            result = subprocess.run(['ldd', cache_path], capture_output=True, text=True)
            print(f"Shared library dependencies for {cache_path}:")
            print(result.stdout)
            if result.stderr:
                print(f"ldd errors: {result.stderr}")
            raise e

        return mod
    except Exception as e:
        print(f"Error loading module {name} from {cache_path}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        raise


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void *"
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


def make_launcher(constants, signature, ids, metadata):
    xpu_arch = metadata.xpu_arch
    is_sdnn = metadata.is_sdnn
    ewtable = metadata.ewtable
    tensor_args = metadata.tensor_args
    kernel_name = metadata.name

    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors, such as tensor descriptors for CUDA.
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" + (f", int64_t arg{i}_numel" if ty[0] == "*" else "")
                          for i, ty in signature.items())

    def _extracted_type(ty):
        if ty[0] == '*':
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
            "int64_t": "l",  # TODO[dyq]: L?
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty]

    def generate_kernel_params(signature, constants, type_mapping=None):
        params = []
        if type_mapping is None:
            type_mapping = {
                '*fp32': 1,
                '*f64': 2,
                '*int32': 3,
                '*int64': 4,
                'bool': 5,
                '*fp16': 6,
                '*bf16': 7,
                'int32': 8,
                'int64': 9,
                'int8': 10,
                'int16': 11,
            }

        for i, ty in signature.items():
            if i in constants:
                continue
            type_enum = type_mapping.get(ty, 0)
            params.append(f"{{{i}l, (int64_t)&arg{i}, (int64_t)(sizeof(arg{i})), "
                          f"{f'arg{i}_numel' if i in tensor_args else '0l'}, "
                          f"{type_enum}l}}")
        return f"{{ {', '.join(params) + (', ' if params else ' ')}{{0l, 0l, 0l, 0l, 0l}} }}"

    def generate_const_params(constants):
        params = []
        for i, val in constants.items():
            if isinstance(val, bool):
                params.append(f"{{{i}l, 5, {1 if val else 0}l}}")
            elif isinstance(val, int):
                params.append(f"{{{i}l, 4, {val}l}}")
        return f"{{ {', '.join(params) + (', ' if params else ' ')}{{0l, 0l, 0l}} }}"

    # drop type info
    args_format = ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])
    format = "iiiKKOOOO" + args_format

    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    def read_data_to_hexstr(file_name):
        if not file_name:
            return ""
        with open(file_name, 'rb') as f:
            data = f.read()
            hex_lines = []
            for i in range(0, len(data), 128):
                chunk = data[i:i + 128]
                hex_string = ','.join(f'0x{byte:02x}' for byte in chunk)
                hex_lines.append(hex_string)
        return ',\n    '.join(hex_lines)

    # generate glue code
    src = f"""
#include <xpu/runtime.h>
#include <stdbool.h>
#include <Python.h>
#include <dlfcn.h>

// XPU_SPEC_START
static inline void xpuAssert(int code, const char *file, int line,
                             const char *call)
{{
   if (code != XPU_SUCCESS)
   {{
      const char* err_msg = xpu_strerror(code);
      char buf[1024] = {{0}};
      sprintf(buf, "%s:%d: %s -> %s(err_code: %d)",
              file, line, call, err_msg, code);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, buf);
      PyGILState_Release(gil_state);
   }}
}}

#define XPU_CHECK(ans) {{ xpuAssert((ans), __FILE__, __LINE__, #ans); }}

enum {{
  kINVALID = 0,
  kL3,
  kGM
}};

static inline int xpu2PointerCheck(void *ptr) {{
  unsigned int ptr_high = (((unsigned long long) ptr) >> 32);
  unsigned int ptr_low = (((unsigned long long) ptr));
  if (ptr_high == 0 && ptr_low >= 0xC0000000 && ptr_low <= 0xC3FFFFFF) {{
      return kL3;
  }}
  if (ptr_high >= 8 && ptr_high <= 15) {{
      return kGM;
  }}
  printf("ptr_high = %u\\n", ptr_high);
  printf("ptr_low = %u\\n", ptr_low);
  return kINVALID;
}}

static inline int xpu3PointerCheck(void *ptr) {{
  // TODO: do it for XPU3.
  return kGM;
}}

static inline int xpu4PointerCheck(void *ptr) {{
  // TODO: do it for XPU4.
  return kGM;
}}

unsigned char ewtable[] = {{ {read_data_to_hexstr(ewtable)} }};

int xpuLaunchKernel(const char *kernel_name, void *func, int gridX, int gridY, int gridZ, int ncluster,
                    int ncores, void *stream, void **kernelParams, void **kernelConsts, void **extra);

static void _launch(int gridX, int gridY, int gridZ, int clusterDimX, int clusterDimY, int clusterDimZ, XPUStream stream, XPUFunc function{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  if (gridX*gridY*gridZ > 0) {{
    // printf("gridX: %d, gridY: %d, gridZ: %d\\n", gridX, gridY, gridZ);
    int nclusters = {get_xpu_spec(xpu_arch, is_sdnn)[0]};
    int ncores = {get_xpu_spec(xpu_arch, is_sdnn)[1]};
    int64_t kernel_params[][5] = {generate_kernel_params(signature, constants)};
    int64_t kernel_consts[][3] = {generate_const_params(constants)};
    void *extra_data[] = {{ {"&ewtable[0]" if ewtable else "nullptr"} }};

    // asm("int3");
    XPU_CHECK(xpuLaunchKernel(\"{kernel_name}\", function, gridX, gridY, gridZ, nclusters, ncores, stream, (void **)&kernel_params[0][0], (void **)&kernel_consts[0][0], &extra_data[0]));
  }}
}}
// XPU_SPEC_END

typedef struct _DevicePtrInfo {{
    void *dev_ptr;
    int64_t numel;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = PyLong_AsVoidPtr(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  PyObject *len = PyObject_GetAttrString(obj, "numel");
  if(ptr && len){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret_ptr = PyObject_Call(ptr, empty_tuple, NULL);
    PyObject *ret_len = PyObject_Call(len, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    Py_DECREF(len);
    if (!PyLong_Check(ret_ptr)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    if (!PyLong_Check(ret_len)) {{
      PyErr_SetString(PyExc_TypeError, "numel method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = PyLong_AsVoidPtr(ret_ptr);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    void *dev_ptr = PyLong_AsVoidPtr(ret_ptr);
    int64_t numel = PyLong_AsLong(ret_len);
    if (xpu{xpu_arch}PointerCheck(dev_ptr) == kINVALID) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }}
    ptr_info.dev_ptr = dev_ptr;
    ptr_info.numel = numel;
    Py_DECREF(ret_ptr);  // Thanks ChatGPT!
    Py_DECREF(ret_len);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr and numel method");
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

  int clusterDimX, clusterDimY, clusterDimZ;
  if (!PyArg_ParseTuple(kernel_metadata, \"iii\", &clusterDimX, &clusterDimY, &clusterDimZ)) {{
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
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
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, clusterDimX, clusterDimY, clusterDimZ, (XPUStream)_stream, (XPUFunc)_function{', ' + ', '.join(f"ptr_info{i}.dev_ptr, ptr_info{i}.numel" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items()) if len(signature) > 0 else ''});
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
  {{"launch", (PyCFunction)launch, METH_VARARGS | METH_KEYWORDS, "Entry point for all kernels with this signature"}},
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


class XPUUtils(object):
    procedure: List[int] = [0, 0, 0]

    def __init__(self):
        mod = compile_module_from_src(Path(os.path.join(dirname, "driver.c")).read_text(), "xpu_utils")
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


class XPULauncher(object):

    def __init__(self, src, metadata):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        src = make_launcher(constants, signature, ids, metadata)
        mod = compile_module_from_src(src, "__triton_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class XPUDriver(GPUDriver):

    def __init__(self):
        self.utils = XPUUtils()
        self.launcher_cls = XPULauncher
        super().__init__()

    @staticmethod
    def is_active():
        return True

    def get_current_target(self):
        arch = int(os.environ.get('TRITON_XPU_ARCH', '3'))
        warp_size = 1  # we don't have warp
        return GPUTarget("xpu", arch, warp_size)
