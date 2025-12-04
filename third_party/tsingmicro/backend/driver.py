#
# This file implements the triton kernel driver interfaces where are used in
# triton/python/triton/compiler/compiler.py.
# For how the interface in driver class is used, see the implementation of the
# file above.
#
import hashlib
import tempfile
import os
import subprocess
import importlib.util
import shutil
import sysconfig
import atexit
from pathlib import Path
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import GPUDriver
from triton.backends.compiler import GPUTarget


def _get_tx8_deps_path(sub_name: str) -> str:
    path = os.getenv("TX8_DEPS_ROOT", "")
    if path == "":
        raise Exception("TX8_DEPS_ROOT is not set.")
    return os.path.join(path, sub_name)


def _is_use_profile():
    return os.getenv("USE_PROFILE", "").strip() == "1"


dirname = os.path.dirname(os.path.realpath(__file__))
if (os.getenv("USE_SIM_MODE", "0").lower() in ("1", "true", "yes")):
    scheme = sysconfig.get_default_scheme()
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]

    include_dirs = [_get_tx8_deps_path("include"), py_include_dir]
    library_dirs = [_get_tx8_deps_path("lib")]
    libraries = ["triton_cmodel", "tx8be_op_cmodel", "neuralcore_qemu"]
else:
    include_dirs = [
        os.path.join(dirname, "include"),
        _get_tx8_deps_path("include"),
        os.path.join(sysconfig.get_path('platlib'), "pybind11", "include"),
        os.path.join(sysconfig.get_path('platlib'), "torch", "include"),
        os.path.join(sysconfig.get_path('platlib'), "torch", "include", "torch", "csrc", "api", "include"),
        os.path.join(sysconfig.get_path('platlib'), "numpy", "_core", "include")
    ]
    library_dirs = [
        os.path.join(dirname, "lib"),
        _get_tx8_deps_path("lib"),
        os.path.join(sysconfig.get_path('platlib'), "torch", "lib")
    ]
    libraries = ['tx8_runtime', 'torch', 'torch_cpu', 'torch_python', 'c10']


def extend_torch():
    import torch
    from torch.utils import cpp_extension, rename_privateuse1_backend, generate_methods_for_privateuse1_backend
    cflags = []
    ldflags = ["-L" + os.path.realpath(_get_tx8_deps_path("lib")), "-ltx8_runtime"]
    if _is_use_profile():
        cflags.append("-DUSE_PROFILE")
        ldflags.append("-lprofiler_x86")

    module = cpp_extension.load(
        name="txda",
        sources=[os.path.dirname(__file__) + "/txda_device.cpp"],
        extra_include_paths=[os.path.realpath(_get_tx8_deps_path("include"))],
        extra_ldflags=ldflags,
        extra_cflags=cflags,
        verbose=True,
    )
    torch.utils.rename_privateuse1_backend("txda")
    torch._register_device_module("txda", module)
    generate_methods_for_privateuse1_backend(for_storage=True)


def _dump_ir_if_needed(files):
    path = os.getenv("TRITON_DUMP_PATH", "")
    if not path:
        return

    os.makedirs(path, exist_ok=True)
    for f in files:
        shutil.copy(f, os.path.join(path, os.path.basename(f)))


def _build(name, src, srcdir, library_dirs, include_dirs, libraries):
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    # try to avoid setuptools if possible
    cc = os.environ.get("CC")
    if cc is None:
        # TODO: support more things here.
        clang = shutil.which("clang")
        cc = clang
        if cc is None:
            raise RuntimeError("Failed to find C compiler. Please specify via CC environment variable.")
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
    custom_backend_dirs = set(os.getenv(var) for var in ('TRITON_CUDACRT_PATH', 'TRITON_CUDART_PATH'))
    include_dirs = include_dirs + [srcdir, py_include_dir, *custom_backend_dirs]
    # for -Wno-psabi, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111047
    cc_cmd = [cc, src, "-O3", "-shared", "-fPIC", "-std=c++17", "-Wno-psabi", "-o", so]
    if _is_use_profile():
        cc_cmd += ["-DUSE_PROFILE"]
    cc_cmd += [f'-l{lib}' for lib in libraries]
    if _is_use_profile():
        cc_cmd += ["-lprofiler_x86"]
    cc_cmd += [f"-L{dir}" for dir in library_dirs]
    cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
    subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)
    return so


# Build a native ELF on the platform running this python script
def compile_native(src, name):
    fname = "native_" + name
    key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{fname}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, f"{name}.cpp")
            with open(src_path, "w") as f:
                f.write(src)
                f.flush()
                _dump_ir_if_needed([src_path])
            so = _build(name, src_path, tmpdir, library_dirs, include_dirs, libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{fname}.so", binary=True)
                _dump_ir_if_needed([cache_path])
    else:
        print("cache_path: ", cache_path, flush=True)

    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    if isinstance(ty, tuple):
        val = ','.join(map(_extracted_type, ty))
        return f"[{val}]"
    if ty[0] == '*':
        return "PyObject*"
    if ty == "constexpr":
        return "PyObject*"
    return _ty_to_cpp(ty)


def _format_of(ty):
    if isinstance(ty, tuple):
        val = ''.join(map(_format_of, ty))
        return f"({val})"
    if ty[0] == '*':
        return "O"
    if ty in ("constexpr"):
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
    }[_ty_to_cpp(ty)]


def make_launcher(constants, signature, kernel_name, kernel_path):
    # Basic declarations. Arguments in triton kernel.
    arg_decls = ', '.join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items() if ty != "constexpr")
    args_format = ''.join([_format_of(ty) for ty in signature.values()])
    format = "iiiOKOOOO" + args_format
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    # Parameters to pass to the kernel function. Arguments in triton kernel except constants.
    kernel_arg_decls = ', '.join(
        f"{_ty_to_cpp(ty)} arg{i}" if ty[0] != "*" else f"uint64_t tx81_ptr{i}, {_ty_to_cpp(ty)} ptr_arg{i}"
        for i, ty in signature.items()
        if ty != "constexpr")
    kernel_arg_decls += ', ' if kernel_arg_decls else ''

    kernel_parameters = ', '.join(
        f"static_cast<{_ty_to_cpp(ty)}>(arg{i})" if ty[0] != "*" else f"tx81_ptr{i}, ptr_arg{i}"
        for i, ty in signature.items()
        if ty != "constexpr")
    kernel_parameters += ', ' if kernel_parameters else ''

    # Simulation or hardware
    if (os.getenv("USE_SIM_MODE", "0").lower() in ("1", "true", "yes")):
        # generate glue code for tile-sim
        return f"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdio.h>
#include <string>
#include <memory>
#include <map>
#include "common_base.h"
#include "instr_def.h"
#include "common_tensor.h"
#include "cmodel.h"


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

using kernel_ptr_t = void(*)({kernel_arg_decls}int, int, int, int, int, int);

inline std::string getStringEnv(const std::string &env, std::string defaultVal = "") {{
  const char *s = std::getenv(env.c_str());
  if (!s)
    return defaultVal;
  std::string str(s);
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) {{ return std::tolower(c); }});
  return str;
}}

static void _launch(int gridX, int gridY, int gridZ, {kernel_arg_decls}kernel_ptr_t kernel_ptr) {{
    if (gridX*gridY*gridZ <= 0)
        return;  // No work to do

    // Cast "function" to the real function type.
    for (uint32_t z = 0; z < gridZ; ++z) {{
        for (uint32_t y = 0; y < gridY; ++y) {{
            for (uint32_t x = 0; x < gridX; ++x) {{
                __set_pid(x, y, z);
                (*kernel_ptr)({kernel_parameters}gridX, gridY, gridZ, x, y, z);
            }}
        }}
    }}
}}


typedef struct _DevicePtrInfo {{
    void* dev_ptr;
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
        ptr_info.dev_ptr = (void*) PyLong_AsLongLong(ret);
        if(!ptr_info.dev_ptr) {{
        return ptr_info;
        }}
        Py_DECREF(ret);  // Thanks ChatGPT!
        return ptr_info;
    }}
    PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
    ptr_info.valid = false;
    return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
    std::map<std::string, TileSimLogLevel> TileSimLogLevelMap = {{
        {{"none",   RCESIM_LOG_NONE}},
        {{"info",   RCESIM_LOG_INFO}},
        {{"debug",  RCESIM_LOG_DEBUG}},
        {{"banner", RCESIM_LOG_BANNER}},
        {{"warn",   RCESIM_LOG_WARN}},
        {{"error",  RCESIM_LOG_ERROR}},
        {{"fatal",  RCESIM_LOG_FATAL}}
    }};


    auto str = getStringEnv("SIM_LOG_LEVEL", "fatal");
    TileSimLogLevel log_level = RCESIM_LOG_FATAL;
    if (TileSimLogLevelMap.find(str) != TileSimLogLevelMap.end())
        log_level = TileSimLogLevelMap[str];

    TileSimHandle *sim_handle = q_tilesim_create(log_level);
    set_sim_handle(sim_handle, NULL);
    FILE *fp = fopen("aa.log", "w");
    if (fp == NULL) {{
        perror("fopen failed\\n");
        exit(-1);
    }}
    fclose(fp);

    q_tilesim_set_logFile(sim_handle, "aa.log");

    int gridX, gridY, gridZ;
    PyObject *launch_enter_hook = NULL;
    PyObject *launch_exit_hook = NULL;
    PyObject *kernel_metadata = NULL;
    PyObject *launch_metadata = NULL;

    PyObject * py_obj_stream = NULL;
    void * pKrnl = NULL;

    {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}

    if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &py_obj_stream, &pKrnl,
                                        &kernel_metadata, &launch_metadata,
                                        &launch_enter_hook, &launch_exit_hook
                                        {args_list})) {{
        return NULL;
    }}

    // FIXME: Steam is PyNone
    // void *pStream = PyLong_AsVoidPtr(py_obj_stream);
    kernel_ptr_t kernel_ptr = reinterpret_cast<kernel_ptr_t>((PyObject*)pKrnl);

    // extract launch metadata
    if (launch_enter_hook != Py_None){{
        PyObject* args = Py_BuildValue("(O)", launch_metadata);
        PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
        Py_DECREF(args);
        if (!ret)
        return NULL;
    }}

    {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items() if ty != "constexpr"])};

    _launch(gridX, gridY, gridZ, {', '.join(f"0, ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items() if ty != "constexpr")} {',' if len(kernel_parameters) > 0  else ''} kernel_ptr);


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

    # generate glue code for tx8 board
    return f"""
#include <assert.h>
#include <stdbool.h>
#include <Python.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/numpy.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#include <numpy/arrayobject.h>
#include <stdint.h>
#include <vector>
#include <memory>
#include <string>
#include <filesystem>
#include "hrt_interface.h"
#include "hrt_common.h"
#include "profiler.h"

enum DATA_TYPE {{
    SCALAR,
    POINT,
}};

// A kernel argument
struct KernelArg {{
    // The actual kernel argument: tensor or scalar
    union Data {{
        void* ptr;        // Pointer to the tensor data
        uint64_t scalar;  // Scalar data
    }} data;
    size_t size;  // The size of the kernel argument
    int data_type;

    KernelArg(void *ptr, size_t s) : size(s) {{
        data.ptr = ptr;
        data_type = POINT;
    }}

    KernelArg(uint64_t v, size_t s) : size(0) {{
        data.scalar = v;
        data_type = SCALAR;
    }}

}};

// 释放的时候打印profiler数据
auto g_guard = std::shared_ptr<void>(
    nullptr,
    [](void*) {{
        PROFILE_CALL(printProfileAll);
        printf("guard release.\\n");
    }}
);


// FIXME: Hardcoded path
std::string chip_out = "/tmp/chip_out/node0/";
std::string kernel_file = "{kernel_path}";
std::string kernel_fun_name = "{kernel_name}";
uint32_t sharedMemBytes = 0;
TsmDevice* device;

typedef void* Stream_t;

static void _launch(int gridX, int gridY, int gridZ, std::vector<KernelArg> kargs) {{
    if (gridX*gridY*gridZ <= 0) {{
        return;  // No work to do
    }}

    // TODO::mv
    uint64_t kernel_len = 0;
    uint8_t* kernel_ptr = read_file_data(kernel_file, kernel_len);
    if (kernel_ptr == nullptr) {{
        PyErr_SetString(PyExc_RuntimeError, "Failed to read kernel so");
        TsmDeInitRuntime();
        return;
    }}

    // Allocate the device memory for all kernel arguments
    std::vector<uint64_t> rtKargs;
    for (KernelArg& karg : kargs) {{
        if (karg.data_type == POINT) {{
            rtKargs.push_back(1);
            rtKargs.push_back((uint64_t)(karg.data.ptr));
        }} else {{
            rtKargs.push_back((uint64_t)(karg.data.ptr));
        }}
    }}
    rtKargs.push_back(gridX);
    rtKargs.push_back(gridY);
    rtKargs.push_back(gridZ);
    rtKargs.push_back(0);
    rtKargs.push_back(0);
    rtKargs.push_back(0);

    // TSM_RETCODE TsmKernelLaunch(TsmDevice *dev, const char *func_name, uint64_t kernel_host, uint64_t kernel_len,
    // Dim3 grid_dim, Dim3 block_dim, void *args, uint32_t args_len);
    uint32_t eventId = EVENT_INIT;
    PROFILE_CALL(addOrderPorfile, TIME_RUNTIME, TIME_LAUNCH_START, &eventId);
    if (TsmKernelLaunch(device, kernel_fun_name.c_str(), (uint64_t)kernel_ptr, kernel_len,
        Dim3({{(uint32_t)gridX, (uint32_t)gridY, (uint32_t)gridZ}}), Dim3({{1u, 1u, 1u}}),
        (void*)(&rtKargs[0]), rtKargs.size()*sizeof(uint64_t)) != RET_SUCCESS){{
        PyErr_SetString(PyExc_RuntimeError, "Failed to TsmKernelLaunch");
        TsmDeInitRuntime();
    }}
    PROFILE_CALL(addOrderPorfile, TIME_RUNTIME, TIME_LAUNCH_END, &eventId);
}}

// Structure to represent a device pointer
typedef struct _DevicePtrInfo {{
    void *dev_ptr;
    bool valid;
    size_t size;
}} DevicePtrInfo;

// Function to get tensor size using untyped_storage if available
static inline size_t getTensorSize(PyObject *obj) {{
    // First try to get size via untyped_storage attribute (newer PyTorch versions)


    // Final fallback: calculate size from numel() * element_size()
    PyObject *numel_method = PyObject_GetAttrString(obj, "numel");
    PyObject *element_size_method = PyObject_GetAttrString(obj, "element_size");

    if (numel_method && element_size_method) {{
        printf("============= has numel_method and element_size_method ==============\\n");
        fflush(stdout);
        PyObject *empty_tuple1 = PyTuple_New(0);
        PyObject *empty_tuple2 = PyTuple_New(0);
        PyObject *numel_obj = PyObject_Call(numel_method, empty_tuple1, NULL);
        PyObject *element_size_obj = PyObject_Call(element_size_method, empty_tuple2, NULL);

        Py_DECREF(empty_tuple1);
        Py_DECREF(empty_tuple2);
        Py_DECREF(numel_method);
        Py_DECREF(element_size_method);

        if (numel_obj && element_size_obj && PyLong_Check(numel_obj) && PyLong_Check(element_size_obj)) {{
            size_t numel = (size_t)PyLong_AsLongLong(numel_obj);
            size_t element_size = (size_t)PyLong_AsLongLong(element_size_obj);
            size_t total_size = numel * element_size;

            printf("============= numel size: %ld\\n", total_size);
            Py_DECREF(numel_obj);
            Py_DECREF(element_size_obj);
            return total_size;
        }}

        if (numel_obj) Py_DECREF(numel_obj);
        if (element_size_obj) Py_DECREF(element_size_obj);
    }} else {{
        if (numel_method) Py_DECREF(numel_method);
        if (element_size_method) Py_DECREF(element_size_method);
    }}

    printf("==== zero size ========\\n");
    fflush(stdout);
    return 0;  // Return 0 if unable to determine size
}}

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
    DevicePtrInfo ptr_info;
    ptr_info.dev_ptr = 0;
    ptr_info.valid = true;
    ptr_info.size = 0;  // Initialize size

    printf("idx: %d, PyObject : %p \\n", idx, obj);
    fflush(stdout);
    if (PyLong_Check(obj)) {{
        ptr_info.dev_ptr = (void*) PyLong_AsLongLong(obj);
        printf("PyLong_AsLongLong %p\\n", ptr_info.dev_ptr);
        return ptr_info;
    }}

    if (obj == Py_None) {{
        // valid nullptr

        printf("Py_None\\n");
        fflush(stdout);
        return ptr_info;
    }}


    PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
    if(ptr){{
        printf("PyObject_GetAttrString\\n");
        fflush(stdout);

        PyObject *empty_tuple = PyTuple_New(0);
        PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
        Py_DECREF(empty_tuple);
        Py_DECREF(ptr);
        if (!PyLong_Check(ret)) {{
            PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
            ptr_info.valid = false;
            printf("data_ptr method of Pointer object must return 64-bit int\\n");
            fflush(stdout);
            return ptr_info;
        }}
        ptr_info.dev_ptr = (void*) PyLong_AsLongLong(ret);
        printf("============= ptr_info.dev_ptr: %p\\n",  ptr_info.dev_ptr);
        if(!ptr_info.dev_ptr) {{
            printf("ptr_info.dev_ptr null\\n");
            fflush(stdout);
            return ptr_info;
        }}
        Py_DECREF(ret);  // Thanks ChatGPT!

        // Get tensor size using the new function
        ptr_info.size = getTensorSize(obj);
        fflush(stdout);

        return ptr_info;
    }}
    PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
    ptr_info.valid = false;
    return ptr_info;
}}


static size_t getTensorStorageSize(PyObject* tensor_obj) {{
    const at::Tensor& tensor = THPVariable_Unpack(tensor_obj);
    printf("========== total size ================: %ld\\n", tensor.storage().nbytes());
    return tensor.storage().nbytes();
}}

// Extract tensor raw ptr
static void* extractTensor(PyObject* tensor_obj) {{
    const at::Tensor& tensor = THPVariable_Unpack(tensor_obj);
    torch::Tensor contiguous_tensor = tensor.contiguous();
    printf("========== ptr ================: %p\\n", contiguous_tensor.data_ptr());
    return contiguous_tensor.data_ptr();
}}

static PyObject* get_device_ptr(PyObject* self, PyObject* args) {{
    uint64_t dev_ptr;
    if (!PyArg_ParseTuple(args, "K", &dev_ptr)) {{
        return NULL;
    }}
    device = (TsmDevice *)dev_ptr;
    return Py_None;
}}

static PyObject* release(PyObject* self, PyObject* args) {{
    PROFILE_CALL(printProfileAll);
    return Py_None;
}}

// Python module launch function
static PyObject* launch(PyObject* self, PyObject* args) {{
    int gridX, gridY, gridZ;
    PyObject *launch_enter_hook = NULL;
    PyObject *launch_exit_hook = NULL;
    PyObject *kernel_metadata = NULL;
    PyObject *launch_metadata = NULL;
    PyObject * py_obj_stream = NULL;
    void * pKrnl = NULL;

    // Define the actual kernel arguments
    {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}

    // Init kernel arguments from python side
    if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &py_obj_stream, &pKrnl,
                                        &kernel_metadata, &launch_metadata,
                                        &launch_enter_hook, &launch_exit_hook
                                        {args_list})) {{
        return NULL;
    }}

    // Construct a data kernel arguments list data structure
    std::vector<KernelArg> kargs;
    //{' '.join([f"kargs.emplace_back(_arg{i}, PyObject_Size(_arg{i})*4);" if ty[0]=="*" else f"kargs.emplace_back(*(uint64_t*)&_arg{i}, sizeof(_arg{i}));" for i, ty in signature.items() if ty != "constexpr"])}
    // {' '.join([f"kargs.emplace_back(extractTensor(_arg{i}), getTensorStorageSize(_arg{i}));"
               if ty[0]=="*" else f"kargs.emplace_back(*(uint64_t*)&_arg{i}, sizeof(_arg{i}));"
                  for i, ty in signature.items() if ty != "constexpr"])}


    {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items() if ty != "constexpr"])};
    {' '.join([f"kargs.emplace_back(ptr_info{i}.dev_ptr, ptr_info{i}.size);"
               if ty[0]=="*" else f"kargs.emplace_back(*(uint64_t*)&_arg{i}, sizeof(_arg{i}));"
                  for i, ty in signature.items() if ty != "constexpr"])}

    // Launch the kernel
    _launch(gridX, gridY, gridZ, kargs);
    if (PyErr_Occurred()) {{
        return NULL;
    }}

    // Call the exit hook if provided
    if (launch_exit_hook != Py_None) {{
        PyObject* hook_args = Py_BuildValue("(O)", launch_metadata);
        PyObject* ret = PyObject_CallObject(launch_exit_hook, hook_args);
        Py_DECREF(hook_args);
        if (!ret)
            return NULL;
    }}

    // Return None to Python
    Py_INCREF(Py_None);
    return Py_None;
}}

// Python module method definitions
static PyMethodDef ModuleMethods[] = {{
    {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
    {{"release", release, METH_VARARGS, "Call release function"}},
    {{"get_device_ptr", get_device_ptr, METH_VARARGS, "Get txda current device"}},
    {{NULL, NULL, 0, NULL}} // sentinel
}};

// Python module definition
static struct PyModuleDef ModuleDef = {{
    PyModuleDef_HEAD_INIT,
    \"__triton_launcher\",
    NULL, // documentation
    -1,   // size
    ModuleMethods
}};

// Python module initialization function
PyMODINIT_FUNC PyInit___triton_launcher(void) {{
    PyObject *m = PyModule_Create(&ModuleDef);
    if (m == NULL) {{
        return NULL;
    }}

    PyModule_AddFunctions(m, ModuleMethods);
    return m;
}}
"""


class TXDAUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(TXDAUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        src = Path(os.path.join(dirname, "driver.cpp")).read_text()
        mod = compile_native(src, "tx81_utils")
        # # NOTE: The triton compiler.py framework requires these 2 interface.
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


class SimulatorUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(SimulatorUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    def load_binary(self, name, kernel, shared_mem, device):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".so", delete=False) as f:
            f.write(kernel)
            f.flush()
            import ctypes

            # Load the kernel ptr
            lib = ctypes.cdll.LoadLibrary(f.name)
            fn_ptr = getattr(lib, name)
            fn_ptr_as_void_p = ctypes.cast(fn_ptr, ctypes.c_void_p).value
            return (lib, fn_ptr_as_void_p, 0, 0)

    def get_device_properties(self, *args):
        return {"max_shared_mem": 1024 * 1024 * 3}


# Launch cross compiled runtime program on controller
class TXDALauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}

        # Compiler runtime kernel launcher source code
        kernel_path = metadata.kernel_path
        print("==== kernel_path: ", kernel_path)
        launcher_src = make_launcher(constants, signature, src.fn.__name__, kernel_path)
        mod = compile_native(launcher_src, "__triton_launcher")
        self.get_device_ptr = mod.get_device_ptr
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        # args: 0: gridX, 1: gridY, 2: gridZ,
        #       3: kernel_metadata?, 4: launch_metadata?,
        #       5: a tuple(0, 0, False, 1, 1, 1, 'add_kernel'), # this is probably kernel metadata
        #       6: None, 7: None, 8: None,
        #       9~N: Actual triton kernel args.
        import torch
        device = torch.txda.get_device()
        self.get_device_ptr(device)
        self.launch(*args, **kwargs)


class TXDADriver(GPUDriver):

    def __init__(self):
        import torch
        super().__init__()
        if (os.getenv("USE_SIM_MODE", "0").lower() in ("1", "true", "yes")):
            self.utils = SimulatorUtils()
        else:
            self.utils = TXDAUtils()
        self.launcher_cls = TXDALauncher
        # Needs to overwrite GPUDriver base methods
        self.get_current_stream = self.get_txda_stream
        self.get_current_device = torch.txda.current_device
        self.set_current_device = torch.txda.set_device
        atexit.register(torch.txda.cleanup_device)

    @staticmethod
    def is_active():
        try:
            import torch
            extend_torch()
            return torch.txda.is_available()
        except ImportError:
            return False

    def get_txda_stream(self, device):
        return None

    def get_current_target(self):
        capability = 1
        warp_size = 16
        return GPUTarget("txda", capability, warp_size)

    def get_active_torch_device(self):
        import torch
        chip_out = _get_tx8_deps_path("chip_out")
        chip_out = chip_out + os.sep
        torch.txda.init_device(chip_out)
        return torch.device("txda", self.get_current_device())

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_device_interface(self):
        import torch
        return torch.txda

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int).to("txda")
