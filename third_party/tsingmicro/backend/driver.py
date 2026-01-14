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

from triton.backends.tsingmicro import txda_tools

dirname = os.path.dirname(os.path.realpath(__file__))
if (os.getenv("USE_SIM_MODE", "0").lower() in ("1", "true", "yes")):
    scheme = sysconfig.get_default_scheme()
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]

    include_dirs = [txda_tools.get_kuiper_path("include"), txda_tools.get_tx8_deps_path("include"), py_include_dir]
    library_dirs = [txda_tools.get_kuiper_path("lib"), txda_tools.get_tx8_deps_path("lib")]
    libraries = ["triton_cmodel", "tx8be_op_cmodel", "neuralcore_qemu"]
else:
    include_dirs = [
        os.path.join(dirname, "include"),
        txda_tools.get_kuiper_path("include"),
        txda_tools.get_tx8_deps_path("include"),
        os.path.join(sysconfig.get_path('platlib'), "pybind11", "include"),
        os.path.join(sysconfig.get_path('platlib'), "torch", "include"),
        os.path.join(sysconfig.get_path('platlib'), "torch", "include", "torch", "csrc", "api", "include"),
        os.path.join(sysconfig.get_path('platlib'), "numpy", "_core", "include")
    ]
    library_dirs = [
        os.path.join(dirname, "lib"),
        txda_tools.get_kuiper_path("lib"),
        txda_tools.get_tx8_deps_path("lib"),
        os.path.join(sysconfig.get_path('platlib'), "torch", "lib")
    ]
    libraries = ['hpgr', 'torch', 'torch_cpu', 'torch_python', 'c10']


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
    if txda_tools.is_use_host_profile():
        cc_cmd += ["-DUSE_PROFILE"]
    if txda_tools.is_debug():
        cc_cmd += ["-DCMAKE_BUILD_TYPE=Debug"]
    cc_cmd += [f'-l{lib}' for lib in libraries]
    if txda_tools.is_use_host_profile():
        cc_cmd += ["-lprofiler_x86"]
    cc_cmd += [f"-L{dir}" for dir in library_dirs]
    cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
    txda_tools.runLoweringCmd(so, cc_cmd)
    txda_tools.dump_ir_if_needed([so])
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
                txda_tools.dump_ir_if_needed([src_path])
            so = _build(name, src_path, tmpdir, library_dirs, include_dirs, libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{fname}.so", binary=True)
                txda_tools.dump_ir_if_needed([cache_path])
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
    format = "issis" + "iiiOKOOOO" + args_format
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

extern "C" {{
    int vdk_printf(const char *fmt, ...) {{ return 0; }}
    int tsprintf_core(const char *fmt, ...) {{ return 0; }}
}}

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

    // Create a temporary file. Remember to add XXXXXX in uppercase; when the temporary file is created successfully, the system will automatically fill in the characters
    char name[] = "/tmp/dirXXXXXX";
    int fd = mkstemp(name);
    if(fd == -1) {{
        perror("mkstemp failed\\n");
        exit(-1);
    }}

    q_tilesim_set_logFile(sim_handle, "/dev/null");

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

    // Delete tmp file
    close(fd);
    unlink(name);

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
#include <unistd.h>
#include <thread>
#include <fstream>

#include "tx_runtime.h"
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


static int read_bin_file(const char *file_name, char **content, size_t *length) {{
    FILE *file;
    int64_t file_size;
    size_t bytes_read;

    file = fopen(file_name, "r");

    if (file == NULL) {{
        printf("don't open file %s\\n", file_name);
        return -1;
    }}

    if (fseek(file, 0L, SEEK_END) != 0) {{
        printf("fseek to end failed \\n");
        fclose(file);
        return -1;
    }}
    file_size = ftell(file);
    if (file_size == -1) {{
        printf("ftell file failed\\n");
        fclose(file);
        return -1;
    }}
    rewind(file);
    *length = file_size;
    *content = reinterpret_cast<char *>(malloc(sizeof(char) * (file_size + 1)));
    printf("filename:%s length:%ld\\n", file_name, file_size);
    if (*content == NULL) {{
        fclose(file);
        printf("file content malloc error %s file_size:%ld\\n", file_name, file_size);
        return -1;
    }}

    bytes_read = fread(*content, sizeof(char), file_size, file);
    (*content)[bytes_read] = '\\0';

    fclose(file);
    return 0;
}}

void dump_kernel_args(int gridX, int gridY, int gridZ,
    const std::string &kernel_file, const std::string &kernel_fun_name,
    const std::vector<KernelArg> &kargs, const std::string &dump_path) {{

    std::string dumpfile = dump_path + "/kernel_args.txt";
    std::ofstream outfile(dumpfile, std::ios::app);

    if (!outfile) {{
        printf("error can't open file:%s\\n", dumpfile.c_str());
        return;
    }}

    outfile << "==============================" << std::endl;
    outfile << "kernel_file:";
    outfile << kernel_file << ", ";
    outfile << "kernel_func:";
    outfile << kernel_fun_name << ", ";

    outfile << "gridX:";
    outfile << gridX << ", ";
    outfile << "gridY:";
    outfile << gridY << ", ";
    outfile << "gridZ:";
    outfile << gridZ << ", ";

    outfile << "blockX:";
    outfile << 1 << ", ";
    outfile << "blockY:";
    outfile << 1 << ", ";
    outfile << "blockZ:";
    outfile << 1 << ", ";
    outfile << std::endl;

    std::vector<uint64_t> rtKargs;
    const char* str_args = "args";
    int count = 0;
    for (const KernelArg& karg : kargs) {{
        outfile << str_args << count++ << "_";
        if (karg.data_type == POINT) {{
            outfile << "v:" << 1 << ", ";
            outfile << str_args << count++ << "_"
                    << "p:" << karg.data.ptr << ", ";
        }} else {{
            outfile << "v:" << std::hex << "0x" << karg.data.scalar << ", ";
        }}
    }}
    outfile << str_args << count++ << "_v:" << std::hex << "0x" << gridX << ", ";
    outfile << str_args << count++ << "_v:" << std::hex << "0x" << gridY << ", ";
    outfile << str_args << count++ << "_v:" << std::hex << "0x" << gridZ << ", ";
    outfile << str_args << count++ << "_v:" << std::hex << "0x" << 0 << ", ";
    outfile << str_args << count++ << "_v:" << std::hex << "0x" << 0 << ", ";
    outfile << str_args << count++ << "_v:" << std::hex << "0x" << 0 << ", ";
    outfile << std::endl;

    outfile << "point size: ";
    for (const KernelArg& karg : kargs) {{
        if (karg.data_type == POINT) {{
            outfile << karg.data.ptr << ":" << std::hex << "0x" << karg.size << ", ";
        }}
    }}
    outfile << std::endl;

    outfile.close();
}}

static bool set_device_id(int device_id) {{
    if (txSetDevice(device_id) != TX_SUCCESS) {{
        PyErr_SetString(PyExc_RuntimeError, "Failed to set device");
        return false;
    }}
    return true;
}}

static void _launch(int gridX, int gridY, int gridZ,
    int device_id, std::string kernel_file, std::string kernel_fun_name,
    int is_dump_args, std::string dump_path, txStream_t stream,
    std::vector<KernelArg> kargs) {{
    if (gridX*gridY*gridZ <= 0) {{
        return;  // No work to do
    }}

    if (!set_device_id(device_id)) {{
        return;
    }}


    if (is_dump_args != 0) {{
        dump_kernel_args(gridX, gridY, gridZ,
            kernel_file, kernel_fun_name, kargs, dump_path);
    }}

    // TODO::mv
    uint64_t kernel_len = 0;
    char* kernel_ptr = nullptr;
    int ret = read_bin_file(kernel_file.c_str(), &kernel_ptr, &kernel_len);
    if (ret != 0 || kernel_ptr == nullptr) {{
        PyErr_SetString(PyExc_RuntimeError, "Failed to read kernel so");
        return;
    }}

    // Allocate the device memory for all kernel arguments
    std::vector<uint64_t> rtKargs;
    for (KernelArg& karg : kargs) {{
        if (karg.data_type == POINT) {{
            rtKargs.push_back(1);
            rtKargs.push_back((uint64_t)(karg.data.ptr));
        }} else {{
            rtKargs.push_back((uint64_t)(karg.data.scalar));
        }}
    }}
    rtKargs.push_back(gridX);
    rtKargs.push_back(gridY);
    rtKargs.push_back(gridZ);
    rtKargs.push_back(0);
    rtKargs.push_back(0);
    rtKargs.push_back(0);

    // txError_t txLaunchKernelGGL(const char *funcName, uint64_t elfAddr, uint64_t elfLen, dim3 gridDim, dim3 blockDim,
    //         void *kernelArg, uint32_t kernelArgLen, uint32_t sharedMemBytes,
    //         txStream_t tStream = nullptr);
    uint32_t eventId = EVENT_INIT;
    PROFILE_CALL(addOrderProfile, TIME_RUNTIME, TIME_LAUNCH_START, &eventId);
    if (txLaunchKernelGGL(kernel_fun_name.c_str(), (uint64_t)kernel_ptr, kernel_len,
        dim3({{(uint32_t)gridX, (uint32_t)gridY, (uint32_t)gridZ}}), dim3({{1u, 1u, 1u}}),
        (void*)(&rtKargs[0]), rtKargs.size()*sizeof(uint64_t), 0, stream) != TX_SUCCESS){{
        PyErr_SetString(PyExc_RuntimeError, "Failed to txLaunchKernelGGL");
    }}
    txStreamSynchronize(stream);
    PROFILE_CALL(addOrderProfile, TIME_RUNTIME, TIME_LAUNCH_END, &eventId);
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

static PyObject* release(PyObject* self, PyObject* args) {{
    PROFILE_CALL(printProfileAll);
    Py_RETURN_NONE;
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

    const char* kernel_file = "base_kernel_path";
    const char* kernel_fun_name = "base_kernel_func_name";
    const char* dump_path = "";
    int is_dump_args = 0;
    uint32_t sharedMemBytes = 0;
    int device_id = 0;
    txStream_t stream = nullptr;

    // Define the actual kernel arguments
    {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}

    // Init kernel arguments from python side

    if(!PyArg_ParseTuple(args, \"{format}\", &device_id, &kernel_file,
                                        &kernel_fun_name, &is_dump_args, &dump_path,
                                        &gridX, &gridY, &gridZ, &py_obj_stream, &pKrnl,
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
    _launch(gridX, gridY, gridZ, device_id, std::string(kernel_file),
        std::string(kernel_fun_name), is_dump_args, std::string(dump_path), stream, kargs);
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
        return {"max_shared_mem": 1024 * 1024 * 3 - 0x10000 - 0x10000}


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
        self.launch = mod.launch
        self.kernel_path = kernel_path
        self.func_name = src.fn.__name__

    def __call__(self, *args, **kwargs):
        # args: 0: gridX, 1: gridY, 2: gridZ,
        #       3: kernel_metadata?, 4: launch_metadata?,
        #       5: a tuple(0, 0, False, 1, 1, 1, 'add_kernel'), # this is probably kernel metadata
        #       6: None, 7: None, 8: None,
        #       9~N: Actual triton kernel args.
        import torch
        device_id = torch.txda.current_device()
        self.launch(device_id, self.kernel_path, self.func_name, txda_tools.is_dump_args_profile(),
                    txda_tools.get_dump_dir(), *args, **kwargs)


class TXDADriver(GPUDriver):

    def __init__(self):
        import torch
        super().__init__()
        print("============= call TXDADriver test")
        if (os.getenv("USE_SIM_MODE", "0").lower() in ("1", "true", "yes")):
            self.utils = SimulatorUtils()
        else:
            self.utils = TXDAUtils()
        self.launcher_cls = TXDALauncher
        # Needs to overwrite GPUDriver base methods
        self.get_current_stream = self.get_txda_stream
        self.get_current_device = torch.txda.current_device
        self.set_current_device = torch.txda.set_device

    @staticmethod
    def is_active():
        try:
            import torch
            import torch_txda
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
        # return torch.empty(int(cache_size // 4), dtype=torch.int).to("txda")
        return True

    def clear_cache(self, cache):
        return True
        # print("clear_cache")
        # cache.zero_()
