#
# This file implements the triton kernel driver interfaces where are used in
# triton/python/triton/compiler/compiler.py.
# For how the interface in driver class is used, see the implementation of the
# file above.
#
import hashlib
import logging
import tempfile
import os
import importlib.util
import shutil
import sysconfig
from pathlib import Path
import torch
import torch_txda
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import GPUDriver
from triton.backends.compiler import GPUTarget

from triton.backends.tsingmicro import txda_tools
from triton.backends.tsingmicro.logger_config import setup_logger, logger_to_custom_level_number

logger = setup_logger("tsingmicro_launch")

dirname = os.path.dirname(os.path.realpath(__file__))
profiling_lib_dir = os.path.join(txda_tools.get_tx8_deps_path("profiling_tool"), "lib")
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
        os.path.join(txda_tools.get_tx8_deps_path("profiling_tool"), "include"),
        os.path.join(sysconfig.get_path('platlib'), "pybind11", "include"),
        os.path.join(sysconfig.get_path('platlib'), "torch", "include"),
        os.path.join(sysconfig.get_path('platlib'), "torch", "include", "torch", "csrc", "api", "include"),
        os.path.join(sysconfig.get_path('platlib'), "numpy", "_core", "include")
    ]
    library_dirs = [
        os.path.join(dirname, "lib"),
        txda_tools.get_kuiper_path("lib"),
        txda_tools.get_tx8_deps_path("lib"), profiling_lib_dir,
        os.path.join(sysconfig.get_path('platlib'), "torch", "lib")
    ]
    libraries = ['hpgr', 'torch', 'torch_cpu', 'torch_python', 'c10', 'dl']


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
    if txda_tools.is_use_profile():
        cc_cmd += ["-DENABLE_PROFILING"]
    if txda_tools.is_use_tsm_profiler():
        cc_cmd += ["-DTSM_PROFILER_EN"]
    if txda_tools.is_debug():
        cc_cmd += ["-DCMAKE_BUILD_TYPE=Debug"]
    if txda_tools.is_enable_kernel_file_cache():
        cc_cmd += ["-DENABLE_KERNEL_FILE_CACHE"]
        kernel_size = txda_tools.get_kernel_cache_size()
        cc_cmd += ["-DKERNEL_CACHE_SIZE=" + kernel_size]
    cc_cmd += [f'-l{lib}' for lib in libraries]
    if txda_tools.is_use_profile():
        profiling_flag = "rcs_profiling"
        cc_cmd += [f"-l{profiling_flag}"]
        cc_cmd += [f"-Wl,-rpath={profiling_lib_dir}"]
    cc_cmd += [f"-L{dir}" for dir in library_dirs]
    cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
    txda_tools.runLoweringCmd(src, so, cc_cmd)
    txda_tools.dump_ir_if_needed([so])
    txda_tools.dump_cmd_if_needed(cc_cmd, cc)
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
    format = "isssisii" + "iiiKKOOOO" + args_format
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
#include <dlfcn.h>

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
            Py_DECREF(ret);
            return ptr_info;
        }}
        ptr_info.dev_ptr = (void*) PyLong_AsLongLong(ret);
        if(!ptr_info.dev_ptr) {{
            Py_DECREF(ret);
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
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <torch/extension.h>
#include <torch/csrc/autograd/python_variable.h>
#include <stdint.h>
#include <vector>
#include <memory>
#include <string>
#include <filesystem>
#include <unistd.h>
#include <thread>
#include <fstream>
#include <dlfcn.h>

#include "logger.h"

#include "tx_runtime.h"


#ifdef TSM_PROFILER_EN
#include "tsm_profiler.h"
#endif


#ifdef ENABLE_PROFILING
    #include "hrt_profiler.h"
    #define PROFILE_CALL(func, ...) func(__VA_ARGS__)
#else
    #define PROFILE_CALL(func, ...)
#endif

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

//============================== launch data begin ==============================
struct LaunchRes {{
    int res;
    std::string log_buffer;
    uint64_t kernel_time_ns;
    uint64_t ap_duration_ns;

    LaunchRes() : res(0), log_buffer(), kernel_time_ns(0), ap_duration_ns(0) {{}}
    LaunchRes(int v, const char* n, uint64_t kt = 0, uint64_t ap = 0)
        : res(v), log_buffer(n ? n : ""), kernel_time_ns(kt), ap_duration_ns(ap) {{}}
}};

typedef struct {{
    PyObject_HEAD
    LaunchRes data;
}} LaunchResObj;

// Deallocator
static void LaunchRes_dealloc(LaunchResObj* self) {{
    self->data.~LaunchRes();
    Py_TYPE(self)->tp_free((PyObject*)self);
}}

static PyObject* LaunchRes_repr(LaunchResObj* self) {{
    return PyUnicode_FromFormat("LaunchRes(res=%d, log_buffer='%s', kernel_time_ns=%llu, ap_duration_ns=%llu)",
                                self->data.res,
                                self->data.log_buffer.c_str(),
                                (unsigned long long)self->data.kernel_time_ns,
                                (unsigned long long)self->data.ap_duration_ns);
}}

static PyMemberDef LaunchRes_members[] = {{
    {{"res", T_INT, offsetof(LaunchResObj, data.res), 0, "integer res"}},
    {{"kernel_time_ns", T_ULONGLONG, offsetof(LaunchResObj, data.kernel_time_ns), 0, "kernel device time in ns"}},
    {{"ap_duration_ns", T_ULONGLONG, offsetof(LaunchResObj, data.ap_duration_ns), 0, "AP duration in ns"}},
    {{NULL}}
}};

static PyObject* LaunchRes_get_name(LaunchResObj* self, void* closure) {{
    return PyUnicode_FromString(self->data.log_buffer.c_str());
}}

static PyGetSetDef LaunchRes_getsetters[] = {{
    {{"log_buffer", (getter)LaunchRes_get_name, NULL, "log_buffer of the object", NULL}},
    {{NULL}}
}};

static PyTypeObject LaunchResObjType = {{
    PyVarObject_HEAD_INIT(NULL, 0)
}};

static PyObject* make_LaunchRes(int res, const char* log_buffer = "", uint64_t kernel_time_ns = 0, uint64_t ap_duration_ns = 0) {{
    LaunchResObj* obj = (LaunchResObj*)LaunchResObjType.tp_alloc(&LaunchResObjType, 0);
    if (obj != NULL) {{
        new (&obj->data) LaunchRes(res, log_buffer, kernel_time_ns, ap_duration_ns);
    }}
    return (PyObject*)obj;
}}

static PyObject* make_LaunchRes(const LaunchRes& result) {{
    LaunchResObj* obj = (LaunchResObj*)LaunchResObjType.tp_alloc(&LaunchResObjType, 0);
    if (obj != NULL) {{
        new (&obj->data) LaunchRes(result);
    }}
    return (PyObject*)obj;
}}


//============================== launch data end ==============================

simple_logger::Logger logger(simple_logger::ERROR);

auto g_guard = std::shared_ptr<void>(
    nullptr,
    [](void*) {{
        // printf("guard release.\\n");
    }}
);

struct Launch_args {{
    int gridX = 0;
    int gridY = 0;
    int gridZ = 0;
    int device_id;
    const char* so_key = nullptr;
    const char* kernel_file = nullptr;
    const char* kernel_fun_name = nullptr;
    int is_dump_args = 0;
    const char* dump_path = nullptr;
    int launch_count = 0;
    std::vector<KernelArg> kargs;
    txStream_t stream = nullptr;
    uint64_t function = 0;  // txFunction_t from txModuleGetFunction (0 → GGL fallback)
    int log_level = simple_logger::ERROR;
    LaunchRes result;
}};

#ifdef ENABLE_KERNEL_FILE_CACHE
typedef struct cache_entry {{
    char *path;
    void *data;
    size_t size;
    struct cache_entry *next;
}} cache_entry_t;

static cache_entry_t *cache_table[KERNEL_CACHE_SIZE];

static uint32_t hash_string(const char *str) {{
    uint32_t c = 0;
    uint64_t hash = 5381;
    while ((c = *str++)) {{
        hash = ((hash << 5) + hash) + c;
    }}
    return hash % KERNEL_CACHE_SIZE;
}}

static cache_entry_t *find_entry(const char *path) {{
    unsigned int idx = hash_string(path);
    cache_entry_t *entry = cache_table[idx];
    while (entry) {{
        if (strcmp(entry->path, path) == 0)
            return entry;
        entry = entry->next;
    }}
    return NULL;
}}

static void insert_entry(cache_entry_t *entry) {{
    unsigned int idx = hash_string(entry->path);
    entry->next = cache_table[idx];
    cache_table[idx] = entry;
}}

static cache_entry_t *create_entry_from_file(const char *path) {{
    FILE *fp = fopen(path, "rb");
    if (!fp) {{
        perror("fopen");
        return NULL;
    }}

    if (fseek(fp, 0, SEEK_END) != 0) {{
        perror("fseek");
        fclose(fp);
        return NULL;
    }}
    long size = ftell(fp);
    if (size == -1) {{
        perror("ftell");
        fclose(fp);
        return NULL;
    }}
    rewind(fp);

    void *data = NULL;
    if (size > 0) {{
        data = malloc(size);
        if (!data) {{
            perror("malloc");
            fclose(fp);
            return NULL;
        }}

        size_t read_len = fread(data, 1, size, fp);
        if (read_len != (size_t)size) {{
            if (feof(fp))
                fprintf(stderr, "======== Unexpected EOF =======");
            else if (ferror(fp))
                perror("fread");
            free(data);
            fclose(fp);
            return NULL;
        }}
    }}
    fclose(fp);

    cache_entry_t *entry = reinterpret_cast<cache_entry_t *>(malloc(sizeof(cache_entry_t)));
    if (!entry) {{
        perror("malloc entry");
        free(data);
        return NULL;
    }}

    entry->path = reinterpret_cast<char *>(malloc(strlen(path) + 1));
    if (!entry->path) {{
        perror("malloc path");
        free(data);
        free(entry);
        return NULL;
    }}

    strcpy(entry->path, path);
    entry->data = data;
    entry->size = size;
    entry->next = NULL;
    return entry;
}}

static void cache_cleanup(void) {{
    for (int i = 0; i < KERNEL_CACHE_SIZE; i++) {{
        cache_entry_t *entry = cache_table[i];
        while (entry) {{
            cache_entry_t *next = entry->next;
            free(entry->path);
            free(entry->data);
            free(entry);
            entry = next;
        }}
        cache_table[i] = NULL;
    }}
}}
static int read_bin_file(const char *file_name, void **data, size_t *size) {{
    cache_entry_t *entry = find_entry(file_name);
    if (entry) {{
        *data = entry->data;
        *size = entry->size;
        return 0;
    }}

    entry = create_entry_from_file(file_name);
    if (!entry)
        return -1;

    insert_entry(entry);
    *data = entry->data;
    *size = entry->size;
    return 0;
}}
#else
static int read_bin_file(const char *file_name, void **data, size_t *size) {{
    FILE *file;
    int64_t file_size;
    size_t bytes_read;

    file = fopen(file_name, "r");

    if (file == NULL) {{
        logger.log(simple_logger::ERROR, "don't open file %s\\n", file_name);
        return -1;
    }}

    if (fseek(file, 0L, SEEK_END) != 0) {{
        logger.log(simple_logger::ERROR, "fseek to end failed \\n");
        fclose(file);
        return -1;
    }}
    file_size = ftell(file);
    if (file_size == -1) {{
        logger.log(simple_logger::ERROR, "ftell file failed\\n");
        fclose(file);
        return -1;
    }}
    rewind(file);
    *size = file_size;
    *data = (malloc(file_size + 1));
    if (*data == NULL) {{
        fclose(file);
        logger.log(simple_logger::ERROR,
            "file data malloc error %s file_size:%ld\\n", file_name, file_size);
        return -1;
    }}

    bytes_read = fread(*data, 1, file_size, file);

    fclose(file);
    return 0;
}}
#endif

void dump_kernel_args(Launch_args &l_args) {{

    std::ostringstream oss;

    oss << "==============================" << std::endl;
    oss << "kernel_file:";
    oss << l_args.kernel_file << ", ";
    oss << "kernel_func:";
    oss << l_args.kernel_fun_name << ", ";
    oss << "launch_count:";
    oss << l_args.launch_count << ", ";

    oss << "gridX:";
    oss << l_args.gridX << ", ";
    oss << "gridY:";
    oss << l_args.gridY << ", ";
    oss << "gridZ:";
    oss << l_args.gridZ << ", ";

    oss << "blockX:";
    oss << 1 << ", ";
    oss << "blockY:";
    oss << 1 << ", ";
    oss << "blockZ:";
    oss << 1 << ", ";
    oss << std::endl;

    std::vector<uint64_t> rtKargs;
    const char* str_args = "args";
    int count = 0;
    for (const KernelArg& karg : l_args.kargs) {{
        oss << str_args << count++ << "_";
        if (karg.data_type == POINT) {{
            oss << "v:" << 1 << ", ";
            oss << str_args << count++ << "_"
                    << "p:" << karg.data.ptr << ", ";
        }} else {{
            oss << "v:" << std::hex << "0x" << karg.data.scalar << ", ";
        }}
    }}
    oss << str_args << count++ << "_v:" << std::hex << "0x" << l_args.gridX << ", ";
    oss << str_args << count++ << "_v:" << std::hex << "0x" << l_args.gridY << ", ";
    oss << str_args << count++ << "_v:" << std::hex << "0x" << l_args.gridZ << ", ";
    oss << str_args << count++ << "_v:" << std::hex << "0x" << 0 << ", ";
    oss << str_args << count++ << "_v:" << std::hex << "0x" << 0 << ", ";
    oss << str_args << count++ << "_v:" << std::hex << "0x" << 0 << ", ";
    oss << std::endl;

    oss << "point size: ";
    for (const KernelArg& karg : l_args.kargs) {{
        if (karg.data_type == POINT) {{
            oss << karg.data.ptr << ":" << std::hex << "0x" << karg.size << ", ";
        }}
    }}
    oss << std::endl;
    oss << "stream: " << l_args.stream;

    logger.log(simple_logger::DEBUG, "%s", oss.str().c_str());
}}

static bool set_device_id(int device_id) {{
    // Skip redundant txSetDevice when device is unchanged (common hot path).
    static thread_local int tls_device = -1;
    if (tls_device == device_id) {{
        return true;
    }}
    if (txSetDevice(device_id) != TX_SUCCESS) {{
        PyErr_SetString(PyExc_RuntimeError, "Failed to set device");
        return false;
    }}
    tls_device = device_id;
    return true;
}}

static bool force_ggl_launch() {{
    const char* env = std::getenv("TXDA_LAUNCH_VIA_GGL");
    return env != nullptr && (env[0] == '1' || env[0] == 'y' || env[0] == 'Y');
}}

static void _launch(Launch_args &l_args) {{
    if (l_args.gridX*l_args.gridY*l_args.gridZ <= 0) {{
        return;  // No work to do
    }}

    if (!set_device_id(l_args.device_id)) {{
        return;
    }}

    if (l_args.is_dump_args != 0) {{
        dump_kernel_args(l_args);
    }}

    // Flatten kernel args once for either launch path.
    std::vector<uint64_t> rtKargs;
    rtKargs.reserve(l_args.kargs.size() * 2 + 6);
    for (KernelArg& karg : l_args.kargs) {{
        if (karg.data_type == POINT) {{
            rtKargs.push_back(1);
            rtKargs.push_back((uint64_t)(karg.data.ptr));
        }} else {{
            rtKargs.push_back((uint64_t)(karg.data.scalar));
        }}
    }}
    rtKargs.push_back(l_args.gridX);
    rtKargs.push_back(l_args.gridY);
    rtKargs.push_back(l_args.gridZ);
    rtKargs.push_back(0);
    rtKargs.push_back(0);
    rtKargs.push_back(0);

    dim3 gridDim({{(uint32_t)l_args.gridX, (uint32_t)l_args.gridY, (uint32_t)l_args.gridZ}});
    dim3 blockDim({{1u, 1u, 1u}});
    void* arg_ptr = rtKargs.empty() ? nullptr : (void*)(&rtKargs[0]);
    uint32_t arg_bytes = (uint32_t)(rtKargs.size() * sizeof(uint64_t));

#ifdef ENABLE_PROFILING
    static int run_count = 0;
    run_count++;
    std::string profiling_key(l_args.so_key);
    profiling_key.append("_").append(l_args.kernel_fun_name);
    profiling_key.append("_").append(std::to_string(run_count));
#endif

    PROFILE_CALL(RcsProcessProfData, l_args.device_id, profiling_key, PROF_START, 7);
#ifdef TSM_PROFILER_EN
    static bool _prof_available = false;
    static bool _prof_checked   = false;
    if (!_prof_checked) {{
        _prof_available = resolve_prof_query_api();
        _prof_checked   = true;
    }}
    uint64_t _prof_token = 0;
    if (_prof_available) {{
        _prof_token = resolve_tsm_prof_begin();
    }}
#endif

    // Fast path: launch by pre-loaded module function handle (txModuleLoad once
    // in load_binary). Avoids re-passing the ELF blob via txLaunchKernelGGL.
    // Fallback: TXDA_LAUNCH_VIA_GGL=1 or missing handle → legacy GGL path.
    txError_t launch_st = TX_SUCCESS;
    const bool use_handle = (l_args.function != 0) && !force_ggl_launch();
    if (use_handle) {{
        Py_BEGIN_ALLOW_THREADS;
        launch_st = txLaunchKernel((txFunction_t)l_args.function, gridDim, blockDim,
                                   arg_ptr, arg_bytes, 0, l_args.stream);
        Py_END_ALLOW_THREADS;
        if (launch_st != TX_SUCCESS) {{
            PyErr_Format(PyExc_RuntimeError, "Failed to txLaunchKernel (err=%d)", (int)launch_st);
            return;
        }}
    }} else {{
        uint64_t kernel_len = 0;
        void* kernel_ptr = nullptr;
        int ret = read_bin_file(l_args.kernel_file, &kernel_ptr, &kernel_len);
        if (ret != 0 || kernel_ptr == nullptr) {{
            PyErr_SetString(PyExc_RuntimeError, "Failed to read kernel so");
            return;
        }}
        Py_BEGIN_ALLOW_THREADS;
        launch_st = txLaunchKernelGGL(l_args.kernel_fun_name, (uint64_t)kernel_ptr, kernel_len,
            gridDim, blockDim, arg_ptr, arg_bytes, 0, l_args.stream);
        Py_END_ALLOW_THREADS;
        if (launch_st != TX_SUCCESS) {{
            PyErr_SetString(PyExc_RuntimeError, "Failed to txLaunchKernelGGL");
            return;
        }}
    }}
    const char* env = std::getenv("TXDA_LAUNCH_KERNEL_SYNC");
    if (env != nullptr && std::string(env) == "1") {{
        txStreamSynchronize(l_args.stream);
    }}
    PROFILE_CALL(txStreamSynchronize, l_args.stream);
#ifdef TSM_PROFILER_EN
    txStreamSynchronize(l_args.stream);
    if (_prof_available) {{
        tsm_prof_timing_t _prof_t = {{}};
        int _prof_rc = resolve_tsm_prof_get_timing(_prof_token, &_prof_t, 5000);
        if (_prof_rc == 0) {{
            l_args.result.kernel_time_ns = _prof_t.kcore_duration_nsec;
            l_args.result.ap_duration_ns = _prof_t.ap_duration_nsec;
            logger.log(simple_logger::DEBUG,
                "tsm_prof token=%llu corid=%llu host_ns=%llu ap_ns=%llu kcore_ns=%llu",
                (unsigned long long)_prof_token,
                (unsigned long long)_prof_t.corid,
                (unsigned long long)_prof_t.host_duration_nsec,
                (unsigned long long)_prof_t.ap_duration_nsec,
                (unsigned long long)_prof_t.kcore_duration_nsec);
        }}
    }}
#endif
    PROFILE_CALL(RcsProcessProfData, l_args.device_id, profiling_key, PROF_STOP, 0);
}}

// Structure to represent a device pointer
typedef struct _DevicePtrInfo {{
    void *dev_ptr;
    bool valid;
    size_t size;
}} DevicePtrInfo;

// Function to get tensor size using untyped_storage if available
static inline size_t getTensorSize(PyObject *obj) {{
    // Final fallback: calculate size from numel() * element_size()
    PyObject *numel_method = PyObject_GetAttrString(obj, "numel");
    PyObject *element_size_method = PyObject_GetAttrString(obj, "element_size");

    if (numel_method && element_size_method) {{
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

    return 0;  // Return 0 if unable to determine size
}}

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
    DevicePtrInfo ptr_info;
    ptr_info.dev_ptr = 0;
    ptr_info.valid = true;
    ptr_info.size = 0;  // Initialize size

    if (PyLong_Check(obj)) {{
        ptr_info.dev_ptr = (void*) PyLong_AsLongLong(obj);
        logger.log(simple_logger::DEBUG, "PyLong_AsLongLong %p\\n", ptr_info.dev_ptr);
        return ptr_info;
    }}

    if (obj == Py_None) {{
        // valid nullptr
        logger.log(simple_logger::DEBUG, "Py_None\\n");
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
            logger.log(simple_logger::ERROR, "data_ptr method of Pointer object must return 64-bit int\\n");
            ptr_info.valid = false;
            Py_DECREF(ret);
            return ptr_info;
        }}
        ptr_info.dev_ptr = (void*) PyLong_AsLongLong(ret);
        Py_DECREF(ret);

        // Size is only needed for debug dumps; skip hot-path Python calls.
        // ptr_info.size stays 0 for the launch path (rtKargs never uses it).
        return ptr_info;
    }}
    std::string error_msg = "Pointer argument must be either uint64 or have data_ptr method\\n";
    PyErr_SetString(PyExc_TypeError, error_msg.c_str());
    logger.log(simple_logger::ERROR, error_msg.c_str());
    ptr_info.valid = false;
    return ptr_info;
}}


static size_t getTensorStorageSize(PyObject* tensor_obj) {{
    const at::Tensor& tensor = THPVariable_Unpack(tensor_obj);
    logger.log(simple_logger::DEBUG, "========== total size ================: %ld\\n", tensor.storage().nbytes());
    return tensor.storage().nbytes();
}}

// Extract tensor raw ptr
static void* extractTensor(PyObject* tensor_obj) {{
    const at::Tensor& tensor = THPVariable_Unpack(tensor_obj);
    torch::Tensor contiguous_tensor = tensor.contiguous();
    logger.log(simple_logger::DEBUG, "========== ptr ================: %p\\n", contiguous_tensor.data_ptr());
    return contiguous_tensor.data_ptr();
}}

static PyObject* release(PyObject* self, PyObject* args) {{
#ifdef ENABLE_KERNEL_FILE_CACHE
    cache_cleanup();
#endif
    Py_RETURN_NONE;
}}

// Python module launch function
static PyObject* launch(PyObject* self, PyObject* args) {{
    int gridX, gridY, gridZ;
    PyObject *launch_enter_hook = NULL;
    PyObject *launch_exit_hook = NULL;
    PyObject *kernel_metadata = NULL;
    PyObject *launch_metadata = NULL;
    uint64_t _stream = 0;
    uint64_t _function = 0;

    const char* kernel_file = "base_kernel_path";
    const char* kernel_fun_name = "base_kernel_func_name";
    const char* dump_path = "";
    const char* so_key = "";
    int is_dump_args = 0;
    int device_id = 0;
    int log_level = simple_logger::ERROR;

    Launch_args l_args;

    // Define the actual kernel arguments
    {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}

    // Init kernel arguments from python side
    if(!PyArg_ParseTuple(args, \"{format}\", &l_args.device_id, &l_args.so_key, &l_args.kernel_file,
                                        &l_args.kernel_fun_name, &l_args.is_dump_args, &l_args.dump_path, &l_args.log_level,
                                        &l_args.launch_count,
                                        &l_args.gridX, &l_args.gridY, &l_args.gridZ, &_stream, &_function,
                                        &kernel_metadata, &launch_metadata,
                                        &launch_enter_hook, &launch_exit_hook
                                        {args_list})) {{
        return make_LaunchRes(-1);
    }}

    logger.setLogLevel((simple_logger::LogLevel)l_args.log_level);
    l_args.stream = (txStream_t)_stream;
    l_args.function = _function;

    //{' '.join([f"l_args.kargs.emplace_back(_arg{i}, PyObject_Size(_arg{i})*4);" if ty[0]=="*" else f"l_args.kargs.emplace_back(*(uint64_t*)&_arg{i}, sizeof(_arg{i}));" for i, ty in signature.items() if ty != "constexpr"])}
    // {' '.join([f"l_args.kargs.emplace_back(extractTensor(_arg{i}), getTensorStorageSize(_arg{i}));"
               if ty[0]=="*" else f"l_args.kargs.emplace_back(*(uint64_t*)&_arg{i}, sizeof(_arg{i}));"
                  for i, ty in signature.items() if ty != "constexpr"])}

    uint64_t buf = 0;
    {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items() if ty != "constexpr"])};
    {' '.join([f"l_args.kargs.emplace_back(ptr_info{i}.dev_ptr, ptr_info{i}.size);"
               if ty[0]=="*" else f"buf = 0;std::memcpy(&buf, &_arg{i}, sizeof(_arg{i}));l_args.kargs.emplace_back(buf, sizeof(_arg{i}));"
                  for i, ty in signature.items() if ty != "constexpr"])}

    // Launch the kernel
    _launch(l_args);
    if (PyErr_Occurred()) {{
        return make_LaunchRes(-1);
    }}

    // Call the exit hook if provided
    if (launch_exit_hook != Py_None) {{
        PyObject* hook_args = Py_BuildValue("(O)", launch_metadata);
        PyObject* ret = PyObject_CallObject(launch_exit_hook, hook_args);
        Py_DECREF(hook_args);
        if (!ret)
            return make_LaunchRes(-1);
    }}

    return make_LaunchRes(l_args.result);
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

    LaunchResObjType.tp_name = "__triton_launcher.LaunchRes";
    LaunchResObjType.tp_doc = "Custom LaunchRes objects";
    LaunchResObjType.tp_basicsize = sizeof(LaunchResObj);
    LaunchResObjType.tp_itemsize = 0;
    LaunchResObjType.tp_flags = Py_TPFLAGS_DEFAULT;
    LaunchResObjType.tp_new = PyType_GenericNew;
    LaunchResObjType.tp_dealloc = (destructor)LaunchRes_dealloc;
    LaunchResObjType.tp_repr = (reprfunc)LaunchRes_repr;
    LaunchResObjType.tp_members = LaunchRes_members;
    LaunchResObjType.tp_getset = LaunchRes_getsetters;

    if (PyType_Ready(&LaunchResObjType) < 0)
        return NULL;

    Py_INCREF(&LaunchResObjType);
    if (PyModule_AddObject(m, "LaunchRes", (PyObject*)&LaunchResObjType) < 0) {{
        Py_DECREF(&LaunchResObjType);
        Py_DECREF(m);
        return NULL;
    }}
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
            # (module, function, n_regs, n_spills, n_max_threads).
            return (lib, fn_ptr_as_void_p, 0, 0, 1024)

    def get_device_properties(self, *args):
        return {"max_shared_mem": 1024 * 1024 * 3 - 0x10000 - 0x10000}


_last_launch_res = None
_launch_counter = 0

# Launch cross compiled runtime program on controller
class TXDALauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}

        # Compiler runtime kernel launcher source code
        self.metadata = metadata
        launcher_src = make_launcher(constants, signature, src.fn.__name__, metadata.kernel_path)
        mod = compile_native(launcher_src, "__triton_launcher")
        self.launch = mod.launch
        self.func_name = src.fn.__name__

    def __call__(self, gridX, gridY, gridZ, stream, function, *args, **kwargs):
        global _launch_counter
        _launch_counter += 1
        device_id = torch.txda.current_device()
        log_level = logger_to_custom_level_number(logger)
        logger.info("%s launch card:%s count:%s begin", self.func_name, device_id, _launch_counter)
        launchRes = self.launch(device_id, self.metadata.so_key, self.metadata.kernel_path, self.func_name,
                                txda_tools.is_dump_args_profile(), txda_tools.get_dump_dir(), log_level,
                                _launch_counter,
                                gridX, gridY, gridZ, stream, function, *args, **kwargs)
        logger.info("%s launch card:%s count:%s end", self.func_name, device_id, _launch_counter)
        if launchRes.res != 0:
            logger.error("launch error code:%s", launchRes.res)
        global _last_launch_res
        _last_launch_res = launchRes
        return launchRes


class TXDADriver(GPUDriver):

    def __init__(self):
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

    @staticmethod
    def is_active():
        try:
            return torch.txda.is_available()
        except ImportError:
            return False

    def get_txda_stream(self, device):
        return torch.txda.current_stream(device).txda_stream

    def map_python_to_cpp_type(self, ty: str) -> str:
        return _ty_to_cpp(ty)

    def get_current_target(self):
        capability = 1
        warp_size = 16
        return GPUTarget("txda", capability, warp_size)

    def get_active_torch_device(self):
        return torch.device("txda", self.get_current_device())

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_device_interface(self):
        return torch.txda

    def get_empty_cache_for_benchmark(self):
        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        # return torch.empty(int(cache_size // 4), dtype=torch.int).to("txda")
        return True

    def clear_cache(self, cache):
        return True

    def get_last_launch_res(self):
        global _last_launch_res
        return _last_launch_res
