#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  PyObject *device_obj;

  if (!PyArg_ParseTuple(args, "O", &device_obj)) {
    return NULL;
  }

  if (PyUnicode_Check(device_obj)) {
    const char *device_str = PyUnicode_AsUTF8(device_obj);
    if (device_str == NULL) {
      return NULL;
    }
    if (strcmp(device_str, "cpu") != 0) {
      PyErr_Format(PyExc_ValueError,
                   "Invalid device string: '%s'. Expected 'cpu'.", device_str);
      return NULL;
    }
  } else if (!PyLong_Check(device_obj)) {
    PyErr_SetString(PyExc_TypeError,
                    "device must be a string ('cpu') or an integer");
    return NULL;
  }

  int max_shared_mem = 1 << 20;
  int multiprocessor_count = 1;
  int warp_size = 1;
  int sm_clock_rate = 0;
  int mem_clock_rate = 0;
  int mem_bus_width = 0;
  int max_num_regs = 0;

#ifdef _SC_NPROCESSORS_ONLN
  long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
  if (nprocs > 0) {
    multiprocessor_count = (int)nprocs;
  }
#endif

  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i, s:i, s:i}", "max_shared_mem",
                       max_shared_mem, "max_num_regs", max_num_regs,
                       "multiprocessor_count", multiprocessor_count, "warpSize",
                       warp_size, "sm_clock_rate", sm_clock_rate,
                       "mem_clock_rate", mem_clock_rate, "mem_bus_width",
                       mem_bus_width);
}

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  const char *name;
  const char *data;
  Py_ssize_t data_size;
  int shared;
  PyObject *device_obj;

  if (!PyArg_ParseTuple(args, "ss#iO", &name, &data, &data_size, &shared,
                        &device_obj)) {
    return NULL;
  }

  if (PyUnicode_Check(device_obj)) {
    const char *device_str = PyUnicode_AsUTF8(device_obj);
    if (device_str == NULL) {
      return NULL;
    }
    if (strcmp(device_str, "cpu") != 0) {
      PyErr_Format(PyExc_ValueError,
                   "Invalid device string: '%s'. Expected 'cpu'.", device_str);
      return NULL;
    }
  } else if (!PyLong_Check(device_obj)) {
    PyErr_SetString(PyExc_TypeError,
                    "device must be a string ('cpu') or an integer");
    return NULL;
  }

  char temp_path[] = "/tmp/triton_cpu_kernel_XXXXXX.so";
  int fd = mkstemps(temp_path, 3);
  if (fd == -1) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Failed to create temporary file for kernel binary");
    return NULL;
  }

  ssize_t written = write(fd, data, data_size);
  close(fd);

  if (written != (ssize_t)data_size) {
    unlink(temp_path);
    PyErr_SetString(PyExc_RuntimeError,
                    "Failed to write kernel binary to temporary file");
    return NULL;
  }

  void *lib_handle = dlopen(temp_path, RTLD_NOW | RTLD_LOCAL);

  unlink(temp_path);

  if (!lib_handle) {
    const char *err = dlerror();
    char err_msg[1024];
    snprintf(err_msg, sizeof(err_msg), "Failed to load kernel binary: %s",
             err ? err : "unknown error");
    PyErr_SetString(PyExc_RuntimeError, err_msg);
    return NULL;
  }

  dlerror();

  void *func_ptr = dlsym(lib_handle, name);
  const char *dlsym_err = dlerror();
  if (dlsym_err) {
    char err_msg[1024];
    snprintf(err_msg, sizeof(err_msg),
             "Failed to find function '%s' in kernel binary: %s", name,
             dlsym_err);
    dlclose(lib_handle);
    PyErr_SetString(PyExc_RuntimeError, err_msg);
    return NULL;
  }

  return Py_BuildValue("(KKiii)", (uint64_t)(uintptr_t)lib_handle,
                       (uint64_t)(uintptr_t)func_ptr, 0, 0, 0);
}

typedef void *(*spine_get_current_stream_t)(int64_t);
typedef uint16_t (*spine_get_current_arch_id_t)(void);
typedef int64_t (*spine_get_num_cores_t)(void);
typedef int64_t (*spine_get_stream_threads_t)(void);

static spine_get_current_stream_t g_spine_get_current_stream = NULL;
static spine_get_current_arch_id_t g_spine_get_current_arch_id = NULL;
static spine_get_num_cores_t g_spine_get_num_cores = NULL;
static spine_get_stream_threads_t g_spine_get_stream_threads = NULL;

static bool g_stream_symbol_loaded = false;
static bool g_arch_symbol_loaded = false;
static bool g_get_num_cores_symbol_loaded = false;
static bool g_get_stream_threads_symbol_loaded = false;

static PyObject *getCurrentStream(PyObject *self, PyObject *args) {
  if (!g_stream_symbol_loaded) {
    g_spine_get_current_stream = (spine_get_current_stream_t)dlsym(
        RTLD_DEFAULT, "spine_get_current_stream");

    g_stream_symbol_loaded = true;
  }

  if (g_spine_get_current_stream == NULL) {
    return PyLong_FromLongLong(0);
  }

  void *stream = g_spine_get_current_stream(-1);

  if (stream == NULL) {
    return PyLong_FromLongLong(0);
  }

  return PyLong_FromUnsignedLongLong((uint64_t)(uintptr_t)stream);
}

static PyObject *getArchId(PyObject *self, PyObject *args) {
  char arch_id_str[32];

  if (!g_arch_symbol_loaded) {
    g_spine_get_current_arch_id = (spine_get_current_arch_id_t)dlsym(
        RTLD_DEFAULT, "spine_get_current_arch_id");
    g_arch_symbol_loaded = true;
  }

  if (g_spine_get_current_arch_id != NULL) {
    uint16_t arch_id = g_spine_get_current_arch_id();
    snprintf(arch_id_str, sizeof(arch_id_str), "0x%X", arch_id);
    return PyUnicode_FromString(arch_id_str);
  }

  return PyUnicode_FromString("0xA03C");
}

static PyObject *getNumCores(PyObject *self, PyObject *args) {
  if (!g_get_num_cores_symbol_loaded) {
    g_spine_get_num_cores =
        (spine_get_num_cores_t)dlsym(RTLD_DEFAULT, "spine_get_num_cores");
    g_get_num_cores_symbol_loaded = true;
  }
  if (g_spine_get_num_cores == NULL) {
    return PyLong_FromLongLong(0);
  }

  int64_t num_cores = g_spine_get_num_cores();
  return PyLong_FromLongLong(num_cores);
}

static PyObject *getStreamThreads(PyObject *self, PyObject *args) {
  if (!g_get_stream_threads_symbol_loaded) {
    g_spine_get_stream_threads = (spine_get_stream_threads_t)dlsym(
        RTLD_DEFAULT, "spine_get_stream_threads");
    g_get_stream_threads_symbol_loaded = true;
  }
  if (g_spine_get_stream_threads == NULL) {
    return PyLong_FromLongLong(0);
  }

  int64_t num_threads = g_spine_get_stream_threads();
  return PyLong_FromLongLong(num_threads);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided shared library into memory and return function pointer"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device (CPU)"},
    {"get_current_stream", getCurrentStream, METH_VARARGS,
     "Get the current execution stream"},
    {"get_arch_id", getArchId, METH_VARARGS, "Get AI core architecture ID"},
    {"get_num_cores", getNumCores, METH_VARARGS, "Get the number of cores"},
    {"get_stream_threads", getStreamThreads, METH_VARARGS,
     "Get the number of threads for a stream"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "cpu_utils",
                                       "CPU backend utilities for Triton", -1,
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_cpu_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
