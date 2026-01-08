#include "tang.h"
#include <dlfcn.h>
#include <stdbool.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Raises a Python exception and returns false if code is not TANG_SUCCESS.
static bool gpuAssert(TAresult code, const char *file, int line) {
  if (code == TANG_SUCCESS)
    return true;

  const char *prefix = "Triton Error [TANG]: ";
  const char *str;
  taGetErrorString(code, &str);
  char err[1024] = {0};
  strcat(err, prefix);
  strcat(err, str);
  PyGILState_STATE gil_state;
  gil_state = PyGILState_Ensure();
  PyErr_SetString(PyExc_RuntimeError, err);
  PyGILState_Release(gil_state);
  return false;
}

// To be used only *outside* a Py_{BEGIN,END}_ALLOW_THREADS block.
#define TANG_CHECK_AND_RETURN_NULL(ans)                                        \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__))                                 \
      return NULL;                                                             \
  } while (0)

// To be used inside a Py_{BEGIN,END}_ALLOW_THREADS block.
#define TANG_CHECK_AND_RETURN_NULL_ALLOW_THREADS(ans)                          \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__)) {                               \
      PyEval_RestoreThread(_save);                                             \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;
  // Get device handle
  TAdevice device;
  taDeviceGet(&device, device_id);

  // create a struct to hold device properties
  int max_shared_mem;
  int max_num_regs;
  int multiprocessor_count;
  int warp_size;
  int sm_clock_rate;
  int mem_clock_rate;
  int mem_bus_width;
  TANG_CHECK_AND_RETURN_NULL(taDeviceGetAttribute(
      &max_shared_mem, TA_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  TANG_CHECK_AND_RETURN_NULL(taDeviceGetAttribute(
      &max_num_regs, TA_DEV_ATTR_REGS_PER_BLOCK, device));
  TANG_CHECK_AND_RETURN_NULL(taDeviceGetAttribute(
      &multiprocessor_count, TA_DEV_ATTR_MULTIPROCESSOR_COUNT, device));
  TANG_CHECK_AND_RETURN_NULL(taDeviceGetAttribute(
      &warp_size, TA_DEV_ATTR_WARP_SIZE, device));
  TANG_CHECK_AND_RETURN_NULL(taDeviceGetAttribute(
      &sm_clock_rate, TA_DEV_ATTR_CLOCK_RATE, device));
  TANG_CHECK_AND_RETURN_NULL(taDeviceGetAttribute(
      &mem_clock_rate, TA_DEV_ATTR_MEMORY_CLOCK_RATE, device));
  TANG_CHECK_AND_RETURN_NULL(taDeviceGetAttribute(
      &mem_bus_width, TA_DEV_ATTR_MEMORY_BUS_WIDTH, device));

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
  int device;
  if (!PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared,
                        &device)) {
    return NULL;
  }
  TAfunction fun;
  TAmodule mod;
  int32_t n_regs = 0;
  int32_t n_spills = 0;
  int32_t n_max_threads = 0;
  // create driver handles
  TAcontext pctx = 0;
  TAdevice device_hd;
  taDeviceGet(&device_hd, device);

  Py_BEGIN_ALLOW_THREADS;
  TANG_CHECK_AND_RETURN_NULL_ALLOW_THREADS(taCtxGetCurrent(&pctx));
  if (!pctx) {
    TANG_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        taDevicePrimaryCtxRetain(&pctx, device_hd));
    TANG_CHECK_AND_RETURN_NULL_ALLOW_THREADS(taCtxSetCurrent(pctx));
  }

  TANG_CHECK_AND_RETURN_NULL_ALLOW_THREADS(taModuleLoadData(&mod, data, (size_t)data_size));
  TANG_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      taModuleGetFunction(&fun, mod, name));
  // get allocated registers and spilled registers from the function
  /* 不支持属性获取, 按照默认0处理 */
  // TANG_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
  //     taFuncGetAttribute(&n_regs, TA_FUNC_ATTRIBUTE_NUM_REGS, fun));
  // TANG_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
  //     taFuncGetAttribute(&n_spills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
  // n_spills /= 4;
  TANG_CHECK_AND_RETURN_NULL_ALLOW_THREADS(taFuncGetAttribute(
      &n_max_threads, TA_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fun));
  // set dynamic shared memory if necessary
  // int shared_optin;
  // TANG_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuDeviceGetAttribute(
  //     &shared_optin, TA_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
  //     device));
  // if (shared > 49152 && shared_optin > 49152) {
  //   TANG_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
  //       cuFuncSetCacheConfig(fun, CU_FUNC_CACHE_PREFER_SHARED));
  //   int shared_total, shared_static;
  //   TANG_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuDeviceGetAttribute(
  //       &shared_total, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
  //       device));
  //   TANG_CHECK_AND_RETURN_NULL_ALLOW_THREADS(cuFuncGetAttribute(
  //       &shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun));
  //   TANG_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
  //       cuFuncSetAttribute(fun, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
  //                          shared_optin - shared_static));
  // }
  Py_END_ALLOW_THREADS;

  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("(KKiii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills, n_max_threads);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided cubin into TANG driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "tang_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_tang_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, ModuleMethods);

  return m;
}
