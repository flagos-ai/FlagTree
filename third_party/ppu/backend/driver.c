/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "hggc.h"
#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct {
  PyObject_HEAD;
  _Alignas(128) HGtensorMap tensorMap;
} PyHGtensorMapObject;

// Raises a Python exception and returns false if code is not HGGC_SUCCESS.
static bool gpuAssert(HGresult code, const char *file, int line) {
  if (code == HGGC_SUCCESS)
    return true;

  const char *prefix = "Triton Error [HGGC]: ";
  const char *str;
  hgGetErrorString(code, &str);
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
#define HGGC_CHECK_AND_RETURN_NULL(ans)                                        \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__))                                 \
      goto cleanup;                                                            \
  } while (0)

// To be used inside a Py_{BEGIN,END}_ALLOW_THREADS block.
#define HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(ans)                          \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__)) {                               \
      PyEval_RestoreThread(_save);                                             \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

// Used to check if functions exist in old HGGC driver versions.
#define INITIALIZE_FUNCTION_POINTER_IF_NULL(funcPointer, initializerFunction)  \
  do {                                                                         \
    if ((funcPointer) == NULL) {                                               \
      (funcPointer) = (initializerFunction)();                                 \
      if ((funcPointer) == NULL) {                                             \
        goto cleanup;                                                          \
      }                                                                        \
    }                                                                          \
  } while (0)

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;
  // Get device handle
  HGdevice device;
  hgDeviceGet(&device, device_id);

  // create a struct to hold device properties
  int max_shared_mem;
  int max_num_regs;
  int multiprocessor_count;
  int warp_size;
  int sm_clock_rate;
  int mem_clock_rate;
  int mem_bus_width;
  HGGC_CHECK_AND_RETURN_NULL(hgDeviceGetAttribute(
      &max_shared_mem, HG_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  HGGC_CHECK_AND_RETURN_NULL(hgDeviceGetAttribute(
      &max_num_regs, HG_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device));
  HGGC_CHECK_AND_RETURN_NULL(hgDeviceGetAttribute(
      &multiprocessor_count, HG_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  HGGC_CHECK_AND_RETURN_NULL(
      hgDeviceGetAttribute(&warp_size, HG_DEVICE_ATTRIBUTE_WARP_SIZE, device));
  HGGC_CHECK_AND_RETURN_NULL(hgDeviceGetAttribute(
      &sm_clock_rate, HG_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
  HGGC_CHECK_AND_RETURN_NULL(hgDeviceGetAttribute(
      &mem_clock_rate, HG_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
  HGGC_CHECK_AND_RETURN_NULL(hgDeviceGetAttribute(
      &mem_bus_width, HG_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));

  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i, s:i, s:i}", "max_shared_mem",
                       max_shared_mem, "max_num_regs", max_num_regs,
                       "multiprocessor_count", multiprocessor_count, "warpSize",
                       warp_size, "sm_clock_rate", sm_clock_rate,
                       "mem_clock_rate", mem_clock_rate, "mem_bus_width",
                       mem_bus_width);

cleanup:
  return NULL;
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
  HGfunction fun;
  HGmodule mod;
  int32_t n_regs = 0;
  int32_t n_spills = 0;
  int32_t n_max_threads = 0;
  // create driver handles
  HGcontext pctx = 0;

  Py_BEGIN_ALLOW_THREADS;
  HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(hgCtxGetCurrent(&pctx));
  if (!pctx) {
    HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        hgDevicePrimaryCtxRetain(&pctx, device));
    HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(hgCtxSetCurrent(pctx));
  }

  HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(hgModuleLoadData(&mod, data));
  HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      hgModuleGetFunction(&fun, mod, name));
  // get allocated registers and spilled registers from the function
  HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      hgFuncGetAttribute(&n_regs, HG_FUNC_ATTRIBUTE_NUM_REGS, fun));
  HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      hgFuncGetAttribute(&n_spills, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
  n_spills /= 4;
  HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(hgFuncGetAttribute(
      &n_max_threads, HG_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fun));
  // set dynamic shared memory if necessary
  int shared_optin;
  HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(hgDeviceGetAttribute(
      &shared_optin, HG_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  if (shared > 49152 && shared_optin > 49152) {
    HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        hgFuncSetCacheConfig(fun, HG_FUNC_CACHE_PREFER_SHARED));
    int shared_total, shared_static;
    HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(hgDeviceGetAttribute(
        &shared_total, HG_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        device));
    HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(hgFuncGetAttribute(
        &shared_static, HG_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun));
    HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        hgFuncSetAttribute(fun, HG_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_optin - shared_static));
  }
  Py_END_ALLOW_THREADS;

  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("(KKiii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills, n_max_threads);
}

typedef HGresult (*hgOccupancyMaxActiveClusters_t)(
    int *numClusters, HGfunction func, const HGlaunchConfig *config);

typedef HGresult (*hgTensorMapEncodeTiled_t)(
    HGtensorMap *tensorMap, HGtensorMapDataType tensorDataType,
    hguint32_t tensorRank, void *globalAddress, const hguint64_t *globalDim,
    const hguint64_t *globalStrides, const hguint32_t *boxDim,
    const hguint32_t *elementStrides, HGtensorMapInterleave interleave,
    HGtensorMapSwizzle swizzle, HGtensorMapL2promotion l2Promotion,
    HGtensorMapFloatOOBfill oobFill);

#define defineGetFunctionHandle(name, symbolName)                              \
  static symbolName##_t name() {                                               \
    /* Open the shared library */                                              \
    void *libHandle = dlopen("libhggc.so", RTLD_LAZY);                         \
    if (!libHandle) {                                                          \
      PyErr_SetString(PyExc_RuntimeError, "Failed to open libhggc.so");        \
      return NULL;                                                             \
    }                                                                          \
    /* Clear any existing error */                                             \
    dlerror();                                                                 \
    symbolName##_t funcHandle = (symbolName##_t)dlsym(libHandle, #symbolName); \
    /* Check for errors */                                                     \
    const char *err = dlerror();                                               \
    if (err) {                                                                 \
      PyErr_SetString(PyExc_RuntimeError,                                      \
                      "Failed to retrieve " #symbolName " from libhggc.so");   \
      dlclose(libHandle);                                                      \
      return NULL;                                                             \
    }                                                                          \
    return funcHandle;                                                         \
  }

defineGetFunctionHandle(getHgOccupancyMaxActiveClustersHandle,
                        hgOccupancyMaxActiveClusters);

defineGetFunctionHandle(getHgTensorMapEncodeTiledHandle,
                        hgTensorMapEncodeTiled);

static PyObject *occupancyMaxActiveClusters(PyObject *self, PyObject *args) {
  int clusterDim = -1, maxActiveClusters = -1;
  int shared = 0;
  HGfunction func;

  if (!PyArg_ParseTuple(args, "Kii", &func, &shared, &clusterDim)) {
    return NULL;
  }

  // Let each SM have one block
  int maxActiveBlocks = 1;
  Py_BEGIN_ALLOW_THREADS;
  HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(hgFuncSetAttribute(
      func, HG_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared));
  Py_END_ALLOW_THREADS;

  HGlaunchAttribute launchAttr[1];
  launchAttr[0].id = HG_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  launchAttr[0].value.clusterDim.x = clusterDim;
  launchAttr[0].value.clusterDim.y = 1;
  launchAttr[0].value.clusterDim.z = 1;
  HGlaunchConfig config;
  config.gridDimX = clusterDim * maxActiveBlocks;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 128;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = shared;
  config.hStream = 0;
  config.numAttrs = 1;
  config.attrs = launchAttr;

  static hgOccupancyMaxActiveClusters_t hgOccupancyMaxActiveClusters = NULL;
  INITIALIZE_FUNCTION_POINTER_IF_NULL(hgOccupancyMaxActiveClusters,
                                      getHgOccupancyMaxActiveClustersHandle);

  Py_BEGIN_ALLOW_THREADS;
  HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(hgFuncSetAttribute(
      func, HG_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
  HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      hgOccupancyMaxActiveClusters(&maxActiveClusters, func, &config));
  Py_END_ALLOW_THREADS;
  return PyLong_FromLong(maxActiveClusters);

cleanup:
  return NULL;
}

static PyObject *setPrintfFifoSize(PyObject *self, PyObject *args) {
  long size;
  if (!PyArg_ParseTuple(args, "l", &size)) {
    return NULL;
  }
  if (size < 0) {
    PyErr_SetString(PyExc_ValueError, "fifo size must be non-negative");
    return NULL;
  }

  Py_BEGIN_ALLOW_THREADS;

  // Ensure we have an active context.
  HGcontext ctx = NULL;
  HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(hgCtxGetCurrent(&ctx));
  if (!ctx) {
    HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        hgDevicePrimaryCtxRetain(&ctx, /*device=*/0));
    HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(hgCtxSetCurrent(ctx));
  }

  // We can't set the fifo size after running a kernel that calls printf.  This
  // is true even if the set() call is a nop and the new size is the same as the
  // old size.
  //
  // This is unfriendly, so check if the old size matches the new size, and skip
  // the set() call if so.
  size_t oldSize = 0;
  HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      hgCtxGetLimit(&oldSize, HG_LIMIT_PRINTF_FIFO_SIZE));
  if (oldSize != size) {
    HGGC_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        hgCtxSetLimit(HG_LIMIT_PRINTF_FIFO_SIZE, size));
  }

  Py_END_ALLOW_THREADS;
  Py_RETURN_NONE;
}

static PyObject *PyHGtensorMap_alloc(PyTypeObject *type, Py_ssize_t n_items) {
  PyHGtensorMapObject *self = NULL;
  void *mem = NULL;
  size_t size = type->tp_basicsize;

  if (posix_memalign(&mem, 128, size) != 0) {
    PyErr_NoMemory();
    return NULL;
  }

  self = (PyHGtensorMapObject *)mem;
  PyObject_INIT(self, type);
  return (PyObject *)self;
}

static void PyHGtensorMap_dealloc(PyObject *self) {
  Py_TYPE(self)->tp_free(self);
}

static void PyHGtensorMap_free(void *ptr) { free(ptr); }

// clang-format off
static PyTypeObject PyHGtensorMapType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "triton.backends.ppu.PyHGtensorMap",
    .tp_basicsize = sizeof(PyHGtensorMapObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "<PyHGtensorMap object>",
    .tp_new = PyType_GenericNew,
    .tp_alloc = PyHGtensorMap_alloc,
    .tp_dealloc = (destructor)PyHGtensorMap_dealloc,
    .tp_free = PyHGtensorMap_free,
};
// clang-format on

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided hgbin into HGGC driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {"hgOccupancyMaxActiveClusters", occupancyMaxActiveClusters, METH_VARARGS,
     "Python interface for hgOccupancyMaxActiveClusters function"},
    {"set_printf_fifo_size", setPrintfFifoSize, METH_VARARGS,
     "Python interface for hgCtxSetLimit(HG_LIMIT_PRINTF_FIFO_SIZE, x), which "
     "controls how many bytes can be streamed from kernels before data starts "
     "being dropped.  This inherits all the limitations of this call; in "
     "particular it's an error to change this value after launching any kernel "
     "that calls printf()."},

    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "hggc_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_hggc_utils(void) {
  if (PyType_Ready(&PyHGtensorMapType) < 0) {
    return NULL;
  }

  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, ModuleMethods);
  Py_INCREF(&PyHGtensorMapType);
  PyModule_AddObject(m, "PyHGtensorMap", (PyObject *)&PyHGtensorMapType);

  return m;
}
