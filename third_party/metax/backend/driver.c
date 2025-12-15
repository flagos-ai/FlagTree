/* Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd. All
 * Rights Reserved. */
#include <dlfcn.h>
#include <mcr/mc_runtime.h>
#include <stdbool.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdatomic.h>

// Raises a Python exception and returns false if code is not MC_SUCCESS.
static bool gpuAssert(mcError_t code, const char *file, int line) {
  if (code == mcSuccess)
    return true;

  const char *prefix = "Triton Error [MACA]: ";
  const char *str = mcGetErrorString(code);
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
#define MACA_CHECK_AND_RETURN_NULL(ans)                                        \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__))                                 \
      return NULL;                                                             \
  } while (0)

// To be used inside a Py_{BEGIN,END}_ALLOW_THREADS block.
#define MACA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(ans)                          \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__)) {                               \
      PyEval_RestoreThread(_save);                                             \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

// Used to check if functions exist in old CUDA driver versions.
#define INITIALIZE_FUNCTION_POINTER_IF_NULL(funcPointer, initializerFunction)  \
  do {                                                                         \
    if ((funcPointer) == NULL) {                                               \
      (funcPointer) = (initializerFunction)();                                 \
      if ((funcPointer) == NULL) {                                             \
        return NULL;                                                           \
      }                                                                        \
    }                                                                          \
  } while (0)

static PyObject *getDeviceCapability(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;
  int capability = 0;
  MCdevice device;
  mcDeviceGet(&device, device_id);
  mcDeviceProp_t device_prop;
  MACA_CHECK_AND_RETURN_NULL(mcGetDeviceProperties(&device_prop, device_id));
  int major = device_prop.major;
  switch (major) {
  case 10:
    capability = 80;
    break;
  case 15:
    capability = 86;
    break;
  case 16:
    capability = 89;
    break;
  default:
    assert(false && "init device capabilities failed");
    break;
  }
  return Py_BuildValue("{s:i}", "capability", capability);
}

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;
  // Get device handle
  MCdevice device;
  mcDeviceGet(&device, device_id);

  // create a struct to hold device properties
  int max_shared_mem = 64 * 1024; // 64KB, default C500
  int max_num_regs;
  int multiprocessor_count;
  int warp_size = 64;
  int sm_clock_rate;
  int mem_clock_rate;
  int mem_bus_width;
  MACA_CHECK_AND_RETURN_NULL(mcDeviceGetAttribute(
      &max_shared_mem, mcDeviceAttributeMaxSharedMemoryPerBlock, device));
  MACA_CHECK_AND_RETURN_NULL(mcDeviceGetAttribute(
      &max_num_regs, mcDeviceAttributeMaxRegistersPerBlock, device));
  MACA_CHECK_AND_RETURN_NULL(mcDeviceGetAttribute(
      &multiprocessor_count, mcDeviceAttributeMultiProcessorCount, device));
  MACA_CHECK_AND_RETURN_NULL(
      mcDeviceGetAttribute(&sm_clock_rate, mcDeviceAttributeClockRate, device));
  MACA_CHECK_AND_RETURN_NULL(mcDeviceGetAttribute(
      &mem_clock_rate, mcDeviceAttributeMemoryClockRate, device));
  MACA_CHECK_AND_RETURN_NULL(mcDeviceGetAttribute(
      &mem_bus_width, mcDeviceAttributeMemoryBusWidth, device));

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
  mcFunction_t fun;
  mcModule_t mod;
  int32_t n_regs = 0;
  int32_t n_spills = 0;
  // create driver handles
  MCcontext pctx = 0;

  Py_BEGIN_ALLOW_THREADS;
  // TODO: MCcontext implement not found
  MACA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(mcCtxGetCurrent(&pctx));
  if (!pctx) {
    MACA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        mcDevicePrimaryCtxRetain(&pctx, device));
    MACA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(mcCtxSetCurrent(pctx));
  }
  MACA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(mcModuleLoadData(&mod, data));
  MACA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      mcModuleGetFunction(&fun, mod, name));
  // get allocated registers and spilled registers from the function
  MACA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      mcFuncGetAttribute(&n_regs, MC_FUNC_ATTRIBUTE_NUM_REGS, fun));
  MACA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      mcFuncGetAttribute(&n_spills, MC_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
  n_spills /= 4;
  Py_END_ALLOW_THREADS;

  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("(KKii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills);
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
  // MCcontext ctx = NULL;
  // TODO: CU_LIMIT_PRINTF_FIFO_SIZE implement not found
  // MACA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(mcCtxGetCurrent(&ctx));
  // if (!ctx) {
  //   MACA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
  //       mcDevicePrimaryCtxRetain(&ctx, /*device=*/0));
  //   MACA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(mcCtxSetCurrent(ctx));
  // }

  // // We can't set the fifo size after running a kernel that calls printf.
  // This
  // // is true even if the set() call is a nop and the new size is the same as
  // the
  // // old size.
  // //
  // // This is unfriendly, so check if the old size matches the new size, and
  // skip
  // // the set() call if so.
  // size_t oldSize = 0;
  // MACA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
  //     mcCtxGetLimit(&oldSize, CU_LIMIT_PRINTF_FIFO_SIZE));
  // if (oldSize != size) {
  //   MACA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
  //       mcCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, size));
  // }

  Py_END_ALLOW_THREADS;
  return Py_None;
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided cubin into CUDA driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {"get_device_capability", getDeviceCapability, METH_VARARGS,
     "Get the capabilitity for a given device"},
    {"set_printf_fifo_size", setPrintfFifoSize, METH_VARARGS,
     "Python interface for cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, x), which "
     "controls how many bytes can be streamed from kernels before data starts "
     "being dropped.  This inherits all the limitations of this call; in "
     "particular it's an error to change this value after launching any kernel "
     "that calls printf()."},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "maca_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_maca_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, ModuleMethods);

  return m;
}
