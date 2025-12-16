#include "musa.h"
#include <dlfcn.h>
#include <stdbool.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

// Raises a Python exception and returns false if code is not MUSA_SUCCESS.
static bool gpuAssert(MUresult code, const char *file, int line) {
  if (code == MUSA_SUCCESS)
    return true;

  const char *prefix = "Triton Error [MUSA]: ";
  const char *str;
  muGetErrorString(code, &str);
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
#define MUSA_CHECK_AND_RETURN_NULL(ans)                                        \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__))                                 \
      return NULL;                                                             \
  } while (0)

// To be used inside a Py_{BEGIN,END}_ALLOW_THREADS block.
#define MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(ans)                          \
  do {                                                                         \
    if (!gpuAssert((ans), __FILE__, __LINE__)) {                               \
      PyEval_RestoreThread(_save);                                             \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

// Used to check if functions exist in old MUSA driver versions.
#define INITIALIZE_FUNCTION_POINTER_IF_NULL(funcPointer, initializerFunction)  \
  do {                                                                         \
    if ((funcPointer) == NULL) {                                               \
      (funcPointer) = (initializerFunction)();                                 \
      if ((funcPointer) == NULL) {                                             \
        return NULL;                                                           \
      }                                                                        \
    }                                                                          \
  } while (0)

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;
  // Get device handle
  MUdevice device;
  muDeviceGet(&device, device_id);

  // create a struct to hold device properties
  int max_shared_mem;
  int max_num_regs;
  int multiprocessor_count;
  int warp_size;
  int sm_clock_rate;
  int mem_clock_rate;
  int mem_bus_width;
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &max_shared_mem, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &max_num_regs, MU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device));
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &multiprocessor_count, MU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  MUSA_CHECK_AND_RETURN_NULL(
      muDeviceGetAttribute(&warp_size, MU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &sm_clock_rate, MU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &mem_clock_rate, MU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
  MUSA_CHECK_AND_RETURN_NULL(muDeviceGetAttribute(
      &mem_bus_width, MU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));

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
  MUfunction fun;
  MUmodule mod;
  int32_t n_regs = 0;
  int32_t n_spills = 0;
  // create driver handles
  MUcontext pctx = 0;

  Py_BEGIN_ALLOW_THREADS;
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muCtxGetCurrent(&pctx));
  if (!pctx) {
    MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        muDevicePrimaryCtxRetain(&pctx, device));
    MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muCtxSetCurrent(pctx));
  }

  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muModuleLoadData(&mod, data));
  // MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muModuleLoad(&mod, data));
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      muModuleGetFunction(&fun, mod, name));
  // get allocated registers and spilled registers from the function
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      muFuncGetAttribute(&n_regs, MU_FUNC_ATTRIBUTE_NUM_REGS, fun));
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      muFuncGetAttribute(&n_spills, MU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
  n_spills /= 4;
  // set dynamic shared memory if necessary
  int shared_optin;
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muDeviceGetAttribute(
      &shared_optin, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));

  int max_shared;
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muDeviceGetAttribute(
      &max_shared, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  if (shared > max_shared) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Requested shared memory exceeds device limit");
    return NULL;
  }
  Py_END_ALLOW_THREADS;

  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("(KKii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills);
}

// Simple helper to experiment creating TMA descriptors on the host.
// This is a useful to test TMA operations independently.
static PyObject *fill1DTMEDescriptor(PyObject *self, PyObject *args) {
  unsigned long long global_address;
  uint64_t dim;
  uint32_t tensorDim;
  int elementSize;
  Py_buffer desc_buffer;
  if (!PyArg_ParseTuple(args, "KKiiy*", &global_address, &dim, &tensorDim,
                        &elementSize, &desc_buffer)) {
    return NULL;
  }
  char *desc = (char *)desc_buffer.buf;
  uint64_t dims[1] = {dim};
  uint64_t globalStrides[1] = {dim * elementSize};
  uint32_t elementStrides[1] = {1};
  MUtensorDescriptorDataType type;
  // FIXME: mcj. shall we specify specific type, like u8/s8, fp16/bf16, fp32,
  // tf32?
  switch (elementSize) {
  case 1:
    type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8;
    break;
  case 2:
    type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT16;
    break;
  case 4:
    type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT32;
    break;
  default:
    PyErr_SetString(PyExc_ValueError, "elementSize must be 1, 2, or 4");
  }
  assert((elementSize * tensorDim) >= 32 && "block size too small.");
  int rank = 1;

  MUresult result = muTensorDescriptorEncode(
      (MUtensorDescriptor *)desc, type, rank, (void *)global_address, dims,
      globalStrides, MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE, 0);
  assert(result == MUSA_SUCCESS);
  return Py_None;
}

// Simple helper to experiment creating TME descriptors on the host.
// This is a useful to test TMA operations independently.
static PyObject *fill2DTMEDescriptor(PyObject *self, PyObject *args) {
  unsigned long long global_address;
  uint64_t dims[2];
  uint32_t tensorDims[2];
  int elementSize;
  Py_buffer desc_buffer;
  if (!PyArg_ParseTuple(args, "KKKiiiy*", &global_address, &dims[1], &dims[0],
                        &tensorDims[1], &tensorDims[0], &elementSize,
                        &desc_buffer)) {
    return NULL;
  }
  char *desc = (char *)desc_buffer.buf;
  uint64_t globalStrides[2] = {dims[0] * elementSize,
                               dims[0] * dims[1] * elementSize};
  uint32_t elementStrides[2] = {1, 1};
  MUtensorDescriptorDataType type;
  switch (elementSize) {
  case 1:
    type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8;
    break;
  case 2:
    type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT16;
    break;
  case 4:
    type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT32;
    break;
  default:
    PyErr_SetString(PyExc_ValueError, "elementSize must be 1, 2, or 4");
  }
  int rank = 2;

  MUresult result = muTensorDescriptorEncode(
      (MUtensorDescriptor *)desc, type, rank, (void *)global_address, dims,
      globalStrides, MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE, 0);
  assert(result == MUSA_SUCCESS);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided mubin into MUSA driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {"fill_1d_tma_descriptor", fill1DTMEDescriptor, METH_VARARGS,
     "create a tme 1D descriptor"},
    {"fill_2d_tma_descriptor", fill2DTMEDescriptor, METH_VARARGS,
     "create a tme 2D descriptor"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "musa_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_musa_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, ModuleMethods);

  return m;
}
