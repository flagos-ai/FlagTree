#include "musa.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

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

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;
  MUdevice device;
  muDeviceGet(&device, device_id);

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
  if (data_size == 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Empty MUSA binary: codegen is not available yet.");
    return NULL;
  }
  MUfunction fun;
  MUmodule mod;
  int32_t n_regs = 0;
  int32_t n_spills = 0;
  int32_t n_max_threads = 0;
  MUcontext pctx = 0;

  Py_BEGIN_ALLOW_THREADS;
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muCtxGetCurrent(&pctx));
  if (!pctx) {
    MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        muDevicePrimaryCtxRetain(&pctx, device));
    MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muCtxSetCurrent(pctx));
  }

  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muModuleLoadData(&mod, data));
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      muModuleGetFunction(&fun, mod, name));
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      muFuncGetAttribute(&n_regs, MU_FUNC_ATTRIBUTE_NUM_REGS, fun));
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      muFuncGetAttribute(&n_spills, MU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
  n_spills /= 4;
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muFuncGetAttribute(
      &n_max_threads, MU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fun));

  int shared_optin = 0;
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muDeviceGetAttribute(
      &shared_optin, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));

  int shared_static = 0;
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muFuncGetAttribute(
      &shared_static, MU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun));
  int max_dynamic_shared = shared_optin - shared_static;
  if (max_dynamic_shared < 0)
    max_dynamic_shared = 0;
  int requested_dynamic_shared = shared;
  if (requested_dynamic_shared > max_dynamic_shared)
    requested_dynamic_shared = max_dynamic_shared;
  if (requested_dynamic_shared > 0) {
    MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        muFuncSetAttribute(fun, MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           requested_dynamic_shared));
  }
  Py_END_ALLOW_THREADS;

  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("(KKiii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills, n_max_threads);
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

  MUcontext ctx = NULL;
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muCtxGetCurrent(&ctx));
  if (!ctx) {
    MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        muDevicePrimaryCtxRetain(&ctx, /*device=*/0));
    MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(muCtxSetCurrent(ctx));
  }

  size_t oldSize = 0;
  MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
      muCtxGetLimit(&oldSize, MU_LIMIT_PRINTF_FIFO_SIZE));
  if (oldSize != (size_t)size) {
    MUSA_CHECK_AND_RETURN_NULL_ALLOW_THREADS(
        muCtxSetLimit(MU_LIMIT_PRINTF_FIFO_SIZE, size));
  }

  Py_END_ALLOW_THREADS;
  Py_INCREF(Py_None);
  return Py_None;
}

static bool getTensorDescriptorDataType(int elementSize,
                                        MUtensorDescriptorDataType *type) {
  switch (elementSize) {
  case 1:
    *type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8;
    return true;
  case 2:
    *type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT16;
    return true;
  case 4:
    *type = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT32;
    return true;
  default:
    PyErr_SetString(PyExc_ValueError, "element_size must be 1, 2, or 4 bytes");
    return false;
  }
}

static bool validateTMEDescriptorBlockBytes(unsigned rank,
                                            const uint32_t *block_dims,
                                            int element_size) {
  uint64_t block_bytes = (uint64_t)element_size;
  for (unsigned i = 0; i < rank; ++i)
    block_bytes *= (uint64_t)block_dims[i];
  if (block_bytes >= 32)
    return true;

  char err[64] = {0};
  snprintf(err, sizeof(err), "%uD block bytes must be >= 32", rank);
  PyErr_SetString(PyExc_ValueError, err);
  return false;
}

static PyObject *
fillTMEDescriptorImpl(unsigned rank, unsigned long long global_address,
                      const uint64_t *dims, const uint32_t *block_dims,
                      int element_size, unsigned long long desc_address) {
  MUtensorDescriptorDataType type;
  if (!getTensorDescriptorDataType(element_size, &type))
    return NULL;
  if (!validateTMEDescriptorBlockBytes(rank, block_dims, element_size))
    return NULL;

  uint64_t global_strides[5] = {0};
  global_strides[0] = dims[0] * (uint64_t)element_size;
  for (unsigned i = 1; i < rank; ++i)
    global_strides[i] = global_strides[i - 1] * dims[i];

  MUtensorDescriptor desc;
  MUSA_CHECK_AND_RETURN_NULL(muTensorDescriptorEncode(
      &desc, type, /*tensorRank=*/rank, (void *)global_address, dims,
      global_strides, MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE, /*swizzle=*/0));
  MUSA_CHECK_AND_RETURN_NULL(
      muMemcpyHtoD((MUdeviceptr)desc_address, &desc, sizeof(desc)));
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *fill1DTMEDescriptor(PyObject *self, PyObject *args) {
  unsigned long long global_address = 0;
  uint64_t dims[1];
  uint32_t block_dims[1];
  int element_size = 0;
  unsigned long long desc_address = 0;
  if (!PyArg_ParseTuple(args, "KKiiK", &global_address, &dims[0],
                        &block_dims[0], &element_size, &desc_address))
    return NULL;

  return fillTMEDescriptorImpl(/*rank=*/1, global_address, dims, block_dims,
                               element_size, desc_address);
}

static PyObject *fill2DTMEDescriptor(PyObject *self, PyObject *args) {
  unsigned long long global_address = 0;
  uint64_t dims[2];
  uint32_t block_dims[2];
  int element_size = 0;
  unsigned long long desc_address = 0;
  if (!PyArg_ParseTuple(args, "KKKiiiK", &global_address, &dims[1], &dims[0],
                        &block_dims[1], &block_dims[0], &element_size,
                        &desc_address))
    return NULL;

  return fillTMEDescriptorImpl(/*rank=*/2, global_address, dims, block_dims,
                               element_size, desc_address);
}

static PyObject *fill3DTMEDescriptor(PyObject *self, PyObject *args) {
  unsigned long long global_address = 0;
  uint64_t dims[3];
  uint32_t block_dims[3];
  int element_size = 0;
  unsigned long long desc_address = 0;
  if (!PyArg_ParseTuple(args, "KKKKiiiiK", &global_address, &dims[2], &dims[1],
                        &dims[0], &block_dims[2], &block_dims[1],
                        &block_dims[0], &element_size, &desc_address))
    return NULL;

  return fillTMEDescriptorImpl(/*rank=*/3, global_address, dims, block_dims,
                               element_size, desc_address);
}

static PyObject *fill4DTMEDescriptor(PyObject *self, PyObject *args) {
  unsigned long long global_address = 0;
  uint64_t dims[4];
  uint32_t block_dims[4];
  int element_size = 0;
  unsigned long long desc_address = 0;
  if (!PyArg_ParseTuple(args, "KKKKKiiiiiK", &global_address, &dims[3],
                        &dims[2], &dims[1], &dims[0], &block_dims[3],
                        &block_dims[2], &block_dims[1], &block_dims[0],
                        &element_size, &desc_address))
    return NULL;

  return fillTMEDescriptorImpl(/*rank=*/4, global_address, dims, block_dims,
                               element_size, desc_address);
}

static PyObject *fill5DTMEDescriptor(PyObject *self, PyObject *args) {
  unsigned long long global_address = 0;
  uint64_t dims[5];
  uint32_t block_dims[5];
  int element_size = 0;
  unsigned long long desc_address = 0;
  if (!PyArg_ParseTuple(args, "KKKKKKiiiiiiK", &global_address, &dims[4],
                        &dims[3], &dims[2], &dims[1], &dims[0], &block_dims[4],
                        &block_dims[3], &block_dims[2], &block_dims[1],
                        &block_dims[0], &element_size, &desc_address))
    return NULL;

  return fillTMEDescriptorImpl(/*rank=*/5, global_address, dims, block_dims,
                               element_size, desc_address);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided mubin into MUSA driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {"set_printf_fifo_size", setPrintfFifoSize, METH_VARARGS,
     "Set printf FIFO size"},
    {"fill_1d_tma_descriptor", fill1DTMEDescriptor, METH_VARARGS,
     "Fill a 1D TMA descriptor"},
    {"fill_2d_tma_descriptor", fill2DTMEDescriptor, METH_VARARGS,
     "Fill a 2D TMA descriptor"},
    {"fill_3d_tma_descriptor", fill3DTMEDescriptor, METH_VARARGS,
     "Fill a 3D TMA descriptor"},
    {"fill_4d_tma_descriptor", fill4DTMEDescriptor, METH_VARARGS,
     "Fill a 4D TMA descriptor"},
    {"fill_5d_tma_descriptor", fill5DTMEDescriptor, METH_VARARGS,
     "Fill a 5D TMA descriptor"},
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
