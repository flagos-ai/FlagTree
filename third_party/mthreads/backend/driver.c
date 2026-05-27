#include "musa.h"
#include <stdalign.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct {
  PyObject_HEAD;
  _Alignas(64) MUtensorDescriptor desc;
} PyMUtensorDescriptorObject;

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

static bool encodeTMEDescriptor(unsigned rank,
                                unsigned long long global_address,
                                const uint64_t *dims,
                                const uint32_t *block_dims, int element_size,
                                MUtensorDescriptor *desc) {
  MUtensorDescriptorDataType type;
  if (!getTensorDescriptorDataType(element_size, &type))
    return false;
  if (!validateTMEDescriptorBlockBytes(rank, block_dims, element_size))
    return false;

  uint64_t global_strides[5] = {0};
  global_strides[0] = dims[0] * (uint64_t)element_size;
  for (unsigned i = 1; i < rank; ++i)
    global_strides[i] = global_strides[i - 1] * dims[i];

  return gpuAssert(muTensorDescriptorEncode(
                       desc, type, /*tensorRank=*/rank, (void *)global_address,
                       dims, global_strides,
                       MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE, /*swizzle=*/0),
                   __FILE__, __LINE__);
}

static PyObject *
PyMUtensorDescriptor_tme_desc_cpu_ptr(PyMUtensorDescriptorObject *self,
                                      PyObject *Py_UNUSED(ignored)) {
  return PyLong_FromUnsignedLongLong((unsigned long long)&self->desc);
}

static PyMethodDef PyMUtensorDescriptor_methods[] = {
    {"tme_desc_cpu_ptr", (PyCFunction)PyMUtensorDescriptor_tme_desc_cpu_ptr,
     METH_NOARGS,
     "Return the 64-byte aligned CPU address of the MUSA tensor descriptor"},
    {NULL, NULL, 0, NULL}};

static PyObject *PyMUtensorDescriptor_alloc(PyTypeObject *type,
                                            Py_ssize_t n_items) {
  void *mem = NULL;
  if (posix_memalign(&mem, 64, type->tp_basicsize) != 0) {
    PyErr_NoMemory();
    return NULL;
  }
  PyMUtensorDescriptorObject *self = (PyMUtensorDescriptorObject *)mem;
  PyObject_INIT(self, type);
  memset(&self->desc, 0, sizeof(self->desc));
  return (PyObject *)self;
}

static void PyMUtensorDescriptor_dealloc(PyObject *self) {
  Py_TYPE(self)->tp_free(self);
}

static void PyMUtensorDescriptor_free(void *ptr) { free(ptr); }

static PyTypeObject PyMUtensorDescriptorType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name =
        "triton.backends.musa.PyMUtensorDescriptor",
    .tp_basicsize = sizeof(PyMUtensorDescriptorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "<PyMUtensorDescriptor object>",
    .tp_new = PyType_GenericNew,
    .tp_alloc = PyMUtensorDescriptor_alloc,
    .tp_dealloc = (destructor)PyMUtensorDescriptor_dealloc,
    .tp_free = PyMUtensorDescriptor_free,
    .tp_methods = PyMUtensorDescriptor_methods,
};

static PyObject *fillTMEDescriptor(PyObject *self, PyObject *args) {
  unsigned long long global_address = 0;
  PyObject *shape = NULL;
  PyObject *block_shape = NULL;
  int element_size = 0;
  if (!PyArg_ParseTuple(args, "KOOi", &global_address, &shape, &block_shape,
                        &element_size))
    return NULL;

  PyObject *shape_fast = PySequence_Fast(shape, "shape must be a sequence");
  if (!shape_fast)
    return NULL;
  PyObject *block_fast =
      PySequence_Fast(block_shape, "block_shape must be a sequence");
  if (!block_fast) {
    Py_DECREF(shape_fast);
    return NULL;
  }

  Py_ssize_t rank = PySequence_Fast_GET_SIZE(shape_fast);
  if (rank != PySequence_Fast_GET_SIZE(block_fast)) {
    PyErr_SetString(PyExc_ValueError, "shape and block_shape rank mismatch");
    Py_DECREF(shape_fast);
    Py_DECREF(block_fast);
    return NULL;
  }
  if (rank <= 0 || rank > 5) {
    PyErr_SetString(PyExc_ValueError, "MUSA TME descriptor rank must be 1..5");
    Py_DECREF(shape_fast);
    Py_DECREF(block_fast);
    return NULL;
  }

  uint64_t dims[5] = {0};
  uint32_t block_dims[5] = {0};
  for (Py_ssize_t i = 0; i < rank; ++i) {
    PyObject *dim = PySequence_Fast_GET_ITEM(shape_fast, i);
    PyObject *block_dim = PySequence_Fast_GET_ITEM(block_fast, i);
    if (!PyLong_Check(dim) || !PyLong_Check(block_dim)) {
      PyErr_SetString(PyExc_TypeError,
                      "shape and block_shape values must be integers");
      Py_DECREF(shape_fast);
      Py_DECREF(block_fast);
      return NULL;
    }
    Py_ssize_t desc_i = rank - i - 1;
    uint64_t val = PyLong_AsUnsignedLongLong(dim);
    if (val == (uint64_t)-1 && PyErr_Occurred()) {
      Py_DECREF(shape_fast);
      Py_DECREF(block_fast);
      return NULL;
    }
    dims[desc_i] = val;
    unsigned long bval = PyLong_AsUnsignedLong(block_dim);
    if (bval == (unsigned long)-1 && PyErr_Occurred()) {
      Py_DECREF(shape_fast);
      Py_DECREF(block_fast);
      return NULL;
    }
    block_dims[desc_i] = (uint32_t)bval;
  }
  Py_DECREF(shape_fast);
  Py_DECREF(block_fast);

  PyMUtensorDescriptorObject *desc =
      (PyMUtensorDescriptorObject *)PyObject_CallObject(
          (PyObject *)&PyMUtensorDescriptorType, NULL);
  if (!desc)
    return NULL;

  if (!encodeTMEDescriptor((unsigned)rank, global_address, dims, block_dims,
                           element_size, &desc->desc)) {
    Py_DECREF(desc);
    return NULL;
  }
  return (PyObject *)desc;
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided mubin into MUSA driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {"set_printf_fifo_size", setPrintfFifoSize, METH_VARARGS,
     "Set printf FIFO size"},
    {"fill_tme_descriptor", fillTMEDescriptor, METH_VARARGS,
     "Fill a host by-value TME descriptor"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "musa_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_musa_utils(void) {
  if (PyType_Ready(&PyMUtensorDescriptorType) < 0)
    return NULL;
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  Py_INCREF(&PyMUtensorDescriptorType);
  PyModule_AddObject(m, "PyMUtensorDescriptor",
                     (PyObject *)&PyMUtensorDescriptorType);
  return m;
}
