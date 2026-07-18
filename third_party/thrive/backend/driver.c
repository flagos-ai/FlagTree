#include "driver.hpp"
#include <stdbool.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *getSRAMUsedBytes(PyObject *self, PyObject *args) {
  uint64_t kernel;
  dfcaFuncAttributes attr;

  if (!PyArg_ParseTuple(args, "K", &kernel)) {
    return NULL;
  }

  CHECK_DFCA(dfcaFuncGetAttributes(&attr, (const void *)kernel));
  if (PyErr_Occurred()) {
    return NULL;
  }
  return PyLong_FromUnsignedLongLong(attr.shared_size_bytes);
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

  dfcaLibrary_t program = nullptr;
  CHECK_DFCA(dfcaLibraryLoadFromFile(&program, data));
  dfcaKernel_t kernel = nullptr;
  CHECK_DFCA(dfcaLibraryGetKernel(&kernel, program, name));

  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("(KKiii)", (uint64_t)program, (uint64_t)kernel,
                       /*n_regs=*/0, /*n_spills=*/0, /*n_max_threads=*/0);
}

static PyObject *getDeviceCapability(PyObject *self, PyObject *args) {
  int device;
  if (!PyArg_ParseTuple(args, "i", &device)) {
    return NULL;
  }
  int64_t capabilityMajor;
  CHECK_DFCA(dfcaDeviceGetAttribute(&capabilityMajor,
                                    dfcaDevAttrComputeCapabilityMajor, device));
  int64_t capabilityMinor;
  CHECK_DFCA(dfcaDeviceGetAttribute(&capabilityMinor,
                                    dfcaDevAttrComputeCapabilityMinor, device));
  int64_t dieDimX;
  CHECK_DFCA(dfcaDeviceGetAttribute(&dieDimX, dfcaDevAttrDieGridDimX, device));
  int64_t dieDimY;
  CHECK_DFCA(dfcaDeviceGetAttribute(&dieDimY, dfcaDevAttrDieGridDimY, device));
  int64_t peNum;
  CHECK_DFCA(
      dfcaDeviceGetAttribute(&peNum, dfcaDevAttrTotalPeNumPerDie, device));
  int64_t coreNum;
  CHECK_DFCA(
      dfcaDeviceGetAttribute(&coreNum, dfcaDevAttrTotalRvCoreNumPerPe, device));
  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("(KKKKKK)", capabilityMajor, capabilityMinor, dieDimX,
                       dieDimY, peNum, coreNum);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load a binary object into the DFCA driver."},
    {"get_device_capability", getDeviceCapability, METH_VARARGS,
     "Get the compute capability of the device."},
    {"get_sram_used_bytes", getSRAMUsedBytes, METH_VARARGS,
     "Get the SRAM bytes used by the kernel."},
    {NULL, NULL, 0, NULL} /* sentinel */
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "thrive_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_thrive_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
