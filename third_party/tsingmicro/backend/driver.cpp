//===---------------------------- driver.c --------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Tx81 platform device side runtime interface for python.
//
//===----------------------------------------------------------------------===//
#include <dlfcn.h>
#include <hrt_common.h>
#include <hrt_interface.h>
#include <stdbool.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  // Extract device properties
  // Note: We're mapping Tx81 properties to fields expected by Triton
  int max_shared_mem = 1024 * 1024 * 3; // Default 3MB
  // int multiprocessor_count = device->tile_num;
  int multiprocessor_count = 1;
  int sm_clock_rate = 1000;  // Placeholder
  int mem_clock_rate = 2000; // Placeholder
  int mem_bus_width = 256;   // Placeholder

  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i}", "max_shared_mem",
                       max_shared_mem, "multiprocessor_count",
                       multiprocessor_count, "sm_clock_rate", sm_clock_rate,
                       "mem_clock_rate", mem_clock_rate, "mem_bus_width",
                       mem_bus_width);
}

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  const char *name;
  const char *data;
  Py_ssize_t data_size;
  int shared;
  int device;

  int32_t n_regs = 256;
  int32_t n_spills = 0;
  // Return values to Python including module, function, n_regs, n_spills
  return Py_BuildValue("(KKii)", "module {}", "void @add_kernel() {}", n_regs,
                       n_spills);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided binary into Tx81 driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given Tx81 device"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "tx81_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_tx81_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }

  PyModule_AddFunctions(m, ModuleMethods);

  return m;
}
