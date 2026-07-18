// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  // create a struct to hold device properties
  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i}", "max_shared_mem", 1024,
                       "multiprocessor_count", 16, "sm_clock_rate", 2100,
                       "mem_clock_rate", 2300, "mem_bus_width", 2400);
}

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  // get allocated registers and spilled registers from the function
  int n_regs = 0;
  int n_spills = 0;
  int mod = 0;
  int fun = 0;
  return Py_BuildValue("(KKii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load dummy binary for the extension device"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for the extension device"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "ext_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_ext_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
