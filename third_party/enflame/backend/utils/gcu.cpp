/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <tops/tops_runtime.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

static inline void gcuAssert(topsError_t code, const char *file, int line) {
  {
    if (code != TOPS_SUCCESS) {
      {
        const char *prefix = "Kurama Error [TOPS]: ";
        const char *str = topsGetErrorString(code);
        char err[1024] = {0};
        snprintf(err, sizeof(err), "%s Code: %d, Message: %s", prefix, code,
                 str);
        PyGILState_STATE gil_state;
        gil_state = PyGILState_Ensure();
        PyErr_SetString(PyExc_RuntimeError, err);
        PyGILState_Release(gil_state);
      }
    }
  }
}

#define TOPS_CHECK(ans)                                                        \
  {                                                                            \
    gcuAssert((ans), __FILE__, __LINE__);                                      \
    if (PyErr_Occurred())                                                      \
      return NULL;                                                             \
  }

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;

  topsDeviceProp_t props;
  TOPS_CHECK(topsGetDeviceProperties(&props, device_id));

  // create a struct to hold device properties
  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i, s:i, s:s}", "max_shared_mem",
                       props.sharedMemPerBlock, "multiprocessor_count",
                       props.multiProcessorCount, "max_threads_per_block",
                       props.maxThreadsPerBlock, "sm_clock_rate",
                       props.clockRate, "mem_clock_rate", props.memoryClockRate,
                       "mem_bus_width", props.memoryBusWidth, "arch_name",
                       props.gcuArchName);
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

  topsFunction_t fun;
  topsModule_t mod;
  // create driver handles
  TOPS_CHECK(topsModuleLoadData(&mod, data));
  TOPS_CHECK(topsModuleGetFunction(&fun, mod, name));

  // get allocated registers and spilled registers from the function
  int n_regs = 0;
  int n_spills = 0;
  int n_max_threads = 65536;
  if (PyErr_Occurred()) {
    return NULL;
  }
  return Py_BuildValue("(KKiii)", reinterpret_cast<uint64_t>(mod),
                       reinterpret_cast<uint64_t>(fun), n_regs, n_spills,
                       n_max_threads);
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided kernel into TOPS driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "gcu_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_gcu_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
