#include <pybind11/pybind11.h>

namespace py = pybind11;

// compilation goes to triton-adapter-opt, do nothing here
void init_triton_triton_adapter(py::module &&m) {}