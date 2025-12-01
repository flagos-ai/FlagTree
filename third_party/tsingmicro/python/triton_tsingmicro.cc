#include <pybind11/pybind11.h>

namespace py = pybind11;

// The TsingMicro backend with ztc doesn't do compilation from within python
// but rather externally through ztc-opt, so we leave this function blank.
void init_triton_tsingmicro(py::module &&m) {}
