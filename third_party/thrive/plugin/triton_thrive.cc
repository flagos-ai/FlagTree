#include <pybind11/pybind11.h>

namespace py = pybind11;
#include "thrive_ir.h"

// Currently, the Thrive backend uses thrive-opt for compilation.
void init_triton_thrive(py::module &&m) { init_thrive_ir(m); }
