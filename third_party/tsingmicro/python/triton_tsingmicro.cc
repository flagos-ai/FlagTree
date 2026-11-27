//===- triton_tsingmicro.cc - TsingMicro backend python bindings ----------===//
//
// Backend-owned python bindings. Intentionally empty: backend-private ops are
// produced by the TLEToMK lowering, not injected into the shared builder.
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_triton_tsingmicro(py::module &&m) { (void)m; }
