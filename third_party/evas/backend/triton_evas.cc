#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_triton_evas(py::module &&m) {
  m.doc() = "EVAS backend bindings for Triton";

  m.def("is_evas_available", []() { return true; });
  m.def("load_dialects", [](py::object) {});
}
