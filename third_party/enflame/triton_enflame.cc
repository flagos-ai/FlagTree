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

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Empty initialization function to resolve linking issues
void init_triton_enflame(py::module &&m) {
  // Temporarily provide empty implementation to resolve linking issues
  // TODO: Add GCU-related MLIR dialects and transformation passes later

  m.doc() = "Enflame GCU backend for Triton";

  // Can add some basic utility functions
  m.def("get_gcu_arch", []() {
    return "gcu300"; // Default architecture
  });

  m.def("is_gcu_available", []() {
    // Simple availability check
    return true;
  });

  // Empty dialect loading function
  m.def("load_dialects", [](py::object context) {
    // TODO: Load GCU-related MLIR dialects
    // Temporarily empty implementation, using py::object to avoid MLIR
    // dependency
  });
}
