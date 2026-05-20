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

// triton_gcu300_module.cpp -- Thin pybind11 wrapper around
// lib_triton_gcu300_core.so's C ABI.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "triton_gcu300_core.h"

namespace py = pybind11;

namespace {

uintptr_t pipelineCreate() {
  auto h = gcu300_pipeline_create();
  if (!h)
    throw std::runtime_error("gcu300_pipeline_create failed");
  return reinterpret_cast<uintptr_t>(h);
}

void pipelineAddPass(uintptr_t h, const std::string &name,
                     const std::string &options) {
  gcu300_pipeline_add_pass(reinterpret_cast<Gcu300Pipeline>(h), name.c_str(),
                           options.c_str());
}

std::string pipelineRun(uintptr_t h, const std::string &input) {
  Gcu300String s = gcu300_pipeline_run(reinterpret_cast<Gcu300Pipeline>(h),
                                       input.data(), input.size());
  if (!s.data) {
    std::string err =
        gcu300_pipeline_last_error(reinterpret_cast<Gcu300Pipeline>(h));
    throw std::runtime_error("Pipeline run failed: " + err);
  }
  std::string result(s.data, s.len);
  gcu300_string_free(s);
  return result;
}

void pipelineDestroy(uintptr_t h) {
  if (h)
    gcu300_pipeline_destroy(reinterpret_cast<Gcu300Pipeline>(h));
}

std::string runOpt(const std::string &input,
                   const std::vector<std::string> &args) {
  std::vector<const char *> cargs;
  cargs.reserve(args.size());
  for (const auto &a : args)
    cargs.push_back(a.c_str());

  Gcu300String s = gcu300_run_opt(input.data(), input.size(), cargs.data(),
                                  static_cast<int>(cargs.size()));
  if (!s.data)
    throw std::runtime_error("gcu300_run_opt failed (pass pipeline error)");
  std::string result(s.data, s.len);
  gcu300_string_free(s);
  return result;
}

} // anonymous namespace

PYBIND11_MODULE(_triton_gcu300, m) {
  m.doc() = "GCU300 in-process MLIR opt pipeline (pybind11 thin binding)";

  m.def("pipeline_create", &pipelineCreate,
        "Create a new pipeline, returns opaque handle.");
  m.def("pipeline_add_pass", &pipelineAddPass, py::arg("handle"),
        py::arg("name"), py::arg("options") = "",
        "Add a pass to the pipeline.");
  m.def("pipeline_run", &pipelineRun, py::arg("handle"), py::arg("input"),
        "Execute the configured pipeline on MLIR text input.");
  m.def("pipeline_destroy", &pipelineDestroy, py::arg("handle"),
        "Destroy a pipeline handle.");

  m.def("run_opt", &runOpt, py::arg("input"), py::arg("args"),
        "Run the MLIR opt pipeline on `input` with CLI-style `args` (legacy "
        "interface).");
}
