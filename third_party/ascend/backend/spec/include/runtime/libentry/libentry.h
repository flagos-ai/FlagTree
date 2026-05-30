/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef LIBENTRY_H
#define LIBENTRY_H

#include <any>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using KeyType = py::tuple;

namespace libentry {

class ArgProcessor {
public:
  ArgProcessor(int div) : divisibility_(div) {};

  void
  classifyArguments(const py::list &args, const py::dict &kwargs,
                    const py::list &jit_params,
                    const std::unordered_set<int> &specialize_indices,
                    const std::unordered_set<int> &do_not_specialize_indices);

  KeyType generateKey();

  py::list getKArgs();

private:
  py::list spec_args_;  // specialize args
  py::list dns_args_;   // do not specialize args
  py::list const_args_; // constexpr args
  py::list k_args_;     // kernel args
  int divisibility_;    // 对齐
};

} // namespace libentry

PYBIND11_MODULE(libentryC, m) {
  py::class_<libentry::ArgProcessor>(m, "ArgProcessor")
      .def(py::init<int>())
      .def("classify_arguments", &libentry::ArgProcessor::classifyArguments,
           py::arg("args"), py::arg("kwargs"), py::arg("jit_params"),
           py::arg("specialize_indices"), py::arg("do_not_specialize_indices"),
           "classify arguments")
      .def("get_k_args", &libentry::ArgProcessor::getKArgs, "get kernel")
      .def("generate_key", &libentry::ArgProcessor::generateKey,
           "generate kernel cache key");
}

#endif
