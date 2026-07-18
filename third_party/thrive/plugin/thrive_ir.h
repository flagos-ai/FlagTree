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

#pragma once
#include "ir.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include <pybind11/pybind11.h>

struct ThriveOpBuilder : public TritonOpBuilder {
  using TritonOpBuilder::TritonOpBuilder;
};

void init_thrive_ir(py::module &m) {
  using namespace mlir;
  using namespace mlir::triton;
  using ret = py::return_value_policy;

  py::class_<ThriveOpBuilder, TritonOpBuilder>(
      m, "ThriveBuilder", py::module_local(), py::dynamic_attr())
      // Constructor
      .def(py::init<mlir::MLIRContext *>())

      .def("get_op_builder", &ThriveOpBuilder::getBuilder, ret::reference)

      // ================= Custom operation binding =================

      // Create math ops
      .def("create_asin",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::AsinOp>(val);
           })
      .def("create_asinh",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::AsinhOp>(val);
           })
      .def("create_acos",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::AcosOp>(val);
           })
      .def("create_acosh",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::AcoshOp>(val);
           })
      .def("create_atan",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::AtanOp>(val);
           })
      .def("create_atanh",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::AtanhOp>(val);
           })
      .def("create_sin",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::SinOp>(val);
           })
      .def("create_sinh",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::SinhOp>(val);
           })
      .def("create_cos",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::CosOp>(val);
           })
      .def("create_cosh",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::CoshOp>(val);
           })
      .def("create_tan",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::TanOp>(val);
           })
      .def("create_tanh",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::TanhOp>(val);
           })
      .def("create_exp",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::ExpOp>(val);
           })
      .def("create_exp2",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::Exp2Op>(val);
           })
      .def("create_log",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::LogOp>(val);
           })
      .def("create_log2",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::Log2Op>(val);
           })
      .def("create_sqrt",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::SqrtOp>(val);
           })
      .def("create_rsqrt",
           [](ThriveOpBuilder &self, Value &val) -> Value {
             return self.create<math::RsqrtOp>(val);
           })
      .def("create_pow",
           [](ThriveOpBuilder &self, Value &base, Value &exponent) -> Value {
             return self.create<math::PowFOp>(base, exponent);
           })
      // TODO(thrive): bind to thrive::ExternCallOp once the dialect lands.
      .def("create_extern_call", [](ThriveOpBuilder &self) -> Value {
        throw py::value_error("create_extern_call: not implemented yet");
      });
}
