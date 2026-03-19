//===- triton_tle_dsa.cc - TLE DSA builder injection -------------*- C++
//-*-===//
//
// Template pybind that injects DSA dialect ops into TritonOpBuilder.
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include "ir.h"
#include "tle-dsa/Dialect/IR/DsaDialect.h"

namespace py = pybind11;
using namespace mlir;

namespace dsa = mlir::dsa;

static void init_triton_tle_ir(py::module m) {
  (void)m;
  auto core_ir = py::module::import("triton._C.libtriton.ir");
  auto builder_cls =
      core_ir.attr("builder").cast<py::class_<TritonOpBuilder>>();

  builder_cls
      .def("create_dsa_alloc",
           [](TritonOpBuilder &self, py::object shapeObj,
              py::object elementTyObj) -> Value {
             self.getContext()->getOrLoadDialect<dsa::DsaDialect>();
             auto &b = self.getBuilder();
             std::vector<int64_t> dims;
             if (py::isinstance<py::int_>(shapeObj)) {
               dims.push_back(py::cast<int64_t>(shapeObj));
             } else {
               py::iterable shape =
                   py::reinterpret_borrow<py::iterable>(shapeObj);
               dims.reserve(py::len(shape));
               for (py::handle dim : shape)
                 dims.push_back(py::cast<int64_t>(dim));
             }
             auto shapeAttr = DenseI64ArrayAttr::get(b.getContext(), dims);
             Type elementTy = py::cast<Type>(elementTyObj);
             auto bufTy = MemRefType::get(dims, elementTy);
             auto op = self.getBuilder().create<dsa::AllocOp>(self.getLastLoc(),
                                                              bufTy, shapeAttr);
             return op.getResult();
           })
      .def("create_dsa_copy",
           [](TritonOpBuilder &self, Value src, Value dst) -> void {
             self.getContext()->getOrLoadDialect<dsa::DsaDialect>();
             self.getBuilder().create<dsa::CopyOp>(self.getLastLoc(), src, dst);
           })
      .def("create_dsa_local_pointers",
           [](TritonOpBuilder &self, Type resultTy, Value src,
              py::args args) -> OpState {
             self.getContext()->getOrLoadDialect<dsa::DsaDialect>();
             llvm::SmallVector<Value> indices;
             indices.reserve(args.size());
             for (const auto &arg : args)
               indices.push_back(py::cast<Value>(arg));
             return self.create<dsa::LocalPointersOp>(resultTy, src, indices);
           })
      .def("create_dsa_remote_pointers",
           [](TritonOpBuilder &self, Type resultTy, Value src,
              Value shardId) -> OpState {
             self.getContext()->getOrLoadDialect<dsa::DsaDialect>();
             return self.create<dsa::RemotePointersOp>(resultTy, src, shardId);
           })
      .def("create_dsa_distributed_barrier",
           [](TritonOpBuilder &self, const std::string &groupKind,
              const std::vector<int32_t> &groupShape,
              const std::vector<int32_t> &groupAxes,
              const std::vector<int32_t> &groupMask) -> void {
             self.getContext()->getOrLoadDialect<dsa::DsaDialect>();
             auto &builder = self.getBuilder();
             auto *ctx = builder.getContext();
             StringAttr kindAttr;
             IntegerAttr rankAttr;
             DenseI32ArrayAttr shapeAttr;
             DenseI32ArrayAttr axesAttr;
             DenseI32ArrayAttr maskAttr;

             if (!groupKind.empty()) {
               kindAttr = builder.getStringAttr(groupKind);
               rankAttr = builder.getI32IntegerAttr(
                   static_cast<int32_t>(groupShape.size()));
               shapeAttr = DenseI32ArrayAttr::get(ctx, groupShape);
               axesAttr = DenseI32ArrayAttr::get(ctx, groupAxes);
               if (!groupMask.empty())
                 maskAttr = DenseI32ArrayAttr::get(ctx, groupMask);
             }

             self.create<dsa::DistributedBarrierOp>(
                 kindAttr, rankAttr, shapeAttr, axesAttr, maskAttr);
           });
}

// void init_triton_tle(py::module &&m, const char *submodule_name = nullptr) {
//   if (submodule_name && *submodule_name != '\0')
//     m = m.def_submodule(submodule_name);
//   py::module local_m = std::move(m);
//   local_m.def("load_dialects", [](mlir::MLIRContext &context) {
//     context.getOrLoadDialect<dsa::DsaDialect>();
//   });
//   init_triton_tle_ir(std::move(local_m));
// }

void init_triton_tle(py::module &&m) {
  py::module local_m = std::move(m);

  local_m.def("load_dialects", [](mlir::MLIRContext &context) {
    context.getOrLoadDialect<dsa::DsaDialect>();
  });

  init_triton_tle_ir(std::move(local_m));
}
