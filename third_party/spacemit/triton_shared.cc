#include "include/triton-shared/Dialect/TLE/IR/TLEDialect.h"
#include "include/triton-shared/Dialect/TLE/IR/TLEOps.h"
#include "include/triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "include/triton-shared/Dialect/XSMT/IR/XSMTOps.h"
#include "include/triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncDialect.h"
#include "include/triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncOps.h"
#include "ir.h"
#include "mlir/Pass/PassManager.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <iostream>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace ir;
using namespace mlir;
namespace xsmt = mlir::xsmt;
namespace xsmt_async = mlir::xsmt_async;
namespace tle = mlir::tle;

void init_triton_xsmt_ir(py::module &&m) {
  auto *builder_cls = ir::getBuilderClass();
  builder_cls
      ->def("create_annotation",
            [](TritonOpBuilder &self, Value &ptr, const std::string &attrKey,
               Attribute &attrVal) {
              auto annotationOp = self.create<xsmt::AnnotationOp>(ptr);
              annotationOp->setAttr(self.getBuilder().getStringAttr(attrKey),
                                    attrVal);
            })
      .def("create_descriptor_load",
           [](TritonOpBuilder &self, Value &base,
              std::vector<Value> &offsets) -> Value {
             auto AdvanceOp =
                 self.create<triton::AdvanceOp>(base.getType(), base, offsets);
             auto pointeeType = cast<mlir::triton::PointerType>(base.getType())
                                    .getPointeeType();
             auto resultType = dyn_cast<RankedTensorType>(pointeeType);
             int rank = resultType.getRank();
             std::vector<int32_t> boundary_check;
             for (int i = 0; i < rank; ++i) {
               boundary_check.push_back(i);
             }
             auto LoadOp = self.create<triton::LoadOp>(
                 AdvanceOp.getResult(), boundary_check, std::nullopt,
                 triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL,
                 false);
             return LoadOp;
           })
      .def("create_descriptor_load_to_destination",
           [](TritonOpBuilder &self, Value &base, std::vector<Value> &offsets,
              Value &destination) {
             auto AdvanceOp =
                 self.create<triton::AdvanceOp>(base.getType(), base, offsets);
             auto pointeeType = cast<mlir::triton::PointerType>(base.getType())
                                    .getPointeeType();
             auto resultType = dyn_cast<RankedTensorType>(pointeeType);
             int rank = resultType.getRank();
             std::vector<int32_t> boundary_check;
             for (int i = 0; i < rank; ++i) {
               boundary_check.push_back(i);
             }
             auto LoadOp = self.create<triton::LoadOp>(
                 AdvanceOp.getResult(), boundary_check, std::nullopt,
                 triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL,
                 false);
             self.create<mlir::triton::StoreOp>(
                 destination, LoadOp, boundary_check,
                 triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
           })
      .def(
          "create_pack",
          [](TritonOpBuilder &self, Value &base, std::vector<Value> &offsets,
             std::vector<int32_t> &shape, std::vector<int32_t> &packed_size,
             std::optional<Value> destination) -> Value {
            if (destination.has_value())
              return self.create<xsmt::PackOp>(
                  base, offsets, destination.value(), shape, packed_size);
            return self.create<xsmt::PackOp>(base, offsets, shape, packed_size);
          },
          py::arg("base"), py::arg("offsets"), py::arg("shape"),
          py::arg("packed_size"), py::arg("destination") = py::none())
      .def(
          "create_unpack",
          [](TritonOpBuilder &self, Value &base, std::vector<Value> &offsets,
             std::vector<int32_t> &shape,
             std::optional<Value> destination) -> Value {
            if (destination.has_value())
              return self.create<xsmt::UnpackOp>(base, offsets,
                                                 destination.value(), shape);
            return self.create<xsmt::UnpackOp>(base, offsets, shape);
          },
          py::arg("base"), py::arg("offsets"), py::arg("shape"),
          py::arg("destination") = py::none())
      .def(
          "create_repack",
          [](TritonOpBuilder &self, Value &base, std::vector<Value> &offsets,
             std::vector<int32_t> &shape, std::vector<int32_t> &packed_size,
             std::optional<Value> destination) -> Value {
            if (destination.has_value())
              return self.create<xsmt::RepackOp>(
                  base, offsets, destination.value(), shape, packed_size);
            return self.create<xsmt::RepackOp>(base, offsets, shape,
                                               packed_size);
          },
          py::arg("base"), py::arg("offsets"), py::arg("shape"),
          py::arg("packed_size"), py::arg("destination") = py::none())
      .def("create_subview",
           [](TritonOpBuilder &self, Value &base, std::vector<Value> &offsets,
              std::vector<int32_t> &shape) -> Value {
             return self.create<xsmt::SubviewOp>(base, offsets, shape);
           })
      .def("create_subview_pack",
           [](TritonOpBuilder &self, Value &base, std::vector<Value> &offsets,
              std::vector<int32_t> &shape,
              std::vector<int32_t> &packed_size) -> Value {
             return self.create<xsmt::SubviewPackOp>(base, offsets, shape,
                                                     packed_size);
           })
      .def("create_alloc",
           [](TritonOpBuilder &self, std::vector<int32_t> &shape,
              mlir::Type type, std::string storage) -> Value {
             if (shape.empty()) {
               throw std::runtime_error("alloc shape cannot be empty");
             }
             auto op = self.create<xsmt::AllocOp>(type, shape, storage);
             return op;
           })
      .def("create_alloc_copies",
           [](TritonOpBuilder &self, std::vector<int64_t> &shape,
              mlir::Type elementType, std::string storage) -> mlir::Value {
             if (shape.empty())
               throw std::runtime_error("alloc_copies shape cannot be empty");
             auto op =
                 self.create<xsmt::AllocCopiesOp>(shape, elementType, storage);
             return op;
           })
      .def("create_mmt4d",
           [](TritonOpBuilder &self, Value &a, Value &b,
              std::optional<Value> c = std::nullopt) -> Value {
             auto aType = cast<RankedTensorType>(a.getType());
             auto bType = cast<RankedTensorType>(b.getType());

             assert(aType.getRank() == 4 && "A must be 4D packed tensor");
             assert(bType.getRank() == 4 && "B must be 4D packed tensor");

             auto aShape = aType.getShape();
             auto bShape = bType.getShape();

             SmallVector<int64_t> outputShape;

             if (aShape[1] == bShape[0] && aShape[3] == bShape[2]) {
               outputShape = {
                   aShape[0],
                   bShape[1],
                   aShape[2],
                   bShape[3],
               };
               auto resultType =
                   RankedTensorType::get(outputShape, aType.getElementType());

               auto perm = std::vector<int>{1, 0, 3, 2};
               auto transbOp = self.create<mlir::triton::TransOp>(b, perm);
               mlir::Value transbValue = transbOp->getResult(0);

               mlir::Value cValue;
               if (c.has_value()) {
                 cValue = *c;
               } else {
                 cValue = Value();
               }

               return self.create<xsmt::MMT4DOp>(resultType, a, transbValue,
                                                 cValue);
             } else if (aShape[1] == bShape[1] && aShape[3] == bShape[3]) {
               outputShape = {
                   aShape[0],
                   bShape[0],
                   aShape[2],
                   bShape[2],
               };
               auto resultType =
                   RankedTensorType::get(outputShape, aType.getElementType());

               mlir::Value cValue;
               if (c.has_value()) {
                 cValue = *c;
               } else {
                 cValue = Value();
               }

               return self.create<xsmt::MMT4DOp>(resultType, a, b, cValue);
             } else {
               throw std::runtime_error("Unsupported packing shapes");
             }
           })
      .def("create_mbarrier",
           [](TritonOpBuilder &self, Value &flag, Value &atc, Value &tc,
              Value &exp) -> Value {
             auto barrierType = self.getBuilder().getI64Type();
             return self.create<mlir::xsmt_async::MBarrierAllocOp>(
                 barrierType, flag, atc, tc, exp);
           })
      .def("create_barrier_arrive",
           [](TritonOpBuilder &self, Value &bar) {
             self.create<mlir::xsmt_async::MBarrierArriveOp>(bar);
           })
      .def("create_barrier_wait",
           [](TritonOpBuilder &self, Value &bar, Value &flag, Value &exp) {
             self.create<mlir::xsmt_async::MBarrierWaitOp>(bar, flag, exp);
           })
      .def("create_get_num_of_thread",
           [](TritonOpBuilder &self) { self.create<xsmt::GetThreadOp>(); })
      .def("create_global_mbarrier",
           [](TritonOpBuilder &self, Value &id) -> Value {
             auto barrierType = self.getBuilder().getI64Type();
             return self.create<xsmt::GlobalMBarrierInitOp>(barrierType, id);
           })
      .def("create_barrier_set_expect",
           [](TritonOpBuilder &self, Value &bar, Value &exp) {
             self.create<xsmt::BarrierSetEepectOp>(bar, exp);
           })
      .def("create_smt_buffer_type",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              Type &elementType, int copies, std::string storageKind) -> Type {
             return xsmt::BufferType::get(shape, elementType, copies,
                                          storageKind);
           })
      .def("create_buffer_tensor_subview",
           [](TritonOpBuilder &self, Value buffer, Value bufferIdx) -> Value {
             return self.create<xsmt::BufferTensorViewOp>(buffer, bufferIdx);
           })
      .def("get_mbarrier_type",
           [](TritonOpBuilder &self, int copies) -> mlir::Type {
             auto *ctx = self.getBuilder().getContext();
             return mlir::xsmt::MBarrierType::get(ctx, copies);
           })
      .def("create_mbarrier_copies",
           [](TritonOpBuilder &self, int numCopies, int flag, int arriveCount,
              int transactionCount, int expectCount) -> mlir::Value {
             auto *ctx = self.getBuilder().getContext();

             auto resultTy = mlir::xsmt::MBarrierType::get(ctx, numCopies);

             auto op = self.create<mlir::xsmt::MBarrierCopiesOp>(
                 resultTy, numCopies, flag, arriveCount, transactionCount,
                 expectCount);

             return op.getResult();
           })
      .def("create_mbarrier_subview",
           [](TritonOpBuilder &self, mlir::Value mbarrierHandle,
              mlir::Value indexValue) -> mlir::Value {
             auto i64Type = self.getBuilder().getI64Type();
             auto op = self.create<mlir::xsmt::MBarrierSubviewOp>(
                 i64Type, mbarrierHandle, indexValue);

             return op.getResult();
           })
      .def("create_i64_constant",
           [](TritonOpBuilder &self, int64_t value) -> mlir::Value {
             auto i64Type = self.getBuilder().getI64Type();
             auto attr = self.getBuilder().getI64IntegerAttr(value);
             return self.create<mlir::arith::ConstantOp>(i64Type, attr)
                 .getResult();
           });
}

// ============================================================================
// TLE (Triton Language Extension) IR bindings
// ============================================================================
void init_triton_xtle_ir(py::module &&m) {
  auto *builder_cls = ir::getBuilderClass();
  builder_cls
      ->def(
          "create_extract_tile",
          [](TritonOpBuilder &self, Value &input, Value &index,
             std::vector<int64_t> &tileShape) -> Value {
            auto op = self.create<tle::ExtractTileOp>(input, index, tileShape);
            return op.getResult();
          },
          py::arg("input"), py::arg("index"), py::arg("tileShape"),
          "Create extract_tile operation")
      .def(
          "create_insert_tile",
          [](TritonOpBuilder &self, Value &input, Value &tile,
             Value &index) -> Value {
            auto op = self.create<tle::InsertTileOp>(input, tile, index);
            return op.getResult();
          },
          py::arg("input"), py::arg("tile"), py::arg("index"),
          "Create insert_tile operation");
}

void init_triton_spacemit(py::module &&m) {
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::xsmt::XSMTDialect, mlir::xsmt_async::XSMTAsyncDialect,
                    tensor::TensorDialect, mlir::triton::proton::ProtonDialect,
                    mlir::tle::TLEDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  init_triton_xsmt_ir(m.def_submodule("xsmt_ir"));
  init_triton_xtle_ir(m.def_submodule("tle_ir"));
}
