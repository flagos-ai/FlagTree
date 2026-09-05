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

// Inject a three-operand dsa binary op builder: create_dsa_<name>(lhs, rhs,
// out).
template <typename DsaOpT>
static void defBinaryOp(py::class_<TritonOpBuilder> &builderCls,
                        const char *name) {
  builderCls.def(
      name, [](TritonOpBuilder &self, Value lhs, Value rhs, Value out) -> void {
        self.getContext()->getOrLoadDialect<dsa::DsaDialect>();
        self.getBuilder().create<DsaOpT>(self.getLastLoc(), lhs, rhs, out);
      });
}

// Cast a Python list of dynamic-offset values to a SmallVector<Value>.
static llvm::SmallVector<Value> castDynOffsets(py::list dynOffsets) {
  llvm::SmallVector<Value> dyn;
  dyn.reserve(py::len(dynOffsets));
  for (py::handle arg : dynOffsets)
    dyn.push_back(py::cast<Value>(arg));
  return dyn;
}

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
      .def(
          "create_dsa_remote_pointers",
          [](TritonOpBuilder &self, Type resultTy, Value src, Value shardId,
             py::object scope) -> OpState {
            self.getContext()->getOrLoadDialect<dsa::DsaDialect>();
            DenseI32ArrayAttr meshPhysicalIdsAttr;
            DenseI32ArrayAttr meshShapeAttr;
            if (!scope.is_none() && py::hasattr(scope, "physical_ids")) {
              py::object physicalIds = scope.attr("physical_ids");
              std::vector<int32_t> ids;
              for (auto id : py::reinterpret_borrow<py::iterable>(physicalIds))
                ids.push_back(py::cast<int32_t>(id));
              if (!ids.empty())
                meshPhysicalIdsAttr =
                    DenseI32ArrayAttr::get(self.getBuilder().getContext(), ids);
            }
            if (!scope.is_none() && py::hasattr(scope, "shape")) {
              py::object shape = scope.attr("shape");
              std::vector<int32_t> dims;
              for (auto dim : py::reinterpret_borrow<py::iterable>(shape))
                dims.push_back(py::cast<int32_t>(dim));
              if (!dims.empty())
                meshShapeAttr = DenseI32ArrayAttr::get(
                    self.getBuilder().getContext(), dims);
            }
            return self.create<dsa::RemotePointersOp>(
                resultTy, src, shardId, meshPhysicalIdsAttr, meshShapeAttr);
          },
          py::arg("resultTy"), py::arg("src"), py::arg("shardId"),
          py::arg("scope") = py::none())
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
           })
      .def("create_dsa_cumsum",
           [](TritonOpBuilder &self, Type exclusiveTy, Type totalTy,
              Value input, int32_t axis, bool reverse,
              const std::vector<int64_t> &shape, int64_t pad) -> OpState {
             self.getContext()->getOrLoadDialect<dsa::DsaDialect>();
             auto &builder = self.getBuilder();
             auto *ctx = builder.getContext();
             return builder.create<dsa::CumsumOp>(
                 self.getLastLoc(), TypeRange{exclusiveTy, totalTy}, input,
                 builder.getI32IntegerAttr(axis), builder.getBoolAttr(reverse),
                 DenseI64ArrayAttr::get(ctx, shape),
                 builder.getI64IntegerAttr(pad));
           })
      .def("create_dsa_randgen",
           [](TritonOpBuilder &self, Type outTy, Type seed0OutTy,
              Type seed1OutTy, Value seed0, Value seed1, int32_t byteCount,
              int16_t fmt) -> OpState {
             self.getContext()->getOrLoadDialect<dsa::DsaDialect>();
             auto &builder = self.getBuilder();
             return builder.create<dsa::RandGenOp>(
                 self.getLastLoc(), TypeRange{outTy, seed0OutTy, seed1OutTy},
                 seed0, seed1, builder.getI32IntegerAttr(byteCount),
                 builder.getI16IntegerAttr(fmt));
           })
      // Vendor-neutral same-nbytes type/shape reinterpret (e.g.
      // i64[N]→i32[2N]). Backends lower this (Tsingmicro: mk.bitcast alias;
      // others: tensor.bitcast).
      .def("create_dsa_bitcast",
           [](TritonOpBuilder &self, Type dstTy, Value src) -> Value {
             self.getContext()->getOrLoadDialect<dsa::DsaDialect>();
             return self.create<dsa::BitcastOp>(dstTy, src);
           });

  builder_cls
      .def("create_dsa_to_tensor",
           [](TritonOpBuilder &self, Type resultTy, Value src,
              bool writable) -> Value {
             self.getContext()->getOrLoadDialect<dsa::DsaDialect>();
             auto &b = self.getBuilder();
             auto writableAttr = b.getBoolAttr(writable);
             auto op =
                 self.create<dsa::ToTensorOp>(resultTy, src, writableAttr);
             return op.getResult();
           })
      .def("create_dsa_to_buffer",
           [](TritonOpBuilder &self, Value src, Value dst) -> void {
             self.getContext()->getOrLoadDialect<dsa::DsaDialect>();
             self.create<dsa::ToBufferOp>(src, dst);
           });

  builder_cls
      .def(
          "create_dsa_extract_slice",
          [](TritonOpBuilder &self, Type resultTy, Value src,
             const std::vector<int64_t> &staticOffsets, py::list dynOffsets,
             const std::vector<int64_t> &sizes,
             const std::vector<int64_t> &strides) -> Value {
            self.getContext()->getOrLoadDialect<dsa::DsaDialect>();
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto dyn = castDynOffsets(dynOffsets);
            auto op = self.create<dsa::ExtractSliceOp>(
                resultTy, src, dyn, DenseI64ArrayAttr::get(ctx, staticOffsets),
                DenseI64ArrayAttr::get(ctx, sizes),
                DenseI64ArrayAttr::get(ctx, strides));
            return op.getResult();
          },
          py::arg("resultTy"), py::arg("src"), py::arg("staticOffsets"),
          py::arg("dynOffsets"), py::arg("sizes"), py::arg("strides"))
      .def(
          "create_dsa_insert_slice",
          [](TritonOpBuilder &self, Type resultTy, Value src, Value tile,
             const std::vector<int64_t> &staticOffsets, py::list dynOffsets,
             const std::vector<int64_t> &sizes,
             const std::vector<int64_t> &strides) -> Value {
            self.getContext()->getOrLoadDialect<dsa::DsaDialect>();
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto dyn = castDynOffsets(dynOffsets);
            auto op = self.create<dsa::InsertSliceOp>(
                resultTy, src, tile, dyn,
                DenseI64ArrayAttr::get(ctx, staticOffsets),
                DenseI64ArrayAttr::get(ctx, sizes),
                DenseI64ArrayAttr::get(ctx, strides));
            return op.getResult();
          },
          py::arg("resultTy"), py::arg("src"), py::arg("tile"),
          py::arg("staticOffsets"), py::arg("dynOffsets"), py::arg("sizes"),
          py::arg("strides"));
  // Three-operand binary arithmetic (out = lhs OP rhs).
  defBinaryOp<dsa::AddOp>(builder_cls, "create_dsa_add");
  defBinaryOp<dsa::SubOp>(builder_cls, "create_dsa_sub");
  defBinaryOp<dsa::MulOp>(builder_cls, "create_dsa_mul");
  defBinaryOp<dsa::MaximumOp>(builder_cls, "create_dsa_maximum");
  defBinaryOp<dsa::MinimumOp>(builder_cls, "create_dsa_minimum");
  defBinaryOp<dsa::DivOp>(builder_cls, "create_dsa_div");
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

void init_triton_tle_dsa(py::module m) {
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    context.getOrLoadDialect<dsa::DsaDialect>();
  });

  init_triton_tle_ir(std::move(m));
}
