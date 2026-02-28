// Copyright (c) 2025 XCoreSigma Inc. All rights reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "ir.h"

using namespace mlir;
namespace py = pybind11;

constexpr unsigned kIntegerAttrBitWidth = 64;

struct DSAOpBuilder : public TritonOpBuilder {};

void init_tle_ir(py::module &&m)
{
  m.def("load_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    registry.insert<memref::MemRefDialect>();
    registry.insert<bufferization::BufferizationDialect>();
    registry.insert<triton::TritonDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  py::class_<DSAOpBuilder, TritonOpBuilder>(m, "tle_builder", py::module_local(), py::dynamic_attr())
    .def(py::init<mlir::MLIRContext *>())
    // Add alloc op
    /// .def("create_dsa_alloc",
    ///      [](DSAOpBuilder &self, std::vector<int64_t> &shape,
    ///         std::string &layout, std::string &scope, Type type)-> Value {
    ///        auto shapeAttr = self.getBuilder().getI64ArrayAttr(shape);
    ///        auto layoutAttr = self.getBuilder().getStringAttr(layout);
    ///        auto scopeAttr = self.getBuilder().getStringAttr(scope);

    ///        auto ptrType = triton::PointerType::get(type, 1);
    ///        auto tensorPtrType = RankedTensorType::get(shape, ptrType);
    ///        return self.create<triton::DSAAllocOp>(tensorPtrType, shapeAttr,
    ///            layoutAttr, scopeAttr);
    ///      })
    // Add copy op
    .def("dsa_get_null_attr", [](DSAOpBuilder &self) { return Attribute(); })
    .def("dsa_get_buffer_type",
           [](DSAOpBuilder &self, std::vector<int64_t> &shape,
              Type &elementType, const Attribute &memorySpace) -> Type {
             return MemRefType::get(shape, elementType,
                                    MemRefLayoutAttrInterface{}, memorySpace);
           })
    .def("dsa_get_buffer_type_with_strides",
           [](TritonOpBuilder &self, std::vector<int64_t> &shape,
              Type &elementType, const std::vector<int64_t> &strides,
              const Attribute &memorySpace) -> Type {
             // create a layout with strides, using dynamic offset
             auto layout = StridedLayoutAttr::get(
                 self.getBuilder().getContext(), ShapedType::kDynamic, strides);
             return MemRefType::get(shape, elementType, layout, memorySpace);
           })
    .def("create_dsa_alloc",
          [](DSAOpBuilder &self, Type memrefType) -> Value {
            return self.create<memref::AllocOp>(mlir::cast<MemRefType>(memrefType));
          })
    .def("create_dsa_copy",
         [](DSAOpBuilder &self, Value &src, Value &dst, std::vector<Value> &shape)-> void {
           self.create<DSACopyOp>(src, dst, shape);
         })
    // Add op
    .def("create_dsa_add",
         [](DSAOpBuilder &self, Value &lhs, Value &rhs, Value &res) -> void {
           self.create<DSAAddOp>(lhs, rhs, res);
         })
    // Sub op
    .def("create_dsa_sub",
         [](DSAOpBuilder &self, Value &lhs, Value &rhs, Value &res) -> void {
           self.create<DSASubOp>(lhs, rhs, res);
         })
    // Mul op
    .def("create_dsa_mul",
         [](DSAOpBuilder &self, Value &lhs, Value &rhs, Value &res) -> void {
           self.create<DSAMulOp>(lhs, rhs, res);
         })
    // Div op
    .def("create_dsa_div",
         [](DSAOpBuilder &self, Value &lhs, Value &rhs, Value &res) -> void {
           self.create<DSADivOp>(lhs, rhs, res);
         })
    // Max op
    .def("create_dsa_max",
         [](DSAOpBuilder &self,  Value &lhs, Value &rhs, Value &res) -> void {
           self.create<DSAMaxOp>(lhs, rhs, res);
         })
    // Min op
    .def("create_dsa_min",
         [](DSAOpBuilder &self,  Value &lhs, Value &rhs, Value &res) -> void {
           self.create<DSAMinOp>(lhs, rhs, res);
         })
    // Dot op
    /// .def("create_dsa_dot",
    ///      [](DSAOpBuilder &self, Value &inA, Value &inB, Value &res,
    ///         std::vector<int64_t> &size, bool &initC, bool &traA, bool &traB,
    ///         bool &enable_hf32) -> void {
    ///        auto &builder = self.getBuilder();
    ///        auto sizeAttr = builder.getI64ArrayAttr(size);

    ///        // convert bool to boolattr.
    ///        auto initC_attr = builder.getBoolAttr(initC);
    ///        auto traA_attr = builder.getBoolAttr(traA);
    ///        auto traB_attr = builder.getBoolAttr(traB);
    ///        auto enable_hf32_attr = builder.getBoolAttr(enable_hf32);

    ///        self.create<DSADotOp>(inA, inB, res, sizeAttr, initC_attr,
    ///                              traA_attr, traB_attr, enable_hf32_attr);
    ///      })
    ///   // ToTensor op
    /// .def("dsa_to_tensor",
    ///    [](DSAOpBuilder &self, Value &src) -> Value {
    ///      return self.create<ToTensorOp>(src);
    ///    })
    /// // ToBuffer op
    /// .def("dsa_to_buffer",
    ///    [](DSAOpBuilder &self, Value &src) -> Value {
    ///      auto srcType = src.getType();
    ///      auto tensorTy = cast<RankedTensorType>(srcType);
    ///      Type elementType = tensorTy.getElementType();
    ///      auto ptrType = triton::PointerType::get(elementType, 1);
    ///      auto shape = tensorTy.getShape();
    ///      auto tensorPtrType = RankedTensorType::get(shape, ptrType);
    ///      return self.create<ToBufferOp>(tensorPtrType, src);
    ///    })
    .def("dsa_to_buffer",
         [](DSAOpBuilder &self, Value &src,
            const Attribute &addressSpace) -> Value {
           auto tensorType = dyn_cast<RankedTensorType>(src.getType());
           if (!tensorType) {
             llvm::report_fatal_error("to_buffer: src must be tensor type");
           }
           auto memrefType = MemRefType::get(
               tensorType.getShape(), tensorType.getElementType(),
               MemRefLayoutAttrInterface{}, addressSpace);
           return self.create<bufferization::ToMemrefOp>(memrefType, src);
         })
    .def("dsa_to_tensor",
         [](DSAOpBuilder &self, Value &src, bool writable) -> Value {
           const auto &memrefType = mlir::cast<MemRefType>(src.getType());
           auto hasAddressSpace = memrefType.getMemorySpace();
           if (hasAddressSpace) {
             return self.create<bufferization::ToTensorOp>(
                 self.create<memref::MemorySpaceCastOp>(
                     MemRefType::get(memrefType.getShape(),
                                     memrefType.getElementType(),
                                     memrefType.getLayout()),
                     src),
                 true, writable);
           }
           return self.create<bufferization::ToTensorOp>(src, true, writable);
         })
    ;

}