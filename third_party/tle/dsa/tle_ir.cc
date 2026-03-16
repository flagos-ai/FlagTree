// Copyright 2026- Xcoresigma Technology Co., Ltd

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "tle/dsa/dialect/include/IR/Dialect.h"

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

void init_triton_tle(py::module &&m) {
  m.def("load_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    registry.insert<memref::MemRefDialect>();
    registry.insert<bufferization::BufferizationDialect>();
    registry.insert<triton::tle::TleDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  py::class_<DSAOpBuilder, TritonOpBuilder>(
      m, "tle_builder", py::module_local(), py::dynamic_attr())
      .def(py::init<mlir::MLIRContext *>())
      .def("dsa_get_null_attr", [](DSAOpBuilder &self) { return Attribute(); })
      .def("dsa_get_buffer_type",
           [](DSAOpBuilder &self, std::vector<int64_t> &shape,
              Type &elementType, const Attribute &memorySpace) -> Type {
             return MemRefType::get(shape, elementType,
                                    MemRefLayoutAttrInterface{}, memorySpace);
           })
      .def("dsa_get_buffer_type_with_strides",
           [](DSAOpBuilder &self, std::vector<int64_t> &shape,
              Type &elementType, const std::vector<int64_t> &strides,
              const Attribute &memorySpace) -> Type {
             // create a layout with strides, using dynamic offset
             auto layout = StridedLayoutAttr::get(
                 self.getBuilder().getContext(), ShapedType::kDynamic, strides);
             return MemRefType::get(shape, elementType, layout, memorySpace);
           })
      .def("create_dsa_alloc",
           [](DSAOpBuilder &self, Type memrefType) -> Value {
             return self.create<memref::AllocOp>(
                 mlir::cast<MemRefType>(memrefType));
           })
      // Add copy op
      .def("create_dsa_copy",
           [](DSAOpBuilder &self, Value &src, Value &dst,
              std::vector<Value> &shape, bool inter_no_alias) -> void {
             auto copyOp = self.create<triton::tle::DSACopyOp>(src, dst, shape);
             if (inter_no_alias) {
               copyOp->setAttr("inter_no_alias",
                               self.getBuilder().getBoolAttr(true));
             }
           })
      // Add op
      .def("create_dsa_add",
           [](DSAOpBuilder &self, Value &lhs, Value &rhs, Value &res) -> void {
             self.create<triton::tle::DSAAddOp>(lhs, rhs, res);
           })
      // Sub op
      .def("create_dsa_sub",
           [](DSAOpBuilder &self, Value &lhs, Value &rhs, Value &res) -> void {
             self.create<triton::tle::DSASubOp>(lhs, rhs, res);
           })
      // Mul op
      .def("create_dsa_mul",
           [](DSAOpBuilder &self, Value &lhs, Value &rhs, Value &res) -> void {
             self.create<triton::tle::DSAMulOp>(lhs, rhs, res);
           })
      // Div op
      .def("create_dsa_div",
           [](DSAOpBuilder &self, Value &lhs, Value &rhs, Value &res) -> void {
             self.create<triton::tle::DSADivOp>(lhs, rhs, res);
           })
      // Max op
      .def("create_dsa_max",
           [](DSAOpBuilder &self, Value &lhs, Value &rhs, Value &res) -> void {
             self.create<triton::tle::DSAMaxOp>(lhs, rhs, res);
           })
      // Min op
      .def("create_dsa_min",
           [](DSAOpBuilder &self, Value &lhs, Value &rhs, Value &res) -> void {
             self.create<triton::tle::DSAMinOp>(lhs, rhs, res);
           })
      // Dot op
      /// .def("create_dsa_dot",
      ///      [](DSAOpBuilder &self, Value &inA, Value &inB, Value &res,
      ///         std::vector<int64_t> &size, bool &initC, bool &traA, bool
      ///         &traB, bool &enable_hf32) -> void {
      ///        auto &builder = self.getBuilder();
      ///        auto sizeAttr = builder.getI64ArrayAttr(size);

      ///        // convert bool to boolattr.
      ///        auto initC_attr = builder.getBoolAttr(initC);
      ///        auto traA_attr = builder.getBoolAttr(traA);
      ///        auto traB_attr = builder.getBoolAttr(traB);
      ///        auto enable_hf32_attr = builder.getBoolAttr(enable_hf32);

      ///        self.create<triton::tle::DSADotOp>(inA, inB, res, sizeAttr,
      ///        initC_attr,
      ///                              traA_attr, traB_attr, enable_hf32_attr);
      ///      })
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
               return self.create<bufferization::ToTensorOp>(src, true,
                                                             writable);
             }
             return self.create<bufferization::ToTensorOp>(src, true, writable);
           })
      .def("create_dsa_extract_scalar",
           [](DSAOpBuilder &self, Value &src,
              std::vector<Value> &indices) -> Value {
             llvm::SmallVector<Value> arg_indices;
             for (const auto &i : indices) {
               auto iTy = i.getType();
               if (!iTy.isIndex()) {
                 auto v = self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), i);
                 arg_indices.push_back(v);
               } else {
                 arg_indices.push_back(i);
               }
             }
             auto ret = self.create<tensor::ExtractOp>(src, arg_indices);
             return ret;
           })
      .def("create_dsa_extract_slice",
           [](DSAOpBuilder &self, Value &ful, std::vector<Value> &offs_vec,
              std::vector<int> &sizs_vec, std::vector<int> &strd_vec) -> Value {
             llvm::SmallVector<Value> offsets;
             for (const auto &o : offs_vec) {
               auto oTy = o.getType();
               if (!oTy.isIndex()) {
                 auto v = self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), o);
                 offsets.push_back(v);
               } else {
                 offsets.push_back(o);
               }
             }
             llvm::SmallVector<Value> sizes;
             llvm::SmallVector<int64_t> retSizes;
             for (const auto &s : sizs_vec) {
               auto v = self.create<arith::ConstantIndexOp>(s);
               sizes.push_back(v);
               retSizes.push_back(s);
             }
             llvm::SmallVector<Value> strides;
             for (const auto &s : strd_vec) {
               auto v = self.create<arith::ConstantIndexOp>(s);
               strides.push_back(v);
             }
             auto retTy = RankedTensorType::get(
                 retSizes,
                 cast<RankedTensorType>(ful.getType()).getElementType());

             return self.create<tensor::ExtractSliceOp>(retTy, ful, offsets,
                                                        sizes, strides);
           })
      .def("create_dsa_insert_slice",
           [](DSAOpBuilder &self, Value &ful, Value &sub,
              std::vector<Value> &offs_vec, std::vector<int> &sizs_vec,
              std::vector<int> &strd_vec) -> Value {
             llvm::SmallVector<Value> offsets;
             for (const auto &o : offs_vec) {
               auto oTy = o.getType();
               if (!oTy.isIndex()) {
                 auto v = self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), o);
                 offsets.push_back(v);
               } else {
                 offsets.push_back(o);
               }
             }
             llvm::SmallVector<Value> sizes;
             llvm::SmallVector<int64_t> retSizes;
             for (const auto &s : sizs_vec) {
               auto v = self.create<arith::ConstantIndexOp>(s);
               sizes.push_back(v);
               retSizes.push_back(s);
             }
             llvm::SmallVector<Value> strides;
             for (const auto &s : strd_vec) {
               auto v = self.create<arith::ConstantIndexOp>(s);
               strides.push_back(v);
             }
             auto retTy = RankedTensorType::get(
                 retSizes,
                 cast<RankedTensorType>(ful.getType()).getElementType());
             auto ret = self.create<tensor::InsertSliceOp>(sub, ful, offsets,
                                                           sizes, strides);
             return ret;
           })
      .def("create_dsa_subview",
           [](DSAOpBuilder &self, Value source, std::vector<Value> &offsets,
              const std::vector<int64_t> &sizes,
              const std::vector<int64_t> &strides) -> Value {
             SmallVector<mlir::OpFoldResult> mixedOffsets;
             auto *context = self.getBuilder().getContext();
             auto &builder = self.getBuilder();

             // Get source memref type for validation
             auto sourceType = mlir::cast<MemRefType>(source.getType());
             int64_t rank = sourceType.getRank();
             // Verify the number of parameters
             if (offsets.size() != rank || sizes.size() != rank ||
                 strides.size() != rank) {
               throw std::runtime_error("Number of offsets, sizes, and strides "
                                        "must match memref rank");
             }

             for (const auto &offset : offsets) {
               auto indexType = builder.getIndexType();
               if (offset.getType() != indexType) {
                 Value offset_val =
                     self.create<arith::IndexCastOp>(indexType, offset);
                 mixedOffsets.push_back(offset_val);
               } else {
                 mixedOffsets.push_back(offset);
               }
             }

             SmallVector<mlir::OpFoldResult> mixedSizes;
             SmallVector<mlir::OpFoldResult> mixedStrides;
             for (int64_t i = 0; i < rank; ++i) {
               int64_t size = sizes[i];
               int64_t stride = strides[i];
               int64_t srcDim = sourceType.getDimSize(i);

               // verify sizes cannot be negative or zero
               if (size <= 0) {
                 throw std::runtime_error("Expected sizes to be positive");
               }

               // verify strides cannot be negative or zero
               if (stride <= 0) {
                 throw std::runtime_error("Expected strides to be positive");
               }

               // getDimSize() returns -1 (ShapedType::kDynamic) for dynamic
               // dimensions
               if (!ShapedType::isDynamic(srcDim)) {
                 // verify the subview size does not exceed the source dimension
                 if (size > srcDim) {
                   throw std::runtime_error(
                       "Subview size cannot exceed source dimension size");
                 }

                 // verify strides cannot exceed the source dimension size
                 if (stride > srcDim) {
                   throw std::runtime_error(
                       "Stride cannot exceed source dimension size");
                 }
               }

               mixedSizes.push_back(IntegerAttr::get(
                   IntegerType::get(context, kIntegerAttrBitWidth), size));
               mixedStrides.push_back(IntegerAttr::get(
                   IntegerType::get(context, kIntegerAttrBitWidth), stride));
             }

             return self.create<memref::SubViewOp>(source, mixedOffsets,
                                                   mixedSizes, mixedStrides);
           });
}
