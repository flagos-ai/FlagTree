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
#include <stdexcept>

using namespace mlir;
namespace py = pybind11;

constexpr unsigned kIntegerAttrBitWidth = 64;

void init_tle_dsa_ir(py::module &&m) {
  auto *builder_cls = ir::getBuilderClass();

  builder_cls
      ->def("dsa_get_null_attr",
            [](TritonOpBuilder &self) { return Attribute(); })
      .def("dsa_get_buffer_type",
           [](TritonOpBuilder &self, std::vector<int64_t> &shape,
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
           [](TritonOpBuilder &self, Type memrefType) -> Value {
             return self.create<memref::AllocOp>(
                 mlir::cast<MemRefType>(memrefType));
           })
      // Add copy op
      .def("create_dsa_copy",
           [](TritonOpBuilder &self, Value &src, Value &dst,
              std::vector<Value> &shape, bool inter_no_alias) -> void {
             auto copyOp = self.create<triton::tle::DSACopyOp>(src, dst, shape);
             if (inter_no_alias) {
               copyOp->setAttr("inter_no_alias",
                               self.getBuilder().getBoolAttr(true));
             }
           })
      // Add op
      .def("create_dsa_add",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs, Value &res)
               -> void { self.create<triton::tle::DSAAddOp>(lhs, rhs, res); })
      // Sub op
      .def("create_dsa_sub",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs, Value &res)
               -> void { self.create<triton::tle::DSASubOp>(lhs, rhs, res); })
      // Mul op
      .def("create_dsa_mul",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs, Value &res)
               -> void { self.create<triton::tle::DSAMulOp>(lhs, rhs, res); })
      // Div op
      .def("create_dsa_div",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs, Value &res)
               -> void { self.create<triton::tle::DSADivOp>(lhs, rhs, res); })
      // Max op
      .def("create_dsa_max",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs, Value &res)
               -> void { self.create<triton::tle::DSAMaxOp>(lhs, rhs, res); })
      // Min op
      .def("create_dsa_min",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs, Value &res)
               -> void { self.create<triton::tle::DSAMinOp>(lhs, rhs, res); })
      // Dot op
      /// .def("create_dsa_dot",
      ///      [](TritonOpBuilder &self, Value &inA, Value &inB, Value &res,
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
           [](TritonOpBuilder &self, Value &src,
              const Attribute &addressSpace) -> Value {
             auto tensorType = dyn_cast<RankedTensorType>(src.getType());
             if (!tensorType) {
               llvm::report_fatal_error("to_buffer: src must be tensor type");
             }
             auto memrefType = MemRefType::get(
                 tensorType.getShape(), tensorType.getElementType(),
                 MemRefLayoutAttrInterface{}, addressSpace);
             return self.create<bufferization::ToBufferOp>(memrefType, src);
           })
      .def("dsa_to_tensor",
           [](TritonOpBuilder &self, Value &src, bool writable) -> Value {
             const auto &memrefType = mlir::cast<MemRefType>(src.getType());
             auto tensorType = RankedTensorType::get(
                 memrefType.getShape(), memrefType.getElementType());
             auto hasAddressSpace = memrefType.getMemorySpace();
             if (hasAddressSpace) {
               return self.create<bufferization::ToTensorOp>(tensorType, src,
                                                             true, writable);
             }
             return self.create<bufferization::ToTensorOp>(tensorType, src,
                                                           true, writable);
           })
      .def("create_dsa_extract_scalar",
           [](TritonOpBuilder &self, Value &src,
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
           [](TritonOpBuilder &self, Value &ful, std::vector<Value> &offs_vec,
              std::vector<int> &sizs_vec, std::vector<int> &strd_vec) -> Value {
             llvm::SmallVector<Value> offsets;
             llvm::SmallVector<int64_t> staticOffsets;
             for (const auto &o : offs_vec) {
               auto oTy = o.getType();
               if (!oTy.isIndex()) {
                 auto v = self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), o);
                 offsets.push_back(v);
               } else {
                 offsets.push_back(o);
               }
               staticOffsets.push_back(ShapedType::kDynamic);
             }
             llvm::SmallVector<Value> sizes;
             llvm::SmallVector<int64_t> staticSizes;
             llvm::SmallVector<int64_t> retSizes;
             for (const auto &s : sizs_vec) {
               staticSizes.push_back(s);
               retSizes.push_back(s);
             }
             llvm::SmallVector<Value> strides;
             llvm::SmallVector<int64_t> staticStrides;
             for (const auto &s : strd_vec) {
               auto v = self.create<arith::ConstantIndexOp>(s);
               strides.push_back(v);
               staticStrides.push_back(ShapedType::kDynamic);
             }
             auto retTy = RankedTensorType::get(
                 retSizes,
                 cast<RankedTensorType>(ful.getType()).getElementType());

             return self.create<tensor::ExtractSliceOp>(
                 retTy, ful, offsets, sizes, strides, staticOffsets,
                 staticSizes, staticStrides);
           })
      .def("create_dsa_insert_slice",
           [](TritonOpBuilder &self, Value &ful, Value &sub,
              std::vector<Value> &offs_vec, std::vector<int> &sizs_vec,
              std::vector<int> &strd_vec) -> Value {
             llvm::SmallVector<Value> offsets;
             llvm::SmallVector<int64_t> staticOffsets;
             for (const auto &o : offs_vec) {
               auto oTy = o.getType();
               if (!oTy.isIndex()) {
                 auto v = self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), o);
                 offsets.push_back(v);
               } else {
                 offsets.push_back(o);
               }
               staticOffsets.push_back(ShapedType::kDynamic);
             }
             llvm::SmallVector<Value> sizes;
             llvm::SmallVector<int64_t> staticSizes;
             llvm::SmallVector<int64_t> retSizes;
             for (const auto &s : sizs_vec) {
               staticSizes.push_back(s);
               retSizes.push_back(s);
             }
             llvm::SmallVector<Value> strides;
             llvm::SmallVector<int64_t> staticStrides;
             for (const auto &s : strd_vec) {
               auto v = self.create<arith::ConstantIndexOp>(s);
               strides.push_back(v);
               staticStrides.push_back(ShapedType::kDynamic);
             }
             auto retTy = RankedTensorType::get(
                 retSizes,
                 cast<RankedTensorType>(ful.getType()).getElementType());
             auto ret = self.create<tensor::InsertSliceOp>(
                 sub, ful, offsets, sizes, strides, staticOffsets, staticSizes,
                 staticStrides);
             return ret;
           })
      .def("create_dsa_subview",
           [](TritonOpBuilder &self, Value source, std::vector<Value> &offsets,
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
           })
      // dist
      .def("create_get_rank",
           [](TritonOpBuilder &self, Value axis) -> Value {
             return self.create<triton::tle::GetRankOp>(axis);
           })
      .def("create_symm_at",
           [](TritonOpBuilder &self, Value ptr, Value rank) -> Value {
             return self.create<triton::tle::SymmAtOp>(ptr.getType(), ptr,
                                                       rank);
           })
      .def("create_extern_call",
           [](TritonOpBuilder &self, const std::string &libName,
              const std::string &libPath, const std::string &symbol,
              std::vector<Value> &argList, const std::vector<Type> &retTypes,
              bool isPure) -> OpState {
             return self.create<triton::tle::ExternCallOp>(
                 retTypes, argList, libName, libPath, symbol, isPure);
           });
}
