/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ir.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

using namespace mlir;
namespace py = pybind11;

constexpr unsigned kIntegerAttrBitWidth = 64;

struct BufferOpBuilder : public TritonOpBuilder {};

void init_buffer_ir(py::module &&m) {
  m.def("load_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    registry.insert<memref::MemRefDialect>();
    registry.insert<bufferization::BufferizationDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  py::class_<BufferOpBuilder, TritonOpBuilder>(
      m, "buffer_builder", py::module_local(), py::dynamic_attr())
      .def(py::init<MLIRContext *>())
      .def("get_null_attr", [](BufferOpBuilder &self) { return Attribute(); })
      .def("get_str_array_attr",
           [](BufferOpBuilder &self,
              const std::vector<std::string> &array) -> ArrayAttr {
             auto strRefVec = to_vector(llvm::map_range(
                 array, [](const auto &s) { return llvm::StringRef(s); }));
             return self.getBuilder().getStrArrayAttr(
                 llvm::ArrayRef<StringRef>{strRefVec});
           })
      .def("alloc",
           [](BufferOpBuilder &self, Type memrefType) -> Value {
             return self.create<memref::AllocOp>(
                 mlir::cast<MemRefType>(memrefType));
           })
      .def("to_buffer",
           [](BufferOpBuilder &self, Value &src,
              const Attribute &addressSpace) -> Value {
             auto tensorType = dyn_cast<RankedTensorType>(src.getType());
             if (!tensorType) {
               llvm::report_fatal_error("to_buffer: src must be tensor type");
             }
             auto memrefType = MemRefType::get(tensorType.getShape(),
                                               tensorType.getElementType(),
                                               MemRefLayoutAttrInterface{});
             // TODO: We need to add a pass before OneShotBufferize to generate
             // MemorySpaceCastOp
             Operation *memref =
                 self.create<bufferization::ToBufferOp>(memrefType, src);
             if (addressSpace) {
               memref = self.create<memref::MemorySpaceCastOp>(
                   MemRefType::get(memrefType.getShape(),
                                   memrefType.getElementType(),
                                   memrefType.getLayout(), addressSpace),
                   memref->getResult(0));
             }
             return memref->getResult(0);
           })
      .def("to_tensor",
           [](BufferOpBuilder &self, Value &src, bool writable) -> Value {
             const auto &memrefType = mlir::cast<MemRefType>(src.getType());
             auto tensorType = mlir::RankedTensorType::get(
                 memrefType.getShape(), memrefType.getElementType());
             auto hasAddressSpace = memrefType.getMemorySpace();
             if (hasAddressSpace) {
               MemRefType targetType = MemRefType::get(
                   memrefType.getShape(), memrefType.getElementType(),
                   memrefType.getLayout());
               return self.create<bufferization::ToTensorOp>(
                   tensorType,
                   self.create<memref::MemorySpaceCastOp>(targetType, src),
                   true ? mlir::UnitAttr::get(self.getContext()) : nullptr,
                   writable ? mlir::UnitAttr::get(self.getContext()) : nullptr);
             }
             return self.create<bufferization::ToTensorOp>(
                 tensorType, src,
                 true ? mlir::UnitAttr::get(self.getContext()) : nullptr,
                 writable ? mlir::UnitAttr::get(self.getContext()) : nullptr);
           })
      .def("subview",
           [](BufferOpBuilder &self, Value source, std::vector<Value> &offsets,
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
