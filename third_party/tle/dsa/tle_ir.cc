// Copyright 2026- Xcoresigma Technology Co., Ltd

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "tle/dsa/dialect/include/IR/Dialect.h"

#include "mlir-ext/Dialect/TileIR/IR/TileIRDialect.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

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

// Convert an address-space attribute to a TileIR MemorySpace enum. The DSA
// layer passes a hivm::AddressSpaceAttr (from ascend's get_target_attribute);
// decode it so buffers carry their real space (L1/L0A/...). Falls back to UB.
static mlir::triton::tile::MemorySpace attrToMemSpace(Attribute attr) {
  using MS = mlir::triton::tile::MemorySpace;
  if (auto as = attr.dyn_cast_or_null<mlir::hivm::AddressSpaceAttr>()) {
    switch (as.getAddressSpace()) {
    case mlir::hivm::AddressSpace::GM:  return MS::GM;
    case mlir::hivm::AddressSpace::L1:  return MS::L1;
    case mlir::hivm::AddressSpace::L0A: return MS::L0A;
    case mlir::hivm::AddressSpace::L0B: return MS::L0B;
    case mlir::hivm::AddressSpace::L0C: return MS::L0C;
    case mlir::hivm::AddressSpace::UB:  return MS::UB;
    default: return MS::UB;
    }
  }
  // Legacy / string-based fallback.
  if (auto strAttr = attr.dyn_cast_or_null<StringAttr>()) {
    auto name = strAttr.getValue();
    if (name == "GM")  return MS::GM;
    if (name == "L1")  return MS::L1;
    if (name == "L0A") return MS::L0A;
    if (name == "L0B") return MS::L0B;
    if (name == "L0C") return MS::L0C;
    if (name == "UB")  return MS::UB;
  }
  return MS::UB;
}

void init_tle_dsa_ir(py::module &&m) {
  m.def("load_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    registry.insert<memref::MemRefDialect>();
    registry.insert<bufferization::BufferizationDialect>();
    registry.insert<triton::tle::TleDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  auto tle_cls = py::class_<TritonOpBuilder>(
      m, "tle_builder", py::module_local(), py::dynamic_attr())
      .def(py::init<mlir::MLIRContext *>())
      .def("dsa_get_null_attr", [](TritonOpBuilder &self) { return Attribute(); })
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
             llvm::SmallVector<OpFoldResult> mixedOffsets;
             for (Value offset : offsets) {
               mixedOffsets.push_back(offset);
             }
             llvm::SmallVector<OpFoldResult> mixedSizes;
             llvm::SmallVector<int64_t> staticSizes;
             llvm::SmallVector<int64_t> retSizes;
             for (const auto &s : sizs_vec) {
               mixedSizes.push_back(self.getBuilder().getIndexAttr(s));
               staticSizes.push_back(s);
               retSizes.push_back(s);
             }
             llvm::SmallVector<OpFoldResult> mixedStrides;
             llvm::SmallVector<int64_t> staticStrides;
             for (const auto &s : strd_vec) {
               mixedStrides.push_back(self.getBuilder().getIndexAttr(s));
               staticStrides.push_back(ShapedType::kDynamic);
             }
             auto retTy = RankedTensorType::get(
                 retSizes,
                 cast<RankedTensorType>(ful.getType()).getElementType());

             return self.create<tensor::ExtractSliceOp>(
                 retTy, ful, mixedOffsets, mixedSizes, mixedStrides,
                 staticOffsets, staticSizes, staticStrides);
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
             llvm::SmallVector<OpFoldResult> mixedOffsets;
             for (Value offset : offsets) {
               mixedOffsets.push_back(offset);
             }
             llvm::SmallVector<OpFoldResult> mixedSizes;
             llvm::SmallVector<int64_t> staticSizes;
             for (const auto &s : sizs_vec) {
               staticSizes.push_back(s);
               mixedSizes.push_back(self.getBuilder().getIndexAttr(s));
             }
             llvm::SmallVector<OpFoldResult> mixedStrides;
             llvm::SmallVector<int64_t> staticStrides;
             for (const auto &s : strd_vec) {
               mixedStrides.push_back(self.getBuilder().getIndexAttr(s));
               staticStrides.push_back(ShapedType::kDynamic);
             }
             auto ret = self.create<tensor::InsertSliceOp>(
                 sub, ful, mixedOffsets, mixedSizes, mixedStrides,
                 staticOffsets, staticSizes, staticStrides);
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

  // ============================================================================
  // TileIR builder methods — create tile.* dialect ops
  // ============================================================================

  // Helper: load TileIR dialect into context
  m.def("load_tile_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    registry.insert<mlir::triton::tile::TileIRDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  // TileIR buffer / tensor type construction
  tle_cls.def("tile_get_buffer_type",
       [](TritonOpBuilder &self, std::vector<int64_t> &shape,
          Type &elementType, const Attribute &memorySpace) -> Type {
         auto memSpace = attrToMemSpace(memorySpace);
         auto *ctx = self.getBuilder().getContext();
         return mlir::triton::tile::BufType::get(ctx, shape, elementType, memSpace);
       })
  .def("tile_get_tensor_type",
       [](TritonOpBuilder &self, std::vector<int64_t> &shape,
          Type &elementType, const Attribute &memorySpace) -> Type {
         auto memSpace = attrToMemSpace(memorySpace);
         auto *ctx = self.getBuilder().getContext();
         return mlir::triton::tile::TensorType::get(ctx, shape, elementType, memSpace);
       })

  // tile.alloc — result type carries the memory space; pass it as the $space attr
  .def("create_tile_alloc",
       [](TritonOpBuilder &self, Type tileBufType) -> Value {
         auto bufType = mlir::cast<mlir::triton::tile::BufType>(tileBufType);
         return self.create<mlir::triton::tile::AllocOp>(
             tileBufType, bufType.getMemorySpace(),
             /*shape=*/mlir::ArrayAttr(), /*dtype=*/mlir::TypeAttr(),
             /*policy=*/mlir::triton::tile::PolicyAttr(),
             /*layout=*/mlir::triton::tile::LayoutAttr(),
             /*lifetime=*/mlir::triton::tile::LifetimeAttr(),
             /*comment=*/mlir::StringAttr());
       })
  // tile.copy — shape extents are informational at this layer; the op itself
  // takes only src/dst (+ optional engine/layout attrs).
  .def("create_tile_copy",
       [](TritonOpBuilder &self, Value &src, Value &dst,
          std::vector<Value> & /*shape*/, bool inter_no_alias) -> void {
         auto op = self.create<mlir::triton::tile::CopyOp>(
             src, dst, /*engine=*/mlir::triton::tile::EngineAttr(),
             /*src_layout=*/mlir::triton::tile::LayoutAttr(),
             /*dst_nz_layout=*/mlir::triton::tile::NZLayoutAttr(),
             /*transpose=*/mlir::UnitAttr(), /*comment=*/mlir::StringAttr());
         if (inter_no_alias) {
           op->setAttr("inter_no_alias", self.getBuilder().getBoolAttr(true));
         }
       })
  // tile.subview — result buffer type = sizes + source elt/space
  .def("create_tile_subview",
       [](TritonOpBuilder &self, Value source, std::vector<Value> &offsets,
          const std::vector<int64_t> &sizes,
          const std::vector<int64_t> &strides) -> Value {
         SmallVector<Value> indexOffsets;
         auto &builder = self.getBuilder();
         auto indexType = builder.getIndexType();
         for (Value offset : offsets) {
           if (offset.getType() != indexType) {
             offset = self.create<arith::IndexCastOp>(indexType, offset);
           }
           indexOffsets.push_back(offset);
         }
         auto *ctx = self.getBuilder().getContext();
         auto srcBuf = mlir::cast<mlir::triton::tile::BufType>(source.getType());
         auto resTy = mlir::triton::tile::BufType::get(
             ctx, sizes, srcBuf.getElementType(), srcBuf.getMemorySpace());
         auto op = self.create<mlir::triton::tile::SubViewOp>(
             resTy, source, indexOffsets,
             self.getBuilder().getI64ArrayAttr(sizes),
             self.getBuilder().getI64ArrayAttr(strides));
         return op.getResult();
       })
  // tile.to_tensor — result is a standard ranked tensor (so tt.dot etc. accept
  // it), mirroring the source buffer's shape and element type.
  .def("create_tile_to_tensor",
       [](TritonOpBuilder &self, Value &src, bool /*writable*/) -> Value {
         auto srcBuf = mlir::cast<mlir::triton::tile::BufType>(src.getType());
         auto resTy = mlir::RankedTensorType::get(srcBuf.getShape(),
                                                  srcBuf.getElementType());
         auto op = self.create<mlir::triton::tile::ToTensorOp>(resTy, src);
         return op.getResult();
       })
  // tile.store_tensor
  .def("create_tile_store_tensor",
       [](TritonOpBuilder &self, Value &src, Value &dst) -> void {
         self.create<mlir::triton::tile::StoreTensorOp>(src, dst);
       })
  // tile.set_flag
  .def("create_tile_set_flag",
       [](TritonOpBuilder &self, int64_t producer, int64_t consumer,
          int64_t event) -> void {
         self.create<mlir::triton::tile::SetFlagOp>(
             static_cast<mlir::triton::tile::Pipe>(producer),
             static_cast<mlir::triton::tile::Pipe>(consumer),
             static_cast<mlir::triton::tile::EventID>(event));
       })
  // tile.wait_flag
  .def("create_tile_wait_flag",
       [](TritonOpBuilder &self, int64_t producer, int64_t consumer,
          int64_t event) -> void {
         self.create<mlir::triton::tile::WaitFlagOp>(
             static_cast<mlir::triton::tile::Pipe>(producer),
             static_cast<mlir::triton::tile::Pipe>(consumer),
             static_cast<mlir::triton::tile::EventID>(event));
       })
  // tile.pipe_barrier
  .def("create_tile_pipe_barrier",
       [](TritonOpBuilder &self, int64_t pipe) -> void {
         self.create<mlir::triton::tile::PipeBarrierOp>(
             static_cast<mlir::triton::tile::Pipe>(pipe));
       })
  // tile.gm_offset — result pointer type matches the base
  .def("create_tile_gm_offset",
       [](TritonOpBuilder &self, Value &base, std::vector<Value> &indices,
          std::vector<Value> &strides) -> Value {
         SmallVector<Value> indexValues;
         SmallVector<Value> strideValues;
         auto &builder = self.getBuilder();
         auto indexType = builder.getIndexType();
         for (Value index : indices) {
           if (index.getType() != indexType) {
             index = self.create<arith::IndexCastOp>(indexType, index);
           }
           indexValues.push_back(index);
         }
         for (Value stride : strides) {
           if (stride.getType() != indexType) {
             stride = self.create<arith::IndexCastOp>(indexType, stride);
           }
           strideValues.push_back(stride);
         }
         auto op = self.create<mlir::triton::tile::GmOffsetOp>(
             base.getType(), base, indexValues, strideValues);
         return op.getResult();
       })
  // tile.cube_launch — async Cube matmul hook. The current TileIR op has no
  // token result, so the matching wait is emitted as a standalone op.
  .def("create_tile_cube_launch",
       [](TritonOpBuilder &self, Value &a, Value &b, Value &acc, Value &stageA,
          Value &stageB, Value &dst, bool transposeA, bool transposeB,
          bool init, std::string mma) -> void {
         auto &builder = self.getBuilder();
         auto unitAttr = builder.getUnitAttr();
         self.create<mlir::triton::tile::CubeLaunchOp>(
             a, b, acc, stageA, stageB, dst,
             transposeA ? unitAttr : mlir::UnitAttr(),
             transposeB ? unitAttr : mlir::UnitAttr(),
             init ? unitAttr : mlir::UnitAttr(),
             mma.empty() ? mlir::StringAttr() : builder.getStringAttr(mma),
             /*comment=*/mlir::StringAttr());
       })
  // tile.cube_wait
  .def("create_tile_cube_wait",
       [](TritonOpBuilder &self) -> void {
         self.create<mlir::triton::tile::CubeWaitOp>();
       });
}
