// Copyright 2026- Xcoresigma Technology Co., Ltd

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#ifdef __TLE_DSA__
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#endif

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "ir.h"
#include <stdexcept>

using namespace mlir;
namespace py = pybind11;

constexpr unsigned kIntegerAttrBitWidth = 64;

#ifdef __TLE_DSA__
// Convert an address-space attribute to a CommonIR MemorySpace enum. The DSA
// layer passes a hivm::AddressSpaceAttr (from ascend's get_target_attribute);
// decode it so buffers carry their real space (L1/L0A/...). Falls back to UB.
static mlir::triton::tile::MemorySpace attrToMemSpace(Attribute attr) {
  using MS = mlir::triton::tile::MemorySpace;
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  if (auto as = attr.dyn_cast_or_null<mlir::hivm::AddressSpaceAttr>()) {
    switch (as.getAddressSpace()) {
    case mlir::hivm::AddressSpace::GM:
      return MS::GM;
    case mlir::hivm::AddressSpace::L1:
      return MS::L1;
    case mlir::hivm::AddressSpace::L0A:
      return MS::L0A;
    case mlir::hivm::AddressSpace::L0B:
      return MS::L0B;
    case mlir::hivm::AddressSpace::L0C:
      return MS::L0C;
    case mlir::hivm::AddressSpace::UB:
      return MS::UB;
    default:
      return MS::UB;
    }
  }
  // Legacy / string-based fallback.
  if (auto strAttr = attr.dyn_cast_or_null<StringAttr>()) {
    auto name = strAttr.getValue();
    if (name == "GM")
      return MS::GM;
    if (name == "L1")
      return MS::L1;
    if (name == "L0A")
      return MS::L0A;
    if (name == "L0B")
      return MS::L0B;
    if (name == "L0C")
      return MS::L0C;
    if (name == "UB")
      return MS::UB;
  }
#else
  if (auto as = llvm::dyn_cast_or_null<mlir::hivm::AddressSpaceAttr>(attr)) {
    switch (as.getAddressSpace()) {
    case mlir::hivm::AddressSpace::GM:
      return MS::GM;
    case mlir::hivm::AddressSpace::L1:
      return MS::L1;
    case mlir::hivm::AddressSpace::L0A:
      return MS::L0A;
    case mlir::hivm::AddressSpace::L0B:
      return MS::L0B;
    case mlir::hivm::AddressSpace::L0C:
      return MS::L0C;
    case mlir::hivm::AddressSpace::UB:
      return MS::UB;
    default:
      return MS::UB;
    }
  }
  // Legacy / string-based fallback.
  if (auto strAttr = llvm::dyn_cast_or_null<StringAttr>(attr)) {
    auto name = strAttr.getValue();
    if (name == "GM")
      return MS::GM;
    if (name == "L1")
      return MS::L1;
    if (name == "L0A")
      return MS::L0A;
    if (name == "L0B")
      return MS::L0B;
    if (name == "L0C")
      return MS::L0C;
    if (name == "UB")
      return MS::UB;
  }
#endif
  return MS::UB;
}

#endif

#ifdef __TLE_DSA__
static std::string getShmemTypeName(Type type) {
  if (auto fpTy = llvm::dyn_cast<FloatType>(type)) {
    if (fpTy.isBF16())
      return "bfloat16";
    if (fpTy.isF16())
      return "half";
    if (fpTy.isF32())
      return "float";
  }
  if (auto intTy = llvm::dyn_cast<IntegerType>(type)) {
    switch (intTy.getWidth()) {
    case 8:
      return "int8";
    case 16:
      return "int16";
    case 32:
      return "int32";
    case 64:
      return "int64";
    }
  }
  return "unknown";
}

// Lower the 3.5 distributed primitives directly to hivm.custom ops, matching
// the conversion that ran in the 3.5 TleToHIVM pass.
static OpState createDistributedCustomOp(TritonOpBuilder &self,
                                         TypeRange resultTypes,
                                         const std::string &symbol,
                                         ValueRange operands, bool isPure,
                                         const std::string &libName = "",
                                         const std::string &libPath = "") {
  auto &builder = self.getBuilder();
  auto *ctx = builder.getContext();
  auto op = self.create<hivm::CustomOp>(resultTypes, "dist." + symbol,
                                        operands, ValueRange{}, ValueRange{});
  Operation *customOp = op.getOperation();
  auto funcOp = customOp->getParentOfType<triton::FuncOp>();

  bool hasDot = false;
  if (funcOp) {
    funcOp.walk([&](Operation *nested) {
      if (isa<triton::DotOp>(nested))
        hasDot = true;
    });
  }
  auto coreType =
      hasDot ? hivm::TCoreType::CUBE_AND_VECTOR : hivm::TCoreType::VECTOR;
  op->setAttr(hivm::TCoreTypeAttr::name,
              hivm::TCoreTypeAttr::get(ctx, coreType));
  op->setAttr("hivm.is_distributed", builder.getUnitAttr());
  op.setPipe(hivm::PIPE::PIPE_S);
  op.setVFMode(hivm::VFMode::SIMD);
  op->setAttr("symbol", builder.getStringAttr(symbol));
  if (!libName.empty())
    op->setAttr("libname", builder.getStringAttr(libName));
  if (!libPath.empty())
    op->setAttr("libpath", builder.getStringAttr(libPath));
  if (isPure)
    op->setAttr("no_side_effect", builder.getUnitAttr());

  llvm::SmallVector<int32_t> gmAddrArgs;
  if (funcOp) {
    for (auto &&[idx, operand] : llvm::enumerate(customOp->getOperands())) {
      if (!llvm::isa<triton::PointerType>(operand.getType()))
        continue;
      if (auto arg = llvm::dyn_cast<BlockArgument>(operand);
          arg && arg.getOwner() == &funcOp.getFunctionBody().front())
        gmAddrArgs.push_back(idx);
    }
  }
  op->setAttr("gm_addr_args_indices",
              DenseI32ArrayAttr::get(ctx, gmAddrArgs));
  return op;
}
#endif

void init_tle_dsa_ir(py::module &&m) {
  m.def("load_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    registry.insert<memref::MemRefDialect>();
    registry.insert<bufferization::BufferizationDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  auto *tle_cls = ir::getBuilderClass();

  tle_cls
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
             llvm::SmallVector<OpFoldResult> mixedOffsets;
             for (Value offset : offsets) {
               mixedOffsets.push_back(offset);
             }
             llvm::SmallVector<OpFoldResult> mixedSizes;
             llvm::SmallVector<int64_t> retSizes;
             for (const auto &s : sizs_vec) {
               mixedSizes.push_back(self.getBuilder().getIndexAttr(s));
               retSizes.push_back(s);
             }
             llvm::SmallVector<OpFoldResult> mixedStrides;
             for (const auto &s : strd_vec) {
               mixedStrides.push_back(self.getBuilder().getIndexAttr(s));
             }
             auto retTy = RankedTensorType::get(
                 retSizes,
                 cast<RankedTensorType>(ful.getType()).getElementType());

             return self.create<tensor::ExtractSliceOp>(
                 retTy, ful, mixedOffsets, mixedSizes, mixedStrides);
           })
      .def("create_dsa_insert_slice",
           [](TritonOpBuilder &self, Value &ful, Value &sub,
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
             llvm::SmallVector<OpFoldResult> mixedOffsets;
             for (Value offset : offsets) {
               mixedOffsets.push_back(offset);
             }
             llvm::SmallVector<OpFoldResult> mixedSizes;
             for (const auto &s : sizs_vec) {
               mixedSizes.push_back(self.getBuilder().getIndexAttr(s));
             }
             llvm::SmallVector<OpFoldResult> mixedStrides;
             for (const auto &s : strd_vec) {
               mixedStrides.push_back(self.getBuilder().getIndexAttr(s));
             }
             auto ret = self.create<tensor::InsertSliceOp>(
                 sub, ful, mixedOffsets, mixedSizes, mixedStrides);
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
           });

  // ============================================================================
  // CommonIR builder methods — create tile.* dialect ops
  // ============================================================================

#ifdef __TLE_DSA__
  // 3.5 distributed primitives. These used to be TleDSA dialect ops converted
  // by TleToHIVM; emit the resulting hivm.custom op directly instead.
  tle_cls
      ->def("create_get_rank",
            [](TritonOpBuilder &self, Value axis) -> Value {
              auto resultTy = self.getBuilder().getI32Type();
              auto op = createDistributedCustomOp(self, TypeRange{resultTy},
                                                  "aclshmem_my_pe",
                                                  ValueRange{axis}, true);
              return op->getResult(0);
            })
      .def("create_symm_at",
           [](TritonOpBuilder &self, Value ptr, Value rank) -> Value {
             auto elemTy =
                 llvm::cast<triton::PointerType>(ptr.getType()).getPointeeType();
             auto symbol = "aclshmem_ptr_" + getShmemTypeName(elemTy);
             auto op = createDistributedCustomOp(self, TypeRange{ptr.getType()},
                                                 symbol, ValueRange{ptr, rank},
                                                 true);
             return op->getResult(0);
           })
      .def("create_extern_call",
           [](TritonOpBuilder &self, const std::string &libName,
              const std::string &libPath, const std::string &symbol,
              std::vector<Value> &argList, const std::vector<Type> &retTypes,
              bool isPure) -> OpState {
             return createDistributedCustomOp(self, TypeRange(retTypes), symbol,
                                              ValueRange(argList), isPure,
                                              libName, libPath);
           });

  // Helper: load CommonIR dialect into context
  m.def("load_tile_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    registry.insert<mlir::triton::tile::CommonIRDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  // CommonIR buffer / tensor type construction
  tle_cls
      ->def("tile_get_buffer_type",
            [](TritonOpBuilder &self, std::vector<int64_t> &shape,
               Type &elementType, const Attribute &memorySpace) -> Type {
              auto memSpace = attrToMemSpace(memorySpace);
              auto *ctx = self.getBuilder().getContext();
              return mlir::triton::tile::BufType::get(ctx, shape, elementType,
                                                      memSpace);
            })
      .def("tile_get_tensor_type",
           [](TritonOpBuilder &self, std::vector<int64_t> &shape,
              Type &elementType, const Attribute &memorySpace) -> Type {
             auto memSpace = attrToMemSpace(memorySpace);
             auto *ctx = self.getBuilder().getContext();
             return mlir::triton::tile::TensorType::get(ctx, shape, elementType,
                                                        memSpace);
           })

      // tile.alloc — result type carries the memory space; pass it as the
      // $space attr
      .def("create_tile_alloc",
           [](TritonOpBuilder &self, Type tileBufType) -> Value {
             auto bufType =
                 mlir::cast<mlir::triton::tile::BufType>(tileBufType);
             return self.create<mlir::triton::tile::AllocOp>(
                 tileBufType, bufType.getMemorySpace(),
                 /*shape=*/mlir::ArrayAttr(), /*dtype=*/mlir::TypeAttr(),
                 /*policy=*/mlir::triton::tile::PolicyAttr(),
                 /*layout=*/mlir::triton::tile::LayoutAttr(),
                 /*lifetime=*/mlir::triton::tile::LifetimeAttr(),
                 /*comment=*/mlir::StringAttr());
           })
      // tile.copy — shape extents are informational at this layer; the op
      // itself takes only src/dst (+ optional engine/layout attrs).
      .def("create_tile_copy",
           [](TritonOpBuilder &self, Value &src, Value &dst,
              std::vector<Value> & /*shape*/, bool inter_no_alias) -> void {
             auto op = self.create<mlir::triton::tile::CopyOp>(
                 src, dst, /*engine=*/mlir::triton::tile::EngineAttr(),
                 /*src_layout=*/mlir::triton::tile::LayoutAttr(),
                 /*dst_nz_layout=*/mlir::triton::tile::NZLayoutAttr(),
                 /*transpose=*/mlir::UnitAttr(),
                 /*comment=*/mlir::StringAttr());
             if (inter_no_alias) {
               op->setAttr("inter_no_alias",
                           self.getBuilder().getBoolAttr(true));
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
             auto srcBuf =
                 mlir::cast<mlir::triton::tile::BufType>(source.getType());
             auto resTy = mlir::triton::tile::BufType::get(
                 ctx, sizes, srcBuf.getElementType(), srcBuf.getMemorySpace());
             auto op = self.create<mlir::triton::tile::SubViewOp>(
                 resTy, source, indexOffsets,
                 self.getBuilder().getI64ArrayAttr(sizes),
                 self.getBuilder().getI64ArrayAttr(strides));
             return op.getResult();
           })
      // tile.to_tensor — result is a standard ranked tensor (so tt.dot etc.
      // accept it), mirroring the source buffer's shape and element type.
      // `writable` records that the caller (e.g. tle.dsa.to_tensor, which
      // defaults to writable=True) expects an in-place view: custom ops write
      // their results back into the source buffer. CommonIRToHIVM lowers the
      // attribute to `bufferization.to_tensor ... restrict writable`; without
      // it One-Shot Bufferization treats every custom out as a value and
      // copies it into a fresh, unplannable UB allocation.
      .def("create_tile_to_tensor",
           [](TritonOpBuilder &self, Value &src, bool writable) -> Value {
             auto srcBuf =
                 mlir::cast<mlir::triton::tile::BufType>(src.getType());
             auto resTy = mlir::RankedTensorType::get(srcBuf.getShape(),
                                                      srcBuf.getElementType());
             auto op = self.create<mlir::triton::tile::ToTensorOp>(resTy, src);
             if (writable)
               op->setAttr("writable",
                           UnitAttr::get(self.getBuilder().getContext()));
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
      // tile.cube_launch — async Cube matmul hook. The current CommonIR op has
      // no token result, so the matching wait is emitted as a standalone op.
      .def("create_tile_cube_launch",
           [](TritonOpBuilder &self, Value &a, Value &b, Value &acc,
              Value &stageA, Value &stageB, Value &dst, bool transposeA,
              bool transposeB, bool init, std::string mma) -> void {
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
      .def("create_tile_cube_wait", [](TritonOpBuilder &self) -> void {
        self.create<mlir::triton::tile::CubeWaitOp>();
      })
#endif
  ;
}
