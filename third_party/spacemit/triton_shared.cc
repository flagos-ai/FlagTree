#include "include/triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "include/triton-shared/Dialect/XSMT/IR/XSMTOps.h"
#include "include/triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncDialect.h"
#include "include/triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncOps.h"
#include "include/triton-shared/Utils/MemorySpaceUtils.h"
#include "ir.h"
#include "mlir/AsmParser/AsmParser.h" // mlir::parseType/parseAttribute (smt_rvisa builder helpers)
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "tle/dialect/include/IR/Dialect.h"
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
// Alias onto the generic third_party/tle dialect namespace so the
// original call sites read unchanged.
namespace tle = ::mlir::triton::tle;

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
              mlir::Type type, std::string scope) -> Value {
             if (shape.empty()) {
               throw std::runtime_error("alloc shape cannot be empty");
             }
             auto op = self.create<xsmt::AllocOp>(type, shape, scope);
             return op;
           })
      .def("create_alloc_copies",
           [](TritonOpBuilder &self, std::vector<int64_t> &shape,
              mlir::Type elementType, std::string scope) -> mlir::Value {
             if (shape.empty())
               throw std::runtime_error("alloc_copies shape cannot be empty");
             auto op =
                 self.create<xsmt::AllocCopiesOp>(shape, elementType, scope);
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
      .def("create_vmt4d",
           [](TritonOpBuilder &self, Value &a, Value &b,
              std::optional<Value> c = std::nullopt) -> Value {
             auto aType = cast<RankedTensorType>(a.getType());
             auto bType = cast<RankedTensorType>(b.getType());

             assert(aType.getRank() == 2 && "A must be a 2D [1, K] tensor");
             assert(bType.getRank() == 2 && "B must be a 2D [K, N] tensor");
             assert(aType.getDimSize(0) == 1 && "A must have M == 1");
             assert(aType.getDimSize(1) == bType.getDimSize(0) &&
                    "K dims of A and B must match");

             // f16 inputs without an explicit init accumulate in f32 (engine
             // widening); with an init the init element type wins.
             Type elemTy = aType.getElementType();
             if (c.has_value())
               elemTy = cast<RankedTensorType>(c->getType()).getElementType();
             else if (elemTy.isF16())
               elemTy = Float32Type::get(aType.getContext());

             auto resultType =
                 RankedTensorType::get({1, bType.getDimSize(1)}, elemTy);
             return self.create<xsmt::VMT4DOp>(resultType, a, b,
                                               c.value_or(Value()));
           },
           py::arg("a"), py::arg("b"), py::arg("c") = py::none())
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
              Type &elementType, int copies, std::string scopeKind) -> Type {
             return xsmt::BufferType::get(shape, elementType, copies,
                                          scopeKind);
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
  // The tt.ptr <-> memref bridge memory-space attr is exposed structurally as
  // ir.builder.get_bridge_memory_space() (init_triton_smt_rvisa_ir below): the
  // C++ header MemorySpaceUtils.h getDefaultBridgeMemorySpace stays the single
  // source of truth, and the attr crosses the boundary as an Attribute object
  // (no printed-text channel).
}

// ============================================================================
// TLE (Triton Language Extension) IR bindings
// ============================================================================
void init_triton_spacemit_tle_ir(py::module &&m) {
  auto *builder_cls = ir::getBuilderClass();
  builder_cls->def(
      "create_tle_dsl_region_direct",
      [](TritonOpBuilder &self, const std::string &fn_name,
         std::vector<Value> &inputs, std::vector<Type> &paramTypes,
         py::function bodyBuilder) {
        // codegen.py builds the raw fn body straight into dsl_region's
        // region via bodyBuilder, using the same builder instance — no
        // parseSourceString, no raw_linalg string.
        auto &builder = self.getBuilder();
        auto fnAttr = builder.getStringAttr(fn_name);
        SmallVector<Value> operands(inputs.begin(), inputs.end());
        // Generic third_party/tle DSLRegionOp: region_dialect/arg_dialect/
        // output_operand_indices are required builder operands, but only
        // that dialect's own consumers read them — emit the
        // self-consistent no-output configuration. fn_name travels as a
        // free extra attribute (the op has no assemblyFormat, so its
        // generic printer round-trips the full attr dict); it is what
        // TLEToLinalg.cpp's DSLRegionOpPattern reads back.
        auto op = self.create<tle::DSLRegionOp>(
            mlir::TypeRange{}, operands, "linalg", "memref",
            llvm::ArrayRef<int32_t>{}, mlir::StringAttr());
        op->setAttr("fn_name", fnAttr);

        mlir::Region &body = op.getBody();
        mlir::Block *block = new mlir::Block();
        body.push_back(block);
        for (mlir::Type paramTy : paramTypes)
          block->addArgument(paramTy, op.getLoc());

        mlir::OpBuilder::InsertionGuard guard(builder);
        self.setInsertionPointToStart(*block);
        std::vector<Value> blockArgs(block->getArguments().begin(),
                                     block->getArguments().end());
        bodyBuilder(std::ref(self), blockArgs);

        // bodyBuilder emits the raw fn's ops but not the terminator
        // (there's no func.return here to translate) — always close with
        // spine_ext.return; the TLEToLinalg pattern expects that
        // terminator.
        mlir::OperationState retState(op.getLoc(), "spine_ext.return");
        builder.create(retState);
      },
      py::arg("fn_name"), py::arg("inputs"), py::arg("param_types"),
      py::arg("body_builder"),
      "Create tle.dsl_region by invoking body_builder(builder, block_args) "
      "directly — no MLIR text parse round trip");
}

// ============================================================================
// Spine Raw IR Builder Bindings
// Provides the standard MLIR builder API for smt_rvisa codegen, replacing
// string concatenation
// ============================================================================

// Helper: parse MLIR type string
static Type parseTypeString(OpBuilder &builder, const std::string &typeStr) {
  MLIRContext *ctx = builder.getContext();
  return mlir::parseType(typeStr, ctx);
}

void init_triton_smt_rvisa_ir(py::module &&m) {
  auto *builder_cls = ir::getBuilderClass();

  // ========================================================================
  // Type utilities
  // ========================================================================
  builder_cls
      ->def("get_index_type",
            [](TritonOpBuilder &self) -> Type {
              return self.getBuilder().getIndexType();
            })
      .def("get_i32_type",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getI32Type();
           })
      .def("get_i64_type",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getI64Type();
           })
      .def("get_f16_type",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getF16Type();
           })
      .def("get_f32_type",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getF32Type();
           })
      .def(
          "parse_type",
          [](TritonOpBuilder &self, const std::string &typeStr) -> Type {
            return parseTypeString(self.getBuilder(), typeStr);
          },
          py::arg("type_str"),
          "Parse MLIR type string (e.g., 'vector<32xf32>', 'memref<?xf16>')")

      // ======================================================================
      // Structured type constructors (official MLIR C++ APIs). The smt_rvisa
      // Python type layer (types.py Ty.build) passes objects across the
      // boundary instead of assembling type text, so these are the single
      // place where MLIR types get constructed for raw kernels.
      // ======================================================================
      .def(
          "get_integer_type",
          [](TritonOpBuilder &self, unsigned bits) -> Type {
            return IntegerType::get(self.getBuilder().getContext(), bits);
          },
          py::arg("bits"))
      .def(
          "get_float_type",
          [](TritonOpBuilder &self, unsigned bits, bool is_bf) -> Type {
            auto *ctx = self.getBuilder().getContext();
            if (is_bf) {
              if (bits != 16) {
                throw std::invalid_argument(
                    "get_float_type: bfloat only supports 16 bits");
              }
              return BFloat16Type::get(ctx);
            }
            switch (bits) {
            case 16:
              return Float16Type::get(ctx);
            case 32:
              return Float32Type::get(ctx);
            case 64:
              return Float64Type::get(ctx);
            default:
              throw std::invalid_argument(
                  "get_float_type: unsupported float width");
            }
          },
          py::arg("bits"), py::arg("is_bf"))
      .def(
          "get_vector_type",
          [](TritonOpBuilder &self, std::vector<int64_t> dims,
             std::vector<bool> scalableDims, Type elemType) -> Type {
            // std::vector<bool> is bit-packed (no .data()); copy into a
            // SmallVector<bool> for VectorType::get.
            SmallVector<bool> scalable(scalableDims.begin(),
                                       scalableDims.end());
            return VectorType::get(dims, elemType, scalable);
          },
          py::arg("dims"), py::arg("scalable_dims"), py::arg("elem_type"))
      .def(
          "get_memref_type",
          [](TritonOpBuilder &self, std::vector<std::optional<int64_t>> dims,
             Type elemType,
             std::optional<std::vector<std::optional<int64_t>>> strides,
             std::optional<int64_t> offset,
             std::optional<Attribute> memorySpace) -> Type {
            SmallVector<int64_t> shape;
            shape.reserve(dims.size());
            for (auto d : dims) {
              shape.push_back(d.value_or(ShapedType::kDynamic));
            }
            Attribute ms = memorySpace.value_or(Attribute());
            if (strides.has_value()) {
              SmallVector<int64_t> stridesVec;
              stridesVec.reserve(strides->size());
              for (auto s : *strides) {
                stridesVec.push_back(s.value_or(ShapedType::kDynamic));
              }
              auto layout = StridedLayoutAttr::get(
                  self.getBuilder().getContext(),
                  offset.value_or(ShapedType::kDynamic), stridesVec);
              return MemRefType::get(shape, elemType, layout, ms);
            }
            return MemRefType::get(shape, elemType, MemRefLayoutAttrInterface{},
                                   ms);
          },
          py::arg("dims"), py::arg("elem_type"),
          py::arg("strides") = py::none(), py::arg("offset") = py::none(),
          py::arg("memory_space") = py::none())
      .def(
          "get_unranked_memref_type",
          [](TritonOpBuilder &self, Type elemType,
             std::optional<Attribute> memorySpace) -> Type {
            return UnrankedMemRefType::get(elemType,
                                           memorySpace.value_or(Attribute()));
          },
          py::arg("elem_type"), py::arg("memory_space") = py::none())
      .def(
          "get_tensor_type",
          [](TritonOpBuilder &self, std::vector<std::optional<int64_t>> dims,
             Type elemType) -> Type {
            SmallVector<int64_t> shape;
            shape.reserve(dims.size());
            for (auto d : dims) {
              shape.push_back(d.value_or(ShapedType::kDynamic));
            }
            return RankedTensorType::get(shape, elemType);
          },
          py::arg("dims"), py::arg("elem_type"))
      .def("get_bridge_memory_space",
           [](TritonOpBuilder &self) -> Attribute {
             // tt.ptr <-> memref bridge memory space (MemorySpaceUtils.h);
             // returned as an Attribute in the builder's context so Python
             // never handles attr text.
             return mlir::triton::getDefaultBridgeMemorySpace(
                 self.getBuilder().getContext());
           })

      // ======================================================================
      // Structured type introspection (types.py classify_type). Lets Python
      // re-derive its Ty hierarchy from a Type parsed via the official MLIR
      // parser (builder.parse_type) without regex on printed text.
      // ======================================================================
      .def(
          "type_is_vector",
          [](TritonOpBuilder &self, Type t) -> bool {
            return mlir::isa<VectorType>(t);
          },
          py::arg("type"))
      .def(
          "type_vector_dims",
          [](TritonOpBuilder &self, Type t) -> std::vector<int64_t> {
            auto vt = mlir::dyn_cast<VectorType>(t);
            if (!vt) {
              throw std::invalid_argument("type_vector_dims: not a VectorType");
            }
            return std::vector<int64_t>(vt.getShape().begin(),
                                        vt.getShape().end());
          },
          py::arg("type"))
      .def(
          "type_vector_scalable_dims",
          [](TritonOpBuilder &self, Type t) -> std::vector<bool> {
            auto vt = mlir::dyn_cast<VectorType>(t);
            if (!vt) {
              throw std::invalid_argument(
                  "type_vector_scalable_dims: not a VectorType");
            }
            return std::vector<bool>(vt.getScalableDims().begin(),
                                     vt.getScalableDims().end());
          },
          py::arg("type"))
      .def(
          "type_vector_elem",
          [](TritonOpBuilder &self, Type t) -> Type {
            auto vt = mlir::dyn_cast<VectorType>(t);
            if (!vt) {
              throw std::invalid_argument("type_vector_elem: not a VectorType");
            }
            return vt.getElementType();
          },
          py::arg("type"))
      .def(
          "type_is_index",
          [](TritonOpBuilder &self, Type t) -> bool {
            return mlir::isa<IndexType>(t);
          },
          py::arg("type"))
      .def(
          "type_int_width",
          [](TritonOpBuilder &self, Type t) -> unsigned {
            if (auto it = mlir::dyn_cast<IntegerType>(t)) {
              return it.getWidth();
            }
            return 0;
          },
          py::arg("type"))
      .def(
          "type_float_info",
          [](TritonOpBuilder &self,
             Type t) -> std::optional<std::pair<unsigned, bool>> {
            if (auto ft = mlir::dyn_cast<FloatType>(t)) {
              return std::make_pair(ft.getWidth(), mlir::isa<BFloat16Type>(ft));
            }
            return std::nullopt;
          },
          py::arg("type"))

      // ========================================================================
      // Arith dialect - constants
      // ========================================================================
      .def(
          "create_arith_constant_index",
          [](TritonOpBuilder &self, int64_t value) -> Value {
            auto type = self.getBuilder().getIndexType();
            auto attr = self.getBuilder().getIntegerAttr(type, value);
            return self.create<arith::ConstantOp>(type, attr).getResult();
          },
          py::arg("value"))
      .def(
          "create_arith_constant_int",
          [](TritonOpBuilder &self, int64_t value, Type intType) -> Value {
            auto attr = self.getBuilder().getIntegerAttr(intType, value);
            return self.create<arith::ConstantOp>(intType, attr).getResult();
          },
          py::arg("value"), py::arg("int_type"))
      .def(
          "create_arith_constant_float",
          [](TritonOpBuilder &self, double value, Type floatType) -> Value {
            auto attr = self.getBuilder().getFloatAttr(floatType, value);
            return self.create<arith::ConstantOp>(floatType, attr).getResult();
          },
          py::arg("value"), py::arg("float_type"))

      // ========================================================================
      // Arith dialect - integer ops
      // ========================================================================
      .def(
          "create_arith_addi",
          [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
            return self.create<arith::AddIOp>(lhs, rhs).getResult();
          },
          py::arg("lhs"), py::arg("rhs"))
      .def("create_arith_subi",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::SubIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_muli",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::MulIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_divsi",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::DivSIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_divui",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::DivUIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_remsi",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::RemSIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_minsi",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::MinSIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_maxsi",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::MaxSIOp>(lhs, rhs).getResult();
           })

      // ========================================================================
      // Arith dialect - float ops
      // ========================================================================
      .def("create_arith_addf",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::AddFOp>(lhs, rhs).getResult();
           })
      .def("create_arith_subf",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::SubFOp>(lhs, rhs).getResult();
           })
      .def("create_arith_mulf",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::MulFOp>(lhs, rhs).getResult();
           })
      .def("create_arith_divf",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::DivFOp>(lhs, rhs).getResult();
           })
      .def("create_arith_negf",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<arith::NegFOp>(operand).getResult();
           })
      .def("create_arith_minimumf",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::MinimumFOp>(lhs, rhs).getResult();
           })
      .def("create_arith_maximumf",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::MaximumFOp>(lhs, rhs).getResult();
           })

      // ========================================================================
      // Arith dialect - type conversion
      // ========================================================================
      .def(
          "create_arith_extf",
          [](TritonOpBuilder &self, Value input, Type targetType) -> Value {
            return self.create<arith::ExtFOp>(targetType, input).getResult();
          },
          py::arg("input"), py::arg("target_type"))
      .def("create_arith_truncf",
           [](TritonOpBuilder &self, Value input, Type targetType) -> Value {
             return self.create<arith::TruncFOp>(targetType, input).getResult();
           })
      .def("create_arith_extsi",
           [](TritonOpBuilder &self, Value input, Type targetType) -> Value {
             return self.create<arith::ExtSIOp>(targetType, input).getResult();
           })
      .def("create_arith_trunci",
           [](TritonOpBuilder &self, Value input, Type targetType) -> Value {
             return self.create<arith::TruncIOp>(targetType, input).getResult();
           })
      .def("create_arith_sitofp",
           [](TritonOpBuilder &self, Value input, Type targetType) -> Value {
             return self.create<arith::SIToFPOp>(targetType, input).getResult();
           })
      .def("create_arith_fptosi",
           [](TritonOpBuilder &self, Value input, Type targetType) -> Value {
             return self.create<arith::FPToSIOp>(targetType, input).getResult();
           })
      .def("create_arith_index_cast",
           [](TritonOpBuilder &self, Value input, Type targetType) -> Value {
             return self.create<arith::IndexCastOp>(targetType, input)
                 .getResult();
           })

      // ========================================================================
      // Arith dialect - comparison
      // ========================================================================
      .def(
          "create_arith_cmpi",
          [](TritonOpBuilder &self, const std::string &predicate, Value lhs,
             Value rhs) -> Value {
            arith::CmpIPredicate pred;
            if (predicate == "eq")
              pred = arith::CmpIPredicate::eq;
            else if (predicate == "ne")
              pred = arith::CmpIPredicate::ne;
            else if (predicate == "slt")
              pred = arith::CmpIPredicate::slt;
            else if (predicate == "sle")
              pred = arith::CmpIPredicate::sle;
            else if (predicate == "sgt")
              pred = arith::CmpIPredicate::sgt;
            else if (predicate == "sge")
              pred = arith::CmpIPredicate::sge;
            else
              throw std::runtime_error("Unknown cmpi predicate: " + predicate);
            return self.create<arith::CmpIOp>(pred, lhs, rhs).getResult();
          },
          py::arg("predicate"), py::arg("lhs"), py::arg("rhs"))
      .def("create_arith_cmpf",
           [](TritonOpBuilder &self, const std::string &predicate, Value lhs,
              Value rhs) -> Value {
             arith::CmpFPredicate pred;
             if (predicate == "oeq")
               pred = arith::CmpFPredicate::OEQ;
             else if (predicate == "one")
               pred = arith::CmpFPredicate::ONE;
             else if (predicate == "olt")
               pred = arith::CmpFPredicate::OLT;
             else if (predicate == "ole")
               pred = arith::CmpFPredicate::OLE;
             else if (predicate == "ogt")
               pred = arith::CmpFPredicate::OGT;
             else if (predicate == "oge")
               pred = arith::CmpFPredicate::OGE;
             else
               throw std::runtime_error("Unknown cmpf predicate: " + predicate);
             return self.create<arith::CmpFOp>(pred, lhs, rhs).getResult();
           })
      .def("create_arith_select",
           [](TritonOpBuilder &self, Value condition, Value trueValue,
              Value falseValue) -> Value {
             return self
                 .create<arith::SelectOp>(condition, trueValue, falseValue)
                 .getResult();
           })

      // ========================================================================
      // Arith dialect - bitwise ops
      // ========================================================================
      .def("create_arith_andi",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::AndIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_ori",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::OrIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_xori",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::XOrIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_shli",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::ShLIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_shrsi",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::ShRSIOp>(lhs, rhs).getResult();
           })

      // ========================================================================
      // Math dialect
      // ========================================================================
      .def(
          "create_math_fma",
          [](TritonOpBuilder &self, Value a, Value b, Value c) -> Value {
            return self.create<math::FmaOp>(a, b, c).getResult();
          },
          py::arg("a"), py::arg("b"), py::arg("c"))
      .def("create_math_sqrt",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<math::SqrtOp>(operand).getResult();
           })
      .def("create_math_rsqrt",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<math::RsqrtOp>(operand).getResult();
           })
      .def("create_math_exp",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<math::ExpOp>(operand).getResult();
           })
      .def("create_math_exp2",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<math::Exp2Op>(operand).getResult();
           })
      .def("create_math_log",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<math::LogOp>(operand).getResult();
           })
      .def("create_math_log2",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<math::Log2Op>(operand).getResult();
           })
      .def("create_math_absf",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<math::AbsFOp>(operand).getResult();
           })
      .def("create_math_absi",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<math::AbsIOp>(operand).getResult();
           })

      // ========================================================================
      // Vector dialect - basics
      // ========================================================================
      .def(
          "create_vector_splat",
          [](TritonOpBuilder &self, Value input, Type vectorType) -> Value {
            // splat = broadcast a scalar to all lanes (SplatOp is deprecated;
            // use BroadcastOp)
            return self.create<vector::BroadcastOp>(vectorType, input)
                .getResult();
          },
          py::arg("input"), py::arg("vector_type"))
      .def(
          "create_vector_broadcast",
          [](TritonOpBuilder &self, Value source, Type destType) -> Value {
            return self.create<vector::BroadcastOp>(destType, source)
                .getResult();
          },
          py::arg("source"), py::arg("dest_type"))
      .def("create_vector_shape_cast",
           [](TritonOpBuilder &self, Value source, Type resultType) -> Value {
             return self.create<vector::ShapeCastOp>(resultType, source)
                 .getResult();
           })
      .def(
          "create_vector_step",
          [](TritonOpBuilder &self, Type resultType) -> Value {
            // vector.step : vector<Nxindex> → [0, 1, .., N-1] (iota).
            // Used for index-tracking reductions (argmax/argmin).
            return self.create<vector::StepOp>(cast<VectorType>(resultType));
          },
          py::arg("result_type"))

      // ========================================================================
      // Vector dialect - Load/Store
      // ========================================================================
      .def(
          "create_vector_load",
          [](TritonOpBuilder &self, Type vectorType, Value base,
             std::vector<Value> indices) -> Value {
            return self.create<vector::LoadOp>(vectorType, base, indices)
                .getResult();
          },
          py::arg("vector_type"), py::arg("base"), py::arg("indices"))
      .def("create_vector_store",
           [](TritonOpBuilder &self, Value valueToStore, Value base,
              std::vector<Value> indices) {
             self.create<vector::StoreOp>(valueToStore, base, indices);
           })
      .def(
          "create_vector_transfer_read",
          [](TritonOpBuilder &self, Type vectorType, Value source,
             std::vector<Value> indices, Value padding,
             std::optional<std::vector<bool>> inBounds =
                 std::nullopt) -> Value {
            SmallVector<bool> inBoundsVec;
            if (inBounds.has_value()) {
              inBoundsVec =
                  SmallVector<bool>(inBounds->begin(), inBounds->end());
            } else {
              // all in_bounds=true by default
              auto vecTy = cast<VectorType>(vectorType);
              inBoundsVec = SmallVector<bool>(vecTy.getRank(), true);
            }
            return self
                .create<vector::TransferReadOp>(cast<VectorType>(vectorType),
                                                source, indices, padding,
                                                ArrayRef<bool>(inBoundsVec))
                .getResult();
          },
          py::arg("vector_type"), py::arg("source"), py::arg("indices"),
          py::arg("padding"), py::arg("in_bounds") = py::none())
      .def(
          "create_vector_transfer_write",
          [](TritonOpBuilder &self, Value vector, Value dest,
             std::vector<Value> indices,
             std::optional<std::vector<bool>> inBounds = std::nullopt) {
            SmallVector<bool> inBoundsVec;
            if (inBounds.has_value()) {
              inBoundsVec =
                  SmallVector<bool>(inBounds->begin(), inBounds->end());
            } else {
              auto vecTy = cast<VectorType>(vector.getType());
              inBoundsVec = SmallVector<bool>(vecTy.getRank(), true);
            }
            self.create<vector::TransferWriteOp>(vector, dest, indices,
                                                 ArrayRef<bool>(inBoundsVec));
          },
          py::arg("vector"), py::arg("dest"), py::arg("indices"),
          py::arg("in_bounds") = py::none())

      // ========================================================================
      // Vector dialect - Reduction
      // ========================================================================
      .def(
          "create_vector_reduction",
          [](TritonOpBuilder &self, const std::string &kind,
             Value source) -> Value {
            vector::CombiningKind combKind;
            if (kind == "add")
              combKind = vector::CombiningKind::ADD;
            else if (kind == "mul")
              combKind = vector::CombiningKind::MUL;
            else if (kind == "minf")
              combKind = vector::CombiningKind::MINIMUMF;
            else if (kind == "maxf")
              combKind = vector::CombiningKind::MAXIMUMF;
            else
              throw std::runtime_error("Unknown reduction kind: " + kind);
            return self.create<vector::ReductionOp>(combKind, source)
                .getResult();
          },
          py::arg("kind"), py::arg("source"))

      // ========================================================================
      // SCF dialect - control flow
      // ========================================================================
      .def(
          "create_scf_for",
          [](TritonOpBuilder &self, Value lb, Value ub, Value step,
             std::vector<Value> iterArgs,
             py::function bodyBuilder) -> std::vector<Value> {
            // create scf.for
            ValueRange iterArgsRange(iterArgs);
            auto forOp = self.create<scf::ForOp>(lb, ub, step, iterArgsRange);

            // set the body builder's insertion point
            OpBuilder::InsertionGuard guard(self.getBuilder());
            Block *bodyBlock = forOp.getBody();

            // ForOp::build with empty initArgs and no bodyBuilder
            // auto-inserts an empty scf.yield via ensureTerminator; if not
            // erased, our later create<YieldOp> would produce two yields
            // (triggering 'scf.yield must be the last operation'). This
            // function owns the yield, so erase the auto-inserted terminator
            // first.
            if (!bodyBlock->empty() &&
                bodyBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
              bodyBlock->back().erase();

            self.getBuilder().setInsertionPointToEnd(bodyBlock);

            // invoke the Python-provided body builder
            // the Python side must return the list of values to yield
            Value iv = forOp.getInductionVar();
            std::vector<Value> regionIterArgs(
                bodyBlock->getArguments().begin() + 1,
                bodyBlock->getArguments().end());

            py::object yieldValsObj =
                bodyBuilder(std::ref(self), iv, regionIterArgs);
            std::vector<Value> yieldVals =
                yieldValsObj.cast<std::vector<Value>>();

            // create scf.yield (the sole yield source in this function)
            self.getBuilder().setInsertionPointToEnd(bodyBlock);
            self.create<scf::YieldOp>(yieldVals);

            // return the for's results
            return std::vector<Value>(forOp.getResults().begin(),
                                      forOp.getResults().end());
          },
          py::arg("lb"), py::arg("ub"), py::arg("step"), py::arg("iter_args"),
          py::arg("body_builder"))
      .def("create_scf_yield",
           [](TritonOpBuilder &self, std::vector<Value> results) {
             self.create<scf::YieldOp>(results);
           })

      // ========================================================================
      // MemRef dialect
      // ========================================================================
      .def(
          "create_memref_alloc",
          [](TritonOpBuilder &self, Type memrefType,
             std::optional<std::vector<Value>> dynamicSizes = std::nullopt,
             std::optional<int64_t> alignment = std::nullopt) -> Value {
            SmallVector<Value> dynSizes;
            if (dynamicSizes.has_value()) {
              dynSizes = SmallVector<Value>(dynamicSizes->begin(),
                                            dynamicSizes->end());
            }
            // Must cast<MemRefType>: the parameter is a Type; passing it
            // directly to create<AllocOp> would match the generic
            // build(TypeRange, ValueRange, attrs) overload, which does not set
            // operandSegmentSizes → 'operand count does not match total size in
            // operandSegmentSizes'. After the cast it hits
            // AllocOp::build(MemRefType, ValueRange dynamicSizes, ...).
            auto op = self.create<memref::AllocOp>(cast<MemRefType>(memrefType),
                                                   dynSizes);
            if (alignment.has_value()) {
              op->setAttr("alignment", self.getBuilder().getI64IntegerAttr(
                                           alignment.value()));
            }
            return op.getResult();
          },
          py::arg("memref_type"), py::arg("dynamic_sizes") = py::none(),
          py::arg("alignment") = py::none())
      .def("create_memref_load",
           [](TritonOpBuilder &self, Value memref,
              std::vector<Value> indices) -> Value {
             return self.create<memref::LoadOp>(memref, indices).getResult();
           })
      .def("create_memref_store",
           [](TritonOpBuilder &self, Value value, Value memref,
              std::vector<Value> indices) {
             self.create<memref::StoreOp>(value, memref, indices);
           })
      .def(
          "create_memref_reinterpret_cast",
          [](TritonOpBuilder &self, Type resultType, Value source,
             std::vector<Value> offsets, std::vector<Value> sizes,
             std::vector<Value> strides) -> Value {
            // ReinterpretCastOp needs OpFoldResult (Value or static int).
            // When all-dynamic, the static arrays use ShapedType::kDynamic
            // placeholders.
            auto toOFR = [](const std::vector<Value> &vals)
                -> SmallVector<OpFoldResult> {
              SmallVector<OpFoldResult> result;
              for (Value v : vals)
                result.push_back(v);
              return result;
            };
            return self
                .create<memref::ReinterpretCastOp>(cast<MemRefType>(resultType),
                                                   source, toOFR(offsets)[0],
                                                   toOFR(sizes), toOFR(strides))
                .getResult();
          },
          py::arg("result_type"), py::arg("source"), py::arg("offsets"),
          py::arg("sizes"), py::arg("strides"))
      .def(
          "create_memref_reinterpret_cast_mixed",
          [](TritonOpBuilder &self, Type resultType, Value source,
             py::list offsets, py::list sizes, py::list strides) -> Value {
            // Mixed static/dynamic version: each py::list element is an
            // int (static, goes into the static_* arrays) or a Value
            // (dynamic, static slot filled with kDynamic). For result types
            // with static dims (e.g. memref<?x64xf16>), static_sizes must
            // match the result type dim-by-dim, otherwise 'expected result
            // type with size = dynamic instead of N'.
            auto &b = self.getBuilder();
            auto toOFR = [&b](py::list items) -> SmallVector<OpFoldResult> {
              SmallVector<OpFoldResult> result;
              for (py::handle it : items) {
                if (py::isinstance<py::int_>(it))
                  result.push_back(b.getIndexAttr(it.cast<int64_t>()));
                else
                  result.push_back(it.cast<Value>());
              }
              return result;
            };
            return self
                .create<memref::ReinterpretCastOp>(cast<MemRefType>(resultType),
                                                   source, toOFR(offsets)[0],
                                                   toOFR(sizes), toOFR(strides))
                .getResult();
          },
          py::arg("result_type"), py::arg("source"), py::arg("offsets"),
          py::arg("sizes"), py::arg("strides"))
      .def(
          "create_memref_collapse_shape",
          [](TritonOpBuilder &self, Value src,
             const std::vector<std::vector<int64_t>> &reassociation) -> Value {
            SmallVector<ReassociationIndices> reassoc;
            for (const auto &group : reassociation) {
              reassoc.push_back(
                  ReassociationIndices(group.begin(), group.end()));
            }
            return self.create<memref::CollapseShapeOp>(src, reassoc)
                .getResult();
          },
          py::arg("src"), py::arg("reassociation"))
      .def("create_memref_cast",
           [](TritonOpBuilder &self, Type targetType, Value source) -> Value {
             return self.create<memref::CastOp>(targetType, source).getResult();
           })

      // ========================================================================
      // Tensor dialect (for bufferization path)
      // ========================================================================
      .def(
          "create_tensor_empty",
          [](TritonOpBuilder &self, Type tensorType,
             std::optional<std::vector<Value>> dynamicSizes =
                 std::nullopt) -> Value {
            SmallVector<Value> dynSizes;
            if (dynamicSizes.has_value()) {
              dynSizes = SmallVector<Value>(dynamicSizes->begin(),
                                            dynamicSizes->end());
            }
            return self.create<tensor::EmptyOp>(tensorType, dynSizes)
                .getResult();
          },
          py::arg("tensor_type"), py::arg("dynamic_sizes") = py::none())
      .def("create_tensor_insert_slice",
           [](TritonOpBuilder &self, Value source, Value dest,
              std::vector<Value> offsets, std::vector<Value> sizes,
              std::vector<Value> strides) -> Value {
             return self
                 .create<tensor::InsertSliceOp>(source, dest, offsets, sizes,
                                                strides)
                 .getResult();
           })
      .def("create_tensor_insert_slice_mixed",
           [](TritonOpBuilder &self, Value source, Value dest, py::list offsets,
              py::list sizes, py::list strides) -> Value {
             // Mixed static/dynamic: int→static OFR, Value→dynamic. When
             // the source has static dims (e.g. tensor<?x64xf16>),
             // insert_slice static_sizes must match the source dim-by-dim,
             // otherwise 'expected type to be tensor<?x?xf16>'
             // rank/size mismatch。
             auto &b = self.getBuilder();
             auto toOFR = [&b](py::list items) -> SmallVector<OpFoldResult> {
               SmallVector<OpFoldResult> result;
               for (py::handle it : items) {
                 if (py::isinstance<py::int_>(it))
                   result.push_back(b.getIndexAttr(it.cast<int64_t>()));
                 else
                   result.push_back(it.cast<Value>());
               }
               return result;
             };
             return self
                 .create<tensor::InsertSliceOp>(source, dest, toOFR(offsets),
                                                toOFR(sizes), toOFR(strides))
                 .getResult();
           })
      .def("create_tensor_collapse_shape",
           [](TritonOpBuilder &self, Value src,
              const std::vector<std::vector<int64_t>> &reassociation) -> Value {
             SmallVector<ReassociationIndices> reassoc;
             for (const auto &group : reassociation) {
               reassoc.push_back(
                   ReassociationIndices(group.begin(), group.end()));
             }
             return self.create<tensor::CollapseShapeOp>(src, reassoc)
                 .getResult();
           })

      // ========================================================================
      // Linalg dialect
      // ========================================================================
      .def("create_linalg_fill",
           [](TritonOpBuilder &self, Value value, Value output) -> Value {
             return self.create<linalg::FillOp>(value, output).getResult(0);
           })
      .def(
          "create_linalg_pack",
          [](TritonOpBuilder &self, Value source, Value dest,
             Value paddingValue, const std::vector<int64_t> &innerTiles,
             const std::vector<int64_t> &outerDimsPerm,
             const std::vector<int64_t> &innerDimsPos) -> Value {
            // Per the MLIR signature: create(builder, loc, type, source, dest,
            // padding_value,
            //                        outer_dims_perm, inner_dims_pos,
            //                        inner_tiles, static_inner_tiles)
            auto &builder = self.getBuilder();
            SmallVector<int64_t> staticInnerTiles(innerTiles.begin(),
                                                  innerTiles.end());
            SmallVector<Value> innerTilesValues; // empty dynamic tiles

            return self
                .create<linalg::PackOp>(
                    dest.getType(), source, dest, paddingValue, outerDimsPerm,
                    innerDimsPos, innerTilesValues, staticInnerTiles)
                .getResult();
          },
          py::arg("source"), py::arg("dest"), py::arg("padding_value"),
          py::arg("inner_tiles"), py::arg("outer_dims_perm"),
          py::arg("inner_dims_pos"))

      // ========================================================================
      // Bufferization dialect
      // ========================================================================
      .def("create_bufferization_to_tensor",
           [](TritonOpBuilder &self, Value memref, Type tensorType) -> Value {
             auto op =
                 self.create<bufferization::ToTensorOp>(tensorType, memref);
             // Match text path "bufferization.to_tensor ... restrict"
             op->setAttr("restrict", self.getBuilder().getUnitAttr());
             return op.getResult();
           })
      .def("create_bufferization_to_memref",
           [](TritonOpBuilder &self, Value tensor, Type memrefType) -> Value {
             // ToMemrefOp was renamed ToBufferOp (newer MLIR); the memref
             // result type must be passed explicitly
             return self.create<bufferization::ToBufferOp>(memrefType, tensor)
                 .getResult();
           })

      // ========================================================================
      // Generic (unregistered) op builder — for ops in dialects that are never
      // registered/loaded (e.g. vector_ext.*), which parse only in generic form
      // under the context's allowUnregisteredDialects flag (see load_dialects).
      // int_attrs covers the only attr shape smt_rvisa's generic ops need today
      // (vector_ext.cross_batch_matmul's m/n/k, vector_ext.group_interleave's
      // groupLen — all i64 integer attrs).
      // ========================================================================
      .def(
          "create_generic_op",
          [](TritonOpBuilder &self, const std::string &opName,
             std::vector<Value> operands,
             std::map<std::string, int64_t> intAttrs,
             std::vector<Type> resultTypes) -> std::vector<Value> {
            auto &builder = self.getBuilder();
            mlir::OperationState state(self.getLastLoc(), opName);
            state.addOperands(operands);
            state.addTypes(resultTypes);
            for (auto &[name, val] : intAttrs)
              state.addAttribute(name, builder.getI64IntegerAttr(val));
            mlir::Operation *op = builder.create(state);
            return std::vector<Value>(op->getResults().begin(),
                                      op->getResults().end());
          },
          py::arg("op_name"), py::arg("operands"), py::arg("int_attrs"),
          py::arg("result_types"),
          "Create an unregistered/generic-form op (e.g. vector_ext.*)")

      // ========================================================================
      // Generic op builder with text-parsed attributes — for LLVM-dialect
      // llvm-direct kernels (call_intrinsic / mlir.constant / extractvalue /
      // getelementptr). Each attr value is an MLIR attribute in text form,
      // parsed via parseAttribute, so string / dense / array / type attrs are
      // all covered by one method (create_generic_op only handles i64 attrs).
      // ========================================================================
      .def(
          "create_op_textattr",
          [](TritonOpBuilder &self, const std::string &opName,
             std::vector<Value> operands,
             std::map<std::string, std::string> textAttrs,
             std::vector<Type> resultTypes) -> std::vector<Value> {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            mlir::OperationState state(self.getLastLoc(), opName);
            state.addOperands(operands);
            state.addTypes(resultTypes);
            for (auto &[name, txt] : textAttrs) {
              mlir::Attribute attr = mlir::parseAttribute(txt, ctx);
              if (!attr)
                throw std::runtime_error(
                    "create_op_textattr: failed to parse attribute '" + name +
                    "' = " + txt);
              state.addAttribute(name, attr);
            }
            mlir::Operation *op = builder.create(state);
            return std::vector<Value>(op->getResults().begin(),
                                      op->getResults().end());
          },
          py::arg("op_name"), py::arg("operands"), py::arg("text_attrs"),
          py::arg("result_types"),
          "Create an op with text-parsed attributes (LLVM dialect "
          "llvm-direct)");
}

void init_triton_spacemit(py::module &&m) {
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::xsmt::XSMTDialect, mlir::xsmt_async::XSMTAsyncDialect,
                    tensor::TensorDialect, mlir::triton::proton::ProtonDialect,
                    mlir::triton::tle::TleDialect,
                    // Payload dialects for tle.dsl_region's real region body,
                    // so the raw-kernel MLIR text parses in-process (custom ops
                    // like vector_ext.matmul stay generic via the context's
                    // allow-unregistered flag).
                    mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect, mlir::vector::VectorDialect,
                    mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                    mlir::math::MathDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::ptr::PtrDialect, mlir::LLVM::LLVMDialect>();
    // Registering func dialect above makes TTIR's InlinerPass query func's
    // DialectInlinerInterface; that interface lives in a separate extension
    // that must be registered explicitly, or the inliner aborts with
    // "interface promised by dialect 'func' but never implemented".
    mlir::func::registerInlinerExtension(registry);
    context.appendDialectRegistry(registry);
    context.allowUnregisteredDialects();
    context.loadAllAvailableDialects();
  });

  init_triton_xsmt_ir(m.def_submodule("xsmt_ir"));
  init_triton_spacemit_tle_ir(m.def_submodule("tle_ir"));
  init_triton_smt_rvisa_ir(m.def_submodule("smt_rvisa_ir"));
}
