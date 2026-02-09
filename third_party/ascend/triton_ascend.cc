#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "incubated/Conversion/DiscreteMaskAccessConversion/Passes.h"
#include "incubated/Conversion/TritonToAnnotation/Passes.h"
#include "incubated/Conversion/TritonToLinalgIncubated/Passes.h"
#include "incubated/Conversion/TritonToStructuredIncubated/Passes.h"
#include "incubated/Conversion/TritonToUnstructureIncubated/Passes.h"
#include "ascend/include/TritonToHFusion/Passes.h"
#include "ascend/include/TritonToHIVM/Passes.h"
#include "ascend/include/TritonToLLVM/Passes.h"
#include "npu/Dialect/TritonAscend/IR/TritonAscendDialect.h"

#include "ir.h" // TritonOpBuilder
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ir;
using namespace mlir;

void init_triton_ascend_ir(py::module &&m) {
  auto *builder_cls = ir::getBuilderClass();
  builder_cls
      ->def("create_extract_scalar",
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
      .def("create_extract_slice",
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
      .def("create_insert_slice",
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
      .def("create_custom_op_for_inter_core_sync",
           [](TritonOpBuilder &self, std::string &op_name,
              std::string &mode_or_sender, int id) -> void {
             auto args = self.getBuilder().getArrayAttr(
                 {self.getBuilder().getStringAttr(mode_or_sender),
                  self.getBuilder().getI32IntegerAttr(id)});
             self.create<triton::ascend::CustomOp>(op_name, args, ValueRange());
           })
      .def("create_index_select_simd",
           [](TritonOpBuilder &self, Value &src, Value &index, int32_t dim,
              std::vector<Value> &srcShape, std::vector<Value> &srcOffset,
              std::vector<int32_t> &readShape,
              std::vector<int32_t> &returnShape) -> Value {
             auto &builder = self.getBuilder();
             auto loc = self.getLastLoc();

             // Get element type from source pointer
             Type elemType;
             if (auto ptrTy = dyn_cast<triton::PointerType>(src.getType())) {
               elemType = ptrTy.getPointeeType();
             } else {
               llvm::report_fatal_error(
                   "index_select_simd: src must be pointer type");
             }

             // Create return tensor type
             llvm::SmallVector<int64_t> retShape;
             for (const auto &s : returnShape) {
               retShape.push_back(s);
             }
             auto retTensorType = RankedTensorType::get(retShape, elemType);

             // Convert srcShape and srcOffset values to index type if needed
             llvm::SmallVector<Value> srcShapeIndex;
             for (auto val : srcShape) {
               if (!val.getType().isIndex()) {
                 val = self.create<arith::IndexCastOp>(builder.getIndexType(),
                                                       val);
               }
               srcShapeIndex.push_back(val);
             }

             llvm::SmallVector<Value> srcOffsetIndex;
             for (auto val : srcOffset) {
               if (!val.getType().isIndex()) {
                 val = self.create<arith::IndexCastOp>(builder.getIndexType(),
                                                       val);
               }
               srcOffsetIndex.push_back(val);
             }

             // Create attributes
             auto dimAttr = builder.getI32IntegerAttr(dim);
             auto readShapeAttr = builder.getDenseI32ArrayAttr(readShape);

             // Create the IndexSelectSimdOp
             // Parameter order must match TritonOps.td definition:
             // src, index, dim, src_shape, src_offset, read_shape
             auto indexSelectSimdOp =
                 builder.create<triton::ascend::IndexSelectSimdOp>(
                     loc,
                     retTensorType,  // result type
                     src,            // src pointer
                     index,          // index tensor
                     dimAttr,        // dim attribute
                     srcShapeIndex,  // src_shape (variadic, index type)
                     srcOffsetIndex, // src_offset (variadic, index type)
                     readShapeAttr   // read_shape attribute
                 );

             return indexSelectSimdOp.getResult();
           })
      .def("create_embedding_gather",
           [](TritonOpBuilder &self, Value &src, Value &idx,
              const int64_t bound, const int64_t blksiz,
              std::vector<Value> &offsets,
              std::vector<Value> &numels) -> Value {
             auto elemTy = cast<PointerType>(src.getType()).getPointeeType();
             auto idxTy = cast<RankedTensorType>(idx.getType());
             auto idxShape = idxTy.getShape();
             std::vector<int64_t> retShape(idxShape.begin(), idxShape.end());
             retShape.push_back(blksiz);
             auto resType = RankedTensorType::get(retShape, elemTy);
             auto idxBitWidth = idxTy.getElementType().getIntOrFloatBitWidth();
             auto bound_val =
                 self.create<arith::ConstantIntOp>(bound, idxBitWidth);
             auto blksiz_val =
                 self.create<arith::ConstantIntOp>(blksiz, idxBitWidth);

             return self.create<triton::ascend::EmbeddingGatherOp>(
                 resType, src, idx, bound_val, blksiz_val, offsets, numels);
           })
      .def("create_index_put",
           [](TritonOpBuilder &self, Value &ptr, Value &index, Value &value,
              const int32_t dim, const int64_t indexBoundary,
              std::vector<Value> &endOffset, std::vector<Value> &startOffset,
              std::vector<Value> &dstStride) -> void {
             // dim need to be i32 type
             auto dimI32Ty = self.getBuilder().getI32Type();
             auto dim_val = self.create<arith::ConstantIntOp>(dim, dimI32Ty);
             // indexBoundary need to be i64 type
             auto BoundI64Ty = self.getBuilder().getI64Type();
             auto bound_val =
                 self.create<arith::ConstantIntOp>(indexBoundary, BoundI64Ty);

             self.create<triton::ascend::IndexPutOp>(ptr, index, value, dim_val,
                                                     bound_val, endOffset,
                                                     startOffset, dstStride);
           })
      .def("create_gather_out_to_ub",
           [](TritonOpBuilder &self, Value &src, Value &index,
              const int64_t indexBoundary, const int32_t dim,
              std::vector<Value> &srcStride, std::vector<Value> &endOffset,
              std::vector<Value> &startOffset,
              std::optional<Value> &other) -> Value {
             auto elemTy = cast<PointerType>(src.getType()).getPointeeType();
             auto idxTy = cast<RankedTensorType>(index.getType());
             auto idxShape = idxTy.getShape();
             std::vector<int64_t> retShape(idxShape.begin(), idxShape.end());
             auto resType = RankedTensorType::get(retShape, elemTy);

             // indexBoundary need to be i64 type
             auto BoundI64Ty = self.getBuilder().getI64Type();
             auto bound_val =
                 self.create<arith::ConstantIntOp>(indexBoundary, BoundI64Ty);
             // dim need to be i32 type
             auto dimI32Ty = self.getBuilder().getI32Type();
             auto dim_val = self.create<arith::ConstantIntOp>(dim, dimI32Ty);
             return self.create<triton::ascend::GatherOutToUbOp>(
                 resType, src, index, bound_val, dim_val, srcStride, endOffset,
                 startOffset, other.value_or(Value()));
           })
      .def("create_scatter_ub_to_out",
           [](TritonOpBuilder &self, Value &ptr, Value &value, Value &index,
              const int64_t indexBoundary, const int32_t dim,
              std::vector<Value> &dstStride, std::vector<Value> &endOffset,
              std::vector<Value> &startOffset) -> void {
             auto idxTy = cast<RankedTensorType>(index.getType());

             // indexBoundary need to be i64 type
             auto BoundI64Ty = self.getBuilder().getI64Type();
             auto bound_val =
                 self.create<arith::ConstantIntOp>(indexBoundary, BoundI64Ty);
             // dim need to be i32 type
             auto dimI32Ty = self.getBuilder().getI32Type();
             auto dim_val = self.create<arith::ConstantIntOp>(dim, dimI32Ty);

             self.create<triton::ascend::ScatterUbToOutOp>(
                 ptr, value, index, bound_val, dim_val, dstStride, endOffset,
                 startOffset);
           })
      // Add sort
      .def("create_sort",
           [](TritonOpBuilder &self, Value src, int64_t dim,
              bool descending) -> Value {
             auto &builder = self.getBuilder();
             auto loc = self.getLastLoc();

             auto dimAttr = builder.getI64IntegerAttr(dim);
             auto descendingAttr = builder.getBoolAttr(descending);

             auto op = builder.create<triton::ascend::SortOp>(loc, src, dimAttr,
                                                              descendingAttr);

             return op->getResult(0);
           })
      // Add flip
      .def("create_flip",
           [](TritonOpBuilder &self, Value src, int64_t dim) -> Value {
             auto &builder = self.getBuilder();
             auto loc = self.getLastLoc();

             auto dimAttr = builder.getI64IntegerAttr(dim);

             auto op =
                 builder.create<triton::ascend::FlipOp>(loc, src, dimAttr);

             return op->getResult(0);
           })
      .def("create_tanh",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::TanhOp>(val);
           })
      // Add an annotation
      .def("create_annotation",
           [](TritonOpBuilder &self, Value &ptr, const std::string &attrKey,
              Attribute &attrVal) {
             auto annotationOp = self.create<triton::ascend::AnnotationOp>(ptr);
             annotationOp->setAttr(self.getBuilder().getStringAttr(attrKey),
                                   attrVal);
           });
}

void init_triton_ascend_passes_ttir(py::module &&m) {
  m.def("add_triton_to_structure_incubated", [](mlir::PassManager &pm,
                                      bool enableMaskFallbackConversion,
                                      bool optimizeDynamicOffset,
                                      bool compileOn91095) {
    pm.addPass(mlir::triton::createTritonToStructuredIncubatedPass(
        enableMaskFallbackConversion, optimizeDynamicOffset, compileOn91095));
  });

  m.def("add_triton_to_annotation", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToAnnotationPass());
  });

  m.def("add_triton_to_linalg_incubated",
        [](mlir::PassManager &pm, bool globalKernel, bool namedOps,
           bool enableNd2nzOnVector, bool enableSelectAnalysis,
           bool compileOn91095) {
          pm.addPass(mlir::triton::Incubated::createTritonToLinalgIncubatedPass(
              globalKernel, namedOps, enableNd2nzOnVector, enableSelectAnalysis,
              compileOn91095));
        });

  m.def("add_triton_to_unstructure_incubated",
        [](mlir::PassManager &pm, bool compileOn91095, bool forceSimtTemplate) {
          TritonToUnstructureIncubatedOptions opts;
          opts.compileOn91095 = compileOn91095;
          opts.forceSimtTemplate = forceSimtTemplate;
          pm.addPass(mlir::triton::createTritonToUnstructureIncubatedPass(opts));
        });

  m.def("add_triton_to_hfusion", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToHFusionPass());
  });

  m.def("add_discrete_mask_access_conversion",
        [](mlir::PassManager &pm, bool compileOn91095, bool forceSimtTemplate) {
          DiscreteMaskAccessConversionOptions opts;
          opts.compileOn91095 = compileOn91095;
          opts.forceSimtTemplate = forceSimtTemplate;
          pm.addPass(
              mlir::triton::createDiscreteMaskAccessConversionPass(opts));
        });

  m.def("add_triton_to_hivm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToHIVMPass());
  });

  m.def("add_triton_to_llvm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToLLVMPass());
  });

  m.def("add_bubble_up_operation", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createBubbleUpOperationPass());
  });
}

// Forward declaration for ascend_ir bindings (defined in ascend_ir.cc)
void init_ascend_ir(py::module &&m);

void init_triton_ascend(py::module &&m) {
  auto passes = m.def_submodule("passes");
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    context.allowUnregisteredDialects();
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::ascend::TritonAscendDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  init_triton_ascend_passes_ttir(passes.def_submodule("ttir"));
  init_triton_ascend_ir(m.def_submodule("ascend_ir"));

  // Initialize ascend IR bindings (ascendnpu_ir_builder, scope/hivm dialects)
  init_ascend_ir(m.def_submodule("ir"));
}
