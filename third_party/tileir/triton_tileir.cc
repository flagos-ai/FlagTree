#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/TargetSelect.h"

#include "Transform/Passes.h"
#include "TritonToTileIR/Passes.h"
#include "Utils/Utils.h"
#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"
#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"
#include "ir.h"
#include "passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace mlir;
using namespace triton;

namespace {

Value castToCudaTileType(TritonOpBuilder &self, Value value) {
  auto *ctx = self.getBuilder().getContext();
  Type elemTy = getElementTypeOrSelf(value);
  SmallVector<int64_t> shape;
  if (auto rankedTy = dyn_cast<RankedTensorType>(value.getType())) {
    shape = llvm::to_vector(rankedTy.getShape());
  }
  return self
      .create<UnrealizedConversionCastOp>(
          cuda_tile::TileType::get(ctx, shape, elemTy), value)
      .getResult(0);
}

SmallVector<Value> castToTileVec(TritonOpBuilder &self,
                                 std::vector<Value> &values) {
  SmallVector<Value> dst;
  for (auto v : values)
    dst.push_back(castToCudaTileType(self, v));
  return dst;
}

Value castFromCudaTileAndExtract(TritonOpBuilder &self, Type resultType,
                                 Value value) {
  auto cTileTy = cast<cuda_tile::TileType>(resultType);
  auto resTy =
      RankedTensorType::get(cTileTy.getShape(), cTileTy.getElementType());
  Value castedValue =
      self.create<UnrealizedConversionCastOp>(resTy, value).getResult(0);
  if (resTy.getRank() == 0) {
    castedValue = self.create<UnrealizedConversionCastOp>(
                          resTy.getElementType(), castedValue)
                      .getResult(0);
  }
  return castedValue;
}

cuda_tile::MemoryOrderingSemanticsAttr
getMemoryOrderingAttr(MLIRContext *ctx, uint32_t semantic) {
  return cuda_tile::MemoryOrderingSemanticsAttr::get(
      ctx, cuda_tile::symbolizeMemoryOrderingSemantics(semantic).value());
}

cuda_tile::MemoryScopeAttr getMemoryScopeAttr(MLIRContext *ctx, int32_t scope) {
  if (scope < 0)
    return cuda_tile::MemoryScopeAttr();
  return cuda_tile::MemoryScopeAttr::get(
      ctx, cuda_tile::symbolizeMemoryScope(scope).value());
}

void init_tileir_tko_ir(py::module &&) {
  auto &builder_cls = *ir::getBuilderClass();
  builder_cls
      .def("get_tileir_mem_token_ty",
           [](TritonOpBuilder &self) -> Type {
             return cuda_tile::TokenType::get(self.getBuilder().getContext());
           })
      .def("get_tensor_view_ty",
           [](TritonOpBuilder &self, Type &elementType,
              std::vector<int64_t> &shape,
              std::vector<int64_t> &stride) -> Type {
             auto ctx = self.getBuilder().getContext();
             return cuda_tile::TensorViewType::get(ctx, elementType, shape,
                                                   stride);
           })
      .def("create_make_tensor_view",
           [](TritonOpBuilder &self, Type &type, Value &ptr,
              std::vector<Value> &shapes,
              std::vector<Value> &strides) -> Value {
             auto tvTy = dyn_cast<cuda_tile::TensorViewType>(type);
             if (!tvTy)
               throw std::runtime_error(
                   "create_make_tensor_view expects tensor_view result type");

             Value cudaTilePtrOperand;
             if (auto tritonPtrTy = dyn_cast<PointerType>(ptr.getType())) {
               Type cudaTilePtrTy =
                   cuda_tile::PointerType::get(tritonPtrTy.getPointeeType());
               cudaTilePtrOperand =
                   self.create<UnrealizedConversionCastOp>(cudaTilePtrTy, ptr)
                       .getResult(0);
             } else if (isa<cuda_tile::PointerType>(ptr.getType())) {
               cudaTilePtrOperand = ptr;
             } else {
               throw std::runtime_error("expect a pointer type");
             }

             return self
                 .create<cuda_tile::MakeTensorViewOp>(
                     type, castToCudaTileType(self, cudaTilePtrOperand),
                     castToTileVec(self, shapes), castToTileVec(self, strides))
                 .getResult();
           })
      .def("create_make_partition_view",
           [](TritonOpBuilder &self, Value &src,
              std::vector<int32_t> &tileShape, std::vector<int32_t> &tileDimMap,
              std::string paddingValue) -> std::tuple<Value, Type> {
             auto tensorViewTy = cast<cuda_tile::TensorViewType>(src.getType());
             auto ctx = self.getBuilder().getContext();
             auto builder = self.getBuilder();
             auto partitionViewTy = cuda_tile::PartitionViewType::get(
                 ctx, builder.getDenseI32ArrayAttr(tileShape), tensorViewTy,
                 tileDimMap,
                 cuda_tile::PaddingValueAttr::get(
                     ctx,
                     cuda_tile::symbolizePaddingValue(paddingValue).value()));
             auto op = self.create<cuda_tile::MakePartitionViewOp>(
                 partitionViewTy, src);
             return std::make_tuple(op.getResult(), op.getResult().getType());
           })
      .def("create_dim",
           [](TritonOpBuilder &self, Value &src, int64_t dim) -> Value {
             auto ctx = self.getBuilder().getContext();
             auto builder = self.getBuilder();
             Type intTy = builder.getIntegerType(32);
             auto scalarTileTy = cuda_tile::TileType::get(ctx, {}, intTy);

             if (auto partTy =
                     dyn_cast<cuda_tile::PartitionViewType>(src.getType())) {
               SmallVector<Type> resTys(partTy.getViewIndexRank(),
                                        scalarTileTy);
               auto shapeOp =
                   self.create<cuda_tile::GetIndexSpaceShapeOp>(resTys, src);
               return castFromCudaTileAndExtract(self, scalarTileTy,
                                                 shapeOp.getResults()[dim]);
             }
             if (auto tensorViewTy =
                     dyn_cast<cuda_tile::TensorViewType>(src.getType())) {
               SmallVector<Type> resTys(tensorViewTy.getShape().size(),
                                        scalarTileTy);
               auto shapeOp =
                   self.create<cuda_tile::GetTensorShapeOp>(resTys, src);
               return castFromCudaTileAndExtract(self, scalarTileTy,
                                                 shapeOp.getResults()[dim]);
             }
             throw std::runtime_error(
                 "src expected to be a partition view or tensor view");
           })
      .def("create_load_view_tko",
           [](TritonOpBuilder &self, Value &view, std::vector<Value> &coords,
              std::optional<Value> &token, uint32_t semantic, int32_t scope,
              [[maybe_unused]] bool hasResultToken, int32_t cost,
              int32_t capability)
               -> std::tuple<Value, Value, std::vector<int64_t>, Type> {
             auto *ctx = self.getBuilder().getContext();
             auto optHint = mlir::triton::utils::cvtNumStagesToOptHintAttr(
                 ctx, capability, cost);
             auto semanticAttr = getMemoryOrderingAttr(ctx, semantic);
             auto scopeAttr = getMemoryScopeAttr(ctx, scope);

             cuda_tile::TileType resultTileTy;
             if (auto partTy =
                     dyn_cast<cuda_tile::PartitionViewType>(view.getType())) {
               resultTileTy =
                   cast<cuda_tile::TileType>(partTy.getViewTileType());
             } else {
               throw std::runtime_error(
                   "expect a cuda_tile.partition_view type");
             }

             auto retOp = self.create<cuda_tile::LoadViewTkoOp>(
                 resultTileTy, cuda_tile::TokenType::get(ctx), semanticAttr,
                 scopeAttr, view, castToTileVec(self, coords),
                 token.value_or(Value()), optHint.value_or(nullptr));

             std::vector<int64_t> shape(resultTileTy.getShape().begin(),
                                        resultTileTy.getShape().end());
             auto rankedTy =
                 RankedTensorType::get(shape, resultTileTy.getElementType());
             auto tensor = self.create<UnrealizedConversionCastOp>(
                                   rankedTy, retOp.getTile())
                               .getResult(0);
             return std::make_tuple(tensor, retOp.getResultToken(), shape,
                                    resultTileTy.getElementType());
           })
      .def("create_store_view_tko",
           [](TritonOpBuilder &self, Value &view, Value &tile,
              std::vector<Value> &coords, std::optional<Value> &token,
              uint32_t semantic, int32_t scope,
              [[maybe_unused]] bool hasResultToken, int32_t cost,
              int32_t capability) -> Value {
             auto *ctx = self.getBuilder().getContext();
             auto optHint = mlir::triton::utils::cvtNumStagesToOptHintAttr(
                 ctx, capability, cost);
             Value castedTile = tile;
             if (!isa<cuda_tile::TileType>(getElementTypeOrSelf(tile)))
               castedTile = castToCudaTileType(self, tile);

             auto retOp = self.create<cuda_tile::StoreViewTkoOp>(
                 cuda_tile::TokenType::get(ctx),
                 getMemoryOrderingAttr(ctx, semantic),
                 getMemoryScopeAttr(ctx, scope), castedTile, view,
                 castToTileVec(self, coords), token.value_or(Value()),
                 optHint.value_or(nullptr));
             return retOp.getResultToken();
           })
      .def("create_mem_token",
           [](TritonOpBuilder &self) -> Value {
             return self.create<cuda_tile::MakeTokenOp>().getResult();
           })
      .def("create_join_mem_tokens",
           [](TritonOpBuilder &self, Value &tokenA, Value &tokenB) -> Value {
             assert(isa<cuda_tile::TokenType>(tokenA.getType()) &&
                    "tokenA must be a cuda_tile::TokenType");
             assert(isa<cuda_tile::TokenType>(tokenB.getType()) &&
                    "tokenB must be a cuda_tile::TokenType");
             return self
                 .create<cuda_tile::JoinTokensOp>(ValueRange{tokenA, tokenB})
                 .getResult();
           });
}

} // namespace

void init_triton_to_cudatile_passes(py::module &&m) {
  using namespace mlir::triton;
  // TODO: it is weird to pass mlir::triton::NVVM here since the conversion is
  // nvidia-specificontext
  m.def("add_triton_to_cudatile",
        [](mlir::PassManager &pm, bool approx, bool ftz, int capability,
           int num_ctas, int simt_num_warps, int occupancy,
           std::optional<int> num_stages) {
          pm.addPass(mlir::triton::createConvertTritonToCudaTilePass(
              approx, ftz, capability, num_ctas, simt_num_warps, occupancy,
              num_stages));
        });
  m.def("add_fma_fusion", [](mlir::PassManager &pm) {
    // Add FMA fusion pass to cuda tile entry operations
    auto &mpm = pm.nest<cuda_tile::ModuleOp>();
    auto &epm = mpm.nest<cuda_tile::EntryOp>();
    epm.addPass(cuda_tile::createFuseFMAPass());
  });
  m.def("add_loop_split", [](mlir::PassManager &pm, int threshold = 1) {
    // Add Loop Split pass to cuda tile entry operations
    auto &mpm = pm.nest<cuda_tile::ModuleOp>();
    auto &epm = mpm.nest<cuda_tile::EntryOp>();
    epm.addPass(cuda_tile::createLoopSplitPass({threshold}));
  });
  m.def("add_lift_tt_cf_to_scf", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createLiftTTCFToSCFPass());
  });
  m.def("add_strip_debuginfo", [](mlir::PassManager &pm) {
    // Strip debug info
    auto &mpm = pm.nest<cuda_tile::ModuleOp>();
    mpm.addPass(mlir::createStripDebugInfoPass());
  });
  m.def("add_synthesize_debug_info_scopes", [](mlir::PassManager &pm) {
    // Synthesize scoped debug info
    auto &mpm = pm.nest<cuda_tile::ModuleOp>();
    mpm.addPass(cuda_tile::createSynthesizeDebugInfoScopesPass());
  });
  m.def("add_rewrite_tensor_pointers_to_ldst", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonRewriteTensorPointer());
  });
  m.def("add_assume_to_tileir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createRewriteAssumeWithCudaTilePass());
  });
  m.def("add_auto_gen_memtoken",
        [](mlir::PassManager &pm, bool enable_autogen_alias_mem_token) {
          pm.addPass(mlir::triton::createAutoGenMemoryTokenPass(
              enable_autogen_alias_mem_token));
        });
}

void init_triton_tileir(py::module &&m) {
  init_triton_to_cudatile_passes(m.def_submodule("passes"));
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::cuda_tile::CudaTileDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::cf::ControlFlowDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    // Register cuda_tile passes to enable nested pass manager parsing
    cuda_tile::registerCudaTilePasses();
  });
  m.def("only_contain_legal_dialects", [](mlir::ModuleOp mod) {
    bool only_contain_legal_dialects = true;
    mod->walk([&](mlir::Operation *op) {
      if (!llvm::isa<mlir::ModuleOp>(op) &&
          (op->getName().getDialectNamespace() !=
           mlir::cuda_tile::CudaTileDialect::getDialectNamespace())) {
        only_contain_legal_dialects = false;
      }
    });
    return only_contain_legal_dialects;
  });
  m.def("write_bytecode", [](mlir::ModuleOp mod) {
    // Find the cuda_tile::ModuleOp within the mlir::ModuleOp.
    cuda_tile::ModuleOp cudaTileModule;
    if (!mod.getBody()->empty())
      if (auto nestedCudaTileModule =
              dyn_cast<cuda_tile::ModuleOp>(&mod.getBody()->front()))
        cudaTileModule = nestedCudaTileModule;

    if (!cudaTileModule)
      throw std::runtime_error(
          "No cuda_tile::ModuleOp found in the input module");

    std::string buffer;
    llvm::raw_string_ostream ostream(buffer);
    if (failed(cuda_tile::writeBytecode(
            ostream, cudaTileModule,
            cuda_tile::BytecodeVersion::kCurrentVersion)))
      throw std::runtime_error("Failed to write cuda_tile bytecode");
    py::bytes bytes(buffer.data(), buffer.size());
    return bytes;
  });
  init_tileir_tko_ir(m.def_submodule("ir"));
}
