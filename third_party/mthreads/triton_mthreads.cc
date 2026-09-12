#include "Dialect/MTGPU/IR/Dialect.h"
#include "Dialect/MUSA/IR/Dialect.h"
#include "MTGPUToLLVM/Passes.h"
#include "TritonMUSAGPUToLLVM/Passes.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/MTVM/MTVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

#ifdef __TLE__
void init_triton_musa_tle_ir(py::module m);
void init_triton_musa_tle_frontend_passes_ttgpuir(py::module m);
void init_triton_musa_tle_dialect_passes_ttgpuir(py::module m);
void register_triton_musa_tle_dialects(mlir::DialectRegistry &registry);
#endif

namespace {

llvm::Function *findPrimaryKernel(llvm::Module &module,
                                  llvm::StringRef kernelNameHint) {
  if (!kernelNameHint.empty()) {
    if (llvm::Function *fn = module.getFunction(kernelNameHint)) {
      if (!fn->isDeclaration())
        return fn;
    }
  }
  for (llvm::Function &fn : module) {
    if (!fn.isDeclaration() &&
        fn.getLinkage() == llvm::GlobalValue::ExternalLinkage)
      return &fn;
  }
  for (llvm::Function &fn : module) {
    if (!fn.isDeclaration())
      return &fn;
  }
  return nullptr;
}

bool hasMusaAnnotation(llvm::NamedMDNode *annotations, const llvm::Function &fn,
                       llvm::StringRef key) {
  if (!annotations)
    return false;
  for (llvm::MDNode *node : annotations->operands()) {
    if (!node || node->getNumOperands() < 3)
      continue;
    auto *valueMD = llvm::dyn_cast<llvm::ValueAsMetadata>(node->getOperand(0));
    auto *keyMD = llvm::dyn_cast<llvm::MDString>(node->getOperand(1));
    if (!valueMD || !keyMD)
      continue;
    auto *annotatedFn = llvm::dyn_cast<llvm::Function>(valueMD->getValue());
    if (annotatedFn != &fn)
      continue;
    if (keyMD->getString() == key)
      return true;
  }
  return false;
}

void addMusaAnnotation(llvm::Module &module, llvm::Function &fn,
                       llvm::StringRef key, int32_t value) {
  llvm::NamedMDNode *annotations =
      module.getOrInsertNamedMetadata("musa.annotations");
  if (hasMusaAnnotation(annotations, fn, key))
    return;

  llvm::LLVMContext &ctx = module.getContext();
  llvm::MDNode *node = llvm::MDNode::get(
      ctx, {llvm::ValueAsMetadata::get(&fn), llvm::MDString::get(ctx, key),
            llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), value))});
  annotations->addOperand(node);
}

bool moduleUsesMulhiHelper(const llvm::Module &module) {
  for (const llvm::Function &fn : module) {
    if (fn.isDeclaration())
      continue;
    for (const llvm::BasicBlock &block : fn) {
      for (const llvm::Instruction &inst : block) {
        auto *call = llvm::dyn_cast<llvm::CallBase>(&inst);
        if (!call)
          continue;
        const llvm::Function *callee = call->getCalledFunction();
        if (!callee)
          continue;
        llvm::StringRef calleeName = callee->getName();
        if (calleeName == "__mt_umulhi" || calleeName == "__mt_umul64hi")
          return true;
      }
    }
  }
  return false;
}

constexpr llvm::StringLiteral kTMETailDivisibilityAttr =
    "musa.tme_tail_divisibility";

std::optional<int64_t> getElementSizeBytes(mlir::Type type) {
  int64_t bits = type.getIntOrFloatBitWidth();
  if (bits <= 0 || bits % 8 != 0)
    return std::nullopt;
  return bits / 8;
}

bool blockShapeSatisfiesTME(mlir::triton::TensorDescType descTy,
                            int64_t elemBytes) {
  auto shape = descTy.getBlockType().getShape();
  return !shape.empty() && shape.back() > 0 &&
         (shape.back() * elemBytes) % 16 == 0;
}

bool valueIsKnownMultipleOf(mlir::Value value, int64_t divisor,
                            unsigned callDepth = 0) {
  if (divisor <= 1)
    return true;
  if (callDepth > 16)
    return false;

  mlir::Attribute constantAttr;
  if (mlir::matchPattern(value, mlir::m_Constant(&constantAttr))) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constantAttr))
      return intAttr.getInt() % divisor == 0;
  }

  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    auto func = mlir::dyn_cast_or_null<mlir::triton::FuncOp>(
        arg.getOwner()->getParentOp());
    if (!func)
      return false;
    auto attr = func.getArgAttrOfType<mlir::IntegerAttr>(arg.getArgNumber(),
                                                         "tt.divisibility");
    if (attr && attr.getInt() % divisor == 0)
      return true;

    auto module = func->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return false;
    bool foundCall = false;
    bool allCallOperandsSatisfy = true;
    module.walk([&](mlir::triton::CallOp call) {
      if (call.resolveCallable() != func.getOperation())
        return;
      foundCall = true;
      if (arg.getArgNumber() >= call.getNumOperands() ||
          !valueIsKnownMultipleOf(call.getOperand(arg.getArgNumber()), divisor,
                                  callDepth + 1))
        allCallOperandsSatisfy = false;
    });
    return foundCall && allCallOperandsSatisfy;
  }

  if (mlir::Operation *def = value.getDefiningOp()) {
    auto attr = def->getAttrOfType<mlir::IntegerAttr>("tt.divisibility");
    return attr && attr.getInt() % divisor == 0;
  }
  return false;
}

bool requiresTMETailPointerFallback(mlir::ModuleOp module) {
  bool requiresFallback = false;

  module.walk([&](mlir::triton::MakeTensorDescOp op) {
    if (requiresFallback)
      return;
    auto descTy = op.getType();
    auto elemBytes =
        getElementSizeBytes(descTy.getBlockType().getElementType());
    if (!elemBytes || !blockShapeSatisfiesTME(descTy, *elemBytes)) {
      requiresFallback = true;
      return;
    }
    if (*elemBytes >= 4)
      return;
    int64_t elementDivisor = 4 / std::gcd<int64_t>(4, *elemBytes);
    if (op.getShape().empty() ||
        !valueIsKnownMultipleOf(op.getShape().back(), elementDivisor))
      requiresFallback = true;
  });

  if (requiresFallback)
    return true;

  module.walk([&](mlir::triton::FuncOp func) {
    if (requiresFallback)
      return;
    if (func.getVisibility() != mlir::SymbolTable::Visibility::Public)
      return;
    for (auto [idx, arg] : llvm::enumerate(func.getArguments())) {
      auto descTy = mlir::dyn_cast<mlir::triton::TensorDescType>(arg.getType());
      if (!descTy)
        continue;
      auto elemBytes =
          getElementSizeBytes(descTy.getBlockType().getElementType());
      if (!elemBytes || !blockShapeSatisfiesTME(descTy, *elemBytes)) {
        requiresFallback = true;
        return;
      }
      if (*elemBytes >= 4)
        continue;
      auto attr = func.getArgAttrOfType<mlir::IntegerAttr>(
          idx, kTMETailDivisibilityAttr);
      if (!attr || attr.getInt() % 4 != 0) {
        requiresFallback = true;
        return;
      }
    }
  });

  return requiresFallback;
}

} // namespace

void init_triton_musa_passes_ttgpuir(py::module m) {
  using namespace mlir::triton;
  m.def("add_mtgpu_to_llvm", [](mlir::PassManager &pm, int32_t capability) {
    pm.addPass(mlir::triton::createConvertMTGPUToLLVMPass(capability));
  });
  m.def(
      "add_to_llvmir",
      [](mlir::PassManager &pm, int32_t capability, bool enableFp8Burst2) {
        pm.addPass(mlir::triton::createConvertTritonMUSAGPUToLLVMPass(
            capability, enableFp8Burst2));
      },
      py::arg("pm"), py::arg("capability"),
      py::arg("enable_fp8_burst2") = false);
  m.def("add_allocate_shared_memory", [](mlir::PassManager &pm,
                                         int32_t capability) {
    pm.addPass(mlir::triton::createAllocateMUSASharedMemoryPass(capability));
  });
  ADD_PASS_OPTION_WRAPPER_2("add_pipeline", mlir::createTritonMUSAGPUPipeline,
                            int, bool);
  ADD_PASS_WRAPPER_0("add_accelerate_matmul",
                     mlir::createTritonMUSAGPUAccelerateMatmul);
  ADD_PASS_WRAPPER_0(
      "add_canonicalize_sqmma_result_conversions",
      mlir::createTritonMUSAGPUCanonicalizeSqmmaResultConversions);
  ADD_PASS_WRAPPER_0("add_convert_sqmma_to_mtgpu",
                     mlir::createTritonMUSAGPUConvertSqmmaToMTGPU);
  ADD_PASS_WRAPPER_0("add_finalize_barriers",
                     mlir::createTritonMUSAGPUFinalizeBarriers);
  ADD_PASS_WRAPPER_0("add_issue_barrier_insertion",
                     mlir::createTritonMUSAGPUIssueBarrierInsertion);
  ADD_PASS_OPTION_WRAPPER_1("add_mark_inplace_loads",
                            mlir::createTritonMUSAGPUMarkInplaceLoads,
                            std::string);
  ADD_PASS_WRAPPER_0("add_optimize_accumulator_init",
                     mlir::createTritonMUSAGPUOptimizeAccumulatorInit);
  ADD_PASS_WRAPPER_0("add_optimize_dot_operands",
                     mlir::createTritonMUSAGPUOptimizeDotOperands);
  ADD_PASS_WRAPPER_0("add_tme_lowering", mlir::createTritonMUSAGPUTMELowering);
  ADD_PASS_WRAPPER_0("add_optimize_descriptor_encoding",
                     mlir::createTritonMUSAGPUOptimizeDescriptorEncoding);
}

void init_triton_mthreads(py::module &&m) {
#ifdef __TLE__
  init_triton_musa_tle_ir(m.def_submodule("ir"));
#endif // __TLE__

  auto passes = m.def_submodule("passes");
  auto ttgpuir = passes.def_submodule("ttgpuir");
  init_triton_musa_passes_ttgpuir(ttgpuir);
#ifdef __TLE__
  init_triton_musa_tle_frontend_passes_ttgpuir(ttgpuir);
  init_triton_musa_tle_dialect_passes_ttgpuir(ttgpuir);
#endif // __TLE__

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry
        .insert<mlir::triton::mtgpu::MTGPUDialect,
                mlir::triton::musa::MUSADialect, mlir::vector::VectorDialect>();
#ifdef __TLE__
    register_triton_musa_tle_dialects(registry);
#endif
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerMTVMDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("requires_tme_tail_pointer_fallback", [](mlir::ModuleOp &module) {
    return requiresTMETailPointerFallback(module);
  });

  m.def("attach_datalayout", [](llvm::Module &module) {
    const std::string dataLayout = "e-p:64:64:64:64-"
                                   "p1:64:64:64:64-"
                                   "p2:64:64:64:64-"
                                   "p3:32:32-"
                                   "p4:32:32-"
                                   "p5:64:64-"
                                   "i64:64-"
                                   "v16:16-"
                                   "v24:32-"
                                   "v32:32-"
                                   "v48:64-"
                                   "v96:128";
    module.setDataLayout(dataLayout);
  });

  m.def("decorate_kernel_abi",
        [](llvm::Module &module, const std::string &kernelNameHint,
           int32_t maxntidx) -> std::string {
          llvm::Function *kernel = findPrimaryKernel(module, kernelNameHint);
          if (!kernel)
            return "";

          kernel->setCallingConv(llvm::CallingConv::MTGPU_KERNEL);
          addMusaAnnotation(module, *kernel, "kernel", 1);
          addMusaAnnotation(module, *kernel, "maxntidx",
                            std::max<int32_t>(1, maxntidx));
          return kernel->getName().str();
        });

  m.def("module_uses_mulhi_helper",
        [](llvm::Module &module) { return moduleUsesMulhiHelper(module); });
}
