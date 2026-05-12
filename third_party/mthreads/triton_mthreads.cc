#include "Dialect/MTGPU/IR/Dialect.h"
#include "Dialect/MUSA/IR/Dialect.h"
#include "MTGPUToLLVM/Passes.h"
#include "TritonMUSAGPUToLLVM/Passes.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "passes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include <algorithm>
#include <cstdint>
#include <pybind11/pybind11.h>

namespace py = pybind11;

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

} // namespace

void init_triton_musa_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;
  m.def("add_mtgpu_to_llvm", [](mlir::PassManager &pm, int32_t capability) {
    pm.addPass(mlir::triton::createConvertMTGPUToLLVMPass(capability));
  });
  m.def("add_to_llvmir", [](mlir::PassManager &pm, int32_t capability) {
    pm.addPass(mlir::triton::createConvertTritonMUSAGPUToLLVMPass(capability));
  });
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
  ADD_PASS_WRAPPER_0("add_mark_inplace_loads",
                     mlir::createTritonMUSAGPUMarkInplaceLoads);
  ADD_PASS_WRAPPER_0("add_optimize_accumulator_init",
                     mlir::createTritonMUSAGPUOptimizeAccumulatorInit);
  ADD_PASS_WRAPPER_0("add_optimize_dot_operands",
                     mlir::createTritonMUSAGPUOptimizeDotOperands);
  ADD_PASS_WRAPPER_0("add_tme_lowering", mlir::createTritonMUSAGPUTMELowering);
  ADD_PASS_WRAPPER_0("add_optimize_descriptor_encoding",
                     mlir::createTritonMUSAGPUOptimizeDescriptorEncoding);
  ADD_PASS_WRAPPER_0("add_optimize_sqmma_accumulator_layout",
                     mlir::createTritonMUSAGPUOptimizeSqmmaAccumulatorLayout);
}

void init_triton_mthreads(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_musa_passes_ttgpuir(passes.def_submodule("ttgpuir"));

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry
        .insert<mlir::triton::mtgpu::MTGPUDialect,
                mlir::triton::musa::MUSADialect, mlir::vector::VectorDialect>();
    mlir::registerLLVMDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
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
