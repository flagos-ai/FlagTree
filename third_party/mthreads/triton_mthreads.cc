#include "Dialect/MTGPU/IR/Dialect.h"
#include "Dialect/MUSA/IR/Dialect.h"
#include "MTGPUToLLVM/Passes.h"
#include "TritonMUSAGPUToLLVM/Passes.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#ifdef __TLE__
#include "ir.h"
#endif
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/MTVM/MTVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "passes.h"
#ifdef __TLE__
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#endif
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include <algorithm>
#include <cstdint>
#include <pybind11/pybind11.h>
#ifdef __TLE__
#include <pybind11/stl.h>
#include <stdexcept>
#endif

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

#ifdef __TLE__
namespace ttg = mlir::triton::gpu;

void checkCtaRank(llvm::ArrayRef<unsigned> order,
                  llvm::ArrayRef<unsigned> ctasPerCGA,
                  llvm::ArrayRef<unsigned> ctaSplitNum,
                  llvm::ArrayRef<unsigned> ctaOrder) {
  if (order.size() != ctasPerCGA.size() || order.size() != ctaSplitNum.size() ||
      order.size() != ctaOrder.size())
    throw py::value_error("shared layout rank mismatch in CTA parameters");
}

ttg::CGAEncodingAttr makeCgaLayout(mlir::MLIRContext *context,
                                   llvm::ArrayRef<unsigned> ctasPerCGA,
                                   llvm::ArrayRef<unsigned> ctaSplitNum,
                                   llvm::ArrayRef<unsigned> ctaOrder) {
  return ttg::CGAEncodingAttr::fromSplitParams(context, ctasPerCGA, ctaSplitNum,
                                               ctaOrder);
}

mlir::Attribute getSharedMemorySpace(mlir::MLIRContext *context,
                                     const std::string &storage) {
  if (storage == "smem" || storage == "share_memory" ||
      storage == "shared_memory")
    return ttg::SharedMemorySpaceAttr::get(context);
  if (storage == "tmem" || storage == "tensor_memory")
    throw py::value_error("mthreads TLE alloc does not support tmem storage");
  throw py::value_error("mthreads TLE alloc only supports smem storage");
}
#endif // __TLE__

} // namespace

#ifdef __TLE__
void init_triton_mthreads_ir(py::module &&m) {
  (void)m;

  auto *builderClsPtr = ir::getBuilderClass();
  if (!builderClsPtr)
    throw std::runtime_error("triton IR builder class is not initialized");

  auto &builderCls = *builderClsPtr;
  builderCls
      .def("make_swizzled_shared_encoding_attr",
           [](TritonOpBuilder &self, unsigned vectorSize, unsigned perPhase,
              unsigned maxPhase, std::vector<unsigned> order,
              std::vector<unsigned> CTAsPerCGA,
              std::vector<unsigned> CTASplitNum,
              std::vector<unsigned> CTAOrder) -> mlir::Attribute {
             checkCtaRank(order, CTAsPerCGA, CTASplitNum, CTAOrder);
             auto *context = self.getBuilder().getContext();
             auto cgaLayout =
                 makeCgaLayout(context, CTAsPerCGA, CTASplitNum, CTAOrder);
             return ttg::SwizzledSharedEncodingAttr::get(
                 context, vectorSize, perPhase, maxPhase, order, cgaLayout);
           })
      .def("make_nv_mma_shared_encoding_attr",
           [](TritonOpBuilder &, std::vector<int64_t>, std::vector<unsigned>,
              mlir::Type &, std::vector<unsigned>, std::vector<unsigned>,
              std::vector<unsigned>, bool, bool) -> mlir::Attribute {
             throw py::value_error("mthreads TLE alloc does not support "
                                   "nv_mma_shared_layout=True");
           })
      .def("make_tensor_memory_encoding_attr",
           [](TritonOpBuilder &, unsigned, unsigned, unsigned, unsigned,
              unsigned, bool) -> mlir::Attribute {
             throw py::value_error(
                 "mthreads TLE alloc does not support tmem storage");
           })
      .def("create_local_alloc",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              mlir::Type &elementType,
              mlir::Attribute &encoding) -> mlir::Value {
             auto *context = self.getBuilder().getContext();
             auto memorySpace = ttg::SharedMemorySpaceAttr::get(context);
             auto memDesc = ttg::MemDescType::get(shape, elementType, encoding,
                                                  memorySpace,
                                                  /*mutableMemory=*/true);
             return self.create<ttg::LocalAllocOp>(memDesc);
           })
      .def("create_local_alloc",
           [](TritonOpBuilder &self, mlir::Type resultTy,
              mlir::Value value) -> mlir::Value {
             return self.create<ttg::LocalAllocOp>(resultTy, value);
           })
      .def("get_memdesc_type",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              mlir::Type &elementType, mlir::Attribute &encoding,
              std::string storage) -> mlir::Type {
             auto *context = self.getBuilder().getContext();
             auto memorySpace = getSharedMemorySpace(context, storage);
             return ttg::MemDescType::get(shape, elementType, encoding,
                                          memorySpace,
                                          /*mutableMemory=*/true);
           })
      .def("get_memdesc_type",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              mlir::Type &elementType, mlir::Attribute &encoding,
              std::string storage,
              std::vector<int64_t> allocShape) -> mlir::Type {
             auto *context = self.getBuilder().getContext();
             auto memorySpace = getSharedMemorySpace(context, storage);
             return ttg::MemDescType::get(shape, elementType, encoding,
                                          memorySpace,
                                          /*mutableMemory=*/true, allocShape);
           });
}
#endif // __TLE__

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
#ifdef __TLE__
  ADD_PASS_WRAPPER_0("add_tle_early_assign_memory_space",
                     mlir::createTritonMUSAGPUTLEEarlyAssignMemorySpace);
  ADD_PASS_WRAPPER_0("add_tle_lower_async_load",
                     mlir::createTritonMUSAGPUTLELowerAsyncLoad);
#endif // __TLE__
}

void init_triton_mthreads(py::module &&m) {
#ifdef __TLE__
  init_triton_mthreads_ir(m.def_submodule("ir"));
#endif // __TLE__

  auto passes = m.def_submodule("passes");
  init_triton_musa_passes_ttgpuir(passes.def_submodule("ttgpuir"));

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry
        .insert<mlir::triton::mtgpu::MTGPUDialect,
                mlir::triton::musa::MUSADialect, mlir::vector::VectorDialect>();
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerMTVMDialectTranslation(registry);
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
