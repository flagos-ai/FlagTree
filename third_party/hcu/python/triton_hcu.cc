#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "TritonHCUGPUToLLVM/Passes.h"
#include "TritonHCUGPUToLLVM/TargetUtils.h"
#include "TritonHCUGPUTransforms/Passes.h"
#include "hcu/include/hipblas_instance.h"
#include "hcu/include/hipblas_types.h"
#include "lib/TritonHCUGPUToLLVM/TargetInfo.h"
#include "lld/Common/Driver.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TargetParser/TargetParser.h"
#include <algorithm>
#include <array>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

#include "ReadOnlyLoadInterleave.inc"
#include "TupledRegisterFences.inc"

namespace {
const char *const hcuTargetTriple = "amdgcn-amd-amdhsa";

void init_triton_hcu_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;
  m.def("add_to_llvmir",
        [](mlir::PassManager &pm, const std::string &arch, bool ftz) {
          pm.addPass(createConvertTritonHCUGPUToLLVMPass(arch, ftz));
        });
  m.def("add_builtin_func_to_llvmir", [](mlir::PassManager &pm, bool ftz) {
    pm.addPass(createConvertBuiltinFuncToLLVMPass(ftz));
  });
  m.def("insert_instruction_sched_hints", [](mlir::PassManager &pm,
                                             const std::string &variant) {
    pm.addPass(createTritonHCUGPUInsertInstructionSchedHintsPass(variant));
  });
  m.def("lower_instruction_sched_hints",
        [](mlir::PassManager &pm, const std::string &arch, int32_t numStages) {
          pm.addPass(createTritonHCUGPULowerInstructionSchedHintsPass(
              arch, numStages));
        });
  ADD_PASS_WRAPPER_2("add_optimize_lds_usage",
                     mlir::triton::HCU::createOptimizeLDSUsagePass,
                     const std::string &, int32_t);
  ADD_PASS_WRAPPER_0("add_allocate_shared_memory",
                     mlir::triton::createAllocateHCUGPUSharedMemory);
  ADD_PASS_OPTION_WRAPPER_4("add_accelerate_matmul",
                            mlir::createTritonHCUGPUAccelerateMatmul,
                            const std::string, int, int, int);
  ADD_PASS_WRAPPER_0("add_optimize_epilogue",
                     mlir::createTritonHCUGPUOptimizeEpilogue);
  ADD_PASS_OPTION_WRAPPER_1(
      "add_optimize_dot_operands",
      mlir::triton::hcugpu::createTritonHCUGPUOptimizeDotOperands,
      const std::string &);
  m.def("add_hoist_layout_conversions", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::triton::FuncOp>(
        mlir::createTritonHCUGPUHoistLayoutConversions());
  });
  m.def("add_canonicalize_pointers", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::triton::FuncOp>(
        mlir::createTritonHCUGPUCanonicalizePointers());
  });
  m.def(
      "add_convert_to_buffer_ops",
      [](mlir::PassManager &pm, const std::string &arch, bool allowAtomics,
         bool analyzeSmallTensorOfst, bool asyncOnly) {
        pm.addPass(mlir::createTritonHCUGPUConvertToBufferOps(
            {arch, allowAtomics, analyzeSmallTensorOfst, asyncOnly}));
      },
      py::arg("pm"), py::arg("arch"), py::arg("allow_buffer_atomics"),
      py::arg("analyze_small_tensor_ofst"), py::arg("async_only") = false);
  ADD_PASS_WRAPPER_0("add_reorder_instructions",
                     mlir::createTritonHCUGPUReorderInstructions);
  ADD_PASS_WRAPPER_0("add_fold_true_cmpi", mlir::createTritonHCUFoldTrueCmpI);
  ADD_PASS_OPTION_WRAPPER_1("add_block_pingpong",
                            mlir::createTritonHCUGPUBlockPingpong, int32_t);
  ADD_PASS_OPTION_WRAPPER_1("add_schedule_loops",
                            mlir::createTritonHCUGPUScheduleLoops, int);
  ADD_PASS_OPTION_WRAPPER_2("add_pipeline", mlir::createTritonHCUGPUPipeline,
                            bool, bool);
  ADD_PASS_OPTION_WRAPPER_1("add_coalesce_async_copy",
                            mlir::createTritonHCUGPUCoalesceAsyncCopy,
                            std::string);
  ADD_PASS_OPTION_WRAPPER_1("add_update_async_wait_count",
                            mlir::createTritonHCUGPUUpdateAsyncWaitCount,
                            std::string);
  m.def("add_in_thread_transpose", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::triton::FuncOp>(
        mlir::createTritonHCUGPUInThreadTranspose());
  });
  ADD_PASS_OPTION_WRAPPER_3("add_mls_stream_pipeline",
                            mlir::createTritonHCUGPUMlsStreamPipeline, int, int,
                            int);
  ADD_PASS_WRAPPER_0("add_mls_encoding_insertion",
                     mlir::createTritonHCUGPUMlsEncodingInsertion);
  ADD_PASS_WRAPPER_0("add_mls_lowering_pass",
                     mlir::createTritonHCUGPUMlsLowering);
  m.def("add_warp_specialize_to_llvm",
        [](mlir::PassManager &pm, const std::string &arch, int waspNumLoadWarps,
           int waspNumMmaWarps, bool wdraEnabled, int wdraNumLoadRegs,
           int wdraNumMmaRegsMain, int wdraNumMmaRegsTail) {
          pm.addPass(createHCUGPUConvertWarpSpecializeToLLVM(
              arch, waspNumLoadWarps, waspNumMmaWarps, wdraEnabled,
              wdraNumLoadRegs, wdraNumMmaRegsMain, wdraNumMmaRegsTail));
        });
  ADD_PASS_WRAPPER_1(
      "set_func_inline_attrs",
      mlir::triton::HCU::createTritonHCUGPUSetFuncInlineAttrsPass,
      const std::string &);
}

void addControlConstant(llvm::Module *module, const char *name,
                        uint32_t bitwidth, uint32_t value) {
  using llvm::GlobalVariable;

  llvm::IntegerType *type =
      llvm::IntegerType::getIntNTy(module->getContext(), bitwidth);
  auto *initializer = llvm::ConstantInt::get(type, value, /*isSigned=*/false);
  auto *constant = new llvm::GlobalVariable(
      *module, type, /*isConstant=*/true,
      GlobalVariable::LinkageTypes::LinkOnceODRLinkage, initializer, name,
      /*before=*/nullptr, GlobalVariable::ThreadLocalMode::NotThreadLocal,
      /*addressSpace=*/4);
  constant->setAlignment(llvm::MaybeAlign(bitwidth / 8));
  constant->setUnnamedAddr(GlobalVariable::UnnamedAddr::Local);
  constant->setVisibility(GlobalVariable::VisibilityTypes::ProtectedVisibility);
}

} // namespace

LLD_HAS_DRIVER(elf)

static void checkMatmulConstraints(const std::string &A_dtype,
                                   const std::string &B_dtype,
                                   const std::string &C_dtype,
                                   const std::vector<int> &A_shape,
                                   const std::vector<int> &B_shape,
                                   const std::vector<int> &C_shape) {
  // Support FP32/FP16/BF16 and 8-bit FP8 (e4m3fn/e4m3fnuz) and BF8
  // (e5m2fn/e5m2fnuz).
  auto is_fp8 = [](const std::string &dtype) {
    return dtype == "torch.float8_e4m3fn" || dtype == "torch.float8_e5m2fn" ||
           dtype == "torch.float8_e4m3fnuz" || dtype == "torch.float8_e5m2fnuz";
  };
  auto is_fp16_family = [](const std::string &dtype) {
    return dtype == "torch.float16" || dtype == "torch.bfloat16";
  };
  const bool A_is_fp8 = is_fp8(A_dtype);
  const bool B_is_fp8 = is_fp8(B_dtype);
  const bool A_supported =
      (A_is_fp8 || is_fp16_family(A_dtype) || A_dtype == "torch.float32");
  const bool B_supported =
      (B_is_fp8 || is_fp16_family(B_dtype) || B_dtype == "torch.float32");
  const bool C_supported = (is_fp16_family(C_dtype) ||
                            C_dtype == "torch.float32" || is_fp8(C_dtype));

  if (!A_supported || !B_supported || !C_supported) {
    std::ostringstream oss;
    oss << "Unsupported data type. Got A=" << A_dtype << ", B=" << B_dtype
        << ", C=" << C_dtype
        << ". Supported: float32, float16, bfloat16, float8_e4m3fn, "
           "float8_e5m2fn, float8_e4m3fnuz, float8_e5m2fnuz.";
    throw std::runtime_error(oss.str());
  }

  if (A_is_fp8 && B_is_fp8) {
    if (C_dtype != "torch.float16" && C_dtype != "torch.float32" &&
        C_dtype != "torch.bfloat16") {
      std::ostringstream oss;
      oss << "When A/B are 8-bit (float8_e4m3fn/e4m3fnuz or "
             "float8_e5m2fn/e5m2fnuz), C must"
          << " be torch.float16, torch.float32, or torch.bfloat16.";
      throw std::runtime_error(oss.str());
    }
  } else {
    if (!(A_dtype == B_dtype && A_dtype == C_dtype)) {
      std::ostringstream oss;
      oss << "Data types do not match: A=" << A_dtype << ", B=" << B_dtype
          << ", C=" << C_dtype << ". Expected all equal when not using 8-bit"
          << " inputs.";
      throw std::runtime_error(oss.str());
    }
  }

  if (A_shape.size() != 2 || B_shape.size() != 2 || C_shape.size() != 2) {
    throw std::runtime_error("Only 2D matrices are supported.");
  }

  int k = A_shape[1];
  if (k != B_shape[1]) {
    std::ostringstream oss;
    oss << "Matrix dimensions do not match. A is [" << A_shape[0] << ", "
        << A_shape[1] << "], B is [" << B_shape[0] << ", " << B_shape[1]
        << "]. Expected A.shape[1] == B.shape[1]. Note that B needs to be "
           "transposed.";
    throw std::runtime_error(oss.str());
  }

  int m = A_shape[0];
  if (m != C_shape[0]) {
    std::ostringstream oss;
    oss << "Matrix dimensions do not match. A is [" << A_shape[0] << ", "
        << A_shape[1] << "], C is [" << C_shape[0] << ", " << C_shape[1]
        << "]. Expected A.shape[0] == C.shape[0].";
    throw std::runtime_error(oss.str());
  }

  int n = B_shape[0];
  if (n != C_shape[1]) {
    std::ostringstream oss;
    oss << "Matrix dimensions do not match. B is [" << B_shape[0] << ", "
        << B_shape[1] << "], C is [" << C_shape[0] << ", " << C_shape[1]
        << "]. Expected B.shape[0] == C.shape[1]. Note that B needs to be "
           "transposed.";
    throw std::runtime_error(oss.str());
  }
}

struct HipBlasInit {
  int m;
  int n;
  int k;
  hipDataType dtype;
  hipDataType out_dtype;
};

static HipBlasInit initialize_hipblas_op(py::object &A, py::object &B,
                                         py::object &out,
                                         std::optional<py::object> accumOpt) {
  auto A_shape = A.attr("shape").cast<std::vector<int>>();
  auto B_shape = B.attr("shape").cast<std::vector<int>>();
  auto OUT_shape = out.attr("shape").cast<std::vector<int>>();

  auto A_dtype = A.attr("dtype").attr("__str__")().cast<std::string>();
  auto B_dtype = B.attr("dtype").attr("__str__")().cast<std::string>();
  auto OUT_dtype = out.attr("dtype").attr("__str__")().cast<std::string>();

  if (accumOpt.has_value()) {
    auto C = accumOpt.value();
    auto C_shape = C.attr("shape").cast<std::vector<int>>();
    auto C_dtype = C.attr("dtype").attr("__str__")().cast<std::string>();

    checkMatmulConstraints(A_dtype, B_dtype, OUT_dtype, A_shape, B_shape,
                           OUT_shape);
    if (C_dtype != OUT_dtype) {
      throw std::runtime_error("C dtype must match output dtype, got C=" +
                               C_dtype + ", D=" + OUT_dtype);
    }
    if (C_shape != OUT_shape) {
      throw std::runtime_error("C and D shapes must match");
    }
  } else {
    checkMatmulConstraints(A_dtype, B_dtype, OUT_dtype, A_shape, B_shape,
                           OUT_shape);
  }

  hipDataType dtype;
  if (A_dtype == "torch.float8_e4m3fn") {
    // Supported for GFX950.
    dtype = HIP_R_8F_E4M3;
  } else if (A_dtype == "torch.float8_e5m2fn") {
    // supported for GFX950.
    dtype = HIP_R_8F_E5M2;
  } else if (A_dtype == "torch.float8_e4m3fnuz") {
    // Supported for GFX942.
    dtype = HIP_R_8F_E4M3_FNUZ;
  } else if (A_dtype == "torch.float8_e5m2fnuz") {
    // Supported for GFX942.
    dtype = HIP_R_8F_E5M2_FNUZ;
  } else if (A_dtype == "torch.float16") {
    dtype = HIP_R_16F;
  } else if (A_dtype == "torch.float32") {
    dtype = HIP_R_32F;
  } else if (A_dtype == "torch.bfloat16") {
    dtype = HIP_R_16BF;
  } else {
    throw std::runtime_error("Unsupported dtype for hipblasLt: " + A_dtype);
  }

  hipDataType out_dtype;
  if (OUT_dtype == "torch.float16") {
    out_dtype = HIP_R_16F;
  } else if (OUT_dtype == "torch.float32") {
    out_dtype = HIP_R_32F;
  } else if (OUT_dtype == "torch.bfloat16") {
    out_dtype = HIP_R_16BF;
  } else {
    throw std::runtime_error("Unsupported output dtype for hipblasLt: " +
                             OUT_dtype);
  }

  int m = A_shape[0];
  int n = B_shape[0];
  int k = A_shape[1];

  return HipBlasInit{m, n, k, dtype, out_dtype};
}

static std::optional<std::string> lldInvoke(const char *inPath,
                                            const char *outPath) {
  // Workaround: Disable parallelism to avoid hangs caused by LLVM's thread pool
  // when the following code is executed in a forked child process.
  // Context: lld::elf::LinkerDriver::link uses parallelFor which uses the
  // LLVM's thread pool. During cleanup at ~TaskGroup() the child process hangs
  // waiting.
  std::array args{"ld.lld", "--threads=1", "-shared", inPath, "-o", outPath};
  std::string errString;
  llvm::raw_string_ostream errStream(errString);
  auto lldRes = lld::lldMain(args, llvm::outs(), llvm::errs(),
                             {{lld::Gnu, &lld::elf::link}});
  bool noErrors = (!lldRes.retCode && lldRes.canRunAgain);
  if (!noErrors) {
    errStream.flush();
    return errString;
  }
  return {};
}

// DTK LLVM 18 MachineSink can move an LDS read past s.barrier into a
// successor block. Reuse the empty memory-clobber fence from Triton's AMD
// workaround (PRs 9624 and 11096; LLVM issue 181708). This emits no ISA.
// Run after LLVM optimization and only for the affected native compiler.
static unsigned preserveLdsBarrierOrder(llvm::Module *module,
                                        const std::string &arch) {
  if (arch != "gfx936")
    return 0;
  auto *type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(module->getContext()), false);
  auto *fence = llvm::InlineAsm::get(type, "", "~{memory}", true);
  unsigned count = 0;
  for (auto &fn : *module)
    for (auto &block : fn)
      for (auto &inst : block) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
        auto *callee = call ? call->getCalledFunction() : nullptr;
        if (!callee || callee->getName() != "llvm.amdgcn.s.barrier")
          continue;
        llvm::IRBuilder<> builder(call);
        builder.SetCurrentDebugLocation(call->getDebugLoc());
        builder.CreateCall(fence);
        ++count;
      }
  return count;
}

// LocalBarrierOp synchronizes shared/register operations, not unrelated VMEM.
// DTK LLVM 18 conservatively drains VMEM at the generic barrier intrinsic.
// Preserve its provenance through optimization, then select the same hardware
// barrier with explicit convergence and memory effects. Global/other barriers
// keep their intrinsic. Explicit async waits and LDS waits are untouched.
// Keep synchronous functions on the established native barrier path. This
// optimization targets outstanding global-to-LDS copies, not GPR loads.
static unsigned stripSynchronousBarrierMarkers(llvm::Module *module,
                                               const std::string &arch) {
  if (arch != "gfx936")
    return 0;
  auto *marker = module->getFunction("__triton_hcu_local_barrier_marker");
  if (!marker)
    return 0;
  llvm::DenseMap<llvm::Function *, bool> hasAsyncCopy;
  for (auto &fn : *module)
    for (auto &block : fn)
      for (auto &inst : block)
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst))
          if (auto *callee = call->getCalledFunction())
            if (callee->getName().starts_with(
                    "llvm.amdgcn.raw.ptr.buffer.load.lds") ||
                callee->getName().starts_with(
                    "llvm.amdgcn.struct.ptr.buffer.load.lds") ||
                callee->getName().starts_with(
                    "llvm.hcu.raw.buffer.load.lds"))
              hasAsyncCopy[&fn] = true;
  llvm::SmallVector<llvm::CallInst *, 8> remove;
  for (auto *user : marker->users()) {
    auto *call = llvm::dyn_cast<llvm::CallInst>(user);
    if (!call || call->getCalledFunction() != marker || call->arg_size() != 0)
      throw std::invalid_argument("invalid HCU local barrier marker use");
    if (!hasAsyncCopy.lookup(call->getFunction()))
      remove.push_back(call);
  }
  for (auto *call : remove)
    call->eraseFromParent();
  if (marker->use_empty())
    marker->eraseFromParent();
  return remove.size();
}

static unsigned lowerTaggedLocalBarriers(llvm::Module *module,
                                         const std::string &arch) {
  if (arch != "gfx936")
    return 0;
  auto *marker = module->getFunction("__triton_hcu_local_barrier_marker");
  if (!marker)
    return 0;
  llvm::SmallVector<llvm::CallInst *, 8> tags;
  for (auto *user : marker->users()) {
    auto *call = llvm::dyn_cast<llvm::CallInst>(user);
    if (!call || call->getCalledFunction() != marker || call->arg_size() != 0)
      throw std::invalid_argument("invalid HCU local barrier marker use");
    tags.push_back(call);
  }
  unsigned changed = 0;
  for (auto *tag : tags) {
    llvm::CallInst *barrier = nullptr;
    for (auto *inst = tag->getPrevNode(); inst; inst = inst->getPrevNode()) {
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(inst)) {
        auto *callee = call->getCalledFunction();
        if (callee && callee->getName() == "llvm.amdgcn.s.barrier") {
          barrier = call;
          break;
        }
        if (callee && callee->getName() == "llvm.amdgcn.sched.barrier")
          continue;
      }
      if (inst->mayReadOrWriteMemory() || inst->mayHaveSideEffects())
        break;
    }
    if (barrier && barrier->getNumOperandBundles() == 0) {
      auto isRange = [](llvm::Instruction *inst) {
        auto *call = llvm::dyn_cast_or_null<llvm::CallInst>(inst);
        auto *callee = call ? call->getCalledFunction() : nullptr;
        return callee && callee->getName() == "llvm.amdgcn.sched.barrier" &&
               call->arg_size() == 1 &&
               llvm::isa<llvm::ConstantInt>(call->getArgOperand(0));
      };
      bool iglp = false;
      for (auto &inst : *barrier->getParent())
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst))
          if (auto *callee = call->getCalledFunction()) {
            auto name = callee->getName();
            iglp |= name == "llvm.amdgcn.iglp.opt";
          }
      bool paired = isRange(barrier->getPrevNode()) &&
                    isRange(barrier->getNextNode());
      // IGroupLP rebuilds native instruction range edges. Opaque barriers
      // require the existing AMD full-range contract; permissive MFMA masks
      // can detach their memory dependencies. Unsupported contracts keep the
      // conservative intrinsic, as do convergence/deoptimization bundles.
      if (iglp) {
        tag->eraseFromParent();
        continue;
      }
      if (!paired) {
        // Establish AMD's full-range contract even without prior scheduler
        // hints. Otherwise native scheduling can move publication waits
        // across preceding MMACs. Keep fences and IGroupLP fallback unchanged.
        auto *boundary = llvm::Intrinsic::getOrInsertDeclaration(
            module, llvm::Intrinsic::amdgcn_sched_barrier);
        llvm::IRBuilder<> before(barrier), after(barrier->getNextNode());
        before.SetCurrentDebugLocation(barrier->getDebugLoc());
        after.SetCurrentDebugLocation(barrier->getDebugLoc());
        before.CreateCall(boundary, {before.getInt32(0)});
        after.CreateCall(boundary, {after.getInt32(0)});
        paired = true;
      }
      if (paired) {
        auto *zero = llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(module->getContext()), 0);
        llvm::cast<llvm::CallInst>(barrier->getPrevNode())->setArgOperand(0, zero);
        llvm::cast<llvm::CallInst>(barrier->getNextNode())->setArgOperand(0, zero);
        // Match AMD BlockPingpong's boundary before async_wait. The emitted
        // LocalBarrier DS0 wait belongs to this publication, not to preceding
        // MMACs; keep it inside the native scheduling range as well.
        auto *begin = llvm::cast<llvm::CallInst>(barrier->getPrevNode());
        auto getWait = [](llvm::Instruction *inst) -> llvm::CallInst * {
          auto *call = llvm::dyn_cast_or_null<llvm::CallInst>(inst);
          auto *callee = call ? call->getCalledFunction() : nullptr;
          if (!callee || callee->getName() != "llvm.amdgcn.s.waitcnt" ||
              call->arg_size() != 1 || call->getNumOperandBundles() ||
              !call->getArgOperand(0)->getType()->isIntegerTy(32) ||
              !llvm::isa<llvm::ConstantInt>(call->getArgOperand(0)))
            return nullptr;
          return call;
        };
        auto *previous = begin->getPrevNode();
        if (auto *fence = llvm::dyn_cast_or_null<llvm::CallInst>(previous))
          if (auto *code = llvm::dyn_cast<llvm::InlineAsm>(
                  fence->getCalledOperand()))
            if (code->getAsmString().empty() &&
                code->getConstraintString() == "~{memory}" &&
                code->hasSideEffects() && fence->arg_empty() &&
                !fence->getNumOperandBundles())
              previous = previous->getPrevNode();
        auto *wait = getWait(previous);
        // Canonical gfx936 LocalBarrier wait: vmcnt(63), lgkmcnt(0).
        if (wait && llvm::cast<llvm::ConstantInt>(wait->getArgOperand(0))
                            ->getZExtValue() == 0xc07f) {
          if (auto *asyncWait = getWait(wait->getPrevNode()))
            wait = asyncWait;
          begin->moveBefore(wait->getIterator());
        }
      }
      llvm::IRBuilder<> builder(barrier);
      builder.SetCurrentDebugLocation(barrier->getDebugLoc());
      auto *code = llvm::InlineAsm::get(barrier->getFunctionType(),
                                       "s_barrier", "~{memory}", true);
      auto *call = builder.CreateCall(code);
      call->setConvergent();
      call->setDoesNotThrow();
      barrier->eraseFromParent();
      ++changed;
    }
    // An unrecognized ordering keeps the conservative intrinsic lowering.
    tag->eraseFromParent();
  }
  if (marker->use_empty())
    marker->eraseFromParent();
  return changed;
}


// DTK DAG selection can discard the x2 byte scale of a prefetched buffer
// load. Protect its element offset after LLVM optimization: earlier freezes
// are propagated away by LLVM 22 before the module reaches DTK.
static unsigned preserveBufferLoadByteOffsets(llvm::Module *module,
                                             const std::string &arch) {
  if (arch != "gfx936")
    return 0;
  llvm::SmallSetVector<llvm::BinaryOperator *, 16> shifts;
  for (auto &fn : *module) {
    if (fn.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL)
      continue;
    for (auto &block : fn)
      for (auto &inst : block) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
        auto *callee = call ? call->getCalledFunction() : nullptr;
        if (!callee || !callee->getName().starts_with(
                           "llvm.amdgcn.raw.ptr.buffer.load.") ||
            callee->getName().starts_with(
                "llvm.amdgcn.raw.ptr.buffer.load.lds"))
          continue;
        llvm::SmallVector<llvm::Value *, 4> offsets{call->getArgOperand(1)};
        while (!offsets.empty()) {
          auto *offset = offsets.pop_back_val();
          if (auto *select = llvm::dyn_cast<llvm::SelectInst>(offset)) {
            offsets.push_back(select->getTrueValue());
            offsets.push_back(select->getFalseValue());
          } else if (auto *shift = llvm::dyn_cast<llvm::BinaryOperator>(offset)) {
            auto *amount = llvm::dyn_cast<llvm::ConstantInt>(shift->getOperand(1));
            if (shift->getOpcode() == llvm::Instruction::Shl &&
                shift->getType()->isIntegerTy(32) && amount && amount->isOne() &&
                !llvm::isa<llvm::FreezeInst>(shift->getOperand(0)))
              shifts.insert(shift);
          }
        }
      }
  }
  for (auto *shift : shifts) {
    llvm::IRBuilder<> builder(shift);
    builder.SetCurrentDebugLocation(shift->getDebugLoc());
    shift->setOperand(0, builder.CreateFreeze(shift->getOperand(0),
                                             "buffer.element.offset"));
  }
  return shifts.size();
}

// Keep ready LDS reads ahead of future direct-to-LDS copies. Older gfx936
// codegen otherwise schedules the copies first and drains them before the reads,
// even when the existing pipeline alias scopes prove that their slots differ.
// This runs after LLVM optimization, under the opt-in local-prefetch hint.
static unsigned preferReadySharedLoads(llvm::Module *module,
                                      const std::string &arch) {
  if (arch != "gfx936")
    return 0;

  auto isCopy = [](llvm::Instruction &inst) {
    auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
    auto *fn = call ? call->getCalledFunction() : nullptr;
    return fn && fn->getName() == "llvm.amdgcn.raw.ptr.buffer.load.lds" &&
           call->getType()->isVoidTy();
  };
  auto isNativeRead = [](llvm::Instruction &inst) {
    auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
    auto *fn = call ? call->getCalledFunction() : nullptr;
    return fn && fn->getName().starts_with("llvm.hcu.ds.read.m32x16.b16.");
  };
  auto isRead = [&](llvm::Instruction &inst) {
    if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst))
      return load->getPointerAddressSpace() == 3 && !load->isVolatile() &&
             !load->isAtomic();
    return isNativeRead(inst);
  };
  auto disjoint = [](llvm::Instruction *copy, llvm::Instruction *read) {
    auto *scopes = copy->getMetadata(llvm::LLVMContext::MD_alias_scope);
    auto *noalias = read->getMetadata(llvm::LLVMContext::MD_noalias);
    if (!scopes || !scopes->getNumOperands() || !noalias)
      return false;
    // Require every existing copy scope to be excluded by every ready read.
    // Never manufacture alias information from pointer arithmetic or tile sizes.
    for (const auto &scope : scopes->operands()) {
      bool found = false;
      for (const auto &excluded : noalias->operands())
        found |= scope.get() == excluded.get();
      if (!found)
        return false;
    }
    return true;
  };

  unsigned changed = 0;
  for (auto &fn : *module) {
    if (fn.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL)
      continue;
    for (auto &block : fn) {
      auto *branch = llvm::dyn_cast<llvm::BranchInst>(block.getTerminator());
      if (!branch || !branch->isConditional() ||
          (branch->getSuccessor(0) != &block &&
           branch->getSuccessor(1) != &block))
        continue;
      llvm::SmallVector<llvm::Instruction *> copies, reads;
      bool native = false;
      for (auto &inst : block) {
        if (isCopy(inst))
          copies.push_back(&inst);
        else if (isRead(inst)) {
          reads.push_back(&inst);
          native |= isNativeRead(inst);
        }
      }
      if (copies.size() < 2 || reads.empty() || !native ||
          !copies.back()->comesBefore(reads.back()))
        continue;
      bool legal = true;
      for (auto *copy : copies)
        for (auto *read : reads)
          legal &= disjoint(copy, read);
      if (!legal)
        continue;
      auto *lastRead = reads.back();
      for (auto *inst = copies.front(); inst != lastRead;
           inst = inst->getNextNode()) {
        if (isCopy(*inst) || isRead(*inst))
          continue;
        // In particular, do not cross synchronization, unknown memory effects,
        // inline assembly or calls which may throw. Keep all original waits.
        if (inst->mayReadOrWriteMemory() || inst->mayHaveSideEffects()) {
          legal = false;
          break;
        }
      }
      if (!legal)
        continue;
      llvm::IRBuilder<> builder(lastRead->getNextNode());
      auto *fence = builder.CreateCall(llvm::Intrinsic::getOrInsertDeclaration(
          module, llvm::Intrinsic::amdgcn_sched_barrier),
          {builder.getInt32(0)});
      llvm::Instruction *after = fence;
      for (auto *copy : copies) {
        copy->moveAfter(after);
        after = copy;
      }
      ++changed;
    }
  }
  return changed;
}

namespace {
// Use physical LDS objects rather than one scope for all DMA operations.
// LLVM's AMDGPU wait pass tracks representative stores by complete AAInfo;
// separate direct-G2L packets therefore need distinct scopes. This follows
// AMDGPULowerModuleLDS's object-scope convention without changing any waits.
struct LdsOffsetRange {
  uint64_t low;
  uint64_t high;
};

// Layout addresses combine masked bitfields and fragment additions. Recurse
// through a bounded OR/add/sub tree before generic ValueTracking exhausts its
// depth budget. KnownBits transfers model modular arithmetic; no no-wrap flags,
// launch assumptions or frontend witnesses are required. Unknown expressions
// and exhausted budgets retain their original conservative facts.
static llvm::KnownBits getLdsIndexKnownBitsImpl(
    llvm::Value *value, const llvm::DataLayout &layout, unsigned &budget) {
  auto known = llvm::computeKnownBits(value, layout);
  // A bound inside all LDS can still cross a smaller physical object.
  // Refine nonconstant expressions within the node budget before partitioning.
  if (!budget || known.isConstant())
    return known;
  --budget;
  auto *op = llvm::dyn_cast<llvm::Operator>(value);
  if (!op)
    return known;
  unsigned code = op->getOpcode();
  if (code == llvm::Instruction::Mul) {
    // Multiplication by a constant can hide disjoint shifted bitfields.
    // Sum its set-bit shifts in modular arithmetic, without no-wrap flags.
    auto *constant = llvm::dyn_cast<llvm::ConstantInt>(op->getOperand(1));
    llvm::Value *input = op->getOperand(0);
    if (!constant) {
      constant = llvm::dyn_cast<llvm::ConstantInt>(op->getOperand(0));
      input = op->getOperand(1);
    }
    if (!constant)
      return known;
    auto lhs = getLdsIndexKnownBitsImpl(input, layout, budget);
    unsigned width = lhs.getBitWidth();
    llvm::SmallVector<unsigned, 8> shifts;
    for (unsigned bit = 0; bit < width; ++bit)
      if (constant->getValue()[bit]) {
        shifts.push_back(bit);
        if (shifts.size() > budget)
          return known;
      }
    budget -= shifts.size();
    llvm::KnownBits product(width);
    product.Zero.setAllBits();
    for (unsigned shift : shifts) {
      auto term = lhs;
      term.Zero = (lhs.Zero << shift) |
                  llvm::APInt::getLowBitsSet(width, shift);
      term.One = lhs.One << shift;
      product = llvm::KnownBits::computeForAddSub(
          true, false, false, product, term);
    }
    return known.unionWith(product);
  }
  if (code != llvm::Instruction::Or && code != llvm::Instruction::And &&
      code != llvm::Instruction::Xor && code != llvm::Instruction::Add &&
      code != llvm::Instruction::Sub)
    return known;
  auto lhs = getLdsIndexKnownBitsImpl(op->getOperand(0), layout, budget);
  auto rhs = getLdsIndexKnownBitsImpl(op->getOperand(1), layout, budget);
  auto refined = lhs;
  if (code == llvm::Instruction::Or)
    refined |= rhs;
  else if (code == llvm::Instruction::And) {
    refined.Zero = lhs.Zero | rhs.Zero;
    refined.One = lhs.One & rhs.One;
  } else if (code == llvm::Instruction::Xor) {
    refined.Zero = (lhs.Zero & rhs.Zero) | (lhs.One & rhs.One);
    refined.One = (lhs.Zero & rhs.One) | (lhs.One & rhs.Zero);
  } else
    refined = llvm::KnownBits::computeForAddSub(
        code == llvm::Instruction::Add, false, false, lhs, rhs);
  return known.unionWith(refined);
}

static llvm::KnownBits getLdsIndexKnownBits(llvm::Value *value,
                                           const llvm::DataLayout &layout) {
  unsigned budget = 31;
  return getLdsIndexKnownBitsImpl(value, layout, budget);
}

static std::optional<LdsOffsetRange>
getBoundedLdsOffset(llvm::Value *ptr, const llvm::DataLayout &layout,
                    unsigned depth = 0) {
  if (depth > 16 || !ptr->getType()->isPointerTy() ||
      ptr->getType()->getPointerAddressSpace() != 3)
    return std::nullopt;
  if (auto *global = llvm::dyn_cast<llvm::GlobalVariable>(ptr)) {
    if (global->getName() == "global_smem")
      return LdsOffsetRange{0, 0};
    return std::nullopt;
  }
  auto *gep = llvm::dyn_cast<llvm::GEPOperator>(ptr);
  if (!gep)
    return std::nullopt;
  auto base = getBoundedLdsOffset(gep->getPointerOperand(), layout, depth + 1);
  if (!base)
    return std::nullopt;
  llvm::SmallMapVector<llvm::Value *, llvm::APInt, 4> variables;
  unsigned indexBits = layout.getIndexSizeInBits(3);
  if (indexBits == 0 || indexBits > 64)
    return std::nullopt;
  llvm::APInt constant(indexBits, 0);
  if (!gep->collectOffset(layout, indexBits, variables, constant))
    return std::nullopt;
  int64_t low = int64_t(base->low) + constant.getSExtValue();
  int64_t high = int64_t(base->high) + constant.getSExtValue();
  for (auto &[value, scale] : variables) {
    if (!value->getType()->isIntegerTy() || scale.isNegative() ||
        scale.getLimitedValue(65537) > 65536)
      return std::nullopt;
    llvm::KnownBits bits = getLdsIndexKnownBits(value, layout);
    auto maximum = bits.getMaxValue();
    // GEP indices are signed. Reject negative or wrapping index ranges.
    if (!bits.Zero.isSignBitSet() || maximum.getActiveBits() > 16)
      return std::nullopt;
    low += bits.getMinValue().getZExtValue() * scale.getZExtValue();
    high += maximum.getZExtValue() * scale.getZExtValue();
    if (low < 0 || high >= 65536)
      return std::nullopt;
  }
  if (low < 0 || high < low || high >= 65536)
    return std::nullopt;
  return LdsOffsetRange{uint64_t(low), uint64_t(high)};
}

static bool hasHcuAliasScope(llvm::MDNode *list, llvm::StringRef name) {
  if (!list)
    return false;
  for (auto &operand : list->operands()) {
    auto *scope = llvm::dyn_cast_or_null<llvm::MDNode>(operand.get());
    if (!scope)
      continue;
    for (auto &field : scope->operands())
      if (auto *text = llvm::dyn_cast_or_null<llvm::MDString>(field.get()))
        if (text->getString() == name)
          return true;
  }
  return false;
}

static unsigned annotateLdsObjectScopes(llvm::Module *module,
                                        const std::string &arch,
                                        unsigned waveSize) {
  if (arch != "gfx936" || waveSize != 64)
    return 0;
  unsigned changed = 0;
  for (auto &fn : *module) {
    uint64_t objectBytes = 4096;
    struct Access {
      llvm::Instruction *inst;
      LdsOffsetRange range;
    };
    llvm::SmallVector<Access, 32> copies, reads;
    llvm::SmallSetVector<uint64_t, 8> objects;
    for (auto &block : fn)
      for (auto &inst : block) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
        auto *callee = call ? call->getCalledFunction() : nullptr;
        llvm::Value *pointer = nullptr;
        uint64_t bytes = 0;
        bool raw = callee && callee->getName() ==
                                  "llvm.amdgcn.raw.ptr.buffer.load.lds" &&
                   call->arg_size() == 7;
        bool structured = callee && callee->getName() ==
                                  "llvm.amdgcn.struct.ptr.buffer.load.lds" &&
                          call->arg_size() == 8;
        bool copy = raw || structured;
        if (copy) {
          if (!hasHcuAliasScope(inst.getMetadata(llvm::LLVMContext::MD_alias_scope),
                                "hcug.AsyncCopies"))
            continue;
          auto *size = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(2));
          if (!size || (size->getZExtValue() != 8 && size->getZExtValue() != 16) ||
              (structured && size->getZExtValue() != 16))
            continue; // Structured hardware footprint is verified for16B only.
          pointer = call->getArgOperand(1);
          // Byte width is per lane, as in AMD BufferEmitter::emitLoadToLds.
          // The scalar dst selects a complete wave packet.
          bytes = waveSize * size->getZExtValue();
        } else {
          if (!hasHcuAliasScope(inst.getMetadata(llvm::LLVMContext::MD_alias_scope),
                                "hcug.LocalLoads") ||
              !hasHcuAliasScope(inst.getMetadata(llvm::LLVMContext::MD_noalias),
                                "hcug.AsyncCopies"))
            continue;
          if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst)) {
            pointer = load->getPointerOperand();
            auto size = module->getDataLayout().getTypeStoreSize(load->getType());
            if (size.isScalable() || size.getFixedValue() > 16)
              continue;
            bytes = size.getFixedValue();
          } else if (callee && (callee->getName() ==
                                    "llvm.hcu.ds.read.m32x16.b16.v4i32.p3" ||
                                callee->getName() ==
                                    "llvm.hcu.ds.read.m32x16.b16.alt.v4i32.p3")) {
            if (call->arg_size() != 2)
              continue;
            auto *mode = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(1));
            if (!mode || !mode->isZero())
              continue;
            pointer = call->getArgOperand(0);
            // Normal and ALT both gather halfwords within an issuing
            // lane's 16-byte row (the finite-read planner proves the
            // distinct cross-lane permutations). Bound the union of all
            // issuing-lane addresses plus [0,14].
            // The range bounds all lanes, including native cross-lane reads.
            bytes = 16;
          }
        }
        if (!pointer || !bytes)
          continue;
        auto range = getBoundedLdsOffset(pointer, module->getDataLayout());
        if (!range || range->high + bytes > 65536)
          continue;
        range->high += bytes; // high is now an exclusive access bound.
        if (copy) {
          bool fits4KiB = range->low / 4096 == (range->high - 1) / 4096;
          bool full8KiB = range->low % 8192 == 0 &&
                          range->high - range->low == 8192;
          if (!fits4KiB && !full8KiB)
            continue;
          // Large tiles span a proven, aligned 8KiB region across all waves.
          // Coarsening retains aliases; arbitrary crossing ranges still bail.
          if (full8KiB)
            objectBytes = 8192;
          copies.push_back({&inst, *range});
        } else {
          reads.push_back({&inst, *range});
        }
      }
    for (const auto &copy : copies)
      objects.insert(copy.range.low / objectBytes);
    // LLVM18 reserves eight object-score slots. Coarsen proven byte ranges
    // when 4KiB partitions exceed that budget; merging regions only reduces
    // alias precision. Unsupported dynamic addresses retain conservative scopes.
    llvm::DenseMap<uint64_t, uint64_t> representatives;
    unsigned scopeCount = objects.size();
    if (objects.size() > 8) {
      // Objects indistinguishable to every proven read can share a scoreboard
      // slot without losing any of that read-disjointness information.
      llvm::SmallVector<llvm::SmallVector<uint8_t, 32>, 8> patterns;
      llvm::SmallVector<uint64_t, 8> leaders;
      for (uint64_t object : objects) {
        llvm::SmallVector<uint8_t, 32> pattern;
        for (const auto &read : reads)
          pattern.push_back((object + 1) * objectBytes <= read.range.low ||
                            object * objectBytes >= read.range.high);
        unsigned group = 0;
        while (group < patterns.size() && patterns[group] != pattern)
          ++group;
        if (group == patterns.size()) {
          patterns.push_back(std::move(pattern));
          leaders.push_back(object);
        }
        representatives[object] = leaders[group];
      }
      scopeCount = leaders.size();
    }
    if (scopeCount > 8) {
      representatives.clear();
      objectBytes = 8192;
      objects.clear();
      for (const auto &copy : copies)
        objects.insert(copy.range.low / objectBytes);
      scopeCount = objects.size();
    }
    if (scopeCount < 2 || scopeCount > 8 || reads.empty())
      continue;
    bool useful = llvm::any_of(reads, [&](const Access &read) {
      return llvm::any_of(objects, [&](uint64_t object) {
        return (object + 1) * objectBytes <= read.range.low ||
               object * objectBytes >= read.range.high;
      });
    });
    if (!useful)
      continue;
    llvm::MDBuilder metadata(module->getContext());
    auto *domain = metadata.createAnonymousAliasScopeDomain("gfx936 LDS objects");
    llvm::DenseMap<uint64_t, llvm::MDNode *> scopes;
    llvm::DenseMap<uint64_t, llvm::MDNode *> groupScopes;
    for (auto object : objects) {
      uint64_t leader = representatives.count(object)
                            ? representatives.lookup(object) : object;
      auto *&scope = groupScopes[leader];
      if (!scope)
        scope = metadata.createAnonymousAliasScope(
            domain, "LDS group led by bytes " +
                        std::to_string(leader * objectBytes));
      scopes[object] = scope;
    }
    auto *localScope = metadata.createAnonymousAliasScope(domain, "LDS reads");
    auto *localList = llvm::MDNode::get(module->getContext(), {localScope});
    for (auto &copy : copies) {
      auto *scope = scopes.lookup(copy.range.low / objectBytes);
      copy.inst->setMetadata(llvm::LLVMContext::MD_alias_scope,
                             llvm::MDNode::get(module->getContext(), {scope}));
      ++changed;
    }
    for (auto &read : reads) {
      llvm::SmallVector<llvm::Metadata *, 8> disjoint;
      for (auto object : objects)
        if ((object + 1) * objectBytes <= read.range.low ||
            object * objectBytes >= read.range.high)
          if (!llvm::is_contained(disjoint, scopes.lookup(object)))
            disjoint.push_back(scopes.lookup(object));
      if (disjoint.empty())
        continue;
      read.inst->setMetadata(llvm::LLVMContext::MD_alias_scope, localList);
      read.inst->setMetadata(llvm::LLVMContext::MD_noalias,
                             llvm::MDNode::get(module->getContext(), disjoint));
      ++changed;
    }
  }
  return changed;
}

} // namespace


// DTK groups LDS DMA writes by AAMDNodes but keeps one pointer per group.
// BasicAA can disambiguate a read from that representative even when it reads
// another fragment of the same group. The wait pass then falls back to its
// general LDS slot and unnecessarily drains future buffers. Forget only the
// first DMA pointer's provenance, retaining the scoped-noalias contract and
// BasicAA everywhere else. A tied empty asm preserves all 32 pointer bits;
// it does not assert non-aliasing or remove any explicit synchronization.
static unsigned preserveLdsDmaAliasScopes(llvm::Module *module,
                                          const std::string &arch) {
  if (arch != "gfx936" || module->getDataLayout().getPointerSizeInBits(3) != 32)
    return 0;
  auto *i32 = llvm::Type::getInt32Ty(module->getContext());
  auto *type = llvm::FunctionType::get(i32, {i32}, false);
  auto *identity = llvm::InlineAsm::get(type, "", "=v,0", false);
  unsigned changed = 0;
  for (auto &fn : *module) {
    llvm::SmallVector<llvm::AAMDNodes, 8> seen;
    for (auto &block : fn)
      for (auto &inst : block) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
        auto *callee = call ? call->getCalledFunction() : nullptr;
        if (!callee || call->getNumOperandBundles())
          continue;
        auto name = callee->getName();
        bool raw = (name == "llvm.amdgcn.raw.ptr.buffer.load.lds" ||
                    name.starts_with("llvm.amdgcn.raw.ptr.buffer.load.lds.")) &&
                   call->arg_size() == 7;
        bool structured =
            (name == "llvm.amdgcn.struct.ptr.buffer.load.lds" ||
             name.starts_with("llvm.amdgcn.struct.ptr.buffer.load.lds.")) &&
            call->arg_size() == 8;
        // Both typed DMA forms carry the exact same 32-bit LDS operand.
        // Preserve the existing scope contract and all raw-path behavior.
        if (!raw && !structured)
          continue;
        auto *ptr = call->getArgOperand(1);
        auto *ptrType = llvm::dyn_cast<llvm::PointerType>(ptr->getType());
        auto aa = call->getAAMetadata();
        if (!ptrType || ptrType->getAddressSpace() != 3 || !aa.Scope ||
            std::find(seen.begin(), seen.end(), aa) != seen.end())
          continue;
        seen.push_back(aa);
        // Avoid adding another identity if this module is visited again.
        if (auto *cast = llvm::dyn_cast<llvm::IntToPtrInst>(ptr))
          if (auto *opaque = llvm::dyn_cast<llvm::CallInst>(cast->getOperand(0)))
            if (auto *code = llvm::dyn_cast<llvm::InlineAsm>(opaque->getCalledOperand()))
              if (code->getAsmString().empty() &&
                  code->getConstraintString() == "=v,0" &&
                  !code->hasSideEffects() && !opaque->getNumOperandBundles() &&
                  opaque->arg_size() == 1 && opaque->getType() == i32 &&
                  opaque->getArgOperand(0)->getType() == i32)
                continue;
        llvm::IRBuilder<> builder(call);
        builder.SetCurrentDebugLocation(llvm::DebugLoc());
        auto *address = builder.CreatePtrToInt(ptr, i32, "lds.dma.alias.address");
        auto *opaque = builder.CreateCall(identity, {address}, "lds.dma.alias.opaque");
        call->setArgOperand(1, builder.CreateIntToPtr(opaque, ptrType, "lds.dma.alias.ptr"));
        ++changed;
      }
  }
  return changed;
}

void init_triton_hcu(py::module &&m) {
  m.def("create_local_memory_wait", [](mlir::OpBuilder &builder,
                                        mlir::Location loc) {
    // Reuse the target counter operation so native wait insertion recognizes
    // the completed LDS responses; an opaque asm leaves redundant waits.
    builder.create<mlir::triton::hcugpu::MemoryCounterWaitOp>(
        loc, /*load=*/nullptr, /*store=*/nullptr,
        /*ds=*/builder.getI32IntegerAttr(0));
  });
  m.def("has_warp_specialize", [](mlir::ModuleOp mod) {
    bool found = false;
    mod.walk([&](mlir::triton::gpu::WarpSpecializeOp) { found = true; });
    return found;
  });
  m.def("has_cta_barrier", [](mlir::ModuleOp mod) {
    bool found = false;
    mod.walk([&](mlir::triton::gpu::LocalBarrierOp op) {
      found |= op->hasAttr("hcu.cta_scope");
    });
    return found;
  });
  m.def("create_cta_barrier", [](mlir::OpBuilder &builder, mlir::Location loc) {
    // GPU dialect's adjacent-barrier folding is invalid for independently
    // phased WS actors. The standard TTG local barrier retains each rendezvous.
    auto barrier = builder.create<mlir::triton::gpu::LocalBarrierOp>(loc);
    barrier->setAttr("hcu.cta_scope", builder.getUnitAttr());
  });
  m.def("create_warp_group_id", [](mlir::OpBuilder &builder, mlir::Location loc,
                                    int32_t size) -> mlir::Value {
    if (size <= 0 || !llvm::isPowerOf2_32(size))
      throw std::invalid_argument("warp group size must be a positive power of two");
    return builder.create<mlir::triton::hcugpu::WarpGroupIdOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(size)).getResult();
  });
  // Target scheduling boundary, independent of any shared-memory protocol.
  m.def("create_sched_barrier", [](mlir::OpBuilder &builder, mlir::Location loc,
                                    int32_t mask) {
    if (mask < 0)
      throw std::invalid_argument("scheduler barrier mask must be nonnegative");
    mlir::ROCDL::SchedBarrier::create(builder, loc, mask);
  }, py::arg("builder"), py::arg("loc"), py::arg("mask") = 0);
  using namespace mlir;
  namespace tt = mlir::triton;
  namespace tg = mlir::triton::gpu;
  auto scalarTypes = [](ValueRange values, unsigned width, unsigned count) {
    if (values.size() != count)
      throw std::invalid_argument("structured register coordinate/extent/stride count mismatch");
    for (Value value : values)
      if (!value || !value.getType().isInteger(width))
        throw std::invalid_argument("structured register coordinate/extent/stride has wrong scalar type");
  };
  m.def("create_structured_register_load",[scalarTypes](OpBuilder &b,Location loc,Value base,
      Type resultType,const std::vector<Value> &shape,
      const std::vector<Value> &strides,const std::vector<Value> &indices)->Value {
    auto result=dyn_cast<RankedTensorType>(resultType);
    auto ptr=dyn_cast<tt::PointerType>(base.getType());
    if(!result || result.getRank()!=2 ||
       !(result.getElementType().isF16() || result.getElementType().isBF16()) ||
       !isa<tg::DistributedEncodingTrait>(result.getEncoding()) || !ptr ||
       ptr.getAddressSpace()!=1 || ptr.getPointeeType()!=result.getElementType())
      throw std::invalid_argument("structured register load needs matching AS1 f16/bf16 pointer and rank-two distributed result");
    scalarTypes(shape,32,2);scalarTypes(strides,64,2);scalarTypes(indices,32,3);
    SmallVector<int32_t> tile;
    for(int64_t d:result.getShape()) {
      if(d<1 || d>INT32_MAX) throw std::invalid_argument("invalid structured register tile");
      tile.push_back(int32_t(d));
    }
    auto op=b.create<tt::MatrixLoadOp>(loc,resultType,base,shape,strides,tile,
        indices,ArrayRef<int32_t>{},tt::CacheModifier::NONE,
        tt::EvictionPolicy::NORMAL,false);
    op->setAttr("hcu.structured_register",b.getUnitAttr());
    return op.getResult();
  });
  // Reuse MLIR's standard LLVM loop metadata, as other backends do. Preserve
  // other loop annotations; explicit LLVM unroll hints take precedence.
  m.def("disable_llvm_loop_unroll", [](mlir::ModuleOp mod) {
    auto *ctx = mod.getContext();
    mod.walk([&](mlir::scf::ForOp loop) {
      auto old = loop->getAttrOfType<mlir::LLVM::LoopAnnotationAttr>(
          "llvm.loop_annotation");
      if (old && old.getUnroll())
        return;
      if (!old)
        old = mlir::LLVM::LoopAnnotationAttr::get(
            ctx, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {});
      auto unroll = mlir::LLVM::LoopUnrollAttr::get(
          ctx, mlir::BoolAttr::get(ctx, true), {}, {}, {}, {}, {}, {});
      auto annotation = mlir::LLVM::LoopAnnotationAttr::get(
          ctx, old.getDisableNonforced(), old.getVectorize(),
          old.getInterleave(), unroll, old.getUnrollAndJam(), old.getLicm(),
          old.getDistribute(), old.getPipeline(), old.getPeeled(),
          old.getUnswitch(), old.getMustProgress(), old.getIsVectorized(),
          old.getStartLoc(), old.getEndLoc(), old.getParallelAccesses());
      loop->setAttr("llvm.loop_annotation", annotation);
    });
  });


  m.def("preserve_buffer_load_byte_offsets", &preserveBufferLoadByteOffsets);
  m.def("preserve_lds_barrier_order", &preserveLdsBarrierOrder);
  m.def("strip_synchronous_barrier_markers", &stripSynchronousBarrierMarkers);
  m.def("lower_tagged_local_barriers", &lowerTaggedLocalBarriers);
  m.def("annotate_lds_object_scopes", &annotateLdsObjectScopes);
  m.def("preserve_lds_dma_alias_scopes", &preserveLdsDmaAliasScopes);
  m.def("prefer_ready_shared_loads", &preferReadySharedLoads);
  m.def("interleave_global_lds_reads",
        [](llvm::Module *module, const std::string &arch, bool addFences) {
          auto s = interleaveGlobalLdsReads(module, arch, addFences);
          py::dict d;
          d["transformed_windows"] = s.windows;
          d["moved_global_reads"] = s.globalReads;
          d["moved_pure_dependents"] = s.pureDependents;
          d["split_lds_reads"] = s.ldsReads;
          d["sched_barriers_added"] = s.fences;
          d["rejected_dependencies"] = s.rejectedDependencies;
          d["invalid_functions"] = s.invalidFunctions;
          d["iglp_functions"] = s.iglpFunctions;
          return d;
        });
  m.def("tuple_register_fences",
        [](llvm::Module *module, const std::string &arch) {
          auto stats = tupleRegisterFences(module, arch);
          py::dict result;
          result["sites"] = stats.sites;
          result["scalar_lanes"] = stats.scalarLanes;
          result["rejected"] = stats.rejected;
          return result;
        });
  m.doc() = "Python bindings to the HCU Triton backend";

  auto passes = m.def_submodule("passes");
  init_triton_hcu_passes_ttgpuir(passes.def_submodule("ttgpuir"));

  m.attr("TARGET_TRIPLE") = hcuTargetTriple;
  m.attr("CALLING_CONV_AMDGPU_KERNEL") =
      (unsigned)llvm::CallingConv::AMDGPU_KERNEL;

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::hcugpu::TritonHCUGPUDialect>();
    // registry.insert<mlir::ROCDL::ROCDLDialect>();
    mlir::registerROCDLDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("attach_target_triple", [](llvm::Module *module) {
    module->setTargetTriple(llvm::Triple(hcuTargetTriple));
  });

  // Set target architecture ISA version
  m.def("set_isa_version", [](llvm::Module *module, const std::string &arch) {
    llvm::AMDGPU::IsaVersion version = llvm::AMDGPU::getIsaVersion(arch);
    addControlConstant(module, "__oclc_ISA_version", /*bitwidth=*/32,
                       version.Major * 1000 + version.Minor * 100 +
                           version.Stepping);
  });

  // Set boolean control constant
  m.def("set_bool_control_constant",
        [](llvm::Module *module, const std::string &name, bool enable) {
          addControlConstant(module, name.c_str(), /*bitwidth=*/8, enable);
        });

  // Set code object ABI version
  m.def("set_abi_version", [](llvm::Module *module, int version) {
    // Inject the control constant into the LLVM module so that device libraries
    // linked against module can resolve their references to it.
    llvm::Type *i32Ty = llvm::Type::getInt32Ty(module->getContext());
    llvm::GlobalVariable *abi = new llvm::GlobalVariable(
        *module, i32Ty, /*isConstant=*/true,
        llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
        llvm::ConstantInt::get(i32Ty, version), "__oclc_ABI_version", nullptr,
        llvm::GlobalValue::ThreadLocalMode::NotThreadLocal, 4);
    abi->setVisibility(llvm::GlobalValue::VisibilityTypes::ProtectedVisibility);
    abi->setAlignment(llvm::MaybeAlign(4));
    abi->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);

    // Also attach the control attribute on the LLVM module. This is also needed
    // in addition to the above for various transformations to know what code
    // object version we are targeting at.
    module->addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version",
                          version);
  });

  m.def("cleanup_bitcode_metadata", [](llvm::Module *module) {
    // We can have Clang version metadata from device libraries linked in. We
    // don't care about them so drop them.
    if (auto *ident = module->getNamedMetadata("llvm.ident"))
      module->eraseNamedMetadata(ident);
    // Also various OpenCL version details.
    if (auto *openclVersion = module->getNamedMetadata("opencl.ocl.version"))
      module->eraseNamedMetadata(openclVersion);
  });

  m.def("disable_print_inline", [](llvm::Module *module) {
    // List of functions name prefixes we want to forbid inline.
    std::array<const char *, 2> prefixes = {"__ockl_fprintf", "__ockl_printf"};

    for (llvm::Function &f : module->functions()) {
      if (!f.hasName())
        continue;
      llvm::StringRef name = f.getName();

      auto isNamePrefixed = [&name](const char *prefix) {
        return name.starts_with(prefix);
      };

      if (llvm::any_of(prefixes, isNamePrefixed))
        f.addFnAttr(llvm::Attribute::NoInline);
    }
  });

  m.def(
      "assemble_amdgcn",
      [](const std::string &assembly, const std::string &arch,
         const std::string &features) {
        std::string error;

        llvm::Triple triple(hcuTargetTriple);
        const llvm::Target *target =
            llvm::TargetRegistry::lookupTarget(triple, error);
        if (!target)
          throw std::runtime_error("target lookup error: " + error);

        llvm::SourceMgr srcMgr;
        srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(assembly),
                                  llvm::SMLoc());

        const llvm::MCTargetOptions mcOptions;
        std::unique_ptr<llvm::MCRegisterInfo> mri(
            target->createMCRegInfo(triple));
        std::unique_ptr<llvm::MCAsmInfo> mai(
            target->createMCAsmInfo(*mri, triple, mcOptions));
        std::unique_ptr<llvm::MCSubtargetInfo> sti(
            target->createMCSubtargetInfo(triple, arch, features));

        llvm::MCContext ctx(triple, mai.get(), mri.get(), sti.get(), &srcMgr,
                            &mcOptions);
        std::unique_ptr<llvm::MCObjectFileInfo> mofi(
            target->createMCObjectFileInfo(ctx, /*PIC=*/false,
                                           /*LargeCodeModel=*/false));
        ctx.setObjectFileInfo(mofi.get());

        llvm::SmallString<128> cwd;
        if (!llvm::sys::fs::current_path(cwd))
          ctx.setCompilationDir(cwd);

        llvm::SmallVector<char, 0> result;
        llvm::raw_svector_ostream svos(result);

        std::unique_ptr<llvm::MCStreamer> mcStreamer;
        std::unique_ptr<llvm::MCInstrInfo> mcii(target->createMCInstrInfo());

        std::unique_ptr<llvm::MCCodeEmitter> ce(
            target->createMCCodeEmitter(*mcii, ctx));
        std::unique_ptr<llvm::MCAsmBackend> mab(
            target->createMCAsmBackend(*sti, *mri, mcOptions));
        std::unique_ptr<llvm::MCObjectWriter> ow(mab->createObjectWriter(svos));
        mcStreamer.reset(target->createMCObjectStreamer(
            triple, ctx, std::move(mab), std::move(ow), std::move(ce), *sti));

        std::unique_ptr<llvm::MCAsmParser> parser(
            createMCAsmParser(srcMgr, ctx, *mcStreamer, *mai));
        std::unique_ptr<llvm::MCTargetAsmParser> tap(
            target->createMCAsmParser(*sti, *parser, *mcii, mcOptions));
        if (!tap)
          throw std::runtime_error("assembler initializtion error");

        parser->setTargetParser(*tap);
        parser->Run(/*NoInitialTextSection=*/false);

        return py::bytes(std::string(result.begin(), result.end()));
      },
      py::return_value_policy::take_ownership);

  m.def("has_architected_sgprs", [](const std::string &arch) {
    std::string error;
    llvm::Triple triple(hcuTargetTriple);
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target)
      throw std::runtime_error("target lookup error: " + error);
    std::unique_ptr<llvm::MCSubtargetInfo> sti(
        target->createMCSubtargetInfo(triple, arch, ""));
    return sti->checkFeatures("+architected-sgprs");
  });

  m.def("supports_multi_cta_launch", [](const std::string &arch) {
    return mlir::triton::HCU::TargetInfo(arch).supportsMultiCTALaunch();
  });

  m.def("need_extern_lib", [](llvm::Module *module, const std::string &lib) {
    for (llvm::Function &f : module->functions()) {
      if (f.hasExternalLinkage() && f.hasName() && !f.hasExactDefinition()) {
        llvm::StringRef funcName = f.getName();
        // The rule for linking the extern lib:
        //    if the function name includes ocml or ockl, link
        //    ocml or ockl accordingly.
        if (funcName.contains(lib))
          return true;
        if (funcName.contains("__nv_")) {
          std::stringstream message;
          message << "Implicit conversion of CUDA " << funcName.str()
                  << " device function has been dropped; "
                  << "please, update your source program to use "
                     "triton.language.extra.<op> "
                  << "to replace triton.language.extra.cuda.<op>";
          throw std::runtime_error(message.str());
        }
      }
    }
    return false;
  });

  m.def("set_all_fn_arg_inreg", [](llvm::Function *fn) {
    for (llvm::Argument &arg : fn->args()) {
      // Check for incompatible attributes.
      if (arg.hasByRefAttr() || arg.hasNestAttr())
        continue;
      arg.addAttr(llvm::Attribute::InReg);
    }
  });

  m.def("link_hsaco",
        [](const std::string &inPath, const std::string &outPath) {
          if (auto errString = lldInvoke(inPath.c_str(), outPath.c_str()))
            throw std::runtime_error("LLD failed to link hsaco source " +
                                     inPath + " into object file " + outPath +
                                     " because " + errString.value());
        });

  m.def("add_scalarize_packed_fops_llvm_pass", [](llvm::Function *fn) {
    mlir::triton::HCU::runScalarizePackedFOpsPass(*fn);
  });

  auto hipBlas = m.def_submodule("hipblas");
  py::class_<HipblasLtInstance>(hipBlas, "HipblasLt")
      .def(py::init<>([&](py::object &workspace) {
        auto wrk_ptr = workspace.attr("data_ptr")().cast<uint64_t>();
        auto wrk_size = workspace.attr("numel")().cast<size_t>() *
                        workspace.attr("element_size")().cast<size_t>();
        return new HipblasLtInstance(wrk_ptr, wrk_size);
      }))
      .def("matmul",
           [](HipblasLtInstance &self, py::object &A, py::object &B,
              py::object &C) {
             auto A_ptr = A.attr("data_ptr")().cast<uint64_t>();
             auto B_ptr = B.attr("data_ptr")().cast<uint64_t>();
             auto C_ptr = C.attr("data_ptr")().cast<uint64_t>();
             auto init = initialize_hipblas_op(A, B, C, std::nullopt);
             self.matmul(init.m, init.n, init.k, A_ptr, B_ptr, C_ptr,
                         init.dtype, init.out_dtype);
           })
      .def("gemm", [](HipblasLtInstance &self, py::object &A, py::object &B,
                      py::object &C, py::object &D, float alpha, float beta) {
        auto A_ptr = A.attr("data_ptr")().cast<uint64_t>();
        auto B_ptr = B.attr("data_ptr")().cast<uint64_t>();
        auto C_ptr = C.attr("data_ptr")().cast<uint64_t>();
        auto D_ptr = D.attr("data_ptr")().cast<uint64_t>();
        auto init = initialize_hipblas_op(A, B, D, C);
        self.gemm(init.m, init.n, init.k, A_ptr, B_ptr, C_ptr, D_ptr,
                  init.dtype, init.out_dtype, alpha, beta);
      });
}
