#include "mlir/IR/BuiltinOps.h" // mlir::ModuleOp
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "triton/Target/LLVMIR/IntrinsicAttrTable.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolSize.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Pass.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#if __has_include("llvm/Passes/PassPlugin.h")
#include "llvm/Passes/PassPlugin.h"
#else
// LLVM >= 22.1.x moved PassPlugin.h under llvm/Plugins/.
#include "llvm/Plugins/PassPlugin.h"
#endif
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"
// XTDK LLVM22 ships the private XPU printf/assert lowering pass; public
// LLVM22 does not.  The XPU backends lower asserts/printfs
// themselves (TargetInfo::assertFail is a no-op and printf goes through the
// printf runtime helpers), so the pass is compiled out for public
// builds rather than reimplemented.
#if __has_include("llvm/Transforms/Utils/XPULowerPrintfAssert.h")
#define TRITON_HAVE_XPU_PRINTF_ASSERT 1
#include "llvm/Transforms/Utils/XPULowerPrintfAssert.h"
#else
#define TRITON_HAVE_XPU_PRINTF_ASSERT 0
#endif
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include <csignal>
#include <cstdio>
#include <cstring>
#include <memory>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

#define DEFAULTLOCALLIMIT 8000

namespace llvm {
struct BreakStructPhiNodesPass : PassInfoMixin<BreakStructPhiNodesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "BreakStructPhiNodesPass"; }
};
} // namespace llvm

using namespace llvm;

static Expected<std::unique_ptr<object::ObjectFile>>
initObjectFileByMemory(unsigned char *ObjArray, uint64_t ObjLen) {
  if (!ObjArray) {
    return errorCodeToError(object::object_error::section_stripped);
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr = MemoryBuffer::getMemBuffer(
      StringRef((char *)ObjArray, ObjLen), ".debug", false);
  if (!FileOrErr) {
    llvm::report_fatal_error(errorCodeToError(FileOrErr.getError()));
  }

  std::unique_ptr<MemoryBuffer> Buffer = std::move(FileOrErr.get());
  Expected<std::unique_ptr<object::ObjectFile>> ObjOrErr =
      object::ObjectFile::createObjectFile(Buffer->getMemBufferRef());
  if (!ObjOrErr) {
    llvm::report_fatal_error(ObjOrErr.takeError());
  }

  return ObjOrErr;
}

// True for the XPU-family target triples (xpu3-, xpu4-, ...).  Public-LLVM
// builds do not link those XTDK-private targets, so several LLVM lookups have
// to fall back for them -- but only for them, so that a genuine lookup failure
// on a target we *do* link (x86/nvptx/amdgpu) still surfaces as an error.
static bool isXpuFamilyTriple(const llvm::Triple &triple) {
  llvm::StringRef arch = triple.getArchName();
  return arch.starts_with("xpu") || arch.starts_with("xcn");
}

// `Margin` bytes are held back from the budget: a caller that wants headroom
// asks with a margin, and the hard limit is the same call with Margin = 0.
static bool isElfStackSizeOOB(std::string ElfObject, uint32_t Margin = 0) {
  uint32_t StackSizeLimit = DEFAULTLOCALLIMIT;
  std::string StackSizeLimitStr =
      mlir::triton::tools::getStrEnv("TRITON_TUNE_BUFFER_LM_SIZE");
  if (!StackSizeLimitStr.empty()) {
    llvm::StringRef StackSizeLimitSRf = StackSizeLimitStr;
    if (StackSizeLimitSRf.getAsInteger(10, StackSizeLimit)) {
      llvm::report_fatal_error(
          "Invalid value for TRITON_TUNE_BUFFER_LM_SIZE: " + StackSizeLimitSRf);
    }
  }

  Expected<std::unique_ptr<object::ObjectFile>> ObjFile =
      initObjectFileByMemory(
          (unsigned char *)const_cast<char *>(ElfObject.data()),
          ElfObject.size());
  if (!ObjFile) {
    llvm::report_fatal_error(ObjFile.takeError());
  }

  uint32_t StackSize = 0;

  std::vector<std::pair<object::SymbolRef, uint64_t>> SymbolSizes =
      object::computeSymbolSizes(*ObjFile.get());
  for (std::pair<object::SymbolRef, uint64_t> &SymbolSize : SymbolSizes) {
    const object::SymbolRef &Symbol = SymbolSize.first;
    Expected<StringRef> NameOrErr = Symbol.getName();
    if (!NameOrErr) {
      errorToErrorCode(NameOrErr.takeError());
      continue;
    }
    StringRef Name = *NameOrErr;
    if (!Name.contains("KERNEL_STACK_SIZE")) {
      continue;
    }

    Expected<llvm::object::section_iterator> SymSectionOrErr =
        Symbol.getSection();
    if (!SymSectionOrErr) {
      llvm::report_fatal_error(SymSectionOrErr.takeError());
    }

    Expected<StringRef> ContentsOrErr = SymSectionOrErr.get()->getContents();
    if (!ContentsOrErr) {
      llvm::report_fatal_error(ContentsOrErr.takeError());
    }

    if (SymbolSize.second != 4) {
      llvm::report_fatal_error("Symbol Size Error");
    }
    // The XPU.KERNEL_STACK_SIZE section is 4 bytes in LLVM 19 XTDK llc objects
    // and 8 bytes in XTDK LLVM 22 ones; the u32 stack size is the first 4
    // bytes in both cases.  The public-LLVM22 libtriton parses the LLVM 19
    // backend's objects, so accept either size.
    uint64_t SectionSize = SymSectionOrErr.get()->getSize();
    if (SectionSize != 4 && SectionSize != 8) {
      llvm::report_fatal_error("Section Size Error");
    }
    if (ContentsOrErr->size() < sizeof(uint32_t)) {
      llvm::report_fatal_error("KERNEL_STACK_SIZE section contents too short");
    }

    // memcpy rather than a pointer cast: the section contents are not
    // guaranteed to be 4-byte aligned in the mapped object.
    std::memcpy(&StackSize, ContentsOrErr->data(), sizeof(StackSize));
    break;
  }
  return StackSize + Margin > StackSizeLimit;
}

static void applyXpuErrorLmSizeEnv() {
  uint32_t LMSizeLimit = -1;
  std::string LMSizeLimitStr =
      mlir::triton::tools::getStrEnv("LLVM_ERROR_LM_SIZE");
  if (!LMSizeLimitStr.empty()) {
    llvm::StringRef LMSizeLimitSRf = LMSizeLimitStr;
    if (LMSizeLimitSRf.getAsInteger(10, LMSizeLimit)) {
      llvm::report_fatal_error("Invalid value for LLVM_ERROR_LM_SIZE: " +
                               LMSizeLimitSRf);
    }
  }
  // The container type returned by getRegisteredOptions() differs between
  // public LLVM (DenseMap-like) and XTDK (StringMap); use auto.
  auto optMap = llvm::cl::getRegisteredOptions();
  auto optIt = optMap.find("xpu-error-lm-size");
  if (optIt != optMap.end()) {
    llvm::cl::opt<uint64_t> *optPtr =
        static_cast<llvm::cl::opt<uint64_t> *>(optIt->second);
    *optPtr = LMSizeLimit;
  }
}

std::unique_ptr<TargetMachine>
createTargetMachine(llvm::Module *module, std::string proc,
                    bool enable_fp_fusion, const std::string &features) {
  std::string error;
  const llvm::Triple &triple = module->getTargetTriple();
  auto target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target && isXpuFamilyTriple(triple)) {
    // Public-LLVM builds: the XPU-family targets are XTDK-private and unlinked,
    // so their triples cannot resolve.  The TargetMachine here only feeds the
    // optimization pipeline below; the actual codegen runs through the staged
    // LLVM 19 toolchain (see python/triton/backends/llvm19_toolchain.py).
    // Substitute a host machine so optimization can proceed.
    //
    // Caveat: the optimizer then sees x86's TargetTransformInfo, so its
    // vectorization/unrolling cost model is not the XPU one.  Accepted
    // deliberately -- the alternative is no O3 at all, since llc 19 runs at
    // -O0.  The XPU-specific decisions that do matter are made explicitly
    // (SLPVectorization off, InferAddressSpaces forced, see optimize_module).
    //
    // The module's functions may still carry GPU-style
    // "target-features" attributes (e.g. from libdevice bitcode: +sm_75,
    // +16-bit-insts); the X86 subtarget rejects those ("64-bit code requested
    // on a subtarget that doesn't support it"), so scrub them.  The LLVM 19
    // backend re-derives features from its own target tables anyway.
    for (auto &fn : *module) {
      if (fn.hasFnAttribute("target-features"))
        fn.removeFnAttr("target-features");
      if (fn.hasFnAttribute("target-cpu"))
        fn.removeFnAttr("target-cpu");
    }
    llvm::Triple fallback("x86_64-unknown-linux-gnu");
    target = llvm::TargetRegistry::lookupTarget(fallback, error);
    if (!target)
      throw std::runtime_error("target lookup error (x86 fallback for " +
                               triple.str() + "): " + error);
    llvm::TargetOptions fopt;
    std::unique_ptr<llvm::TargetMachine> fmachine{target->createTargetMachine(
        fallback, "generic", "", fopt, llvm::Reloc::PIC_, std::nullopt,
        llvm::CodeGenOptLevel::Aggressive)};
    return fmachine;
  }
  if (!target)
    throw std::runtime_error("target lookup error: " + error);
  llvm::TargetOptions opt;
  bool disableLLVMOpt = mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  if (enable_fp_fusion)
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
#if defined(TRITON_HAVE_XTDK_TUNING_OPTIONS)
  opt.UnsafeFPMath = false;
#endif
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  opt.TrapUnreachable = true;
  opt.MCOptions.AsmVerbose = true;
  opt.MCOptions.PreserveAsmComments = true;
  std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
      module->getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
      std::nullopt,
      disableLLVMOpt ? llvm::CodeGenOptLevel::None
                     : llvm::CodeGenOptLevel::Aggressive)};
  return machine;
}

void dumpSchedulingDAG(llvm::Module &module, const std::string &triple,
                       const std::string &proc, const std::string &features,
                       const std::vector<std::string> &flags,
                       bool enable_fp_fusion, const std::string &dumpFileId) {
  using namespace mlir;

  // Check if we should dump sched DAG
  std::string dumpMirBase = triton::tools::getStrEnv("TRITON_DUMP_MIR");
  bool dumpMir = !dumpMirBase.empty();
  if (!dumpMir) {
    return;
  }

  // options
  auto options = llvm::cl::getRegisteredOptions();
  for (std::string flag : flags) {
    auto *shortPtr = static_cast<llvm::cl::opt<bool> *>(options[flag]);
    assert(shortPtr);
    shortPtr->setValue(true);
  }
  bool disableLLVMOpt = triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  if (!disableLLVMOpt) {
    // Check to see if we are passing a list of flags to disable optimizations.
    auto flagList = triton::tools::getStrEnv("DISABLE_LLVM_OPT");
    if (!flagList.empty()) {
      llvm::SmallVector<StringRef, 3> split;
      StringRef(flagList.c_str()).split(split, ',');
      for (auto flag : split) {
        auto optIt = options.find(flag);
        if (optIt != options.end()) {
          auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
          *optPtr = true;
        }
      }
    }
  }

  // inline everything
  for (llvm::Function &f : module.functions())
    if (!f.hasFnAttribute(llvm::Attribute::NoInline))
      f.addFnAttr(llvm::Attribute::AlwaysInline);
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createAlwaysInlinerLegacyPass());
  pm.add(llvm::createVerifierPass());

  pm.run(module);

  // create machine
  module.setTargetTriple(Triple(triple));
  auto machine = createTargetMachine(&module, proc, enable_fp_fusion, features);
  // set data layout
  module.setDataLayout(machine->createDataLayout());

  int saved_stderr_fd = -1;
  std::string dumpFilename = dumpMirBase + "/" + dumpFileId + ".txt";

  // Save and set stop-after
  std::string originalStopAfter;
  auto stopAfterOpt = options.find("stop-after");
  if (stopAfterOpt != options.end()) {
    auto *optPtr =
        static_cast<llvm::cl::opt<std::string> *>(stopAfterOpt->second);
    originalStopAfter = optPtr->getValue();
    optPtr->setValue("machine-scheduler");
  }

  // Enable misched-print-dags for DAG
  auto mischedPrintOpt = options.find("misched-print-dags");
  if (mischedPrintOpt != options.end()) {
    auto *optPtr = static_cast<llvm::cl::opt<bool> *>(mischedPrintOpt->second);
    optPtr->setValue(true);
  }

  // Save original stderr file descriptor
  saved_stderr_fd = dup(fileno(stderr));

  // Redirect stderr to append to dump file
  FILE *redirected = freopen(dumpFilename.c_str(), "a", stderr);
  if (!redirected) {
    llvm::errs() << "Warning: Failed to redirect stderr to " << dumpFilename
                 << "\n";
  }

  // emit machine code
  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager pass;
    // emit
    machine->addPassesToEmitFile(pass, pstream, nullptr,
                                 llvm::CodeGenFileType::AssemblyFile);
    pass.run(module);
  }

  // Restore stderr and reset options
  fflush(stderr);
  if (saved_stderr_fd != -1) {
    dup2(saved_stderr_fd, fileno(stderr));
    close(saved_stderr_fd);
    clearerr(stderr);
  }

  if (stopAfterOpt != options.end()) {
    auto *optPtr =
        static_cast<llvm::cl::opt<std::string> *>(stopAfterOpt->second);
    optPtr->setValue(originalStopAfter);
  }

  if (mischedPrintOpt != options.end()) {
    auto *optPtr = static_cast<llvm::cl::opt<bool> *>(mischedPrintOpt->second);
    optPtr->setValue(false);
  }

  llvm::errs() << "MIR and DAG dumped to: " << dumpFilename << "\n";
}

std::string
translateLLVMIRToMIR(llvm::Module &module, const std::string &triple,
                     const std::string &proc, const std::string &features,
                     const std::vector<std::string> &flags,
                     bool enable_fp_fusion, const std::string &dumpFileId) {
  using namespace mlir;

  // Check if we should dump MIR
  std::string dumpMirBase = triton::tools::getStrEnv("TRITON_DUMP_MIR");
  bool dumpMir = !dumpMirBase.empty();
  if (!dumpMir) {
    return "";
  }

  // options
  auto options = llvm::cl::getRegisteredOptions();
  for (std::string flag : flags) {
    auto *shortPtr = static_cast<llvm::cl::opt<bool> *>(options[flag]);
    assert(shortPtr);
    shortPtr->setValue(true);
  }
  bool disableLLVMOpt = triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  if (!disableLLVMOpt) {
    // Check to see if we are passing a list of flags to disable optimizations.
    auto flagList = triton::tools::getStrEnv("DISABLE_LLVM_OPT");
    if (!flagList.empty()) {
      llvm::SmallVector<StringRef, 3> split;
      StringRef(flagList.c_str()).split(split, ',');
      for (auto flag : split) {
        auto optIt = options.find(flag);
        if (optIt != options.end()) {
          auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
          *optPtr = true;
        }
      }
    }
  }

  // Save and set stop-before if needed (for MIR output or custom stop point)
  std::string originalStopBefore;
  auto stopBeforeOpt = options.find("stop-before");
  if (stopBeforeOpt != options.end()) {
    auto *optPtr =
        static_cast<llvm::cl::opt<std::string> *>(stopBeforeOpt->second);
    originalStopBefore = optPtr->getValue();
    optPtr->setValue("machine-scheduler");
  }

  // inline everything
  for (llvm::Function &f : module.functions())
    if (!f.hasFnAttribute(llvm::Attribute::NoInline))
      f.addFnAttr(llvm::Attribute::AlwaysInline);
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createAlwaysInlinerLegacyPass());
  pm.add(llvm::createVerifierPass());

  pm.run(module);

  // create machine
  module.setTargetTriple(Triple(triple));
  auto machine = createTargetMachine(&module, proc, enable_fp_fusion, features);
  // set data layout
  module.setDataLayout(machine->createDataLayout());

  // emit machine code
  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager pass;
    // emit
    machine->addPassesToEmitFile(pass, pstream, nullptr,
                                 llvm::CodeGenFileType::AssemblyFile);
    pass.run(module);
  }

  if (stopBeforeOpt != options.end()) {
    auto *optPtr =
        static_cast<llvm::cl::opt<std::string> *>(stopBeforeOpt->second);
    optPtr->setValue(originalStopBefore);
  }

  std::string dumpFilename = dumpMirBase + "/" + dumpFileId + ".txt";
  {
    std::error_code EC;
    llvm::raw_fd_ostream outFile(dumpFilename, EC, llvm::sys::fs::OF_None);
    if (EC) {
      llvm::errs() << "Error opening file " << dumpFilename << ": "
                   << EC.message() << "\n";
    } else {
      outFile << result;
      outFile << "---";
      outFile << "\n========== SCHEDULING DAG ==========\n";
    }
  }

  return result;
}

std::string translateLLVMIRToASM(llvm::Module &module,
                                 const std::string &triple,
                                 const std::string &proc,
                                 const std::string &features,
                                 const std::vector<std::string> &flags,
                                 bool enable_fp_fusion, bool isObject) {
  using namespace mlir;
  // options
  auto options = llvm::cl::getRegisteredOptions();
  for (std::string flag : flags) {
    auto *shortPtr = static_cast<llvm::cl::opt<bool> *>(options[flag]);
    assert(shortPtr);
    shortPtr->setValue(true);
  }
#if !defined(TRITON_CONCEAL_IR) || (TRITON_CONCEAL_IR == 0)
  if (triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP")) {
    auto optIt = options.find("print-after-all");
    if (optIt != options.end()) {
      auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
      *optPtr = true;
    }
  }
#endif
  applyXpuErrorLmSizeEnv();
  bool disableLLVMOpt = triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  if (!disableLLVMOpt) {
    // Check to see if we are passing a list of flags to disable optimizations.
    auto flagList = triton::tools::getStrEnv("DISABLE_LLVM_OPT");
    if (!flagList.empty()) {
      llvm::SmallVector<StringRef, 3> split;
      StringRef(flagList.c_str()).split(split, ',');
      for (auto flag : split) {
        auto optIt = options.find(flag);
        if (optIt != options.end()) {
          auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
          *optPtr = true;
        }
      }
    }
  }

  // inline everything
  for (llvm::Function &f : module.functions())
    if (!f.hasFnAttribute(llvm::Attribute::NoInline))
      f.addFnAttr(llvm::Attribute::AlwaysInline);
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createAlwaysInlinerLegacyPass());
  pm.add(llvm::createVerifierPass());

  const bool enabledTiming = triton::tools::getBoolEnv("LLVM_ENABLE_TIMING");
  if (enabledTiming) {
    llvm::TimePassesIsEnabled = true;
    llvm::TimePassesPerRun = true;
  }

  pm.run(module);

  SmallString<0> timePassesStr;
  raw_svector_ostream reportStream(timePassesStr);

  if (enabledTiming) {
    reportAndResetTimings(&reportStream);
    llvm::dbgs() << reportStream.str();
    timePassesStr.clear();
  }

  // create machine
  module.setTargetTriple(Triple(triple));
  auto machine = createTargetMachine(&module, proc, enable_fp_fusion, features);
  // set data layout
  module.setDataLayout(machine->createDataLayout());
  // emit machine code
  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream pstream(stream);
    for (llvm::Function &f : module.functions())
      f.addFnAttr(llvm::Attribute::AlwaysInline);
    llvm::legacy::PassManager pass;
    // emit
    auto fileType = isObject ? llvm::CodeGenFileType::ObjectFile
                             : llvm::CodeGenFileType::AssemblyFile;
    machine->addPassesToEmitFile(pass, pstream, nullptr, fileType);
    pass.run(module);

    if (enabledTiming) {
      reportAndResetTimings(&reportStream);
      llvm::dbgs() << reportStream.str();
      timePassesStr.clear();
    }
  }
  return result;
}

using ret = py::return_value_policy;

// Expand constant-size llvm.memset calls into straight-line scalar stores.
//
// Rationale (see optimize_module): public-LLVM O3's MemCpyOptPass folds the
// SDNN buffer-counter init stores (CounterOpConversion emits
// `alloca [cacheNum x i32]` + per-element `store i32 -1`) into llvm.memset
// calls, which the LLVM 19 backend then lowers to byte loops.  On tiny shapes
// (grid=1 tile) that byte loop dominates the kernel prologue (~4x slowdown
// vs the XTDK pipeline, whose private MemCpyOptXPU=false keeps the stores
// scalar).  The XTDK knobs are unavailable in public LLVM, so undo the fold
// after the pipeline: constant-size, non-volatile memsets only, elements
// sized by the memset's alignment, value replicated bytewise.
static void expandConstMemSets(llvm::Module &M) {
  using namespace llvm;
  SmallVector<CallInst *, 8> MemSets;
  for (Function &F : M.functions())
    for (BasicBlock &BB : F)
      for (Instruction &I : llvm::make_early_inc_range(BB))
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (CI->getCalledFunction() &&
              CI->getCalledFunction()->getIntrinsicID() == Intrinsic::memset)
            MemSets.push_back(CI);

  for (CallInst *CI : MemSets) {
    if (CI->use_empty() == false) // memset returns void; defensive
      continue;
    auto *II = cast<MemSetInst>(CI);
    if (II->isVolatile())
      continue;
    ConstantInt *Len = dyn_cast<ConstantInt>(II->getLength());
    ConstantInt *Val = dyn_cast<ConstantInt>(II->getValue());
    if (!Len || !Val)
      continue; // dynamic size/value: leave for llc19
    uint64_t Size = Len->getZExtValue();
    if (Size == 0)
      continue;
    Align Al = II->getDestAlign().valueOrOne();
    Value *Dst = II->getDest();

    unsigned ElemBits = 8;
    if (Al >= Align(4) && Size % 4 == 0)
      ElemBits = 32;
    else if (Al >= Align(2) && Size % 2 == 0)
      ElemBits = 16;
    Type *ElemTy = Type::getIntNTy(M.getContext(), ElemBits);
    uint64_t Rep = Size / (ElemBits / 8);
    // Cap the expansion so pathological large memsets do not explode IR size.
    if (Rep > 4096)
      continue;
    APInt FillVal(ElemBits, 0);
    uint64_t B = (uint64_t)Val->getZExtValue() & 0xFF;
    for (unsigned i = 0; i < ElemBits / 8; ++i)
      FillVal = (FillVal << 8) | B;

    IRBuilder<> Bld(CI);
    for (uint64_t i = 0; i < Rep; ++i) {
      Value *Ptr =
          i == 0 ? Dst
                 : Bld.CreateConstInBoundsGEP1_32(ElemTy, Dst, (unsigned)i);
      Bld.CreateAlignedStore(ConstantInt::get(ElemTy, FillVal), Ptr, Al);
    }
    CI->eraseFromParent();
  }
}

using ret = py::return_value_policy;

void init_triton_llvm(py::module &&m) {

  py::class_<llvm::LLVMContext>(m, "context", py::module_local())
      .def(py::init<>());
  py::class_<llvm::SourceMgr>(m, "source_mgr", py::module_local())
      .def(py::init<>());

  py::class_<llvm::Module::FunctionListType>(m, "function_list")
      .def(
          "__iter__",
          [](llvm::Module::FunctionListType &s) {
            return py::make_iterator(s.begin(), s.end());
          },
          py::keep_alive<0, 1>());

  // Module Flag behavior. See
  // https://llvm.org/doxygen/classllvm_1_1Module.html#a0a5c55e12c97b80021330fe82b642293
  // for details.
  py::class_<llvm::Module::ModFlagBehavior>(m, "module_flag_behavior",
                                            py::module_local());
  m.attr("MODULE_FLAG_BEHAVIOR_ERROR") = llvm::Module::Error;
  m.attr("MODULE_FLAG_BEHAVIOR_WARNING") = llvm::Module::Warning;
  m.attr("MODULE_FLAG_BEHAVIOR_REQUIRE") = llvm::Module::Require;
  m.attr("MODULE_FLAG_BEHAVIOR_OVERRIDE") = llvm::Module::Override;
  m.attr("MODULE_FLAG_BEHAVIOR_APPEND") = llvm::Module::Append;
  m.attr("MODULE_FLAG_BEHAVIOR_APPEND_UNIQUE") = llvm::Module::AppendUnique;
  m.attr("MODULE_FLAG_BEHAVIOR_MAX") = llvm::Module::Max;
  m.attr("MODULE_FLAG_BEHAVIOR_MIN") = llvm::Module::Min;

  py::class_<llvm::Module>(m, "module", py::module_local())
      .def(
          "__str__",
          [](llvm::Module *self) {
            std::string str;
            llvm::raw_string_ostream os(str);
#if !defined(TRITON_CONCEAL_IR) || (TRITON_CONCEAL_IR == 0)
            os << *self;
#endif
            return os.str();
          },
          ret::take_ownership)
      .def(
          "_ir_for_lowering",
          [](llvm::Module *self) {
            std::string str;
            llvm::raw_string_ostream os(str);
            os << *self;
            return os.str();
          },
          ret::take_ownership)
      .def(
          "get_functions",
          [](llvm::Module *mod) -> llvm::Module::FunctionListType & {
            // Note: Backends assume that we are compiling exactly one kernel
            // (i.e. one function that's that's called by the CPU) and that it's
            // the first function in this list.
            return mod->getFunctionList();
          },
          ret::reference_internal)
      .def("add_flag",
           [](llvm::Module *mod, llvm::Module::ModFlagBehavior behavior,
              std::string &key, uint32_t value) {
             return mod->addModuleFlag(behavior, key, value);
           })
      .def("set_target_triple",
           [](llvm::Module *mod, const std::string &triple) {
             mod->setTargetTriple(llvm::Triple(triple));
           });

  py::class_<llvm::Function>(m, "function", py::module_local())
      .def_property_readonly(
          "name", [](llvm::Function *fn) { return fn->getName().str(); })
      .def("set_calling_conv", &llvm::Function::setCallingConv)
      .def("add_fn_attr", [](llvm::Function *fn, std::string &name,
                             std::string &val) { fn->addFnAttr(name, val); })
      .def("remove_fn_attr", [](llvm::Function *fn,
                                std::string &name) { fn->removeFnAttr(name); })
      .def("add_fn_asan_attr",
           [](llvm::Function *fn) {
             fn->addFnAttr(llvm::Attribute::SanitizeAddress);
           })
      .def("add_fn_target_feature",
           [](llvm::Function *fn, std::string &val) {
             fn->addFnAttr("target-features", val);
           })
      // Sets the nvvm.maxreg property on the given function.
      .def("set_nvvm_maxnreg",
           [](llvm::Function *fn, int maxnreg) {
             auto op = MDNode::get(
                 fn->getContext(),
                 {
                     ValueAsMetadata::get(fn),
                     MDString::get(fn->getContext(), "maxnreg"),
                     ConstantAsMetadata::get(ConstantInt::get(
                         Type::getInt32Ty(fn->getContext()), maxnreg)),
                 });
             fn->getParent()
                 ->getOrInsertNamedMetadata("nvvm.annotations")
                 ->addOperand(op);
           })
      // External functions that are definitions (i.e. not declarations) are
      // kernel functions.
      .def("is_declaration", &llvm::Function::isDeclaration)
      .def("is_external_linkage", [](llvm::Function *fn) {
        return fn->getLinkage() == llvm::GlobalValue::ExternalLinkage;
      });

  // optimization levels
  py::class_<llvm::OptimizationLevel>(m, "optimization_level",
                                      py::module_local());
  m.attr("OPTIMIZE_O0") = llvm::OptimizationLevel::O0;
  m.attr("OPTIMIZE_O1") = llvm::OptimizationLevel::O1;
  m.attr("OPTIMIZE_O2") = llvm::OptimizationLevel::O2;
  m.attr("OPTIMIZE_O3") = llvm::OptimizationLevel::O3;
  m.attr("OPTIMIZE_Os") = llvm::OptimizationLevel::Os;
  m.attr("OPTIMIZE_Oz") = llvm::OptimizationLevel::Oz;

  m.def(
      "to_module",
      [](mlir::ModuleOp &mod, llvm::LLVMContext &ctx) {
        std::unique_ptr<llvm::Module> llvmMod =
            mlir::translateModuleToLLVMIR(mod, ctx);
        if (!llvmMod) {
          throw std::runtime_error("failed to translate module to LLVM IR");
        }
        return llvmMod;
      },
      py::keep_alive<0, 2>(), py::call_guard<py::gil_scoped_release>());

  // Landing B' (plan v3 §4-D1): the same table landing B stamps, applied to the
  // *translated* module instead of the MLIR one.  Landing B walks `llvm.func`
  // ops, so it can only reach a declaration that exists in MLIR; the XPU
  // dialect's ops (`llvm.xpu.core_id` and friends) create their
  // `llvm::Function` inside `to_module`, after B has run, and would otherwise
  // reach the in-process O3 carrying no attribute at all.  The payload is
  // `intrinsic_tables.stamp_payload(stamp_tag())`; the sweep leaves a name the
  // linked LLVM already knows alone -- on the fallback leg XTDK's own table has
  // already spoken (see IntrinsicAttrTable.h).
  m.def("add_intrinsic_attrs", [](llvm::Module *mod,
                                  const std::string &payload) {
    std::string error;
    unsigned stamped =
        mlir::intrinsic_attr_table::applyPayloadToModule(*mod, payload, error);
    if (!error.empty())
      throw std::runtime_error("intrinsic attribute stamp (LLVM IR): " + error);
    return stamped;
  });

  // The same table, for the emitters: they consult it while *creating* a
  // declaration, so a fact does not have to be patched up later (and cannot
  // drift from what the `llc` that reads the same table will assert).  Set once
  // per process before the pipeline runs --
  // `llvm19_toolchain.install_intrinsic_attr_table` hands it the same payload
  // landing B' sweeps with.  `setPayload` refuses a second, *different* table
  // (one process = one table; the refusal travels as data because the util is
  // compiled with exceptions off) and that refusal is surfaced here as a Python
  // `RuntimeError`.
  m.def("set_intrinsic_attr_table", [](const std::string &payload) {
    mlir::intrinsic_attr_table::setPayload(payload);
    std::string error = mlir::intrinsic_attr_table::takePayloadError();
    if (!error.empty())
      throw std::runtime_error(error);
  });

  m.def("attach_datalayout", [](llvm::Module *mod, const std::string triple,
                                const std::string proc,
                                const std::string features) {
    std::string error;
    llvm::Triple targetTriple(triple);
    auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    if (!target) {
      // Public-LLVM builds do not link the XTDK-private XPU targets, so
      // the XPU-family triples cannot resolve.  Fall back to the data layout
      // the XTDK 19 target machine produces for them (identical for all of
      // them; verbatim from its emitted .ll: "e-m:e-p:32:32-p1:64:64-..."). The
      // generic pointer MUST be 32-bit (p:32:32) to match the staged LLVM 19
      // llc, otherwise the LLVM 22 optimizer materializes i32->i64 zexts
      // around every generic-pointer inttoptr and the address math in the
      // inner loop gets ~50% slower (CI fc_fusion perf regression).  The
      // actual codegen runs through the staged LLVM 19 toolchain, which
      // derives the layout from its own target tables -- this string only has
      // to keep the LLVM 22 optimization pipeline's view consistent with it.
      //
      // If XTDK ever changes the layout this string silently diverges.  It is
      // covered by docs/pub-llvm22-frontend-knowledge-base.md §2.4 and by the
      // two-leg IR diff described in §4.3, which is where a mismatch would
      // show up.
      static const char *kFallbackLayout =
          "e-m:e-p:32:32-p1:64:64-p2:32:32-p4:32:32-p135:64:64-p134:64:64-"
          "p133:32:32-p131:64:64-i1:8:32-i8:8:32-i16:16:32-i64:64:64-"
          "f64:64:64-v512:512-a:0:32-n32-S32";
      if (isXpuFamilyTriple(targetTriple)) {
        mod->setDataLayout(kFallbackLayout);
        return;
      }
      throw std::runtime_error("target lookup error: " + error);
    }
    llvm::TargetOptions opt;
    // Target machine is only used to create the data layout.
    std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
        targetTriple, proc, features, opt, llvm::Reloc::PIC_, std::nullopt,
        llvm::CodeGenOptLevel::None)};
    // set target triple and data layout
    mod->setTargetTriple(targetTriple);
    mod->setDataLayout(machine->createDataLayout());
  });

  m.def(
      "optimize_module",
      [](llvm::Module *mod, const llvm::OptimizationLevel &opt,
         std::string arch, std::string features, std::vector<std::string> flags,
         bool enable_fp_fusion) {
        if (mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT"))
          return;
        auto options = llvm::cl::getRegisteredOptions();
        // Hack for the 3.6 release only. Vectorization of copyable elements
        // exposed a bug in ptxas. Manually disable it by modifying the command
        // line option for it. Note that we can abuse DISABLE_LLVM_OPT to
        // override this, since setting it to slp-copyable-elements will set the
        // flag back to true.
        auto it = options.find("slp-copyable-elements");
        if (it != options.end())
          *static_cast<llvm::cl::opt<bool> *>(it->second) = false;
        // Disable XPUAddRangeMeta (the "add-range-meta" PipelineStartEP pass).
        // It stamps tight !range metadata on xpu.core_id/xpu.load_param, which
        // lets CorrelatedValuePropagation/InstCombine narrow strided scatter
        // address math to i8/i32 (srem i8, trunc nuw, nuw nsw). On the XPU LLVM
        // 22 backend that narrowing miscompiles the last-dim torch.cat copy
        // kernel, producing an out-of-bounds lm2gm_v3 and a -714 illegal-memory
        // fault at runtime (DISABLE_LLVM_OPT=1 makes it pass). Turn it off
        // here, mirroring the slp-copyable-elements hack above. As with that
        // hack, DISABLE_LLVM_OPT=add-range-meta re-enables it via the loop
        // below.
        auto rangeMetaIt = options.find("add-range-meta");
        if (rangeMetaIt != options.end())
          *static_cast<llvm::cl::opt<bool> *>(rangeMetaIt->second) = false;
        // Check to see if we are passing a list of flags to disable
        // optimizations.
        auto flagList = mlir::triton::tools::getStrEnv("DISABLE_LLVM_OPT");
        if (!flagList.empty()) {
          llvm::SmallVector<StringRef, 3> split;
          StringRef(flagList.c_str()).split(split, ',');
          for (auto flag : split) {
            auto optIt = options.find(flag);
            if (optIt != options.end()) {
              auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
              *optPtr = true;
            }
          }
        }
        using namespace llvm;
        LoopAnalysisManager lam;
        FunctionAnalysisManager fam;
        CGSCCAnalysisManager cgam;
        ModuleAnalysisManager mam;

        if (arch.empty()) {
          llvm::TargetLibraryInfoImpl TLII(mod->getTargetTriple());
          TLII.disableAllFunctions();
          fam.registerPass([TLII = std::move(TLII)] {
            return llvm::TargetLibraryAnalysis(TLII);
          });
        }

        PassInstrumentationCallbacks *instrCbPtr = nullptr;
        PassInstrumentationCallbacks passInstrCb;
        StandardInstrumentations standardInstr(mod->getContext(),
                                               /*DebugLogging*/ true);
        // The XPU TargetMachine's registerPassBuilderCallbacks dereferences
        // PassBuilder::PIC unconditionally, so a null
        // PassInstrumentationCallbacks pointer to PassBuilder will segfault.
        // Always provide a real one; it's harmless when no callbacks are
        // registered.
        instrCbPtr = &passInstrCb;
#if !defined(TRITON_CONCEAL_IR) || (TRITON_CONCEAL_IR == 0)
        if (mlir::triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP")) {
          auto optMap = llvm::cl::getRegisteredOptions();
          auto optIt = optMap.find("print-after-all");
          if (optIt != optMap.end()) {
            auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
            *optPtr = true;
          }
          standardInstr.registerCallbacks(passInstrCb, &mam);
          instrCbPtr = &passInstrCb;
        }
#endif

        applyXpuErrorLmSizeEnv();

        {
          auto optMap = llvm::cl::getRegisteredOptions();
          auto optIt = optMap.find("xpu-ensure-ieee754-semantic");
          if (optIt != optMap.end()) {
            auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
            *optPtr = true;
          }
        }

        PipelineTuningOptions tuningOptions;
        //===-------------------- For Triton XPU -----------------------===//
        tuningOptions.LoopUnrolling = false;
        tuningOptions.LoopInterleaving = true;
        tuningOptions.LoopVectorization = true;
        tuningOptions.SLPVectorization =
            false; // TODO[dyq]: wait for xtdk adaptation
#if defined(TRITON_HAVE_XTDK_TUNING_OPTIONS)
        // XTDK-private pipeline tuning knobs (absent from public LLVM); the
        // defaults there already disable these XPU-specific transforms.
        tuningOptions.SimpleLoopUnswitchingXPU =
            false; // To Avoid Copying When If Else is in For
        tuningOptions.MemCpyOptXPU =
            false; // To Void Selecting Memset Instruction
        tuningOptions.VectorCombineXPU =
            false; // To Void Selecting ShuffleVector Instruction
#endif
        //===-----------------------------------------------------------===//

        std::string pluginFile =
            mlir::triton::tools::getStrEnv("LLVM_PASS_PLUGIN_PATH");

        // We don't pass the targetMachine to the LLVM-IR pass builder, unless
        // `arch` is specified.
        //
        // Don't set target machine in LLVM pass builder when using LLVM IR
        // level plugins. LLVM IR level plugin passes typically want to insert
        // calls to externally generated code (i.e. precompile a Cuda/Hip kernel
        // with Clang and then insert a call to it within an instrumentation
        // pass) setting the targetMachine value here can can cause a mismatch
        // in the target machine between the MLIR and Clang generated kernels
        // and break the lowering of some target specific intrinsics.
        std::unique_ptr<TargetMachine> targetMachine = nullptr;
        if (!arch.empty() && pluginFile.empty())
          targetMachine =
              createTargetMachine(mod, arch, enable_fp_fusion, features);
        PassBuilder pb(/*targetMachine=*/targetMachine.get(), tuningOptions,
                       std::nullopt, instrCbPtr);

        if (!pluginFile.empty()) {
          // TODO: Add some logging here that we inserted a pass into the LLVM
          // pass pipeline
          auto passPlugin = llvm::PassPlugin::Load(pluginFile);
          if (!passPlugin) {
            llvm::Error Err = passPlugin.takeError();
            std::string ErrMsg =
                "Pass Plugin Error: " + llvm::toString(std::move(Err));
            throw std::runtime_error(ErrMsg);
          }
          passPlugin->registerPassBuilderCallbacks(pb);
        }

        pb.registerModuleAnalyses(mam);
        pb.registerCGSCCAnalyses(cgam);
        pb.registerFunctionAnalyses(fam);
        pb.registerLoopAnalyses(lam);
        pb.crossRegisterProxies(lam, fam, cgam, mam);

        ModulePassManager mpm;
        pb.registerPipelineStartEPCallback(
            [](ModulePassManager &PM, OptimizationLevel) {
#if TRITON_HAVE_XPU_PRINTF_ASSERT
              PM.addPass(XPULowerPrintfAssert());
#endif
#if !defined(TRITON_HAVE_XTDKDL)
              // Public-LLVM builds run the optimizer on an x86 fallback
              // TargetMachine (the XPU-family targets are XTDK-private).  The
              // x86 target exposes no address-space assumption map, so the
              // pipeline's InferAddressSpacesPass never fires and
              // `load (addrspacecast p1 -> p0)` survives the pub22 optimizer.
              // Run the pass explicitly with the flat address space (0, per
              // the cluster data layout) so the generic-pointer loads/stores
              // are inferred back to their pointee address space, matching the
              // XTDK pipeline's codegen.
              //
              // NOTE (2026-09-07): the earlier claim that the cast would
              // otherwise reach the LLVM 19 backend and select flat_load
              // instead of global_load was NOT borne out empirically -- the
              // XTDK 19 llc folds `addrspacecast p1->p0` in ISel (emits
              // global_load, identical to a direct AS1 load) for both the
              // xpu3 and the cluster triples.  The XTDK pipeline handles this
              // cast in codegen, not in the IR-level pass.  This explicit pass
              // is therefore redundant under the reopt path (pub22
              // translate-only
              // + XTDK19 opt -O3 + llc19 -O3) and is kept only while the pub22
              // O3 pipeline still runs.  See
              // CI146-FC-FUSION-PERF-ROOT-CAUSE.md §5 (dead-end 4).
              PM.addPass(createModuleToFunctionPassAdaptor(
                  InferAddressSpacesPass(/*AddressSpace=*/0)));
#endif
            });
        pb.registerVectorizerStartEPCallback(
            [&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
              // Triton generates large structure of scalars which may pessimise
              // optimizations, we run a pass to break up phi of struct to make
              // sure all the struct are removed for the following passes.
              fpm.addPass(BreakStructPhiNodesPass());
              fpm.addPass(InstCombinePass());
            });
        bool enableAddressSanitizer =
            mlir::triton::tools::getBoolEnv("TRITON_ENABLE_ASAN");
        if (enableAddressSanitizer) {
          AddressSanitizerOptions Opts;
          mpm.addPass(AddressSanitizerPass(Opts));
        }
        mpm.addPass(pb.buildPerModuleDefaultPipeline(opt));
        mpm.run(*mod, mam);
#if !defined(TRITON_HAVE_XTDK_TUNING_OPTIONS)
        // Undo the MemCpyOpt memset folding for public-LLVM builds (see
        // expandConstMemSets above for the rationale).  Runs after the O3
        // pipeline so it only rewrites what the pipeline itself produced.
        expandConstMemSets(*mod);
#endif
      },
      // Mandatory parameters
      py::arg("mod"), py::arg("opt"),
      // If we want to specify the target machine, we require additional
      // (optional) parameters
      py::arg("arch") = "", py::arg("features") = "",
      py::arg("flags") = std::vector<std::string>{},
      py::arg("enable_fp_fusion") = false,
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "translate_to_asm",
      [](std::string llvmIR, std::string triple, std::string proc,
         std::string features, std::vector<std::string> flags,
         bool enable_fp_fusion, bool isObject) -> py::object {
        std::string obj;
        {
          // when allow_threads goes out of scope, gil will be released
          py::gil_scoped_release allow_threads;
          // create LLVM module from C++
          llvm::LLVMContext context;
          std::unique_ptr<llvm::MemoryBuffer> buffer =
              llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
          llvm::SMDiagnostic error;
          std::unique_ptr<llvm::Module> module =
              llvm::parseIR(buffer->getMemBufferRef(), error, context);
          if (!module) {
            llvm::report_fatal_error(
                "failed to parse IR: " + error.getMessage() +
                "lineno: " + std::to_string(error.getLineNo()));
          }
          obj = translateLLVMIRToASM(*module, triple, proc, features, flags,
                                     enable_fp_fusion, isObject);
        }
        if (isObject)
          return py::bytes(obj);
        else
          return py::str(obj);
      },
      ret::take_ownership);

  m.def(
      "is_elf_stack_size_oob",
      [](std::string ElfObj, uint32_t Margin) -> py::bool_ {
        bool StackSizeOutofBound = isElfStackSizeOOB(ElfObj, Margin);
        return StackSizeOutofBound;
      },
      py::arg("elf_obj"), py::arg("margin") = 0);

  m.def("dump_sched_dag", [](std::string llvmIR, std::string triple,
                             std::string proc, std::string features,
                             std::vector<std::string> flags,
                             bool enable_fp_fusion, std::string dumpFileId) {
    // when allow_threads goes out of scope, gil will be released
    py::gil_scoped_release allow_threads;
    // create LLVM module from C++
    llvm::LLVMContext context;
    std::unique_ptr<llvm::MemoryBuffer> buffer =
        llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
    llvm::SMDiagnostic error;
    std::unique_ptr<llvm::Module> module =
        llvm::parseIR(buffer->getMemBufferRef(), error, context);
    if (!module) {
      llvm::report_fatal_error("failed to parse IR: " + error.getMessage() +
                               "lineno: " + std::to_string(error.getLineNo()));
    }
    dumpSchedulingDAG(*module, triple, proc, features, flags, enable_fp_fusion,
                      dumpFileId);
  });

  m.def(
      "translate_to_mir",
      [](std::string llvmIR, std::string triple, std::string proc,
         std::string features, std::vector<std::string> flags,
         bool enable_fp_fusion, std::string dumpFileId) -> py::object {
        std::string obj;
        {
          // when allow_threads goes out of scope, gil will be released
          py::gil_scoped_release allow_threads;
          // create LLVM module from C++
          llvm::LLVMContext context;
          std::unique_ptr<llvm::MemoryBuffer> buffer =
              llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
          llvm::SMDiagnostic error;
          std::unique_ptr<llvm::Module> module =
              llvm::parseIR(buffer->getMemBufferRef(), error, context);
          if (!module) {
            llvm::report_fatal_error(
                "failed to parse IR: " + error.getMessage() +
                "lineno: " + std::to_string(error.getLineNo()));
          }
          obj = translateLLVMIRToMIR(*module, triple, proc, features, flags,
                                     enable_fp_fusion, dumpFileId);
        }
        return py::str(obj);
      },
      ret::take_ownership);

  m.def("init_targets", []() {
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
    // Only initialize targets we actually link against.  On the XTDK leg
    // the trust LLVM's Targets.h does not list XPU, so InitializeAll*
    // would silently initialize nothing for XPU (target lookup fails at
    // runtime); there we keep the explicit XPU init.  On the public-LLVM
    // leg there is no XPU target at all, so the InitializeAll* set is the
    // only thing that works.
#if defined(TRITON_HAVE_XTDKDL)
      LLVMInitializeXPUTargetInfo();
      LLVMInitializeXPUTarget();
      LLVMInitializeXPUTargetMC();
      LLVMInitializeXPUAsmParser();
      LLVMInitializeXPUAsmPrinter();
#else
      llvm::InitializeAllTargetInfos();
      llvm::InitializeAllTargets();
      llvm::InitializeAllTargetMCs();
      llvm::InitializeAllAsmParsers();
      llvm::InitializeAllAsmPrinters();
#endif
#if defined(__x86_64__)
      LLVMInitializeX86TargetInfo();
      LLVMInitializeX86Target();
      LLVMInitializeX86TargetMC();
      LLVMInitializeX86AsmParser();
      LLVMInitializeX86AsmPrinter();
#elif defined(__aarch64__)
      LLVMInitializeAArch64TargetInfo();
      LLVMInitializeAArch64Target();
      LLVMInitializeAArch64TargetMC();
      LLVMInitializeAArch64AsmParser();
      LLVMInitializeAArch64AsmPrinter();
#endif
    });
  });

  m.def("link_extern_libs", [](llvm::Module *dstMod,
                               const std::vector<std::string> &paths) {
    if (paths.empty())
      return;

    LLVMContext &ctx = dstMod->getContext();
    llvm::Linker linker(*dstMod);
    for (const std::string &path : paths) {
      llvm::SMDiagnostic err;
      std::unique_ptr<llvm::Module> libMod = llvm::parseIRFile(path, err, ctx);
      if (!libMod) {
        std::string message = "Failed to parse library at " + path;
        throw std::invalid_argument(message);
      }
      libMod->setTargetTriple(Triple(dstMod->getTargetTriple()));
      libMod->setDataLayout(dstMod->getDataLayout());

      std::unordered_set<std::string> externalFns;
      for (llvm::Function &fn : libMod->functions()) {
        if (!fn.isDeclaration())
          externalFns.insert(fn.getName().str());
      }

      if (linker.linkInModule(std::move(libMod),
                              llvm::Linker::Flags::LinkOnlyNeeded)) {
        std::string message = "Failed to link library at " + path;
        throw std::invalid_argument(message);
      }

      // Mark linked-in functions as internal because backends use external
      // linkage as a signifier of kernel functions.
      for (llvm::Function &fn : dstMod->functions()) {
        if (externalFns.count(fn.getName().str())) {
          fn.setLinkage(llvm::GlobalValue::InternalLinkage);
        }
      }
    }
  });
}

void triton_stacktrace_signal_handler(void *) {
  llvm::sys::PrintStackTrace(llvm::errs());
  raise(SIGABRT);
}

void init_triton_stacktrace_hook(pybind11::module &m) {
  if (mlir::triton::tools::getBoolEnv("TRITON_ENABLE_PYTHON_STACKTRACE")) {
    llvm::sys::AddSignalHandler(triton_stacktrace_signal_handler, nullptr);
  }
}
