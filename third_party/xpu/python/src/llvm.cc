#include "mlir/IR/BuiltinOps.h" // mlir::ModuleOp
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
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
#include "llvm/Passes/PassPlugin.h"
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
#include "llvm/Transforms/Utils/XPULowerPrintfAssert.h"
#include <csignal>
#include <cstdio>
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

static bool isElfStackSizeOOB(std::string ElfObject) {
  uint32_t StackSizeLimit = DEFAULTLOCALLIMIT;
  const char *_lmSizeEnv = std::getenv("TRITON_TUNE_BUFFER_LM_SIZE");
  std::string StackSizeLimitStr = _lmSizeEnv ? std::string(_lmSizeEnv) : "";
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
    if (SymSectionOrErr.get()->getSize() != 8) {
      llvm::report_fatal_error("Section Size Error");
    }

    StackSize = *((uint32_t *)(ContentsOrErr->data()));
    break;
  }
  return StackSize > StackSizeLimit;
}

static void applyXpuErrorLmSizeEnv() {
  uint32_t LMSizeLimit = -1;
  std::string LMSizeLimitStr =
      (std::getenv("LLVM_ERROR_LM_SIZE")
           ? std::string(std::getenv("LLVM_ERROR_LM_SIZE"))
           : "");
  if (!LMSizeLimitStr.empty()) {
    llvm::StringRef LMSizeLimitSRf = LMSizeLimitStr;
    if (LMSizeLimitSRf.getAsInteger(10, LMSizeLimit)) {
      llvm::report_fatal_error("Invalid value for LLVM_ERROR_LM_SIZE: " +
                               LMSizeLimitSRf);
    }
  }
  llvm::StringMap<llvm::cl::Option *> optMap = llvm::cl::getRegisteredOptions();
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
  auto target =
      llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
  llvm::TargetOptions opt;
  bool disableLLVMOpt = mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  if (enable_fp_fusion)
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
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

  m.def("attach_datalayout", [](llvm::Module *mod, const std::string triple,
                                const std::string proc,
                                const std::string features) {
    std::string error;
    llvm::Triple targetTriple(triple);
    auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    if (!target) {
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
        tuningOptions.SimpleLoopUnswitchingXPU =
            false; // To Avoid Copying When If Else is in For
        tuningOptions.MemCpyOptXPU =
            false; // To Void Selecting Memset Instruction
        tuningOptions.VectorCombineXPU =
            false; // To Void Selecting ShuffleVector Instruction
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
              PM.addPass(XPULowerPrintfAssert());
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

  m.def("is_elf_stack_size_oob", [](std::string ElfObj) -> py::bool_ {
    bool StackSizeOutofBound = isElfStackSizeOOB(ElfObj);
    return StackSizeOutofBound;
  });

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
      // Only initialize targets we actually link against.
      // InitializeAllTargets() would also pull in XCN which is not linked
      // (trust LLVM does not provide libLLVMXCNCodeGen.a), causing
      // "undefined symbol: LLVMInitializeXCNTargetInfo" at dlopen time.
      LLVMInitializeXPUTargetInfo();
      LLVMInitializeXPUTarget();
      LLVMInitializeXPUTargetMC();
      LLVMInitializeXPUAsmParser();
      LLVMInitializeXPUAsmPrinter();
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
