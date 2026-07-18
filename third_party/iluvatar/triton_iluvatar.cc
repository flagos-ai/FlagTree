#include "TritonILUVATARGPUToLLVM/Passes.h"
#ifdef __ILUVATAR_TLE__
#include "Dialect.h"
#endif
// #include "cublas_instance.h"
#include "TritonILUVATARGPUTransforms/Passes.h"

#include "Dialect/TritonILUVATARGPU/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
// #include "triton/Dialect/TritonILUVATARGPU/Transforms/Passes.h"
#include "llvm/IR/Constants.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>

namespace py = pybind11;

#ifdef __ILUVATAR_TLE__
void init_triton_iluvatar_tle_ir(py::module m);
void init_triton_iluvatar_tle_passes(py::module m);
#endif

static std::unique_ptr<llvm::TargetMachine>
createTargetMachine(llvm::Module *module, std::string proc,
                    bool enable_fp_fusion, const std::string &features) {
  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
  llvm::TargetOptions opt;
  bool disableLLVMOpt = mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  if (enable_fp_fusion)
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  opt.TrapUnreachable = true;
  std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
      module->getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
      std::nullopt,
      disableLLVMOpt ? llvm::CodeGenOptLevel::None
                     : llvm::CodeGenOptLevel::Aggressive)};
  return machine;
}

std::string translateLLVMIRToILUVATAR(llvm::Module &module,
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
  if (triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP")) {
    auto optIt = options.find("print-after-all");
    if (optIt != options.end()) {
      auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
      *optPtr = true;
    }
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

  const bool enabledTiming = triton::tools::getBoolEnv("LLVM_ENABLE_TIMING");
  if (enabledTiming) {
    llvm::TimePassesIsEnabled = true;
    llvm::TimePassesPerRun = true;
  }

  pm.run(module);

  SmallString<0> timePassesStr;
  llvm::raw_svector_ostream reportStream(timePassesStr);

  if (enabledTiming) {
    reportAndResetTimings(&reportStream);
    llvm::dbgs() << reportStream.str();
    timePassesStr.clear();
  }

  // create machine
  module.setTargetTriple(llvm::Triple(triple));
  auto machine = createTargetMachine(&module, proc, enable_fp_fusion, features);
  // set data layout
  module.setDataLayout(machine->createDataLayout());

  // Dump 加上 iluvatar 后端信息后的 llvm IR
  if (triton::tools::getBoolEnv("ILUIR_ENABLE_DUMP")) {
    llvm::dbgs()
        << "// -----// Iluvatar LLIR Dump after initialization //----- //\n"
        << module << '\n';
  }

  // create unique dir for kernel's binary
  std::error_code ec;
  std::string kernel_name_base = "iluvatar_triton_kernel";
  std::filesystem::path tmp = std::filesystem::temp_directory_path();
  std::filesystem::path kernel_dir_base(kernel_name_base);
  llvm::SmallString<256> unique_dir;
  ec = llvm::sys::fs::createUniqueDirectory((tmp / kernel_dir_base).string(),
                                            unique_dir);
  if (ec) {
    std::cerr << "Directory for " << kernel_name_base
              << " was not created. error code: " << ec << std::endl;
  }
  std::filesystem::path kernel_dir(unique_dir.data());
  std::string kernel_name = kernel_dir.stem();
  // Save Iluvatar ISA binary.
  std::filesystem::path isa_binary(kernel_name + ".o");
  std::string isabin_path = (kernel_dir / isa_binary).string();
  std::unique_ptr<llvm::raw_fd_ostream> isabin_fs(
      new llvm::raw_fd_ostream(isabin_path, ec, llvm::sys::fs::OF_Text));
  if (ec) {
    llvm::errs() << isabin_path
                 << " was not created. error code: " << ec.category().name()
                 << ':' << ec.value() << '\n';
  }
  // emit
  llvm::legacy::PassManager pass;

  // Fix __nvvm_reflect issue, adopted from tensorflow2.12: gpu_backend_lib.cc
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  fam.registerPass([&] { return machine->getTargetIRAnalysis(); });

  llvm::PipelineTuningOptions pto;
  pto.SLPVectorization = true;
  pto.InlinerThreshold = 0x100000;

  llvm::PassInstrumentationCallbacks pic;

  llvm::StandardInstrumentations si(module.getContext(), false);
  si.registerCallbacks(pic, &mam);

  llvm::PassBuilder pb(machine.get(), pto, std::nullopt, &pic);
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  int32_t opt_level = 3;
  llvm::OptimizationLevel ol;
  switch (opt_level) {
  case 0:
    ol = llvm::OptimizationLevel::O0;
    break;
  case 1:
    ol = llvm::OptimizationLevel::O1;
    break;
  case 2:
    ol = llvm::OptimizationLevel::O2;
    break;
  case 3:
    ol = llvm::OptimizationLevel::O3;
    break;
  }

  llvm::ModulePassManager mpm;
  mpm.addPass(llvm::VerifierPass());
  if (ol == llvm::OptimizationLevel::O0) {
    mpm.addPass(pb.buildO0DefaultPipeline(ol));
  } else {
    mpm.addPass(pb.buildPerModuleDefaultPipeline(ol));
  }
  mpm.addPass(llvm::VerifierPass());

  mpm.run(module, mam);

  // Dump 经过部分优化后的 llvm IR
  if (triton::tools::getBoolEnv("ILUIR_ENABLE_DUMP")) {
    llvm::dbgs()
        << "// -----// Iluvatar LLIR Dump before optimization //----- //\n"
        << module << '\n';
    // module.dump();
  }

  machine->addPassesToEmitFile(pass, *isabin_fs, nullptr,
                               llvm::CodeGenFileType::ObjectFile);

  pass.run(module);

  // Dump 经过整个后端优化后的 llvm IR
  if (triton::tools::getBoolEnv("ILUIR_ENABLE_DUMP")) {
    llvm::dbgs()
        << "// -----// Iluvatar LLIR Dump after optimization //----- //\n"
        << module << '\n';
    // module.dump();
  }

  // generate cubin file
  std::filesystem::path cubin_fname(kernel_name + ".cubin");
  std::string cubin_path = (kernel_dir / cubin_fname).string();
  std::string error_message;
  std::string linker_path = mlir::triton::tools::getLinkerPath().string();
  int lld_result = llvm::sys::ExecuteAndWait(
      linker_path,
      {linker_path, "-flavor", "ld.lld", "--no-warn-missing-entry",
       "--no-undefined", isabin_path, "-o", cubin_path},
      std::nullopt, {}, 0, 0, &error_message);
  if (lld_result) {
    std::cout << "ld.lld execute fail: " << std::endl;
    std::cout << error_message << std::endl;
    std::cout << lld_result << std::endl;
  }

  // Read cubin
  std::ifstream _cubin(cubin_path.c_str(), std::ios::binary);
  std::string cubin(std::istreambuf_iterator<char>(_cubin), {});
  _cubin.close();

  // Remove tmp file
  ec = llvm::sys::fs::remove_directories(kernel_dir.string());
  if (ec) {
    llvm::errs() << "fail to remove tmp kernel: " << kernel_dir
                 << ", error code: " << ec.category().name() << ":"
                 << ec.value() << "\n";
  }
  return cubin;
}

namespace mlir::triton::gpu {
#define GEN_PASS_DECL_TRITONGPUACCELERATEMATMUL
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

static std::unique_ptr<mlir::Pass>
createTritonGPUAccelerateMatmulWithSme(unsigned useSme) {
  mlir::triton::gpu::TritonGPUAccelerateMatmulOptions options;
  options.useSme = useSme;
  return mlir::triton::gpu::createTritonGPUAccelerateMatmul(options);
}

void init_triton_iluvatar_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;
  m.def("add_to_llvmir",
        [](mlir::PassManager &pm, const std::string &arch, bool ftz) {
          pm.addPass(mlir::triton::createConvertTritonILUVATARGPUToLLVMPass(
              arch, ftz));
        });
  // iluvatar-specific passes
  ADD_PASS_WRAPPER_1("add_matmul_smeload",
                     mlir::createTritonILUVATARGPUSmeLoadPass, int);
  ADD_PASS_WRAPPER_0("add_optimize_epilogue",
                     mlir::createTritonILUVATARGPUOptimizeEpiloguePass);
  ADD_PASS_WRAPPER_0("add_mma_reduce_thread_locality",
                     mlir::createTritonILUVATARGPUMMAReduceThreadLocalityPass);
  m.def("add_accelerate_matmul", [](mlir::PassManager &pm, unsigned useSme) {
    pm.addPass(createTritonGPUAccelerateMatmulWithSme(useSme));
  });
}

void init_triton_iluvatar(py::module &&m) {
#ifdef __ILUVATAR_TLE__
  init_triton_iluvatar_tle_ir(m.def_submodule("ir"));
#endif

  auto passes = m.def_submodule("passes");
  init_triton_iluvatar_passes_ttgpuir(passes.def_submodule("ttgpuir"));
#ifdef __ILUVATAR_TLE__
  init_triton_iluvatar_tle_passes(passes.def_submodule("tle"));
#endif

  m.attr("TARGET_TRIPLE") = "bi-iluvatar-ilurt";
  m.attr("CALLING_CONV_ILUVATAR_KERNEL") =
      (unsigned)llvm::CallingConv::ILUVATAR_KERNEL;

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::iluvatargpu::TritonILUVATARGPUDialect>();
#ifdef __ILUVATAR_TLE__
    mlir::triton::iluvatar_tle::registerDialects(registry);
#endif
    mlir::registerNVVMDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("attach_target_triple", [](llvm::Module *module) {
    module->setTargetTriple(llvm::Triple("bi-iluvatar-ilurt"));
  });

  m.def(
      "translate_llvmir_to_cubin",
      [](std::string llvmIR, std::string triple, std::string proc,
         std::string features, std::vector<std::string> flags,
         bool enable_fp_fusion, bool isObject) -> py::object {
        std::string cubin;
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
          cubin = translateLLVMIRToILUVATAR(*module, triple, proc, features,
                                            flags, enable_fp_fusion, isObject);
        }
        py::bytes bytes(cubin);
        return std::move(bytes);
      },
      py::return_value_policy::take_ownership);

  // Set short point option, this needs to be set before setting the data
  // layout.
  m.def("set_short_ptr", []() {
    auto options = llvm::cl::getRegisteredOptions();
    const char *flag = "nvptx-short-ptr";
    auto *shortPtr = static_cast<llvm::cl::opt<bool> *>(options[flag]);
    assert(shortPtr);
    shortPtr->setValue(true);
  });

  // TODO: could be done in python if we had a generic interface to set metadata
  m.def("set_nvvm_reflect_ftz", [](llvm::Module *mod) {
    // please check https://llvm.org/docs/NVPTXUsage.html#reflection-parameters
    // this will enable fast math path in libdevice
    // for example, when enable nvvm-reflect-ftz, sqrt.approx.f32 will change to
    // sqrt.approx.ftz.f32
    using namespace llvm;
    auto &ctx = mod->getContext();
    Type *i32 = Type::getInt32Ty(ctx);
    auto *mdFour = ConstantAsMetadata::get(ConstantInt::getSigned(i32, 4));
    auto *mdName = MDString::get(ctx, "nvvm-reflect-ftz");
    auto *mdOne = ConstantAsMetadata::get(ConstantInt::getSigned(i32, 1));
    auto *reflect = MDNode::get(ctx, {mdFour, mdName, mdOne});
    mod->addModuleFlag(reflect);
  });

  m.def("has_extern_deps", [](llvm::Module *dstMod) -> bool {
    // `global_smem` is special cased in Triton, so we ignore it here.
    for (const auto &g : dstMod->globals()) {
      if (g.hasExternalLinkage() && g.getName() != "global_smem") {
        return true;
      }
    }
    for (const auto &f : *dstMod) {
      if (f.hasExternalLinkage() && !f.hasExactDefinition() &&
          !f.isIntrinsic()) {
        return true;
      }
    }
    return false;
  });
}
