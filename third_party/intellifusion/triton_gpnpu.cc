#include "lib/TritonGPNPUToLLVM/Passes.h"
#include "lib/TritonGPNPUTransforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "passes.h" //python/src/ ADD_PASS_WRAPPER_0
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <fstream>

// Include standard MLIR dialects
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

// Include Triton dialect headers for MLIR parsing
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace mlir;

void init_convert_to_llvm(py::module &&m) {
  m.def("add_convert_to_llvmir", [](mlir::PassManager &pm, int numWarps) {
    pm.addPass(
        mlir::triton::Intellifusion::createConvertTritonGPNPUToLLVM(numWarps));
  });
}

void init_triton_intellifusion_passes_ttgpuir(py::module &&m) {
  m.def(
      "add_dot_strategy",
      [](mlir::PassManager &pm, bool forcePadding, int numWarps, int numCTAs) {
        pm.addPass(mlir::triton::Intellifusion::createGPNPUDotStrategyPass(
            forcePadding, numWarps, numCTAs));
      },
      py::arg("pm"), py::arg("force_padding") = false, py::arg("num_warps") = 4,
      py::arg("num_ctas") = 1);
  m.def("add_accelerate_matmul", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::Intellifusion::createGPNPUAccelerateMatmulPass());
  });
  m.def("add_stage_scaled_fp4", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::Intellifusion::createGPNPUStageScaledFp4Pass());
  });
  m.def("add_allocate_shared_memory", [](mlir::PassManager &pm) {
    pm.addPass(
        mlir::triton::Intellifusion::createGPNPUAllocateSharedMemoryPass());
  });
}

void init_triton_intellifusion_passes_ttnvgpuir(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_plan_cta",
                     mlir::triton::Intellifusion::createGPNPUPlanCTAPass);
  ADD_PASS_WRAPPER_0(
      "add_optimize_descriptor_encoding",
      mlir::triton::Intellifusion::createGPNPUOptimizeDescriptorEncodingPass);
  ADD_PASS_WRAPPER_0(
      "add_promote_lhs_to_tmem",
      mlir::triton::Intellifusion::createGPNPUPromoteLHSToTMemPass);
  ADD_PASS_WRAPPER_0(
      "add_remove_tmem_tokens",
      mlir::triton::Intellifusion::createGPNPURemoveTMEMTokensPass);
  ADD_PASS_WRAPPER_0(
      "add_optimize_tmem_layouts",
      mlir::triton::Intellifusion::createGPNPUOptimizeTMemLayoutsPass);
  ADD_PASS_WRAPPER_0(
      "add_tmem_load_reduce",
      mlir::triton::Intellifusion::createGPNPUFuseTMEMLoadReducePass);
  ADD_PASS_WRAPPER_0("add_tma_lowering",
                     mlir::triton::Intellifusion::createGPNPUTMALoweringPass);
  ADD_PASS_WRAPPER_0(
      "add_interleave_tmem",
      mlir::triton::Intellifusion::createGPNPUInterleaveTMemPass);
  ADD_PASS_WRAPPER_0(
      "add_fence_insertion",
      mlir::triton::Intellifusion::createGPNPUFenceInsertionPass);
  ADD_PASS_WRAPPER_0("add_lower_mma",
                     mlir::triton::Intellifusion::createGPNPUMMALoweringPass);
  ADD_PASS_WRAPPER_0(
      "add_allocate_tensor_memory",
      mlir::triton::Intellifusion::createGPNPUTensorMemoryAllocationPass);
  ADD_PASS_WRAPPER_0(
      "add_proxy_fence_insertion",
      mlir::triton::Intellifusion::createGPNPUProxyFenceInsertionPass);
  ADD_PASS_WRAPPER_0(
      "add_check_matmul_two_cta",
      mlir::triton::Intellifusion::createGPNPUCheckMatmulTwoCTAPass);
  ADD_PASS_WRAPPER_0(
      "add_tmem_barrier_insertion",
      mlir::triton::Intellifusion::createGPNPUTMemBarrierInsertionPass);
  ADD_PASS_WRAPPER_0("add_gpnpu_to_llvm",
                     mlir::triton::Intellifusion::createGPNPUCompatToLLVMPass);
}

void init_triton_intellifusion_common(py::module &&m) {

  m.def("save_mlir", [](ModuleOp mod, const std::string &filepath) -> bool {
    std::string str;
    llvm::raw_string_ostream os(str);
    auto printingFlags = OpPrintingFlags();
    printingFlags.enableDebugInfo();
    mod.print(os, printingFlags);

    std::ofstream file(filepath);
    if (!file.is_open()) {
      llvm::errs() << "Error: Cannot open file " << filepath
                   << " for writing\n";
      return false;
    }
    file << str;
    file.close();
    llvm::outs() << "✓ Saved MLIR to " << filepath << "\n";
    return true;
  });

  m.def(
      "load_mlir",
      [](mlir::MLIRContext &context, const std::string &filepath) -> ModuleOp {
        std::ifstream file(filepath);
        if (!file.is_open()) {
          llvm::errs() << "Error: Cannot open file " << filepath
                       << " for reading\n";
          throw std::runtime_error("Cannot open file: " + filepath);
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        file.close();

        DialectRegistry registry;
        registry.insert<::mlir::triton::TritonDialect,
                        ::mlir::triton::gpu::TritonGPUDialect,
                        ::mlir::triton::instrument::TritonInstrumentDialect,
                        ::mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
                        ::mlir::math::MathDialect, ::mlir::arith::ArithDialect,
                        ::mlir::scf::SCFDialect, ::mlir::gpu::GPUDialect,
                        ::mlir::cf::ControlFlowDialect,
                        ::mlir::LLVM::LLVMDialect, ::mlir::ub::UBDialect,
                        ::mlir::triton::gluon::GluonDialect>();
        context.allowUnregisteredDialects();
        context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();

        OwningOpRef<ModuleOp> result =
            parseSourceString<ModuleOp>(content, &context);
        if (!result) {
          llvm::errs() << "Error: Failed to parse MLIR from " << filepath
                       << "\n";
          throw std::runtime_error("Failed to parse MLIR from: " + filepath);
        }

        llvm::outs() << "✓ Loaded MLIR from " << filepath << "\n";
        return result->clone();
      },
      py::return_value_policy::take_ownership);
}

void init_triton_intellifusion(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_convert_to_llvm(passes.def_submodule("llir"));
  init_triton_intellifusion_passes_ttgpuir(passes.def_submodule("ttgpuir"));
  init_triton_intellifusion_passes_ttnvgpuir(passes.def_submodule("ttnvgpuir"));

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    using namespace mlir;
    DialectRegistry registry;
    registry.insert<::mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
                    ::mlir::triton::nvgpu::NVGPUDialect>();
    mlir::registerNVVMDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  init_triton_intellifusion_common(m.def_submodule("ir"));

  m.def(
      "parse_llvm_ir",
      [](const std::string &llvmIR, llvm::LLVMContext &context,
         const std::string &sourceName) {
        llvm::SMDiagnostic error;
        auto module = llvm::parseIR(llvm::MemoryBufferRef(llvmIR, sourceName),
                                    error, context);
        if (!module) {
          std::string message;
          llvm::raw_string_ostream stream(message);
          error.print("LLVM IR", stream);
          throw std::invalid_argument(message);
        }
        std::string message;
        llvm::raw_string_ostream diagnostics(message);
        if (llvm::verifyModule(*module, &diagnostics))
          throw std::invalid_argument("Invalid LLVM IR: " + message);
        return module;
      },
      py::arg("llvm_ir"), py::arg("context"), py::arg("source_name") = "",
      py::keep_alive<0, 2>(), py::call_guard<py::gil_scoped_release>());

  m.def("serialize_bitcode", [](const llvm::Module &module) {
    // Preserve canonical LLIR and debug information for callers and dumps.
    auto backendModule = llvm::CloneModule(module);
    llvm::StripDebugInfo(*backendModule);
    std::string bitcode;
    llvm::raw_string_ostream output(bitcode);
    llvm::WriteBitcodeToFile(*backendModule, output);
    return py::bytes(bitcode);
  });

  m.def("has_libdevice_deps", [](llvm::Module *module) {
    // Inspect the final module, including LLIR overrides, rather than carrying
    // a dependency flag from an earlier compilation stage.
    for (const auto &symbol : module->global_values()) {
      if (symbol.use_empty() || !symbol.isDeclarationForLinker())
        continue;
      auto name = symbol.getName();
      // The merged IFWA bitcode supplies these ABIs. Runtime externs such as
      // printf are resolved separately and do not require libdevice.
      if (name.starts_with("__nv_") || name.starts_with("ifwa_"))
        return true;
      // IFWA target lowering can introduce libdevice helpers for these NVVM
      // math intrinsics even though they are not ordinary unresolved externs.
      if (name.starts_with("llvm.nvvm.div.") ||
          name.starts_with("llvm.nvvm.rsqrt.") ||
          name.starts_with("llvm.nvvm.sqrt."))
        return true;
    }
    return false;
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
