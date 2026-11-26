//===- HIVMAnalysisPass.cpp - Native HIVM scheduling analysis ------------===//
//
// Runs the HIVM-native analysis on a parsed MLIR module and emits a textual
// report plus an optional Perfetto trace.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Analysis/HIVMAnalysis.h"
#include "AscendModel/HardwareConfig.h"
#include "AscendModel/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_HIVMANALYSISPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

static FailureOr<HIVMSchedulerMode> parseSchedulerMode(llvm::StringRef mode) {
  if (mode.empty() || mode == "static")
    return HIVMSchedulerMode::Static;
  if (mode == "des")
    return HIVMSchedulerMode::DES;
  return failure();
}

struct HIVMAnalysisPass : public impl::HIVMAnalysisPassBase<HIVMAnalysisPass> {
  using HIVMAnalysisPassBase::HIVMAnalysisPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    std::string loadError;
    auto hardwareConfig =
        loadHardwareConfigForAnalysis(hardwareConfigPath, loadError);
    if (!hardwareConfig) {
      module.emitError() << "failed to load hardware config: " << loadError;
      signalPassFailure();
      return;
    }
    const HardwareConfig &config = *hardwareConfig;
    auto schedulerOr = parseSchedulerMode(schedulerMode);
    if (failed(schedulerOr)) {
      module.emitError() << "invalid HIVM scheduler mode `" << schedulerMode
                         << "`; expected `static` or `des`";
      signalPassFailure();
      return;
    }
    HIVMAnalyzer analyzer(config, argBindingsStr, *schedulerOr);
    HIVMAnalysisReport report;
    std::string error;
    if (!analyzer.analyzeModule(module, report, error)) {
      module.emitError() << "HIVM analysis failed: " << error;
      signalPassFailure();
      return;
    }

    if (!reportFile.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream os(reportFile, ec, llvm::sys::fs::OF_Text);
      if (ec) {
        module.emitError() << "failed to open HIVM report file `" << reportFile
                           << "`: " << ec.message();
        signalPassFailure();
        return;
      }
      report.print(os, config);
    } else {
      report.print(llvm::errs(), config);
    }

    if (!perfettoTraceFile.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream os(perfettoTraceFile, ec, llvm::sys::fs::OF_Text);
      if (ec) {
        module.emitError() << "failed to open Perfetto trace file `"
                           << perfettoTraceFile << "`: " << ec.message();
        signalPassFailure();
        return;
      }
      report.emitPerfettoTrace(os, config);
    }
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
