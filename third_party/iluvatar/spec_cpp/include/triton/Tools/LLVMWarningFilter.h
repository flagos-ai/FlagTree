#ifndef TRITON_TOOLS_LLVMWARNINGFILTER_H
#define TRITON_TOOLS_LLVMWARNINGFILTER_H

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

#include "triton/Tools/Sys/GetEnv.hpp"

namespace mlir::triton::tools {

inline bool shouldSuppressLoopUnrollWarning(const llvm::DiagnosticInfo &DI) {
  if (getBoolEnv("TRITON_ENABLE_LOOP_UNROLL_WARNING"))
    return false;

  auto *OptFailure =
      llvm::dyn_cast<llvm::DiagnosticInfoOptimizationFailure>(&DI);
  if (!OptFailure)
    return false;
  if (OptFailure->getPassName() != "loop-unroll")
    return false;
  if (OptFailure->getRemarkName() != "FailedFullyUnrolling")
    return false;

  std::string message;
  llvm::raw_string_ostream stream(message);
  llvm::DiagnosticPrinterRawOStream printer(stream);
  DI.print(printer);
  stream.flush();
  return llvm::StringRef(message).ends_with(
      "loop with constant trip count not unrolled");
}

inline void handleLLVMContextDiagnostic(const llvm::DiagnosticInfo *DI,
                                        void * /*context*/) {
  if (!DI)
    return;
  if (shouldSuppressLoopUnrollWarning(*DI))
    return;

  llvm::DiagnosticPrinterRawOStream printer(llvm::errs());
  llvm::errs() << llvm::LLVMContext::getDiagnosticMessagePrefix(
                      DI->getSeverity())
               << ": ";
  DI->print(printer);
  llvm::errs() << "\n";

  if (DI->getSeverity() == llvm::DS_Error)
    std::exit(1);
}

inline void installLLVMWarningFilter(llvm::LLVMContext &context) {
  context.setDiagnosticHandlerCallBack(handleLLVMContextDiagnostic, nullptr,
                                       /*RespectFilters=*/true);
}

} // namespace mlir::triton::tools

#endif
