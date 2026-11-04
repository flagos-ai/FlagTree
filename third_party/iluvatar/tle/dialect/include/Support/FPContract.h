// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
#ifndef ILUVATAR_TLE_FP_CONTRACT_H
#define ILUVATAR_TLE_FP_CONTRACT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include <mutex>

namespace mlir::triton::iluvatar_tle {

inline std::mutex &fpContractMutex() {
  static std::mutex mutex;
  return mutex;
}

// COREX's FMAD contraction pass uses a separate process-wide option and
// otherwise contracts even when TargetOptions::AllowFPOpFusion is Strict.
// Serialize both native emission paths while applying the caller's contract,
// then restore the option for subsequent compilations.
class ScopedFPContract {
public:
  ScopedFPContract(llvm::StringRef triple, bool enableFusion)
      : lock(fpContractMutex()) {
    if (!triple.starts_with("bi-"))
      return;
    auto &options = llvm::cl::getRegisteredOptions();
    auto it = options.find("iluvatar-contract-fmad");
    if (it == options.end())
      llvm::report_fatal_error("COREX LLVM is missing its FP contraction option");
    option = static_cast<llvm::cl::opt<bool> *>(it->second);
    previous = option->getValue();
    option->setValue(enableFusion);
  }

  ~ScopedFPContract() {
    if (option)
      option->setValue(previous);
  }

private:
  std::unique_lock<std::mutex> lock;
  llvm::cl::opt<bool> *option = nullptr;
  bool previous = false;
};

} // namespace mlir::triton::iluvatar_tle
#endif
