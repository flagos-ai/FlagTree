#ifndef NV_TLE_RAW_DEFERRED_SOURCE_REGISTRY_H
#define NV_TLE_RAW_DEFERRED_SOURCE_REGISTRY_H

#include "llvm/ADT/StringMap.h"
#include <optional>
#include <string>

namespace mlir::triton::nvidia::tle_raw {

struct DeferredRawSourceEntry {
  std::string sourceId;
  std::string regionDialect;
  std::optional<std::string> externFuncName;
  std::string source;
  std::string llvmIr;
  std::string hint;
};

llvm::StringMap<DeferredRawSourceEntry> &getDeferredRawSourceRegistry();
void clearDeferredRawSourceRegistry();

} // namespace mlir::triton::nvidia::tle_raw

#endif
