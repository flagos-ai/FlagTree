#ifndef ILUVATAR_TLE_RAW_COMPILED_RAW_H
#define ILUVATAR_TLE_RAW_COMPILED_RAW_H

#include <string>

namespace mlir::triton::iluvatar::tle_raw {

struct CompiledRaw {
  std::string regionDialect;
  std::string externFuncName;
  std::string llvmIr;
};

} // namespace mlir::triton::iluvatar::tle_raw

#endif
