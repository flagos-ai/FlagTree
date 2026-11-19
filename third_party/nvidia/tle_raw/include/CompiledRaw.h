#ifndef NV_TLE_RAW_COMPILED_RAW_H
#define NV_TLE_RAW_COMPILED_RAW_H

#include <string>

namespace mlir::triton::nvidia::tle_raw {

struct CompiledRaw {
  std::string regionDialect;
  std::string externFuncName;
  std::string llvmIr;
};

} // namespace mlir::triton::nvidia::tle_raw

#endif
