#ifndef HCU_TLE_RAW_COMPILED_RAW_H
#define HCU_TLE_RAW_COMPILED_RAW_H

#include <string>

namespace mlir::triton::HCU::tle_raw {

struct CompiledRaw {
  std::string regionDialect;
  std::string externFuncName;
  std::string llvmIr;
};

} // namespace mlir::triton::HCU::tle_raw

#endif
