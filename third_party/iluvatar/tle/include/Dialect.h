#ifndef TRITON_THIRD_PARTY_ILUVATAR_TLE_DIALECT_H_
#define TRITON_THIRD_PARTY_ILUVATAR_TLE_DIALECT_H_

#include "IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton::iluvatar_tle {

inline void registerDialects(DialectRegistry &registry) {
  registry.insert<mlir::triton::iluvatar_tle::IluvatarTleDialect>();
}

inline void addIllegalDialects(ConversionTarget &target) {
  target.addIllegalDialect<mlir::triton::iluvatar_tle::IluvatarTleDialect>();
}

} // namespace mlir::triton::iluvatar_tle

#endif // TRITON_THIRD_PARTY_ILUVATAR_TLE_DIALECT_H_
