#ifndef TLE_DSA_CONVERSION_DSATOMCORE_H
#define TLE_DSA_CONVERSION_DSATOMCORE_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace mlir::dsa {

std::unique_ptr<mlir::Pass> createDsaMemoryToCorePass();
void registerDsaMemoryToCorePass();

} // namespace mlir::dsa

#endif // TLE_DSA_CONVERSION_DSATOMCORE_H
