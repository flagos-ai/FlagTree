#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class OpPassManager;
namespace rpu {

std::unique_ptr<Pass> createRecognizeRPUPlanPass();
std::unique_ptr<Pass> createLowerSupportedTTIRToRPUExecutablePass();
std::unique_ptr<Pass> createLegalizeRPUExecutableCompactElementwise1DPass();
std::unique_ptr<Pass> createLegalizeRPUExecutableDotPass();
std::unique_ptr<Pass> createLegalizeRPUExecutableSoftmaxPass();
std::unique_ptr<Pass> createLegalizeRPUExecutableValueMapsPass();
void addRPUExecutableHighLevelLegalizationPipeline(OpPassManager &pm);
std::unique_ptr<Pass> createVerifyRPUExecutableRenderablePass();
std::unique_ptr<Pass> createEmitRPUExecutableToRPURCPass();
std::unique_ptr<Pass> createConvertRPUPlanToRPUExecutablePass();

} // namespace rpu
} // namespace mlir
