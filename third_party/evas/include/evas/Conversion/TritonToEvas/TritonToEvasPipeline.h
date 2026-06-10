#ifndef EVAS_CONVERSION_TRITONTOEVAS_TRITONTOEVASPIPELINE_H
#define EVAS_CONVERSION_TRITONTOEVAS_TRITONTOEVASPIPELINE_H

#include "mlir/Pass/PassManager.h"

namespace mlir {
class ModuleOp;
template <typename OpT>
class OperationPass;
namespace triton {
namespace evas {

void buildTritonToEvasPipeline(OpPassManager &pm);
void registerTritonToEvasPipeline();
std::unique_ptr<OperationPass<ModuleOp>>
createEvasTritonArithToLinalgPass(bool tensorPtrToLinalg = true,
                                  bool transposeReduceToRank0 = true);

} // namespace evas
} // namespace triton
} // namespace mlir

#endif
