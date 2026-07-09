#ifndef TRITON_THIRD_PARTY_HCU_LIB_TRITONHCUGPUTRANSFORMS_UTILITY_H_
#define TRITON_THIRD_PARTY_HCU_LIB_TRITONHCUGPUTRANSFORMS_UTILITY_H_

#include "hcu/lib/TritonHCUGPUToLLVM/TargetInfo.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

using namespace mlir;

// DFS the def chain of 'defValue' starting from 'consumer' and will return the
// minimum found when accumulating countFunc(op) for all non control flow ops
// between value and the consumer. This function will traverse through for loop
// iterations and to the outside of the loop to find all its producers.
//    CountOp(Operation*) should return the value to accumulate for the
//    operation
// Returns 0 if there is an error traversing the def chain
int deduceMinCountOnDefChain(Value defValue, Operation *consumerOp,
                             llvm::function_ref<int(Operation *)> countFunc);

FailureOr<std::pair<triton::DotOp, unsigned>>
getDotOpIdxFromMatrixLoad(triton::MatrixLoadOp matrixOp);

// Returns a padded shared encoding minimizing bank conflicts for the given
// tensor and dot encoding.
triton::gpu::PaddedSharedEncodingAttr
composePaddedLayout(const triton::HCU::TargetInfo &targetInfo,
                    triton::gpu::DotOperandEncodingAttr dotOpEnc,
                    triton::gpu::TensorOrMemDesc srcTy,
                    ArrayRef<unsigned> sharedOrder, bool useAsyncCopy);

#endif
