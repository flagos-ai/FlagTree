#ifndef HCU_TLE_RAW_WRAPPER_ADAPTER_H
#define HCU_TLE_RAW_WRAPPER_ADAPTER_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::triton::HCU::tle_raw {

// A nonempty, homogeneous scalar input range computed by materialization.
struct RawFragmentOutput {
  unsigned inputOffset;
  unsigned count;
};

// Build in the imported module, whose data layout describes the source ABI.
// With no fragment outputs, validate the signature and return wrapper.
FailureOr<LLVM::LLVMFuncOp>
createRawWrapperAdapter(ModuleOp module, LLVM::LLVMFuncOp wrapper,
                        unsigned inputCount,
                        ArrayRef<RawFragmentOutput> outputs);

} // namespace mlir::triton::HCU::tle_raw

#endif
