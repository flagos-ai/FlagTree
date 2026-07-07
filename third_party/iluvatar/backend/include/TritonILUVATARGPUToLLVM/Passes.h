#ifndef TRITON_THIRD_PARTY_ILUVATAR_INCLUDE_TRITONILUVATARGPUTOLLVM_PASSES_H_
#define TRITON_THIRD_PARTY_ILUVATAR_INCLUDE_TRITONILUVATARGPUTOLLVM_PASSES_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/Function.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

} // namespace mlir

namespace mlir::triton {

#define GEN_PASS_DECL
#include "TritonILUVATARGPUToLLVM/Passes.h.inc"

} // namespace mlir::triton

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonILUVATARGPUToLLVMPass(StringRef targetArch, bool ftz);
#define GEN_PASS_REGISTRATION
#include "TritonILUVATARGPUToLLVM/Passes.h.inc"

} // namespace mlir::triton

#endif // TRITON_THIRD_PARTY_ILUVATAR_INCLUDE_TRITONILUVATARGPUTOLLVM_PASSES_H_
