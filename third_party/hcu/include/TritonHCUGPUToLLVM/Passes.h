// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_PASSES_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_PASSES_H_

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
#include "TritonHCUGPUToLLVM/Passes.h.inc"

} // namespace mlir::triton

namespace mlir::triton::HCU {
/// @brief Creates pass that keep LDS consumption within specified limits.
/// @param arch target architecture name, for example "gfx940"
/// @param customLDSLimit defines LDS size available for one thread block
/// zero value tells pass that whole LDS is available on a device
/// @return created pass
std::unique_ptr<OperationPass<ModuleOp>>
createOptimizeLDSUsagePass(StringRef arch, int32_t customLDSLimit = 0);

void runScalarizePackedFOpsPass(llvm::Function &F);

std::unique_ptr<OperationPass<ModuleOp>>
createTritonHCUGPUSetFuncInlineAttrsPass(StringRef arch = "");

} // namespace mlir::triton::HCU

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonHCUGPUToLLVMPass(StringRef targetArch, bool ftz);
std::unique_ptr<OperationPass<ModuleOp>>
createConvertBuiltinFuncToLLVMPass(bool ftz);
std::unique_ptr<OperationPass<ModuleOp>>
createTritonHCUGPUInsertInstructionSchedHintsPass(StringRef variant);
std::unique_ptr<OperationPass<ModuleOp>>
createTritonHCUGPULowerInstructionSchedHintsPass(StringRef arch,
                                                 int32_t numStages);

std::unique_ptr<OperationPass<ModuleOp>>
createHCUGPUConvertWarpSpecializeToLLVM(StringRef targetArch,
                                        int waspNumLoadWarps,
                                        int waspNumMmaWarps, bool wdraEnabled,
                                        int wdraNumLoadRegs,
                                        int wdraNumMmaRegsMain,
                                        int wdraNumMmaRegsTail);

#define GEN_PASS_REGISTRATION
#include "TritonHCUGPUToLLVM/Passes.h.inc"

} // namespace mlir::triton

#endif // TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_PASSES_H_
