// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//===----------------------------------------------------------------------===//
//
// This provides registration calls for LLVMXPU dialect to LLVM IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_DIALECT_XPU_XPUTOLLVMIRTRANSLATION_H
#define MLIR_TARGET_LLVMIR_DIALECT_XPU_XPUTOLLVMIRTRANSLATION_H

namespace mlir {

class DialectRegistry;
class MLIRContext;

/// Register the LLVMXPU dialect and the translation from it to the LLVM IR in
/// the given registry;
void registerLLVMXPUDialectTranslation(DialectRegistry &registry);

/// Register the LLVMXPU dialect and the translation from it in the registry
/// associated with the given context.
void registerLLVMXPUDialectTranslation(MLIRContext &context);

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_XPU_XPUTOLLVMIRTRANSLATION_H
