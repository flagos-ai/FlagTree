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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace LLVMDIUtils {
LLVM::DITypeAttr convertType(MLIRContext *context, mlir::Type type);
LLVM::DITypeAttr convertPtrType(MLIRContext *context,
                                LLVM::LLVMPointerType pointerType,
                                mlir::Type pointeeType, DataLayout datalayout);
LLVM::DITypeAttr convertStructType(MLIRContext *context,
                                   LLVM::LLVMStructType structType,
                                   LLVM::DIFileAttr fileAttr,
                                   DataLayout datalayout, int64_t line);
LLVM::DITypeAttr convertArrayType(MLIRContext *context,
                                  LLVM::LLVMArrayType arrayType,
                                  LLVM::DIFileAttr fileAttr,
                                  DataLayout datalayout, int64_t line);
FileLineColLoc extractFileLoc(Location loc, bool getCaller = true);
std::optional<unsigned> calcBitWidth(mlir::Type type);
} // namespace LLVMDIUtils
} // namespace mlir
