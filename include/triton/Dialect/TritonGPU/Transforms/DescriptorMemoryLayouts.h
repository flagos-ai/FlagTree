/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_DESCRIPTOR_MEMORY_LAYOUTS_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_DESCRIPTOR_MEMORY_LAYOUTS_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include <unordered_set>

namespace mlir::triton::gpu {
struct UseInfo;
struct EncodingInfo;

/// Update shared encoding given a new shape
SharedEncodingTrait updateEncodingForShape(Operation *op,
                                           SharedEncodingTrait encoding,
                                           RankedTensorType tensorType);

//===----------------------------------------------------------------------===//
// AssignDescriptorMemoryLayouts
//===----------------------------------------------------------------------===//

/// Assign memory layouts to tensor descriptors in a module.
class AssignDescriptorMemoryLayouts {
public:
  AssignDescriptorMemoryLayouts() = default;
  virtual ~AssignDescriptorMemoryLayouts() = default;
  void assignMemoryLayouts(ModuleOp &mod);

private:
  void runOnFunction(FuncOp &func);
  const EncodingInfo *
  internEncoding(std::unordered_set<EncodingInfo> &encodings,
                 EncodingInfo info);
  EncodingInfo combineEncodings(const EncodingInfo &lhs,
                                const EncodingInfo &rhs, unsigned rank);
  Attribute findLoadEncodingFromUsers(Operation *op);
  std::optional<UseInfo> getUseInfo(Operation *op);
  Attribute getFallbackSharedEncoding(RankedTensorType tensorType,
                                      CGAEncodingAttr cgaLayout,
                                      ArrayRef<int64_t> usageShape,
                                      unsigned numCTAs);

protected:
  virtual Attribute getCompatibleSharedEncoding(Attribute enc,
                                                ArrayRef<int64_t> shape,
                                                Type elementType) {
    return isCompatibleSharedEncoding(enc) ? enc : Attribute();
  }

private:
  // Override with backend specific implementation
  virtual Attribute buildFallbackSharedEncoding(mlir::MLIRContext *,
                                                ArrayRef<int64_t>,
                                                ArrayRef<unsigned>,
                                                CGAEncodingAttr, Type) = 0;
  virtual bool isCompatibleSharedEncoding(Attribute) = 0;
};

} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_DESCRIPTOR_MEMORY_LAYOUTS_H_
