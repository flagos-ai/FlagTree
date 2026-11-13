/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_DYNAMIC_CV_PIPELINE_COMMON_FALLBACKHELPER_H
#define TRITON_DYNAMIC_CV_PIPELINE_COMMON_FALLBACKHELPER_H

#include <memory>

#include "llvm/ADT/STLExtras.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

namespace mlir::CVPipeline {

class FallbackHelper {
  struct OperationEraser {
    void operator()(Operation *op) const {
      if (op) {
        op->erase();
      }
    }
  };

  std::unique_ptr<Operation, OperationEraser> backup;
  ModuleOp original;

public:
  FallbackHelper(ModuleOp original)
      : backup(original->clone()), original(original) {}

  void restore() {
    if (!backup) {
      return;
    }
    original->setLoc(backup->getLoc());
    original->setAttrs(backup->getAttrs());
    if (original->getPropertiesStorageSize() != 0) {
      original->copyProperties(backup->getPropertiesStorage());
    }
    for (auto [oRegion, bRegion] :
         llvm::zip(original->getRegions(), backup->getRegions())) {
      oRegion.takeBody(bRegion);
    }
  }
};

} // namespace mlir::CVPipeline

#endif
