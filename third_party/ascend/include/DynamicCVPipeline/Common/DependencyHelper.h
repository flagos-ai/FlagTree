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

#ifndef TRITON_DYNAMIC_CV_PIPELINE_COMMON_DEPENDENCYHELPER_H
#define TRITON_DYNAMIC_CV_PIPELINE_COMMON_DEPENDENCYHELPER_H

#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"

namespace mlir::CVPipeline {

class DependencyHelper {
public:
  enum class SourceMode { Default, AcrossIterArg };

private:
  using CallbackFn = function_ref<mlir::WalkResult(mlir::Operation *)>;

  mlir::WalkResult forEachUserImpl(mlir::Operation *op, CallbackFn pred) const;

  template <SourceMode SM = SourceMode::Default>
  mlir::WalkResult forEachSourceImpl(mlir::Operation *op,
                                     CallbackFn pred) const;

  template <typename Fn> static auto makeWalkCallback(Fn &&pred) {
    if constexpr (std::is_void_v<std::invoke_result_t<Fn, mlir::Operation *>>) {
      return [pred = std::forward<Fn>(pred)](mlir::Operation *op) mutable {
        pred(op);
        return mlir::WalkResult::advance();
      };
    } else {
      return std::forward<Fn>(pred);
    }
  }

  template <typename Fn>
  static auto mapToAncestorInBlock(mlir::Block *block, Fn &&pred) {
    return [block, pred = std::forward<Fn>(pred)](mlir::Operation *op) mutable {
      if (auto *ancestor = block->findAncestorOpInBlock(*op)) {
        return pred(ancestor);
      }
      if constexpr (std::is_same_v<std::invoke_result_t<Fn, mlir::Operation *>,
                                   mlir::WalkResult>) {
        return mlir::WalkResult::advance();
      }
    };
  }

public:
  const MemoryDependenceGraph &memGraph;

  explicit DependencyHelper(const MemoryDependenceGraph &memGraph)
      : memGraph(memGraph) {}

  template <typename Fn>
  mlir::WalkResult forEachUser(mlir::Operation *op, Fn &&pred) const {
    return forEachUserImpl(op, makeWalkCallback(std::forward<Fn>(pred)));
  }

  template <SourceMode SM = SourceMode::Default, typename Fn>
  mlir::WalkResult forEachSource(mlir::Operation *op, Fn &&pred) const {
    return forEachSourceImpl<SM>(op, makeWalkCallback(std::forward<Fn>(pred)));
  }

  template <typename Fn>
  mlir::WalkResult forEachUserInSameBlock(mlir::Operation *op,
                                          Fn &&pred) const {
    return forEachUser(
        op, mapToAncestorInBlock(op->getBlock(), std::forward<Fn>(pred)));
  }

  template <SourceMode SM = SourceMode::Default, typename Fn>
  mlir::WalkResult forEachSourceInSameBlock(mlir::Operation *op,
                                            Fn &&pred) const {
    return forEachSource<SM>(
        op, mapToAncestorInBlock(op->getBlock(), std::forward<Fn>(pred)));
  }
};

} // namespace mlir::CVPipeline

#endif
