
/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef KURAMA_TRITON_GCU_TO_GCU_VECTORIZE_H_
#define KURAMA_TRITON_GCU_TO_GCU_VECTORIZE_H_

#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gcu {

struct VectorizeConfig {
  unsigned vectorLength;
  bool needCvtDataLayout = false;
};

struct VectorizeContext {
  OpBuilder &builder;
  Location loc;
  IRMapping &mapper;
  unsigned vectorLength;
};

class VectorizeHandlerInterface {
public:
  virtual ~VectorizeHandlerInterface() = default;
  virtual bool match(Operation *op) const = 0;
  virtual Operation *rewriteOp(const VectorizeContext &context,
                               Operation *op) const = 0;
  virtual void finalizeMappedResults(const VectorizeContext &context,
                                     Operation *originalOp,
                                     Operation *clonedOp) const;
};

class DefaultVectorizeHandler : public VectorizeHandlerInterface {
public:
  bool match(Operation *op) const override { return true; }
  Operation *rewriteOp(const VectorizeContext &context,
                       Operation *op) const override {
    auto clonedOp = context.builder.clone(*op, context.mapper);
    return clonedOp;
  }
};

class SelectVectorizeHandler : public VectorizeHandlerInterface {
public:
  bool match(Operation *op) const override { return isa<arith::SelectOp>(op); }
  Operation *rewriteOp(const VectorizeContext &context,
                       Operation *op) const override;
};

class ExternElementwiseVectorizeHandler : public VectorizeHandlerInterface {
public:
  bool match(Operation *op) const override {
    return isa<triton::ExternElementwiseOp>(op);
  }
  Operation *rewriteOp(const VectorizeContext &context,
                       Operation *op) const override;
  void finalizeMappedResults(const VectorizeContext &context,
                             Operation *originalOp,
                             Operation *clonedOp) const override;
};

class VectorizeEngine {
public:
  VectorizeEngine();
  Operation *vectorize(Operation *op, const VectorizeContext &context) const;
  void addHandler(std::unique_ptr<VectorizeHandlerInterface> handler) {
    handlers.emplace_back(std::move(handler));
  }

private:
  SmallVector<std::unique_ptr<VectorizeHandlerInterface>> handlers;
};

} // namespace gcu
} // namespace triton
} // namespace mlir
#endif
