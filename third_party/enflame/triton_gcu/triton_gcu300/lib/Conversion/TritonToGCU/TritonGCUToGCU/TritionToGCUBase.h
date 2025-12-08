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

#ifndef KURAMA_TRITON_TO_GCU_BASE_H_
#define KURAMA_TRITON_TO_GCU_BASE_H_

#include <map>

#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {
namespace gcu {
class FirstLastUserAnalysis;
}
} // namespace triton
} // namespace mlir

using namespace mlir;

template <typename SourceOp>
class SharedConversionPattern : public OpConversionPattern<SourceOp> {
public:
  SharedConversionPattern(const TypeConverter &converter, MLIRContext *ctx,
                          triton::gcu::FirstLastUserAnalysis &userAnalysis,
                          std::map<Operation *, Operation *> &replaced2Origin)
      : OpConversionPattern<SourceOp>(converter, ctx),
        userAnalysis(userAnalysis), replaced2Origin(replaced2Origin) {}

protected:
  triton::gcu::FirstLastUserAnalysis &userAnalysis;
  std::map<Operation *, Operation *> &replaced2Origin;
};
#endif
