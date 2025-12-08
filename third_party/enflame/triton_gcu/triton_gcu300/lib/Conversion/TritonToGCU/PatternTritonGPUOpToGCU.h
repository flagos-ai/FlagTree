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

#ifndef KURANA_TRITON_GPU_OP_TO_GCU_CONVERSION_H
#define KURANA_TRITON_GPU_OP_TO_GCU_CONVERSION_H

#include <map>
#include <string>

#include "TritonGCUToGCU/TritionToGCUBase.h"

namespace {
enum GCUArch { GCU300 = 0, GCU400 };

struct GCUInfo {
  unsigned vaccSizeInBytes;
  bool supportI64;
  unsigned preferVaccNum;
};

static const GCUInfo targetInfo[] = {{128, false, 4}, {512, false, 1}};
} // namespace

namespace mlir {
namespace triton {

namespace gcu {
class FirstLastUserAnalysis;
}

void populateReduceOpToGCUPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin);

void populateElementwiseFusionOpToGCUPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin);

} // namespace triton
} // namespace mlir

#endif
