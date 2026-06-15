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
#ifndef GCU_CONVERSION_TRITONTOTRITONGPU_GCUTRITONGPUCONVERSION_H
#define GCU_CONVERSION_TRITONTOTRITONGPU_GCUTRITONGPUCONVERSION_H

#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {

// Type converter that assigns BlockedEncodingAttr to ranked tensors.
// Unlike the upstream TritonGPUTypeConverter, this version accepts a
// custom `order` vector and a reduce-axis frequency map so that the
// caller can control dimension priority (e.g. deprioritize the reduce
// axis, with higher-frequency axes getting even lower priority).
class GCUTritonGPUTypeConverter : public TypeConverter {
public:
  GCUTritonGPUTypeConverter(
      MLIRContext *context, int numWarps, int threadsPerWarp, int numCTAs,
      ArrayRef<unsigned> defaultOrder,
      const llvm::SmallDenseMap<unsigned, unsigned> &axisFreq);

  int getNumWarps() const { return numWarps; }
  int getThreadsPerWarp() const { return threadsPerWarp; }
  int getNumCTAs() const { return numCTAs; }

private:
  MLIRContext *context;
  int numWarps;
  int threadsPerWarp;
  int numCTAs;
};

class GCUTritonGPUConversionTarget : public ConversionTarget {
public:
  explicit GCUTritonGPUConversionTarget(
      MLIRContext &ctx, GCUTritonGPUTypeConverter &typeConverter);

  static bool isDynamicallyLegal(Operation *op,
                                 const TypeConverter &typeConverter);
};

triton::gpu::BlockedEncodingAttr getBlockedEncodingWithOrder(
    MLIRContext *context, ArrayRef<int64_t> shape, ArrayRef<unsigned> order,
    const llvm::SmallDenseMap<unsigned, unsigned> &axisFreq, int numWarps,
    int threadsPerWarp, int numCTAs);

} // namespace mlir

#endif // GCU_CONVERSION_TRITONTOTRITONGPU_GCUTRITONGPUCONVERSION_H
