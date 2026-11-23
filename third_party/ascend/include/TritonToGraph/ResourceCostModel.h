/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef TRITON_TO_GRAPH_RESOURCE_COST_MODEL_H
#define TRITON_TO_GRAPH_RESOURCE_COST_MODEL_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace ascend {
class HardwareConfig;
}

namespace triton {
namespace cfg {

enum class ResourceCostRejectReason : uint8_t {
  None,
  UnknownResource,
  DynamicShape,
  UnknownElementType,
  Overflow,
  UBOverflow,
  InvalidCandidate,
};

const char *getResourceCostRejectReasonName(ResourceCostRejectReason reason);

struct ResourceSnapshot {
  uint64_t ubCapacityBytes = 0;
  uint64_t reservedUBBytes = 0;
  unsigned ubSafetyPercent = 80;

  bool isKnown() const;
  std::optional<uint64_t> getSafeUBBudget() const;

  static ResourceSnapshot fromExplicit(uint64_t ubCapacityBytes,
                                       unsigned ubSafetyPercent = 80,
                                       uint64_t reservedUBBytes = 0);

  static ResourceSnapshot
  fromHardwareConfig(const ascend::HardwareConfig &hardware,
                     unsigned ubSafetyPercent = 80,
                     uint64_t reservedUBBytes = 0);
};

struct LiveTensorInterval {
  Value value;
  uint64_t begin = 0;
  uint64_t end = 0;
  uint64_t bytes = 0;
};

struct LiveByteEstimate {
  bool known = false;
  bool overflow = false;
  ResourceCostRejectReason reason = ResourceCostRejectReason::UnknownResource;
  uint64_t peakLiveBytes = 0;
  llvm::SmallVector<LiveTensorInterval, 16> intervals;
};

std::optional<uint64_t> getStaticTensorBytes(Type type);

LiveByteEstimate estimatePeakLiveBytes(Operation *root);

struct CandidatePlan {
  uint64_t tensorizeFactor = 1;
  uint64_t blockT = 1;
  uint64_t staticAxisFusionFactor = 1;
  std::string stableId;
  uint64_t sourceOrdinal = 0;
};

struct CandidateCost {
  CandidatePlan plan;
  uint64_t logicalTasksBefore = 0;
  uint64_t logicalTasksAfter = 0;
  uint64_t actualProgramsBefore = 0;
  uint64_t actualProgramsAfter = 0;
  uint64_t physicalWavesBefore = 0;
  uint64_t physicalWavesAfter = 0;
  uint64_t launchesBefore = 0;
  uint64_t launchesAfter = 0;
  uint64_t gmReadBytesBefore = 0;
  uint64_t gmReadBytesAfter = 0;
  uint64_t gmWriteBytesBefore = 0;
  uint64_t gmWriteBytesAfter = 0;
  uint64_t storeCountBefore = 0;
  uint64_t storeCountAfter = 0;
  uint64_t addressCalculationsBefore = 0;
  uint64_t addressCalculationsAfter = 0;
  uint64_t baselinePeakLiveBytes = 0;
  uint64_t estimatedPeakLiveBytes = 0;
  uint64_t persistentLoopTripsBefore = 0;
  uint64_t persistentLoopTripsAfter = 0;
  uint64_t tokenOnlyRepeatedBytesBefore = 0;
  uint64_t tokenOnlyRepeatedBytesAfter = 0;
  uint64_t workPerProgramBefore = 0;
  uint64_t workPerProgramAfter = 0;
  bool hasPeakLiveBytes = false;
  bool hasDynamicShape = false;
  bool hasUnknownResource = false;
  bool legacyAutoMapBefore = false;
  bool legacyAutoMapAfter = false;
  bool persistent = false;
};

struct CostModelWeights {
  uint64_t logicalProgramReduction = 64;
  uint64_t programReduction = 64;
  uint64_t launchReduction = 256;
  uint64_t gmByte = 1;
  uint64_t store = 16;
  uint64_t addressCalculation = 1;
  uint64_t liveByte = 1;
  uint64_t persistentLoopTrip = 512;
  uint64_t tokenOnlyRepeatedByte = 1;
};

struct CandidateEvaluation {
  CandidateCost candidate;
  bool accepted = false;
  ResourceCostRejectReason reason = ResourceCostRejectReason::UnknownResource;
  int64_t benefitScore = 0;
  uint64_t safeUBBudgetBytes = 0;
  uint64_t requiredParallelPrograms = 0;
  uint64_t effectiveWorkPerProgram = 0;
  uint64_t physicalWavesBefore = 0;
  uint64_t physicalWavesAfter = 0;
  uint64_t persistentLoopTripsBefore = 0;
  uint64_t persistentLoopTripsAfter = 0;
  uint64_t tokenOnlyRepeatedBytesBefore = 0;
  uint64_t tokenOnlyRepeatedBytesAfter = 0;
  std::string remark;
};

CandidateEvaluation evaluateCandidateCost(const ResourceSnapshot &resources,
                                          const CandidateCost &candidate,
                                          const CostModelWeights &weights = {});

llvm::SmallVector<CandidatePlan>
enumerateCandidatePlans(llvm::ArrayRef<unsigned> tensorizeFactors,
                        llvm::ArrayRef<unsigned> blockTFactors,
                        llvm::ArrayRef<unsigned> staticAxisFusionFactors,
                        llvm::StringRef stableIdPrefix = "resource");

void sortCandidateEvaluations(
    llvm::MutableArrayRef<CandidateEvaluation> values);

std::string formatCandidateRemark(const CandidateEvaluation &evaluation);
void emitCandidateRemark(Operation *anchor,
                         const CandidateEvaluation &evaluation);

struct ProfilerCalibrationRecord {
  std::string kernelName;
  std::string profilerArtifact;
  uint64_t sampleCount = 0;
  uint64_t medianNanoseconds = 0;
  uint64_t p90Nanoseconds = 0;
  CandidateEvaluation evaluation;
};

std::string
formatProfilerCalibrationRecord(const ProfilerCalibrationRecord &record);

class ResourceCostAnalysis {
public:
  ResourceCostAnalysis(Operation *root, ResourceSnapshot resources);

  const ResourceSnapshot &getResourceSnapshot() const { return resources; }
  const LiveByteEstimate &getLiveByteEstimate() const { return liveBytes; }
  CandidateEvaluation evaluate(const CandidateCost &candidate) const;

private:
  ResourceSnapshot resources;
  LiveByteEstimate liveBytes;
};

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif
