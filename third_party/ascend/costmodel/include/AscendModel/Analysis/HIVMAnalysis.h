//===- HIVMAnalysis.h - HIVM performance analysis --------------*- C++ -*-===//
//
// Optional HIVM-based analysis path for TritonSim.
//
// This analyzer consumes dumped `.npuir.mlir`, normalizes the subset of
// HIVM HIR that is relevant to performance modeling, and schedules the
// resulting execution graph on Ascend execution pipes through MLIR-native
// ingestion.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_ANALYSIS_HIVMANALYSIS_H
#define ASCENDMODEL_ANALYSIS_HIVMANALYSIS_H

#include "AscendModel/Analysis/HardwareConfig.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mlir {
namespace ascend {

enum class HIVMSchedulerMode { Static, DES };

enum class HIVMPipe {
  Unknown,
  Vector,
  VectorMTE2,
  CubeMTE2,
  MTE3,
  Scalar,
  FixPipe,
  Cube,
  MTE1,
  All
};

struct HIVMOp {
  size_t id = 0;
  std::string opName;
  std::string text;
  HIVMPipe pipe = HIVMPipe::Unknown;
  HIVMPipe senderPipe = HIVMPipe::Unknown;
  HIVMPipe receiverPipe = HIVMPipe::Unknown;
  std::string coreType;
  std::string eventId;
  int64_t eventGeneration = 0;
  int64_t duration = 0;
  int64_t bytes = 0;
  int64_t elements = 0;
  int64_t multiBufferSlots = 1;
  int64_t startCycle = 0;
  int64_t endCycle = 0;
  int64_t loopMultiplier = 1;
  int lineNumber = 0;
  bool isSyncOp = false;
  bool isBarrier = false;
  std::vector<std::string> readBuffers;
  std::vector<std::string> writeBuffers;
  std::vector<int64_t> readBufferVersions;
  std::vector<int64_t> writeBufferVersions;
  std::vector<size_t> dependsOn;
};

struct HIVMAnalysisReport {
  std::string sourcePath;
  std::string sourceMode;
  HIVMSchedulerMode schedulerMode = HIVMSchedulerMode::Static;
  int64_t oneIterationCycles = 0;
  int64_t weightedCycles = 0;
  int64_t totalBusyCycles = 0;
  int64_t syncCycles = 0;
  int64_t barrierCycles = 0;
  size_t opCount = 0;
  size_t syncOpCount = 0;
  size_t barrierCount = 0;
  size_t unknownOpCount = 0;
  int64_t maxLoopMultiplier = 1;
  std::map<HIVMPipe, int64_t> pipeBusyCycles;
  std::map<HIVMPipe, int64_t> weightedPipeCycles;
  std::vector<HIVMOp> operations;

  void print(llvm::raw_ostream &os, const HardwareConfig &config) const;
  void emitPerfettoTrace(llvm::raw_ostream &os,
                         const HardwareConfig &config) const;
  void emitDESGraph(llvm::raw_ostream &os, const HardwareConfig &config) const;
};

class HIVMAnalyzer {
public:
  HIVMAnalyzer(const HardwareConfig &config,
               llvm::StringRef argBindings = llvm::StringRef(),
               HIVMSchedulerMode schedulerMode = HIVMSchedulerMode::Static);

  bool analyzeModule(mlir::ModuleOp module, HIVMAnalysisReport &report,
                     std::string &error) const;

  bool analyzeFile(llvm::StringRef path, HIVMAnalysisReport &report,
                   std::string &error) const;

  static llvm::StringRef stringifyPipe(HIVMPipe pipe);
  static llvm::StringRef stringifySchedulerMode(HIVMSchedulerMode mode);

private:
  const HardwareConfig &config;
  std::string argBindingsStr;
  HIVMSchedulerMode schedulerMode;
};

} // namespace ascend
} // namespace mlir

#endif // ASCENDMODEL_ANALYSIS_HIVMANALYSIS_H
