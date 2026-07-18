#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>
#include <string>

#include "RPU/IR/RPUDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "RPU/IR/RPUTypes.h.inc"

#define GET_OP_CLASSES
#include "RPU/IR/RPUOps.h.inc"

namespace mlir {
namespace rpu {
namespace exec {

struct ExecutableKernelArgumentInfo {
  unsigned index;
  std::string name;
  Value value;
  Type type;
};

struct ExecutableKernelContractInfo {
  KernelOp kernel;
  std::string kernelName;
  std::string kind;
  SmallVector<ExecutableKernelArgumentInfo, 4> arguments;
  SmallVector<Operation *> bodyOps;
};

struct ExecutableKernelScaffold {
  KernelOp kernel;
  SmallVector<Value, 5> arguments;
};

struct ExecutableVectorMaskInfo {
  bool masked;
  int64_t logicalN;
  int64_t blockN;
};

enum class ExecutableCompactVectorBinaryOpcode { Add, Mul };

enum class ExecutableOpLoweringClass {
  Unsupported,
  Renderable,
  HighLevelLegalizable,
};

struct ExecutableCompactVectorBinaryBuildOp {
  ExecutableCompactVectorBinaryOpcode opcode =
      ExecutableCompactVectorBinaryOpcode::Add;
  int64_t lhs = -1;
  int64_t rhs = -1;
};

struct ExecutableCompactElementwise1DBuildSpec {
  int64_t n = 0;
  int64_t logicalN = 0;
  bool masked = false;
  unsigned outputArgIndex = 0;
  ArrayRef<unsigned> inputArgIndices;
  ArrayRef<ExecutableCompactVectorBinaryBuildOp> ops;
};

struct ExecutableElementwise16ValueMapBuildSpec {
  unsigned outputArgIndex = 0;
  ArrayRef<unsigned> inputArgIndices;
  ArrayRef<ExecutableCompactVectorBinaryBuildOp> ops;
};

struct ExecutableAddBodyBuildSpec {
  int64_t n = 0;
  int64_t logicalN = 0;
  bool masked = false;
  unsigned outputArgIndex = 0;
  unsigned lhsArgIndex = 1;
  unsigned rhsArgIndex = 2;
};

struct ExecutableGemmBodyBuildSpec {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  unsigned outputArgIndex = 0;
  unsigned lhsArgIndex = 1;
  unsigned rhsArgIndex = 2;
};

struct ExecutableSoftmaxBodyBuildSpec {
  int64_t nvec = 0;
  unsigned outputArgIndex = 0;
  unsigned inputArgIndex = 1;
};

struct ExecutableSqrtBodyBuildSpec {
  int64_t nvec = 0;
  unsigned outputArgIndex = 0;
  unsigned inputArgIndex = 1;
};

struct ExecutableReduceSumAllBodyBuildSpec {
  int64_t nvec = 0;
  unsigned outputArgIndex = 0;
  unsigned inputArgIndex = 1;
};

struct ExecutableReluBodyBuildSpec {
  int64_t nvec = 0;
  unsigned outputArgIndex = 0;
  unsigned inputArgIndex = 1;
};

struct ExecutableMaximumBodyBuildSpec {
  int64_t n = 0;
  unsigned outputArgIndex = 0;
  unsigned lhsArgIndex = 1;
  unsigned rhsArgIndex = 2;
};

struct ExecutableReduceSumAxisBodyBuildSpec {
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t axis = 0;
  unsigned outputArgIndex = 0;
  unsigned inputArgIndex = 1;
};

struct ExecutableBroadcastAddBodyBuildSpec {
  int64_t rows = 0;
  int64_t cols = 0;
  unsigned outputArgIndex = 0;
  unsigned lhsArgIndex = 1;
  unsigned rhsArgIndex = 2;
};

struct ExecutableConvKxKBodyBuildSpec {
  int64_t kernelSize = 0;
  unsigned outputArgIndex = 0;
  unsigned inputArgIndex = 1;
  unsigned weightArgIndex = 2;
};

struct ExecutableResNetBlockBodyBuildSpec {
  int64_t hidden = 0;
  unsigned outputArgIndex = 0;
  unsigned xArgIndex = 1;
  unsigned w1ArgIndex = 2;
  unsigned w2ArgIndex = 3;
};

struct ExecutableResNet50BottleneckBodyBuildSpec {
  int64_t bottleneck = 0;
  unsigned outputArgIndex = 0;
  unsigned inputArgIndex = 1;
  unsigned w1ArgIndex = 2;
  unsigned w2ArgIndex = 3;
  unsigned w3ArgIndex = 4;
};

struct ExecutableAddOpSequenceInfo {
  LoadContigOp lhsLoad;
  LoadContigOp rhsLoad;
  AddOp add;
  StoreContigOp store;
  int64_t n;
  int64_t logicalN;
  bool masked;
};

struct ExecutableGemmOpSequenceInfo {
  LoadMatrixOp lhsLoad;
  LoadMatrixOp rhsLoad;
  ZeroOp zero;
  MmaOp mma;
  StoreMatrixOp store;
  int64_t m;
  int64_t n;
  int64_t k;
};

struct ExecutableSoftmaxOpSequenceInfo {
  LoadContigOp load;
  ReduceMaxAllOp reduceMax;
  SubScalarOp sub;
  ExpOp exp;
  ReduceSumAllOp reduceSum;
  ReciprocalOp reciprocal;
  MulScalarOp mul;
  StoreContigOp store;
  int64_t nvec;
};

struct ExecutableSqrtOpSequenceInfo {
  LoadContigOp load;
  SqrtOp sqrt;
  StoreContigOp store;
  int64_t nvec;
};

struct ExecutableReduceSumAllOpSequenceInfo {
  LoadContigOp load;
  ReduceSumAllOp reduceSum;
  FullOp full;
  StoreContigOp store;
  int64_t nvec;
};

struct ExecutableReluOpSequenceInfo {
  LoadContigOp load;
  ReluOp relu;
  StoreContigOp store;
  int64_t nvec;
};

struct ExecutableMaximumOpSequenceInfo {
  LoadContigOp lhsLoad;
  LoadContigOp rhsLoad;
  MaxOp max;
  StoreContigOp store;
  int64_t n;
};

struct ExecutableReduceSumAxisOpSequenceInfo {
  LoadMatrixOp load;
  ReduceSumAxisOp reduce;
  StoreContigOp store;
  int64_t rows;
  int64_t cols;
  int64_t axis;
};

struct ExecutableBroadcastAddOpSequenceInfo {
  LoadMatrixOp lhsLoad;
  LoadContigOp rhsLoad;
  BroadcastAddOp broadcastAdd;
  StoreMatrixOp store;
  int64_t rows;
  int64_t cols;
};

struct ExecutableConvKxKOpSequenceInfo {
  ZeroOp zero;
  SmallVector<LoadMatrixOp, 16> inputLoads;
  SmallVector<LoadMatrixOp, 16> weightLoads;
  SmallVector<MmaOp, 16> mmas;
  StoreMatrixOp store;
  int64_t kernelSize;
};

struct ExecutableResNetBlockOpSequenceInfo {
  LoadMatrixOp xLoad;
  SmallVector<LoadMatrixOp, 2> w1Loads;
  SmallVector<LoadMatrixOp, 2> w2Loads;
  SmallVector<ZeroOp, 5> zeros;
  SmallVector<MmaOp, 4> mmas;
  SmallVector<MaxOp, 3> relus;
  AddOp residualAdd;
  StoreMatrixOp store;
  int64_t hidden;
};

struct ExecutableResNet50BottleneckOpSequenceInfo {
  LoadMatrixOp skipLoad;
  SmallVector<LoadMatrixOp, 2> w1Loads;
  SmallVector<LoadMatrixOp, 9> xWindowLoads;
  SmallVector<LoadMatrixOp, 36> w2Loads;
  SmallVector<LoadMatrixOp, 2> w3Loads;
  SmallVector<ZeroOp, 42> zeros;
  SmallVector<MmaOp, 56> mmas;
  SmallVector<MaxOp, 21> relus;
  AddOp residualAdd;
  StoreMatrixOp store;
  int64_t bottleneck;
};

bool isExecutableF16PointerType(Type type);

FailureOr<KernelOp>
getSingleExecutableKernelFromModule(ModuleOp module, llvm::StringRef consumer);

FailureOr<KernelOp>
getSingleVerifiedExecutableKernelFromModule(ModuleOp module,
                                            llvm::StringRef consumer);

SmallVector<Operation *> getExecutableKernelBodyOps(KernelOp kernel);

FailureOr<SmallVector<Operation *>>
getNonEmptyExecutableKernelBodyOps(KernelOp kernel, llvm::StringRef consumer);

FailureOr<SmallVector<Operation *>>
getCanonicalExecutableKernelBodyOps(KernelOp kernel, llvm::StringRef consumer);

bool isGenericRenderableExecutableKernelKind(llvm::StringRef kind);

bool isGenericRenderableExecutableOp(Operation *op);

ArrayRef<llvm::StringLiteral> getHighLevelLegalizableExecutableOpNames();

bool isHighLevelLegalizableExecutableOpName(llvm::StringRef opName);

ExecutableOpLoweringClass getExecutableOpLoweringClass(Operation *op);

bool isHighLevelLegalizableExecutableOp(Operation *op);

bool isGenericLegalizableExecutableOp(Operation *op);

LogicalResult verifyExecutableModuleRenderable(ModuleOp module);

LogicalResult verifyGenericRenderableExecutableOpSequence(
    KernelOp kernel, ArrayRef<Operation *> ops, llvm::StringRef consumer);

FailureOr<ExecutableVectorMaskInfo>
getExecutableVectorMaskInfo(Operation *op, int64_t n, llvm::StringRef consumer);

bool isExecutableCompactVectorBinaryOp(Operation *op);

LogicalResult verifyExecutableCompactVectorLoad(
    LoadContigOp load, Value expectedPtr, int64_t expectedN,
    const ExecutableVectorMaskInfo &expectedMask, llvm::StringRef consumer);

LogicalResult verifyExecutableCompactVectorBinaryOp(Operation *op,
                                                    int64_t expectedN,
                                                    llvm::StringRef consumer);

LogicalResult
verifyExecutableCompactVectorStore(StoreContigOp store, Value expectedPtr,
                                   Value expectedValue, int64_t expectedN,
                                   const ExecutableVectorMaskInfo &expectedMask,
                                   llvm::StringRef consumer);

LogicalResult buildExecutableCompactElementwise1DBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableCompactElementwise1DBuildSpec &spec,
    llvm::StringRef consumer);

LogicalResult buildExecutableElementwise16ValueMapBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableElementwise16ValueMapBuildSpec &spec,
    llvm::StringRef consumer);

LogicalResult expandExecutableElementwise16ValueMapOp(
    OpBuilder &builder, Elementwise16ValueMapOp map, llvm::StringRef consumer);

LogicalResult expandExecutableCompactElementwise1DOp(OpBuilder &builder,
                                                     CompactElementwise1DOp op,
                                                     llvm::StringRef consumer);

LogicalResult expandExecutableDotOp(OpBuilder &builder, DotOp op,
                                    llvm::StringRef consumer);

LogicalResult expandExecutableSoftmaxOp(OpBuilder &builder, SoftmaxOp softmax,
                                        llvm::StringRef consumer);

LogicalResult buildExecutableAddBody(OpBuilder &builder, Location loc,
                                     KernelOp kernel,
                                     const ExecutableAddBodyBuildSpec &spec,
                                     llvm::StringRef consumer);

LogicalResult
buildExecutableSoftmaxBody(OpBuilder &builder, Location loc, KernelOp kernel,
                           const ExecutableSoftmaxBodyBuildSpec &spec,
                           llvm::StringRef consumer);

LogicalResult buildExecutableSqrtBody(OpBuilder &builder, Location loc,
                                      KernelOp kernel,
                                      const ExecutableSqrtBodyBuildSpec &spec,
                                      llvm::StringRef consumer);

LogicalResult buildExecutableReduceSumAllBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableReduceSumAllBodyBuildSpec &spec, llvm::StringRef consumer);

LogicalResult buildExecutableReluBody(OpBuilder &builder, Location loc,
                                      KernelOp kernel,
                                      const ExecutableReluBodyBuildSpec &spec,
                                      llvm::StringRef consumer);

LogicalResult
buildExecutableMaximumBody(OpBuilder &builder, Location loc, KernelOp kernel,
                           const ExecutableMaximumBodyBuildSpec &spec,
                           llvm::StringRef consumer);

LogicalResult buildExecutableReduceSumAxisBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableReduceSumAxisBodyBuildSpec &spec, llvm::StringRef consumer);

LogicalResult buildExecutableBroadcastAddBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableBroadcastAddBodyBuildSpec &spec, llvm::StringRef consumer);

LogicalResult buildExecutableGemmBody(OpBuilder &builder, Location loc,
                                      KernelOp kernel,
                                      const ExecutableGemmBodyBuildSpec &spec,
                                      llvm::StringRef consumer);

LogicalResult
buildExecutableConvKxKBody(OpBuilder &builder, Location loc, KernelOp kernel,
                           const ExecutableConvKxKBodyBuildSpec &spec,
                           llvm::StringRef consumer);

LogicalResult buildExecutableResNetBlockBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableResNetBlockBodyBuildSpec &spec, llvm::StringRef consumer);

LogicalResult buildExecutableResNet50BottleneckBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableResNet50BottleneckBodyBuildSpec &spec,
    llvm::StringRef consumer);

LogicalResult verifyExecutableElementwise1DOpSequence(KernelOp kernel,
                                                      ArrayRef<Operation *> ops,
                                                      llvm::StringRef consumer);

FailureOr<SmallVector<Operation *>>
getGenericRenderableExecutableKernelBodyOps(KernelOp kernel,
                                            llvm::StringRef consumer);

std::optional<unsigned> getExecutableKernelArgumentIndex(KernelOp kernel,
                                                         Value value);

FailureOr<SmallVector<unsigned, 4>>
getExecutableKernelArgumentIndices(KernelOp kernel, ArrayRef<Value> values,
                                   llvm::StringRef diagnostic);

FailureOr<SmallVector<ExecutableKernelArgumentInfo, 4>>
getExecutableKernelArguments(KernelOp kernel, llvm::StringRef diagnostic);

FailureOr<ExecutableKernelContractInfo>
getVerifiedExecutableKernelContract(KernelOp kernel, llvm::StringRef consumer);

FailureOr<ExecutableKernelContractInfo>
getVerifiedExecutableKernelContractFromModule(ModuleOp module,
                                              llvm::StringRef consumer);

FailureOr<ExecutableKernelScaffold> buildExecutableF16PointerKernelScaffold(
    OpBuilder &builder, ModuleOp module, Location loc,
    llvm::StringRef kernelName, llvm::StringRef kind, unsigned numArgs,
    llvm::StringRef consumer);

FailureOr<ExecutableAddOpSequenceInfo>
getExecutableAddOpSequenceInfo(KernelOp kernel, ArrayRef<Operation *> ops,
                               llvm::StringRef consumer);

FailureOr<ExecutableGemmOpSequenceInfo>
getExecutableGemmOpSequenceInfo(KernelOp kernel, ArrayRef<Operation *> ops,
                                llvm::StringRef consumer);

FailureOr<ExecutableSoftmaxOpSequenceInfo>
getExecutableSoftmaxOpSequenceInfo(KernelOp kernel, ArrayRef<Operation *> ops,
                                   llvm::StringRef consumer);

FailureOr<ExecutableSqrtOpSequenceInfo>
getExecutableSqrtOpSequenceInfo(KernelOp kernel, ArrayRef<Operation *> ops,
                                llvm::StringRef consumer);

FailureOr<ExecutableReduceSumAllOpSequenceInfo>
getExecutableReduceSumAllOpSequenceInfo(KernelOp kernel,
                                        ArrayRef<Operation *> ops,
                                        llvm::StringRef consumer);

FailureOr<ExecutableReluOpSequenceInfo>
getExecutableReluOpSequenceInfo(KernelOp kernel, ArrayRef<Operation *> ops,
                                llvm::StringRef consumer);

FailureOr<ExecutableMaximumOpSequenceInfo>
getExecutableMaximumOpSequenceInfo(KernelOp kernel, ArrayRef<Operation *> ops,
                                   llvm::StringRef consumer);

FailureOr<ExecutableReduceSumAxisOpSequenceInfo>
getExecutableReduceSumAxisOpSequenceInfo(KernelOp kernel,
                                         ArrayRef<Operation *> ops,
                                         llvm::StringRef consumer);

FailureOr<ExecutableBroadcastAddOpSequenceInfo>
getExecutableBroadcastAddOpSequenceInfo(KernelOp kernel,
                                        ArrayRef<Operation *> ops,
                                        llvm::StringRef consumer);

FailureOr<ExecutableConvKxKOpSequenceInfo>
getExecutableConvKxKOpSequenceInfo(KernelOp kernel, ArrayRef<Operation *> ops,
                                   llvm::StringRef consumer);

FailureOr<ExecutableResNetBlockOpSequenceInfo>
getExecutableResNetBlockOpSequenceInfo(KernelOp kernel,
                                       ArrayRef<Operation *> ops,
                                       llvm::StringRef consumer);

FailureOr<ExecutableResNet50BottleneckOpSequenceInfo>
getExecutableResNet50BottleneckOpSequenceInfo(KernelOp kernel,
                                              ArrayRef<Operation *> ops,
                                              llvm::StringRef consumer);

} // namespace exec

struct RPUExecutableKernelSummary {
  std::string kernelName;
  std::string pattern;
};

FailureOr<RPUExecutableKernelSummary>
getRPUExecutableKernelSummaryFromKernelOp(exec::KernelOp op);

FailureOr<RPUExecutableKernelSummary>
getRPUExecutableKernelSummaryFromModule(ModuleOp module);

} // namespace rpu
} // namespace mlir
