//===- AscendModelOps.cpp - AscendModel operations implementation ---------===//
//
// This file implements the operations in the AscendModel dialect.
// Each operation implements the EstimateCyclesOpInterface to provide
// hardware-aware cycle estimation.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Analysis/PipelineAnalysis.h"
#include "AscendModel/HardwareConfig.h"
#include "AscendModel/IR/AscendModelDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::ascend;

namespace {
/// tilesim bandwidth convention: 1 GB/s == (1024^3)/1e6 B/us. Must match
/// HardwareConfig::estimateTransferCycles / lookupBandwidth exactly.
constexpr double kBytesPerUsPerGbs = (1024.0 * 1024.0 * 1024.0) / 1e6;
} // namespace

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Get number of elements from a tensor type.
static int64_t getNumElementsFromType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    int64_t count = 1;
    for (int64_t dim : tensorType.getShape()) {
      if (dim == ShapedType::kDynamic)
        return 1024; // Default estimate for dynamic shapes
      count *= dim;
    }
    return count;
  }
  return 1;
}

/// Get element bit width from a tensor type.
static int getElementBitsFromType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    Type elemType = tensorType.getElementType();
    if (elemType.isF16() || elemType.isBF16())
      return 16;
    else if (elemType.isF32())
      return 32;
    else if (elemType.isF64())
      return 64;
    else if (auto intType = dyn_cast<IntegerType>(elemType))
      return intType.getWidth();
  }
  return 32; // Default to FP32
}

/// Estimate vector operation cycles.
/// Vector unit is 2048 bits wide = 128 FP16 = 64 FP32 elements per cycle.
static int64_t estimateVectorCycles(int64_t numElements, int cyclesPerVectorOp,
                                    int elementBits, int startupLatency) {
  // Vector width based on element type
  int64_t vectorWidth = 2048 / elementBits;

  // Number of vector operations
  int64_t numVectorOps = (numElements + vectorWidth - 1) / vectorWidth;

  // Total cycles
  return numVectorOps * cyclesPerVectorOp + startupLatency;
}

/// Estimate memory transfer cycles.
static int64_t estimateMemoryCycles(int64_t bytes, const HardwareConfig &config,
                                    int startupLatency) {
  double bandwidth_gbs = config.getHBMBandwidthGBs();
  if (bandwidth_gbs <= 0)
    bandwidth_gbs = 1600.0;

  double time_seconds = static_cast<double>(bytes) / (bandwidth_gbs * 1e9);
  double cycles = time_seconds * config.getClockFrequencyGHz() * 1e9;
  return static_cast<int64_t>(cycles) + startupLatency;
}

/// tilesim-migrated vector estimate (root cause 1, vector side):
///   repeats = ceil(numElems * elemBytes / 256B)   (vec_size_single_repeat)
///   cycles  = compute_cycles(intrinsic, dtype) * repeats + startup
/// Mirrors tilesim plain_simple_op_based_on_cce. The calibrated
/// getVectorStartupLatency() is retained as the per-op pipeline startup
/// anchor; intrinsic compute costs come from the migrated 910B1 table.
static int64_t estimateVectorFromTable(Type type, llvm::StringRef intrinsic,
                                       const HardwareConfig &config) {
  int64_t n = getNumElementsFromType(type);
  int bits = getElementBitsFromType(type);
  int elemBytes = (bits + 7) / 8;
  // 256 B per repeat == 2048 bits / elementBits elements per repeat.
  int64_t repeats = (n * elemBytes + 255) / 256;
  VecCycleEntry e = config.lookupVecCycle(intrinsic, bits);
  int64_t compute = static_cast<int64_t>(e.compute) * repeats;
  return compute + config.getVectorStartupLatency();
}

//===----------------------------------------------------------------------===//
// Include generated definitions
//===----------------------------------------------------------------------===//

#include "AscendModel/IR/AscendModelInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "AscendModel/IR/AscendModelOps.cpp.inc"

//===----------------------------------------------------------------------===//
// EstimateCyclesOpInterface helpers (defined in interface)
//===----------------------------------------------------------------------===//

int EstimateCyclesOpInterface::getElementBits() {
  Operation *op = getOperation();
  if (op->getNumOperands() > 0)
    return getElementBitsFromType(op->getOperand(0).getType());
  if (op->getNumResults() > 0)
    return getElementBitsFromType(op->getResult(0).getType());
  return 32;
}

int64_t EstimateCyclesOpInterface::getNumElements() {
  Operation *op = getOperation();
  if (op->getNumOperands() > 0)
    return getNumElementsFromType(op->getOperand(0).getType());
  if (op->getNumResults() > 0)
    return getNumElementsFromType(op->getResult(0).getType());
  return 1;
}

//===----------------------------------------------------------------------===//
// MatmulOp Interface Implementation
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MatmulOp Interface Implementation (L1-pipe roofline — root cause 2)
//===----------------------------------------------------------------------===//
//
// Migrated from tilesim plain_gemm_based_on_l1_pipe. Models the Cube core as a
// real pipeline: the steady-state latency is the max of (L1->L0 input copy) and
// (cube compute), then the first-tile warmup (pipeline fill) is added on top.
// Input copy uses per-mover L1->L0A/L0B bandwidth (441/220.5 GB/s) instead of
// aggregate HBM. HBM->L1 copy is accounted by the separate cube_load op, so it
// is NOT double-counted here (design 4.2.3 option b).
//
//   k0          = best_k0(m, n)            # largest k0 fitting 32KB L0 tiles
//   copy_time   = m*k*dt/L0A_bw + n*k*dt/L0B_bw
//   compute     = ceil(m/bm)*ceil(k/bk)*ceil(n/bn) * repeat_cycles[dt] / clock
//   first_tile  = (m*k0 + k0*n)*dt / bw
//   latency     = max(copy_time - first_tile, compute) + first_tile

int64_t MatmulOp::estimateCycles(const HardwareConfig &config) {
  int64_t m = getM();
  int64_t n = getN();
  int64_t k = getK();

  int elemBits = getElementBitsFromType(getLhs().getType());
  double dtBytes = (elemBits + 7) / 8;
  double clockMHz = config.getClockFrequencyGHz() * 1000.0;

  // best_k0: tilesim _calculate_best_k0 — largest k0 whose (m,k0) and (k0,n)
  // tiles fit the L0 footprint limit (tilesim uses a fixed *2 byte factor).
  int64_t tileLimit = config.getCubeModelL0TileLimitBytes();
  auto fits = [&](int64_t k0) {
    return m * k0 * 2 <= tileLimit && n * k0 * 2 <= tileLimit;
  };
  int64_t k0 = 32;
  if (fits(256))
    k0 = 256;
  else if (fits(128))
    k0 = 128;
  else if (fits(64))
    k0 = 64;

  int64_t sizeA0 = m * k0 * (int64_t)dtBytes;
  int64_t sizeB0 = k0 * n * (int64_t)dtBytes;

  // L1 -> L0A / L0B copy bandwidth (MTE1), in B/us (tilesim GiB convention).
  // tilesim cube_op passes the k0-tile sizes (size_a0/size_b0) as pkt_size so
  // the small-packet fitting can apply to tiny L1->L0 tiles; mirror that here.
  auto bwBus = [&](llvm::StringRef src, llvm::StringRef dst,
                   int64_t pktBytes) -> double {
    BandwidthEntry bw = config.lookupBandwidth(src, dst, 1, pktBytes);
    double gbps = bw.bwGBs > 0 ? bw.bwGBs : config.getHBMBandwidthGBs();
    return gbps * kBytesPerUsPerGbs;
  };
  double l0aBw = bwBus("l1", "l0a", sizeA0);
  double l0bBw = bwBus("l1", "l0b", sizeB0);

  double copyTime = (m * k * dtBytes) / l0aBw + (n * k * dtBytes) / l0bBw;

  // Cube compute (L1-pipe roofline).
  int basicM = 16, basicK = 16, basicN = 16;
  config.getCubeModelThroughput(elemBits, basicM, basicK, basicN);
  int repeatCycles = config.getCubeModelRepeatCycles(elemBits);
  auto ceilDiv = [](int64_t a, int64_t b) -> int64_t {
    return b > 0 ? (a + b - 1) / b : 0;
  };
  int64_t repeats =
      ceilDiv(m, basicM) * ceilDiv(k, basicK) * ceilDiv(n, basicN);
  double computeTime = static_cast<double>(repeats * repeatCycles) / clockMHz;

  // First-tile warmup (pipeline fill).
  double firstTile =
      static_cast<double>(sizeA0) / l0aBw + static_cast<double>(sizeB0) / l0bBw;

  double latency = std::max(copyTime - firstTile, computeTime) + firstTile;
  int64_t cycles = static_cast<int64_t>(latency * clockMHz);
  if (cycles <= 0)
    cycles = 1;
  return cycles + config.getCubeStartupLatency();
}

HWUnit MatmulOp::getHWUnit() { return HWUnit::Cube; }

int64_t MatmulOp::getFlops() { return 2 * getM() * getN() * getK(); }

//===----------------------------------------------------------------------===//
// Memory Operations Interface Implementation
//===----------------------------------------------------------------------===//
//
// Migrated to tilesim per-(src,dst,corenum) bandwidth lookup (root cause 1).
// Each transfer op carries optional src_space/dst_space attributes (defaulting
// to the op's natural memory path) so estimateTransferCycles can pick the
// correct mover bandwidth instead of the aggregate HBM figure.

int64_t CubeLoadOp::estimateCycles(const HardwareConfig &config) {
  return config.estimateTransferCycles(getSrcMemSpace(), getDstMemSpace(),
                                       getBytes(),
                                       config.getActiveBandwidthCores()) +
         config.getMTE2StartupLatency();
}
llvm::StringRef CubeLoadOp::getSrcMemSpace() {
  if (auto a = getSrcSpaceAttr())
    return a.getValue();
  return "hbm";
}
llvm::StringRef CubeLoadOp::getDstMemSpace() {
  if (auto a = getDstSpaceAttr())
    return a.getValue();
  return "l1";
}
HWUnit CubeLoadOp::getHWUnit() { return HWUnit::CubeMTE2; }
int64_t CubeLoadOp::getTransferBytes() { return getBytes(); }

int64_t CubeStoreOp::estimateCycles(const HardwareConfig &config) {
  return config.estimateTransferCycles(getSrcMemSpace(), getDstMemSpace(),
                                       getBytes(),
                                       config.getActiveBandwidthCores()) +
         config.getFixPipeStartupLatency();
}
llvm::StringRef CubeStoreOp::getSrcMemSpace() {
  if (auto a = getSrcSpaceAttr())
    return a.getValue();
  return "l0c";
}
llvm::StringRef CubeStoreOp::getDstMemSpace() {
  if (auto a = getDstSpaceAttr())
    return a.getValue();
  return "hbm";
}
HWUnit CubeStoreOp::getHWUnit() { return HWUnit::FixPipe; }
int64_t CubeStoreOp::getTransferBytes() { return getBytes(); }

int64_t VectorLoadOp::estimateCycles(const HardwareConfig &config) {
  return config.estimateTransferCycles(getSrcMemSpace(), getDstMemSpace(),
                                       getBytes(),
                                       config.getActiveBandwidthCores()) +
         config.getMTE2StartupLatency();
}
llvm::StringRef VectorLoadOp::getSrcMemSpace() {
  if (auto a = getSrcSpaceAttr())
    return a.getValue();
  return "hbm";
}
llvm::StringRef VectorLoadOp::getDstMemSpace() {
  if (auto a = getDstSpaceAttr())
    return a.getValue();
  return "ub";
}
HWUnit VectorLoadOp::getHWUnit() { return HWUnit::VecMTE2; }
int64_t VectorLoadOp::getTransferBytes() { return getBytes(); }

int64_t VectorStoreOp::estimateCycles(const HardwareConfig &config) {
  return config.estimateTransferCycles(getSrcMemSpace(), getDstMemSpace(),
                                       getBytes(),
                                       config.getActiveBandwidthCores()) +
         config.getMTE3StartupLatency();
}
llvm::StringRef VectorStoreOp::getSrcMemSpace() {
  if (auto a = getSrcSpaceAttr())
    return a.getValue();
  return "ub";
}
llvm::StringRef VectorStoreOp::getDstMemSpace() {
  if (auto a = getDstSpaceAttr())
    return a.getValue();
  return "hbm";
}
HWUnit VectorStoreOp::getHWUnit() { return HWUnit::MTE3; }
int64_t VectorStoreOp::getTransferBytes() { return getBytes(); }

//===----------------------------------------------------------------------===//
// Simple Vector Binary Operations
//===----------------------------------------------------------------------===//
// Migrated to tilesim per-(intrinsic,dtype) cycle table (root cause 1).

#define IMPL_VEC_BINARY(OpClass, Intrinsic)                                    \
  int64_t OpClass::estimateCycles(const HardwareConfig &config) {              \
    return estimateVectorFromTable(getLhs().getType(), Intrinsic, config);     \
  }                                                                            \
  HWUnit OpClass::getHWUnit() { return HWUnit::Vector; }                       \
  int64_t OpClass::getFlops() {                                                \
    return getNumElementsFromType(getLhs().getType());                         \
  }

IMPL_VEC_BINARY(AddOp, "VADD")
IMPL_VEC_BINARY(SubOp, "VSUB")
IMPL_VEC_BINARY(MulOp, "VMUL")
IMPL_VEC_BINARY(MaxOp, "VMAX")
IMPL_VEC_BINARY(MinOp, "VMIN")
// Division is latency-limited; tilesim VDIV table value (fp16:8, fp32:4).
IMPL_VEC_BINARY(DivOp, "VDIV")
// DivOp (VectorBinaryComplex) declares getCyclesPerVectorOp; the macro above
// already provides estimateCycles/getHWUnit/getFlops, so only the latency hook
// is defined separately. tilesim VDIV table value (fp16:8, fp32:4).
int DivOp::getCyclesPerVectorOp() { return 4; }

#undef IMPL_VEC_BINARY

//===----------------------------------------------------------------------===//
// Vector Comparison Operations
//===----------------------------------------------------------------------===//
// tilesim maps equal->CMPV_EQ, not-equal->VCMPV_NE, comparisons->VCMPV_GE
// (all ~2 compute cycles/repeat).

#define IMPL_VEC_CMP(OpClass, Intrinsic)                                       \
  int64_t OpClass::estimateCycles(const HardwareConfig &config) {              \
    return estimateVectorFromTable(getLhs().getType(), Intrinsic, config);     \
  }                                                                            \
  HWUnit OpClass::getHWUnit() { return HWUnit::Vector; }                       \
  int64_t OpClass::getFlops() {                                                \
    return getNumElementsFromType(getLhs().getType());                         \
  }

IMPL_VEC_CMP(CmpEqOp, "CMPV_EQ")
IMPL_VEC_CMP(CmpNeOp, "VCMPV_NE")
IMPL_VEC_CMP(CmpLtOp, "VCMPV_NE")
IMPL_VEC_CMP(CmpLeOp, "VCMPV_NE")
IMPL_VEC_CMP(CmpGtOp, "VCMPV_GE")
IMPL_VEC_CMP(CmpGeOp, "VCMPV_GE")

#undef IMPL_VEC_CMP

//===----------------------------------------------------------------------===//
// Simple Vector Unary Operations
//===----------------------------------------------------------------------===//

#define IMPL_VEC_UNARY_TABLE(OpClass, Intrinsic)                               \
  int64_t OpClass::estimateCycles(const HardwareConfig &config) {              \
    return estimateVectorFromTable(getInput().getType(), Intrinsic, config);   \
  }                                                                            \
  HWUnit OpClass::getHWUnit() { return HWUnit::Vector; }                       \
  int64_t OpClass::getFlops() {                                                \
    return getNumElementsFromType(getInput().getType());                       \
  }

IMPL_VEC_UNARY_TABLE(AbsOp, "VABS")
IMPL_VEC_UNARY_TABLE(ReluOp, "RELU")
IMPL_VEC_UNARY_TABLE(ExpOp, "VEXP")
IMPL_VEC_UNARY_TABLE(LogOp, "LOG")
IMPL_VEC_UNARY_TABLE(SqrtOp, "VSQRT")

#undef IMPL_VEC_UNARY_TABLE

// Ops without a direct tilesim intrinsic keep the legacy calibrated estimate
// (Neg, Cast, Rsqrt, Tanh, Sigmoid). These are minor contributors; the
// bandwidth migration (root cause 1) is the dominant accuracy win. The macro
// deliberately omits getCyclesPerVectorOp: Neg/Cast are VectorUnarySimple and
// do not declare it, so defining it would be an error. The Complex unary ops
// that DO declare it provide it standalone below.
#define IMPL_VEC_UNARY_LEGACY(OpClass, CyclesPerOp)                            \
  int64_t OpClass::estimateCycles(const HardwareConfig &config) {              \
    int64_t n = getNumElementsFromType(getInput().getType());                  \
    int bits = getElementBitsFromType(getInput().getType());                   \
    return estimateVectorCycles(n, CyclesPerOp, bits,                          \
                                config.getVectorStartupLatency());             \
  }                                                                            \
  HWUnit OpClass::getHWUnit() { return HWUnit::Vector; }                       \
  int64_t OpClass::getFlops() {                                                \
    return getNumElementsFromType(getInput().getType());                       \
  }

IMPL_VEC_UNARY_LEGACY(NegOp, 1)
IMPL_VEC_UNARY_LEGACY(CastOp, 1)
IMPL_VEC_UNARY_LEGACY(RsqrtOp, 6)
IMPL_VEC_UNARY_LEGACY(TanhOp, 18)
IMPL_VEC_UNARY_LEGACY(SigmoidOp, 15)

#undef IMPL_VEC_UNARY_LEGACY

// VectorUnaryComplex ops (Sqrt/Rsqrt/Exp/Log/Tanh/Sigmoid) declare
// getCyclesPerVectorOp via their ODS base, so a definition is required for the
// link to succeed. estimateCycles above is table/legacy-based and does not set
// it; values retained from the pre-migration transcendental calibration.
// (getCyclesPerVectorOp is a separate interface hook not consumed by the
// current pipeline metric.)
int SqrtOp::getCyclesPerVectorOp() { return 6; }
int RsqrtOp::getCyclesPerVectorOp() { return 6; }
int ExpOp::getCyclesPerVectorOp() { return 9; }
int LogOp::getCyclesPerVectorOp() { return 12; }
int TanhOp::getCyclesPerVectorOp() { return 18; }
int SigmoidOp::getCyclesPerVectorOp() { return 15; }

//===----------------------------------------------------------------------===//
// Reduce Operations
//===----------------------------------------------------------------------===//

#define IMPL_REDUCE_OP(OpClass)                                                \
  int64_t OpClass::estimateCycles(const HardwareConfig &config) {              \
    int64_t numElems = getNumElementsFromType(getInput().getType());           \
    int bits = getElementBitsFromType(getInput().getType());                   \
    int64_t vectorWidth = 2048 / bits;                                         \
    int64_t numVectors = (numElems + vectorWidth - 1) / vectorWidth;           \
    int vectorReduceCycles = 0;                                                \
    int64_t w = vectorWidth;                                                   \
    while (w > 1) {                                                            \
      w /= 2;                                                                  \
      vectorReduceCycles++;                                                    \
    }                                                                          \
    int crossVectorCycles = 0;                                                 \
    int64_t v = numVectors;                                                    \
    while (v > 1) {                                                            \
      v /= 2;                                                                  \
      crossVectorCycles++;                                                     \
    }                                                                          \
    return numVectors + vectorReduceCycles + crossVectorCycles +               \
           config.getVectorStartupLatency();                                   \
  }                                                                            \
  HWUnit OpClass::getHWUnit() { return HWUnit::Vector; }                       \
  int64_t OpClass::getFlops() {                                                \
    return getNumElementsFromType(getInput().getType());                       \
  }

IMPL_REDUCE_OP(ReduceSumOp)
IMPL_REDUCE_OP(ReduceMaxOp)
IMPL_REDUCE_OP(ReduceMinOp)
IMPL_REDUCE_OP(ReduceProdOp)

#undef IMPL_REDUCE_OP

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

int64_t BroadcastOp::estimateCycles(const HardwareConfig &config) {
  // Broadcast is essentially free, just startup latency
  return 1 + config.getVectorStartupLatency();
}
HWUnit BroadcastOp::getHWUnit() { return HWUnit::Vector; }

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

int64_t SelectOp::estimateCycles(const HardwareConfig &config) {
  int64_t n = getNumElementsFromType(getResult().getType());
  int bits = getElementBitsFromType(getResult().getType());
  return estimateVectorCycles(n, 1, bits, config.getVectorStartupLatency());
}
HWUnit SelectOp::getHWUnit() { return HWUnit::Vector; }
int64_t SelectOp::getFlops() {
  return getNumElementsFromType(getResult().getType());
}
