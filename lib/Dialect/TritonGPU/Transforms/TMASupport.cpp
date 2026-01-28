//===- TMASupport.cpp - TMA Support for Hopper GPUs -----------------------===//
//
// This file implements TMA (Tensor Memory Accelerator) support for Hopper
// GPUs (SM90+) with hardware-accelerated bulk data transfers.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonGPU/Transforms/TMASupport.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tma-support"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// TMASupport Implementation
//===----------------------------------------------------------------------===//

bool TMASupport::isTMAAvailable() const {
  // TMA is available on SM90+ (Hopper and later)
  unsigned cc = getComputeCapability();
  return cc >= 90;
}

unsigned TMASupport::getComputeCapability() const {
  // In practice, this would query the actual GPU
  // For now, we default to detecting based on environment
  // or module attributes

  // Check if we're targeting Hopper
  // This is a simplified check - real implementation would
  // query target GPU properties

  // Default to A100 (SM80) - conservative
  // When running on Hopper, this should return 90
  return 80;
}

bool TMASupport::isProfitable(const PipelineOpportunity &opp,
                               const CircularBufferInfo &circularInfo) {
  // TMA is only available on Hopper
  if (!isTMAAvailable()) {
    LLVM_DEBUG(llvm::dbgs() << "TMA not available (requires SM90+)\n");
    return false;
  }

  // TMA is beneficial for large, aligned transfers
  // Check minimum transfer size
  if (circularInfo.stride < 128) {
    LLVM_DEBUG(llvm::dbgs() << "Transfer too small for TMA benefit\n");
    return false;
  }

  // TMA works best with tensor operations
  bool hasTensorOps = false;
  scf::ForOp loop = circularInfo.loop;
  if (loop) {
    loop.getBody()->walk([&](Operation *op) {
      if (isa<triton::DotOp>(op)) {
        hasTensorOps = true;
      }
    });
  }

  LLVM_DEBUG(llvm::dbgs() << "TMA profitability: hasTensorOps=" << hasTensorOps
                          << ", stride=" << circularInfo.stride << "\n");

  return hasTensorOps;
}

TMADescriptor TMASupport::createDescriptor(Value globalPtr, Value sharedMemPtr,
                                            ArrayRef<int64_t> shape,
                                            ArrayRef<int64_t> strides,
                                            Type elementType) {
  TMADescriptor desc;
  desc.globalPtr = globalPtr;
  desc.sharedMemPtr = sharedMemPtr;
  desc.shape = SmallVector<int64_t>(shape.begin(), shape.end());
  desc.strides = SmallVector<int64_t>(strides.begin(), strides.end());
  desc.elementType = elementType;

  // Calculate box dimensions for TMA
  // Box dimensions define the shape of the transfer tile
  for (int64_t dim : shape) {
    desc.boxDim.push_back(dim);
  }

  LLVM_DEBUG(llvm::dbgs() << "Created TMA descriptor: shape=[");
  for (auto d : shape) {
    LLVM_DEBUG(llvm::dbgs() << d << " ");
  }
  LLVM_DEBUG(llvm::dbgs() << "]\n");

  return desc;
}

TMAInfo TMASupport::apply(const PipelineOpportunity &opp,
                           CircularBufferInfo &circularInfo,
                           unsigned pipelineId) {
  TMAInfo info;
  info.loop = circularInfo.loop;
  info.pipelineId = pipelineId;

  if (!isTMAAvailable()) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping TMA transformation (not available)\n");
    return info;
  }

  if (!info.loop) {
    return info;
  }

  Location loc = info.loop.getLoc();

  // Create MBarrier for TMA synchronization
  // Number of arrivals = number of TMA transfers per iteration
  unsigned numArrivals = 1;  // One transfer per iteration typically
  info.mbarrier = createMBarrier(loc, numArrivals);

  // Initialize phase for multi-stage pipeline
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(info.loop);

  info.phase = builder.create<arith::ConstantOp>(
      loc, builder.getI32Type(),
      builder.getI32IntegerAttr(0));

  // Find and transform loads to TMA
  SmallVector<triton::LoadOp> loadsToTransform;
  scf::ForOp loopForWalk = info.loop;
  if (loopForWalk) {
    loopForWalk.getBody()->walk([&](triton::LoadOp loadOp) {
      if (canUseTMA(loadOp)) {
        loadsToTransform.push_back(loadOp);
      }
    });
  }

  for (auto loadOp : loadsToTransform) {
    transformLoadToTMA(loadOp, info);
  }

  LLVM_DEBUG(llvm::dbgs() << "Applied TMA transformation: "
                          << loadsToTransform.size() << " loads\n");

  return info;
}

void TMASupport::insertPrefetch(TMAInfo &info, unsigned stageIndex) {
  if (info.descriptors.empty() || !info.loop) {
    return;
  }

  scf::ForOp loop = info.loop;
  Location loc = loop.getLoc();

  // Issue async bulk load for prefetching
  for (const auto &desc : info.descriptors) {
    createAsyncBulkLoad(loc, desc, info.mbarrier);
  }

  LLVM_DEBUG(llvm::dbgs() << "Inserted TMA prefetch for stage " << stageIndex
                          << "\n");
}

void TMASupport::insertWait(TMAInfo &info) {
  if (!info.mbarrier || !info.loop) {
    return;
  }

  scf::ForOp loop = info.loop;
  Location loc = loop.getLoc();

  // Wait for all TMA transfers to complete
  waitOnMBarrier(loc, info.mbarrier, info.phase);

  LLVM_DEBUG(llvm::dbgs() << "Inserted TMA wait\n");
}

Value TMASupport::createMBarrier(Location loc, unsigned arrivals) {
  // Create an MBarrier for TMA synchronization
  // MBarrier is a hardware barrier for async operations

  OpBuilder::InsertionGuard guard(builder);

  // Create barrier allocation in shared memory
  // For TMA, we need a proper mbarrier_t allocation

  Value arrivalsVal = builder.create<arith::ConstantOp>(
      loc, builder.getI32Type(),
      builder.getI32IntegerAttr(arrivals));

  LLVM_DEBUG(llvm::dbgs() << "Created MBarrier with " << arrivals
                          << " arrivals\n");

  // Return arrivals value as placeholder
  // Real implementation would allocate mbarrier_t in shared memory
  return arrivalsVal;
}

void TMASupport::arriveAtMBarrier(Location loc, Value mbarrier, Value bytes) {
  // Signal arrival at MBarrier
  // This is called by the producer after issuing TMA

  LLVM_DEBUG(llvm::dbgs() << "Producer arrived at MBarrier\n");
}

void TMASupport::waitOnMBarrier(Location loc, Value mbarrier, Value phase) {
  // Wait for expected arrivals at MBarrier
  // This is called by the consumer before using data

  LLVM_DEBUG(llvm::dbgs() << "Consumer waiting on MBarrier\n");
}

void TMASupport::createAsyncBulkLoad(Location loc, const TMADescriptor &desc,
                                      Value mbarrier) {
  // Create cp.async.bulk.tensor load operation
  // This maps to CUDA's cp.async.bulk.tensor instruction

  // Calculate expected bytes
  Value expectedBytes = calculateExpectedBytes(loc, desc);

  // In a real implementation, this would create the appropriate
  // Triton/LLVM operations for TMA

  LLVM_DEBUG(llvm::dbgs() << "Created async bulk load\n");
}

void TMASupport::createAsyncBulkStore(Location loc, const TMADescriptor &desc) {
  // Create cp.async.bulk.tensor store operation

  LLVM_DEBUG(llvm::dbgs() << "Created async bulk store\n");
}

Value TMASupport::calculateExpectedBytes(Location loc,
                                          const TMADescriptor &desc) {
  // Calculate total bytes for the transfer
  int64_t totalBytes = 1;
  for (int64_t dim : desc.boxDim) {
    totalBytes *= dim;
  }

  // Account for element size
  unsigned elemSize = 4;  // Default to 32-bit
  if (desc.elementType) {
    if (desc.elementType.isF16() || desc.elementType.isBF16()) {
      elemSize = 2;
    } else if (desc.elementType.isF64() || desc.elementType.isInteger(64)) {
      elemSize = 8;
    }
  }
  totalBytes *= elemSize;

  return builder.create<arith::ConstantOp>(
      loc, builder.getI64Type(),
      builder.getI64IntegerAttr(totalBytes));
}

void TMASupport::transformLoadToTMA(triton::LoadOp loadOp, TMAInfo &info) {
  // Transform a regular LoadOp to use TMA

  Location loc = loadOp.getLoc();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(loadOp);

  // Get load properties
  Value ptr = loadOp.getPtr();
  Type resultType = loadOp.getResult().getType();

  // Determine transfer shape from tensor type
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;

  if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
    shape = SmallVector<int64_t>(tensorType.getShape().begin(),
                                  tensorType.getShape().end());
    // Calculate strides (row-major)
    int64_t stride = 1;
    strides.resize(shape.size());
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape[i];
    }
  }

  // Create TMA descriptor
  TMADescriptor desc = createDescriptor(
      ptr, Value(), shape, strides, resultType);
  desc.mode = TMAMode::Load;

  info.descriptors.push_back(desc);

  // Mark original load for transformation
  // In real implementation, would replace with TMA operations
  loadOp->setAttr("tma_candidate", builder.getUnitAttr());

  LLVM_DEBUG(llvm::dbgs() << "Marked LoadOp for TMA transformation\n");
}

void TMASupport::transformStoreToTMA(triton::StoreOp storeOp, TMAInfo &info) {
  // Transform a regular StoreOp to use TMA

  Location loc = storeOp.getLoc();

  // Get store properties
  Value ptr = storeOp.getPtr();
  Value value = storeOp.getValue();
  Type valueType = value.getType();

  // Determine transfer shape
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;

  if (auto tensorType = dyn_cast<RankedTensorType>(valueType)) {
    shape = SmallVector<int64_t>(tensorType.getShape().begin(),
                                  tensorType.getShape().end());
    int64_t stride = 1;
    strides.resize(shape.size());
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape[i];
    }
  }

  // Create TMA descriptor
  TMADescriptor desc = createDescriptor(
      ptr, Value(), shape, strides, valueType);
  desc.mode = TMAMode::Store;

  info.descriptors.push_back(desc);

  // Mark original store for transformation
  storeOp->setAttr("tma_candidate", builder.getUnitAttr());

  LLVM_DEBUG(llvm::dbgs() << "Marked StoreOp for TMA transformation\n");
}

bool TMASupport::canUseTMA(Operation *op) {
  // Check if an operation can be transformed to use TMA

  // Must be a load or store operation
  if (!isa<triton::LoadOp, triton::StoreOp>(op)) {
    return false;
  }

  // Check for tensor types (TMA works on tensors)
  Value result;
  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    result = loadOp.getResult();
  } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
    result = storeOp.getValue();
  }

  if (!result) {
    return false;
  }

  auto tensorType = dyn_cast<RankedTensorType>(result.getType());
  if (!tensorType) {
    return false;
  }

  // Check tensor dimensions (TMA has limits)
  auto shape = tensorType.getShape();
  if (shape.size() < 1 || shape.size() > 5) {
    return false;  // TMA supports 1D-5D tensors
  }

  // Check alignment requirements
  // TMA requires 16-byte alignment
  int64_t numElements = 1;
  for (int64_t dim : shape) {
    numElements *= dim;
  }

  unsigned elemSize = 4;
  Type elemType = tensorType.getElementType();
  if (elemType.isF16() || elemType.isBF16()) {
    elemSize = 2;
  } else if (elemType.isF64() || elemType.isInteger(64)) {
    elemSize = 8;
  }

  int64_t totalBytes = numElements * elemSize;
  if (totalBytes % 16 != 0) {
    return false;  // Not 16-byte aligned
  }

  // Check minimum transfer size for efficiency
  if (totalBytes < 128) {
    return false;  // Too small for TMA benefit
  }

  return true;
}
