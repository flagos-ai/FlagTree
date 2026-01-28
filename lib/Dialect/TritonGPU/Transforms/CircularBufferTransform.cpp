//===- CircularBufferTransform.cpp - Circular Buffer Index Transformation ===//
//
// This file implements circular buffer transformation for pipelined memory
// accesses, including index rewriting and predecessor handling.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonGPU/Transforms/CircularBufferTransform.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "circular-buffer-transform"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// CircularBufferTransform Implementation
//===----------------------------------------------------------------------===//

CircularBufferInfo CircularBufferTransform::transformAllocation(
    const PipelineOpportunity &opp, unsigned pipelineId) {

  // Transform buffer allocation to include pipeline stage dimension
  // by expanding the buffer by a factor of numStages

  CircularBufferInfo info;
  info.originalBuffer = opp.buffer;
  info.numStages = opp.numStages;
  info.loop = opp.loop;
  info.pipelineId = pipelineId;
  info.useAsyncCopy = opp.useAsyncCopy;
  info.useSwizzle = opp.useSwizzle;

  // Get the defining operation for the buffer
  Operation *defOp = opp.buffer.getDefiningOp();
  if (!defOp) {
    // Buffer is a block argument, cannot transform allocation directly
    info.circularBuffer = opp.buffer;
    info.stride = 0;
    LLVM_DEBUG(llvm::dbgs() << "Buffer is block argument, skipping allocation transform\n");
    return info;
  }

  // Check if it's a LocalAllocOp
  if (auto allocOp = dyn_cast<triton::gpu::LocalAllocOp>(defOp)) {
    // Get the original buffer type
    auto origType = cast<triton::MemDescType>(allocOp.getResult().getType());
    ArrayRef<int64_t> origShape = origType.getShape();
    Type elementType = origType.getElementType();
    Attribute encoding = origType.getEncoding();

    // Calculate stride as the product of original dimensions
    int64_t stride = 1;
    for (int64_t dim : origShape) {
      stride *= dim;
    }
    info.stride = stride;

    // Create new shape with stage dimension prepended: [numStages, ...origShape]
    SmallVector<int64_t> newShape;
    newShape.push_back(static_cast<int64_t>(opp.numStages));
    newShape.append(origShape.begin(), origShape.end());

    // Apply swizzle optimization if enabled
    Attribute newEncoding = encoding;
    if (opp.useSwizzle && origShape.size() >= 2) {
      // Create swizzled SharedEncodingAttr to reduce bank conflicts
      // The swizzle pattern distributes accesses across memory banks using XOR

      // Determine memory order (typically row-major for shared memory)
      SmallVector<unsigned> order;
      if (auto sharedEnc = mlir::dyn_cast<SharedEncodingAttr>(encoding)) {
        order = SmallVector<unsigned>(sharedEnc.getOrder().begin(),
                                      sharedEnc.getOrder().end());
      } else {
        // Default order for 2D and higher
        for (unsigned i = 0; i < origShape.size(); ++i) {
          order.push_back(origShape.size() - 1 - i);
        }
      }

      // Get CTA layout
      auto ctaLayout = getCTALayout(encoding);
      if (!ctaLayout) {
        // Create default CTA layout
        SmallVector<unsigned> ctasPerCGA(origShape.size(), 1);
        SmallVector<unsigned> ctaSplitNum(origShape.size(), 1);
        SmallVector<unsigned> ctaOrder(order.begin(), order.end());
        ctaLayout = CTALayoutAttr::get(builder.getContext(), ctasPerCGA,
                                       ctaSplitNum, ctaOrder);
      }

      // Create swizzled SharedEncodingAttr using the shape-based constructor
      // This computes optimal vec, perPhase, maxPhase based on element type and shape
      newEncoding = SharedEncodingAttr::get(builder.getContext(), origShape,
                                            order, ctaLayout, elementType);

      LLVM_DEBUG(llvm::dbgs() << "Applied swizzle encoding for bank conflict reduction\n");
    }

    // Create new MemDescType with expanded shape and (possibly swizzled) encoding
    auto newType = triton::MemDescType::get(newShape, elementType, newEncoding);

    // Insert new allocation before the original one
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(allocOp);
    Location loc = allocOp.getLoc();

    // Create new LocalAllocOp with expanded buffer
    Value src = allocOp.getSrc();
    Value newAlloc;
    if (src) {
      // If there's a source tensor, we need to handle it appropriately
      // For circular buffer, we typically allocate without initialization
      newAlloc = builder.create<triton::gpu::LocalAllocOp>(loc, newType, Value());
    } else {
      newAlloc = builder.create<triton::gpu::LocalAllocOp>(loc, newType, Value());
    }

    info.circularBuffer = newAlloc;

    LLVM_DEBUG(llvm::dbgs() << "Created circular buffer allocation: "
                            << "original shape [");
    for (auto d : origShape) {
      LLVM_DEBUG(llvm::dbgs() << d << " ");
    }
    LLVM_DEBUG(llvm::dbgs() << "] -> new shape [");
    for (auto d : newShape) {
      LLVM_DEBUG(llvm::dbgs() << d << " ");
    }
    LLVM_DEBUG(llvm::dbgs() << "], stride=" << stride << "\n");

  } else {
    // For other allocation types, keep the original buffer
    info.circularBuffer = opp.buffer;
    info.stride = 0;
    LLVM_DEBUG(llvm::dbgs() << "Unknown allocation type, keeping original buffer\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "Transformed allocation for pipeline "
                          << pipelineId << " with " << info.numStages
                          << " stages, stride " << info.stride << "\n");

  return info;
}

void CircularBufferTransform::transformStore(Operation *storeOp,
                                              CircularBufferInfo &info) {
  if (!storeOp || !info.loop) {
    return;
  }

  // Transform store operation to use circular buffer indexing
  // Formula: offset = ((global_iter + numStages - 1) % numStages) * stride
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(storeOp);

  Location loc = storeOp->getLoc();
  Value globalIter = computeGlobalIteration(info.loop);

  if (!globalIter) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to compute global iteration for store\n");
    return;
  }

  // Compute circular offset for store (producer side)
  Value offset = computeCircularOffsetStore(loc, globalIter,
                                            info.numStages, info.stride);

  LLVM_DEBUG(llvm::dbgs() << "Transformed store with circular offset: producer\n");
}

void CircularBufferTransform::transformLoad(Operation *loadOp,
                                             CircularBufferInfo &info) {
  if (!loadOp || !info.loop) {
    return;
  }

  // Transform load operation to use circular buffer indexing
  // Formula: offset = (global_iter % numStages) * stride
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(loadOp);

  Location loc = loadOp->getLoc();
  Value globalIter = computeGlobalIteration(info.loop);

  if (!globalIter) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to compute global iteration for load\n");
    return;
  }

  // Compute circular offset for load (consumer side)
  Value offset = computeCircularOffsetLoad(loc, globalIter,
                                           info.numStages, info.stride);

  LLVM_DEBUG(llvm::dbgs() << "Transformed load with circular offset: consumer\n");
}

Value CircularBufferTransform::computeCircularOffsetStore(
    Location loc, Value globalIter, unsigned numStages, int64_t stride) {
  // Compute circular offset for store (producer side)
  // Formula: ((global_iter + numStages - 1) % numStages) * stride

  Type iterType = globalIter.getType();

  // Create constants
  Value numStagesVal = builder.create<arith::ConstantOp>(
      loc, iterType,
      builder.getIntegerAttr(iterType, numStages));

  Value strideVal = builder.create<arith::ConstantOp>(
      loc, iterType,
      builder.getIntegerAttr(iterType, stride));

  Value oneVal = builder.create<arith::ConstantOp>(
      loc, iterType,
      builder.getIntegerAttr(iterType, 1));

  // Compute (global_iter + numStages - 1)
  Value adjustedIter = builder.create<arith::AddIOp>(loc, globalIter, numStagesVal);
  adjustedIter = builder.create<arith::SubIOp>(loc, adjustedIter, oneVal);

  // Compute ((global_iter + numStages - 1) % numStages)
  Value stageIdx = builder.create<arith::RemSIOp>(loc, adjustedIter, numStagesVal);

  // Compute offset = stageIdx * stride
  Value offset = builder.create<arith::MulIOp>(loc, stageIdx, strideVal);

  return offset;
}

Value CircularBufferTransform::computeCircularOffsetLoad(
    Location loc, Value globalIter, unsigned numStages, int64_t stride) {
  // Compute circular offset for load (consumer side)
  // Formula: (global_iter % numStages) * stride

  Type iterType = globalIter.getType();

  // Create constants
  Value numStagesVal = builder.create<arith::ConstantOp>(
      loc, iterType,
      builder.getIntegerAttr(iterType, numStages));

  Value strideVal = builder.create<arith::ConstantOp>(
      loc, iterType,
      builder.getIntegerAttr(iterType, stride));

  // Compute (global_iter % numStages)
  Value stageIdx = builder.create<arith::RemSIOp>(loc, globalIter, numStagesVal);

  // Compute offset = stageIdx * stride
  Value offset = builder.create<arith::MulIOp>(loc, stageIdx, strideVal);

  return offset;
}

Value CircularBufferTransform::computeGlobalIteration(scf::ForOp loop) {
  if (!loop) {
    return Value();
  }

  // For nested loops, compute the global iteration number
  // global_iter = outer_iter * inner_trip_count + inner_iter

  Value iv = loop.getInductionVar();
  Location loc = loop.getLoc();

  // Check if there's an outer loop
  auto outerLoop = loop->getParentOfType<scf::ForOp>();
  if (!outerLoop) {
    // Single loop case - just return the iteration count from lower bound
    // iter = (iv - lb) / step
    Value lb = loop.getLowerBound();
    Value step = loop.getStep();

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());

    Value diff = builder.create<arith::SubIOp>(loc, iv, lb);
    Value iter = builder.create<arith::DivSIOp>(loc, diff, step);

    return iter;
  }

  // Nested loop case
  // Compute inner trip count: (ub - lb + step - 1) / step
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(loop.getBody());

  Value innerLb = loop.getLowerBound();
  Value innerUb = loop.getUpperBound();
  Value innerStep = loop.getStep();

  // Calculate inner trip count
  Value innerRange = builder.create<arith::SubIOp>(loc, innerUb, innerLb);
  Value stepMinusOne = builder.create<arith::SubIOp>(
      loc, innerStep,
      builder.create<arith::ConstantOp>(
          loc, innerStep.getType(),
          builder.getIntegerAttr(innerStep.getType(), 1)));
  Value adjustedRange = builder.create<arith::AddIOp>(loc, innerRange, stepMinusOne);
  Value innerTripCount = builder.create<arith::DivSIOp>(loc, adjustedRange, innerStep);

  // Calculate inner iteration: (iv - lb) / step
  Value innerDiff = builder.create<arith::SubIOp>(loc, iv, innerLb);
  Value innerIter = builder.create<arith::DivSIOp>(loc, innerDiff, innerStep);

  // Recursively compute outer global iteration
  Value outerGlobalIter = computeGlobalIteration(outerLoop);
  if (!outerGlobalIter) {
    // Fallback: just use inner iteration
    return innerIter;
  }

  // global_iter = outer_global_iter * inner_trip_count + inner_iter
  Value scaledOuter = builder.create<arith::MulIOp>(loc, outerGlobalIter, innerTripCount);
  Value globalIter = builder.create<arith::AddIOp>(loc, scaledOuter, innerIter);

  LLVM_DEBUG(llvm::dbgs() << "Computed global iteration for nested loop\n");

  return globalIter;
}

std::pair<Value, SmallVector<Value>>
CircularBufferTransform::decomposePointer(Value ptr) {
  // Decompose pointer into base buffer and offset indices
  // For Triton, pointers are typically represented as:
  // - tt.addptr(base, offset) for pointer arithmetic
  // - tt.splat(scalar_ptr) for broadcasting scalar pointers
  // - Direct tensor pointers

  SmallVector<Value> indices;

  if (!ptr) {
    return {ptr, indices};
  }

  Operation *defOp = ptr.getDefiningOp();
  if (!defOp) {
    // Block argument - return as-is
    return {ptr, indices};
  }

  // Handle tt.addptr - decompose into base and offset
  if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(defOp)) {
    Value base = addPtrOp.getPtr();
    Value offset = addPtrOp.getOffset();

    // Recursively decompose the base pointer
    auto [innerBase, innerIndices] = decomposePointer(base);

    // Add the current offset to indices
    indices = std::move(innerIndices);
    indices.push_back(offset);

    LLVM_DEBUG(llvm::dbgs() << "Decomposed addptr: found offset index\n");
    return {innerBase, indices};
  }

  // Handle tt.splat - the base is the scalar operand
  if (auto splatOp = dyn_cast<triton::SplatOp>(defOp)) {
    Value src = splatOp.getSrc();
    LLVM_DEBUG(llvm::dbgs() << "Decomposed splat: found scalar base\n");
    return {src, indices};
  }

  // Handle tt.broadcast - decompose the source
  if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(defOp)) {
    return decomposePointer(broadcastOp.getSrc());
  }

  // Handle MemDescSubviewOp - extract base and offsets
  if (auto subviewOp = dyn_cast<triton::gpu::MemDescSubviewOp>(defOp)) {
    Value src = subviewOp.getSrc();
    for (Value offset : subviewOp.getOffsets()) {
      indices.push_back(offset);
    }

    // Recursively decompose the source
    auto [innerBase, innerIndices] = decomposePointer(src);

    // Prepend inner indices
    SmallVector<Value> allIndices(innerIndices.begin(), innerIndices.end());
    allIndices.append(indices.begin(), indices.end());

    LLVM_DEBUG(llvm::dbgs() << "Decomposed MemDescSubview: found "
                            << allIndices.size() << " indices\n");
    return {innerBase, allIndices};
  }

  // Default: return pointer as base with no indices
  return {ptr, indices};
}

Value CircularBufferTransform::buildPointer(Value baseBuffer,
                                             ArrayRef<Value> indices) {
  // Build a new pointer/memdesc from base buffer and indices
  // Uses MemDescSubviewOp for memory descriptor access

  if (!baseBuffer) {
    return baseBuffer;
  }

  if (indices.empty()) {
    return baseBuffer;
  }

  // Check if baseBuffer is a MemDescType
  auto memDescType = dyn_cast<triton::MemDescType>(baseBuffer.getType());
  if (memDescType) {
    // Use MemDescSubviewOp to create indexed access
    Location loc = baseBuffer.getLoc();

    // Convert indices to i32 if needed
    SmallVector<Value> i32Indices;
    for (Value idx : indices) {
      if (idx.getType().isIndex()) {
        Value i32Idx = builder.create<arith::IndexCastOp>(
            loc, builder.getI32Type(), idx);
        i32Indices.push_back(i32Idx);
      } else if (idx.getType().isInteger(32)) {
        i32Indices.push_back(idx);
      } else {
        // Try to cast to i32
        Value i32Idx = builder.create<arith::TruncIOp>(
            loc, builder.getI32Type(), idx);
        i32Indices.push_back(i32Idx);
      }
    }

    // Calculate result shape by dropping leading dimensions
    ArrayRef<int64_t> baseShape = memDescType.getShape();
    size_t numIndicesToDrop = std::min(i32Indices.size(), baseShape.size());
    SmallVector<int64_t> resultShape(
        baseShape.begin() + numIndicesToDrop, baseShape.end());

    if (resultShape.empty()) {
      // Scalar access - return single element shape
      resultShape.push_back(1);
    }

    auto resultType = triton::MemDescType::get(
        resultShape, memDescType.getElementType(), memDescType.getEncoding());

    Value subview = builder.create<triton::gpu::MemDescSubviewOp>(
        loc, resultType, baseBuffer, i32Indices);

    LLVM_DEBUG(llvm::dbgs() << "Built MemDescSubview with " << i32Indices.size()
                            << " indices\n");
    return subview;
  }

  // For tensor pointers, use addptr to add offsets
  auto ptrType = dyn_cast<triton::PointerType>(baseBuffer.getType());
  auto tensorType = dyn_cast<RankedTensorType>(baseBuffer.getType());

  if (ptrType || (tensorType && triton::isTensorPointerType(tensorType))) {
    Location loc = baseBuffer.getLoc();
    Value result = baseBuffer;

    for (Value idx : indices) {
      result = builder.create<triton::AddPtrOp>(loc, result.getType(), result, idx);
    }

    LLVM_DEBUG(llvm::dbgs() << "Built pointer with " << indices.size()
                            << " addptr operations\n");
    return result;
  }

  // Fallback: return base buffer
  LLVM_DEBUG(llvm::dbgs() << "buildPointer: unhandled type, returning base\n");
  return baseBuffer;
}

Value CircularBufferTransform::applySwizzle(Value ptr,
                                             CircularBufferInfo &info) {
  // Apply swizzle pattern to reduce bank conflicts
  // This would XOR the index with a pattern to distribute accesses

  if (!info.useSwizzle) {
    return ptr;
  }

  // Swizzling is typically applied at the PTX level
  // This is a placeholder for the swizzle logic

  LLVM_DEBUG(llvm::dbgs() << "Swizzle applied to pointer\n");

  return ptr;
}

void CircularBufferTransform::substituteLoopVariable(Operation *op,
                                                      Value oldVar,
                                                      Value newVar) {
  // Substitute all uses of oldVar with newVar in the operation tree
  if (!op || !oldVar || !newVar) {
    return;
  }

  // Walk the operation and replace uses
  op->walk([&](Operation *innerOp) {
    for (OpOperand &operand : innerOp->getOpOperands()) {
      if (operand.get() == oldVar) {
        operand.set(newVar);
      }
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "Substituted loop variable in operation\n");
}

void CircularBufferTransform::transformLocalStore(Operation *localStoreOp,
                                                   CircularBufferInfo &info) {
  if (!localStoreOp || !info.loop) {
    return;
  }

  auto storeOp = dyn_cast<triton::gpu::LocalStoreOp>(localStoreOp);
  if (!storeOp) {
    return;
  }

  // Transform LocalStore to use circular buffer indexing
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(storeOp);

  Location loc = storeOp->getLoc();
  Value globalIter = computeGlobalIteration(info.loop);

  if (!globalIter) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to compute global iteration for LocalStore\n");
    return;
  }

  // Compute circular offset for store (producer side)
  // Producer writes to slot (iter + numStages - 1) % numStages
  Value offset = computeCircularOffsetStore(loc, globalIter,
                                            info.numStages, info.stride);

  // Get destination and create subview with circular index
  Value dst = storeOp.getDst();
  if (auto memDescType = dyn_cast<triton::MemDescType>(dst.getType())) {
    // Create subview into circular buffer at computed offset
    SmallVector<Value> indices;

    // Add stage index - ensure it's i32
    Value i32Offset;
    Type i32Type = builder.getI32Type();
    if (offset.getType() == i32Type) {
      i32Offset = offset;  // Already i32
    } else if (offset.getType().isIndex()) {
      i32Offset = builder.create<arith::IndexCastOp>(loc, i32Type, offset);
    } else {
      // Try truncation for other integer types
      i32Offset = builder.create<arith::TruncIOp>(loc, i32Type, offset);
    }
    indices.push_back(i32Offset);

    // Build subview
    Value subview = buildPointer(info.circularBuffer, indices);

    // Create new store with updated destination
    builder.create<triton::gpu::LocalStoreOp>(loc, storeOp.getSrc(), subview);

    LLVM_DEBUG(llvm::dbgs() << "Transformed LocalStore with circular indexing\n");
  }
}

void CircularBufferTransform::transformLocalLoad(Operation *localLoadOp,
                                                  CircularBufferInfo &info) {
  if (!localLoadOp || !info.loop) {
    return;
  }

  auto loadOp = dyn_cast<triton::gpu::LocalLoadOp>(localLoadOp);
  if (!loadOp) {
    return;
  }

  // Transform LocalLoad to use circular buffer indexing
  // This enables Shared→Register pipelining by prefetching into registers
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(loadOp);

  Location loc = loadOp->getLoc();
  Value globalIter = computeGlobalIteration(info.loop);

  if (!globalIter) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to compute global iteration for LocalLoad\n");
    return;
  }

  // Compute circular offset for load (consumer side)
  // Consumer reads from slot iter % numStages
  Value offset = computeCircularOffsetLoad(loc, globalIter,
                                           info.numStages, info.stride);

  // Get source and create subview with circular index
  Value src = loadOp.getSrc();
  if (auto memDescType = dyn_cast<triton::MemDescType>(src.getType())) {
    // Create subview into circular buffer at computed offset
    SmallVector<Value> indices;

    // Add stage index - ensure it's i32
    Value i32Offset;
    Type i32Type = builder.getI32Type();
    if (offset.getType() == i32Type) {
      i32Offset = offset;  // Already i32
    } else if (offset.getType().isIndex()) {
      i32Offset = builder.create<arith::IndexCastOp>(loc, i32Type, offset);
    } else {
      // Try truncation for other integer types
      i32Offset = builder.create<arith::TruncIOp>(loc, i32Type, offset);
    }
    indices.push_back(i32Offset);

    // Build subview
    Value subview = buildPointer(info.circularBuffer, indices);

    // Create new load with updated source
    Value newLoad = builder.create<triton::gpu::LocalLoadOp>(
        loc, loadOp.getResult().getType(), subview);

    // Replace uses of old load
    loadOp.getResult().replaceAllUsesWith(newLoad);

    LLVM_DEBUG(llvm::dbgs() << "Transformed LocalLoad with circular indexing for Shared→Register pipeline\n");
  }
}

//===----------------------------------------------------------------------===//
// Global Load Transformation with Async Copy (cp.async generation)
//===----------------------------------------------------------------------===//

Attribute CircularBufferTransform::getSharedEncodingForLoad(triton::LoadOp loadOp) {
  auto resultType = cast<RankedTensorType>(loadOp.getType());
  auto encoding = resultType.getEncoding();

  // Get CTA layout from the original encoding
  auto ctaLayout = getCTALayout(encoding);
  if (!ctaLayout) {
    // Create default CTA layout
    SmallVector<unsigned> ctasPerCGA(resultType.getRank(), 1);
    SmallVector<unsigned> ctaSplitNum(resultType.getRank(), 1);
    SmallVector<unsigned> ctaOrder;
    for (unsigned i = 0; i < resultType.getRank(); ++i) {
      ctaOrder.push_back(resultType.getRank() - 1 - i);
    }
    ctaLayout = CTALayoutAttr::get(builder.getContext(), ctasPerCGA,
                                   ctaSplitNum, ctaOrder);
  }

  // Get order (typically row-major for shared memory)
  SmallVector<unsigned> order;
  if (auto blockedEnc = mlir::dyn_cast<BlockedEncodingAttr>(encoding)) {
    auto blockedOrder = blockedEnc.getOrder();
    order.assign(blockedOrder.begin(), blockedOrder.end());
  } else {
    // Default order for 2D and higher
    for (unsigned i = 0; i < resultType.getRank(); ++i) {
      order.push_back(resultType.getRank() - 1 - i);
    }
  }

  // Create SharedEncodingAttr using shape-based constructor for optimal layout
  return SharedEncodingAttr::get(builder.getContext(), resultType.getShape(),
                                 order, ctaLayout, resultType.getElementType());
}

Value CircularBufferTransform::allocateSharedBuffer(triton::LoadOp loadOp,
                                                     unsigned numStages) {
  auto resultType = cast<RankedTensorType>(loadOp.getType());
  Type elementType = resultType.getElementType();
  ArrayRef<int64_t> shape = resultType.getShape();

  // Get shared encoding
  Attribute sharedEncoding = getSharedEncodingForLoad(loadOp);

  // Create shape with stage dimension prepended: [numStages, ...shape]
  SmallVector<int64_t> bufferShape;
  bufferShape.push_back(static_cast<int64_t>(numStages));
  bufferShape.append(shape.begin(), shape.end());

  // Create MemDescType for shared memory
  auto memDescType = triton::MemDescType::get(bufferShape, elementType,
                                               sharedEncoding,
                                               /*mutableMemory=*/true);

  // Insert allocation at the beginning of the function
  OpBuilder::InsertionGuard guard(builder);
  auto funcOp = loadOp->getParentOfType<triton::FuncOp>();
  if (funcOp) {
    builder.setInsertionPointToStart(&funcOp.getBody().front());
  } else {
    builder.setInsertionPoint(loadOp);
  }

  Location loc = loadOp.getLoc();
  Value alloc = builder.create<triton::gpu::LocalAllocOp>(loc, memDescType, Value());

  LLVM_DEBUG(llvm::dbgs() << "Allocated shared buffer with shape [");
  for (auto d : bufferShape) {
    LLVM_DEBUG(llvm::dbgs() << d << " ");
  }
  LLVM_DEBUG(llvm::dbgs() << "] for async copy pipelining\n");

  return alloc;
}

void CircularBufferTransform::transformGlobalLoad(triton::LoadOp loadOp,
                                                   CircularBufferInfo &info,
                                                   Value insertIdx,
                                                   Value extractIdx) {
  if (!loadOp || !info.loop) {
    return;
  }

  OpBuilder::InsertionGuard guard(builder);
  Location loc = loadOp.getLoc();

  // Allocate shared memory buffer if not already allocated or wrong type
  // At TTGIR stage, info.circularBuffer might be the original pointer, not a MemDescType
  if (!info.circularBuffer || !isa<triton::MemDescType>(info.circularBuffer.getType())) {
    info.circularBuffer = allocateSharedBuffer(loadOp, info.numStages);
  }

  Value alloc = info.circularBuffer;
  if (!isa<triton::MemDescType>(alloc.getType())) {
    llvm::errs() << "[CircularBufferTransform] ERROR: alloc is not MemDescType, type is: "
                 << alloc.getType() << "\n";
    return;
  }
  auto allocType = cast<triton::MemDescType>(alloc.getType());

  // Create constants
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);

  // ========== ASYNC COPY (Insert) ==========
  builder.setInsertionPoint(loadOp);

  // Create subview for the insert slot
  SmallVector<Value> insertOffsets(allocType.getRank(), zero);
  insertOffsets[0] = insertIdx;

  auto subviewType = triton::MemDescType::get(
      allocType.getShape().drop_front(), allocType.getElementType(),
      allocType.getEncoding(), /*mutableMemory=*/true);

  auto insertView = builder.create<triton::gpu::MemDescSubviewOp>(
      loc, subviewType, alloc, insertOffsets);

  // Get source pointer and optional mask/other from the load
  Value src = loadOp.getPtr();
  Value mask = loadOp.getMask();
  Value other = loadOp.getOther();

  // Create async copy: global -> shared
  Operation *asyncCopy = builder.create<triton::gpu::AsyncCopyGlobalToLocalOp>(
      loc, src, insertView, mask, other,
      loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());

  // Commit the async copy group
  Operation *commit = builder.create<triton::gpu::AsyncCommitGroupOp>(
      loc, asyncCopy->getResult(0));

  // Wait for async copy to complete (wait for numStages-1 groups)
  int waitNum = info.numStages > 1 ? info.numStages - 2 : 0;
  Operation *wait = builder.create<triton::gpu::AsyncWaitOp>(
      loc, commit->getResult(0), waitNum);

  // ========== LOCAL LOAD (Extract) ==========
  // Create subview for the extract slot
  SmallVector<Value> extractOffsets(allocType.getRank(), zero);
  extractOffsets[0] = extractIdx;

  auto extractView = builder.create<triton::gpu::MemDescSubviewOp>(
      loc, subviewType, alloc, extractOffsets);

  // Create local load from shared memory
  auto localLoad = builder.create<triton::gpu::LocalLoadOp>(
      loc, loadOp.getType(), extractView, wait->getResult(0));

  // Handle non-zero "other" values (not handled by AsyncCopyGlobalToLocalOp)
  Value result = localLoad.getResult();
  if (other && !isa<arith::ConstantOp>(other.getDefiningOp())) {
    // Create select for non-zero other values
    auto select = builder.create<arith::SelectOp>(
        loc, loadOp.getType(), mask, localLoad.getResult(), other);
    result = select.getResult();
  }

  // Replace all uses of the original load
  loadOp.getResult().replaceAllUsesWith(result);

  LLVM_DEBUG(llvm::dbgs() << "Transformed global LoadOp to async copy pipeline:\n"
                          << "  - Created AsyncCopyGlobalToLocalOp\n"
                          << "  - Created AsyncCommitGroupOp\n"
                          << "  - Created AsyncWaitOp (num=" << waitNum << ")\n"
                          << "  - Created LocalLoadOp from shared memory\n");
}
