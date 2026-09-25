/*
 * Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */

#include "tle/dialect/include/Transforms/LogicalDomain.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/ExactSMEM.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::triton::tle {
namespace ttg = mlir::triton::gpu;

#define GEN_PASS_DEF_TRITONTLEPLANLOGICALDOMAINS
#include "tle/dialect/include/Transforms/Passes.h.inc"

// Implemented with the tensor actions so it can reuse the predicate/tail
// materialization helpers without exposing them as public APIs.
void applyLogicalTensorActions(LogicalDomainPlan &plan);

namespace {
// Frontend stage initialization can already contain register extracts. Keep
// their lowering consistent with the extracts created while splitting copies.
static void markPlannedInputTiles(const LogicalDomainPlan &plan) {
  SmallVector<Value> worklist;
  for (const LogicalRootRewriteAction &root : plan.roots) {
    if (root.initializer)
      worklist.push_back(root.initializer);
    for (auto store : root.localStores)
      worklist.push_back(store.getSrc());
    for (const auto &copy : root.pointerCopies)
      llvm::append_range(worklist, copy.load->getOperands());
  }
  llvm::SmallPtrSet<Operation *, 32> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!isa<RankedTensorType>(value.getType()))
      continue;
    Operation *op = value.getDefiningOp();
    if (!op || !visited.insert(op).second)
      continue;
    if (isa<ExtractTileOp>(op))
      op->setAttr(kPlannedTileAttr, UnitAttr::get(op->getContext()));
    llvm::append_range(worklist, op->getOperands());
  }
}

constexpr llvm::StringLiteral kTiledFields("tiled_smem_fields");
static void addTiledPipeField(Operation *op, unsigned fieldIndex,
                              OpBuilder &builder) {
  SmallVector<int32_t> fields;
  if (auto old = op->getAttrOfType<DenseI32ArrayAttr>(kTiledFields))
    fields.assign(old.asArrayRef().begin(), old.asArrayRef().end());
  if (!llvm::is_contained(fields, static_cast<int32_t>(fieldIndex)))
    fields.push_back(fieldIndex);
  llvm::sort(fields);
  op->setAttr(kTiledFields, builder.getDenseI32ArrayAttr(fields));
}

static Value addOffset(OpBuilder &builder, Location loc, Value base,
                       int64_t delta) {
  if (delta == 0)
    return base;
  auto intType = cast<IntegerType>(base.getType());
  Value constant =
      arith::ConstantIntOp::create(builder, loc, delta, intType.getWidth());
  return arith::AddIOp::create(builder, loc, base, constant);
}

static Value createStaticExtractTile(OpBuilder &builder, Location loc,
                                     Value value, int64_t linearTile,
                                     ArrayRef<int64_t> tileShape) {
  if (!value)
    return {};
  Value index = arith::ConstantIntOp::create(builder, loc, builder.getI32Type(),
                                             linearTile);
  auto extract = ExtractTileOp::create(builder, loc, value, index, tileShape);
  extract->setAttr(kPlannedTileAttr, builder.getUnitAttr());
  return extract.getResult();
}

static Value createFullTileIndex(OpBuilder &builder, Location loc,
                                 ArrayRef<int64_t> tileShape, unsigned axis,
                                 int64_t offset) {
  auto rangeType =
      RankedTensorType::get({tileShape[axis]}, builder.getI32Type());
  Value index = triton::MakeRangeOp::create(builder, loc, rangeType, offset,
                                            offset + tileShape[axis]);
  SmallVector<int64_t, 2> expandedShape{tileShape[axis]};
  unsigned expandAxis = axis == 0 ? 1u : 0u;
  expandedShape.insert(expandedShape.begin() + expandAxis, 1);
  index = triton::ExpandDimsOp::create(
      builder, loc, RankedTensorType::get(expandedShape, builder.getI32Type()),
      index, expandAxis);
  if (expandedShape != tileShape)
    index = triton::BroadcastOp::create(
        builder, loc, RankedTensorType::get(tileShape, builder.getI32Type()),
        index);
  return index;
}

static void applyLogicalPointerCopy(LogicalPointerCopyAction &action,
                                    ArrayRef<int64_t> storageTileShape,
                                    ArrayRef<int64_t> logicalShape,
                                    unsigned fragmentAxis) {
  LocalPointersOp pointers = action.pointers;
  triton::StoreOp store = action.store;
  triton::LoadOp load = action.load;
  ExactSMEMStage stage = getExactSMEMStage(pointers.getSrc());
  assert(stage && "validated logical pointer destination must be a stage");

  auto carrierType = cast<RankedTensorType>(load.getPtr().getType());
  ArrayRef<int64_t> carrierShape = carrierType.getShape();
  ArrayRef<int64_t> microTileShape = action.microTileShape;
  assert(microTileShape.size() == 2 && "copy work partition must be planned");
  int64_t storageRowTiles = storageTileShape[0] / microTileShape[0];
  int64_t storageColTiles = storageTileShape[1] / microTileShape[1];
  int64_t logicalRowTiles = logicalShape[0] / storageTileShape[0];
  int64_t logicalColTiles = logicalShape[1] / storageTileShape[1];
  int64_t sourceGridCols = carrierShape[1] / microTileShape[1];
  auto originalPtrType = cast<RankedTensorType>(pointers.getResult().getType());
  auto microPtrType =
      RankedTensorType::get(microTileShape, originalPtrType.getElementType(),
                            originalPtrType.getEncoding());

  OpBuilder builder(store);
  Location loc = store.getLoc();
  for (int64_t logicalRowTile = 0; logicalRowTile < logicalRowTiles;
       ++logicalRowTile) {
    for (int64_t logicalColTile = 0; logicalColTile < logicalColTiles;
         ++logicalColTile) {
      int64_t tile = fragmentAxis == 0
                         ? logicalColTile * logicalRowTiles + logicalRowTile
                         : logicalRowTile * logicalColTiles + logicalColTile;
      Value flatIndex = addOffset(builder, loc, stage.atom.getIndex(), tile);
      auto storageTile = ttg::MemDescIndexOp::create(
          builder, loc, stage.atom.getType(), stage.atom.getSrc(), flatIndex);
      storageTile->setAttr(kExactSMEMTileAttr, builder.getI32IntegerAttr(tile));

      for (int64_t storageRowTile = 0; storageRowTile < storageRowTiles;
           ++storageRowTile) {
        for (int64_t storageColTile = 0; storageColTile < storageColTiles;
             ++storageColTile) {
          int64_t sourceRowTile =
              logicalRowTile * storageRowTiles + storageRowTile;
          int64_t sourceColTile =
              logicalColTile * storageColTiles + storageColTile;
          int64_t sourceLinearTile =
              sourceRowTile * sourceGridCols + sourceColTile;
          Value tiledPtr = createStaticExtractTile(
              builder, loc, load.getPtr(), sourceLinearTile, microTileShape);
          Value tiledMask = createStaticExtractTile(
              builder, loc, load.getMask(), sourceLinearTile, microTileShape);
          Value tiledOther = createStaticExtractTile(
              builder, loc, load.getOther(), sourceLinearTile, microTileShape);
          Value tiledLoad = triton::LoadOp::create(
              builder, loc, tiledPtr, tiledMask, tiledOther, load.getCache(),
              load.getEvict(), load.getIsVolatile(),
              load.getFlagtreeHintsAttr());

          SmallVector<Value, 2> indices;
          indices.push_back(
              createFullTileIndex(builder, loc, microTileShape, 0,
                                  storageRowTile * microTileShape[0]));
          indices.push_back(
              createFullTileIndex(builder, loc, microTileShape, 1,
                                  storageColTile * microTileShape[1]));
          Value tiledPointers = LocalPointersOp::create(
              builder, loc, microPtrType, storageTile, indices);
          auto tiledStore =
              triton::StoreOp::create(builder, loc, tiledPointers, tiledLoad,
                                      store.getCache(), store.getEvict());
          tiledStore->setAttr(
              kCopyPlanAttr,
              builder.getDictionaryAttr(
                  {builder.getNamedAttr(
                       "tile", builder.getDenseI64ArrayAttr(microTileShape)),
                   builder.getNamedAttr(
                       "vector_bytes",
                       builder.getI32IntegerAttr(action.vectorBytes)),
                   builder.getNamedAttr(
                       "storage", stage.root.alloc->getAttr(kSMEMPlanAttr))}));
        }
      }
    }
  }

  store.erase();
  pointers.erase();
  load.erase();
}

// local_alloc's initializer is an SSA tensor. Materialize its logical prefix
// with existing register extraction and shared stores, once at the allocation
// site. Every logical stage is fully initialized without storing payload tails.
static void applyLogicalInitializer(LogicalRootRewriteAction &action,
                                    ttg::LocalAllocOp exactAlloc) {
  if (!action.initializer)
    return;
  OpBuilder builder(action.alloc);
  Location loc = action.alloc.getLoc();
  auto type = cast<RankedTensorType>(action.initializer.getType());
  ArrayRef<int64_t> payloadShape = type.getShape().drop_front();
  auto payloadType = RankedTensorType::get(payloadShape, type.getElementType());
  SmallVector<int64_t, 3> sourceStageShape{1, payloadShape[0], payloadShape[1]};
  Value initializer = action.initializer;
  // Share a broadcast initializer's original register payload across slots.
  if (auto broadcast = initializer.getDefiningOp<triton::BroadcastOp>()) {
    auto sourceType = cast<RankedTensorType>(broadcast.getSrc().getType());
    if (sourceType.getShape() == ArrayRef<int64_t>(sourceStageShape))
      initializer = broadcast.getSrc();
  }
  Value broadcastPayload;
  if (cast<RankedTensorType>(initializer.getType()).getShape()[0] == 1)
    broadcastPayload = triton::ReshapeOp::create(builder, loc, payloadType,
                                                 initializer, false);
  const auto &tile = action.storageTileShape;
  int64_t rows = action.logicalShape[1], cols = action.logicalShape[2];
  bool fragmentRows = !llvm::isPowerOf2_64(rows);
  auto tileType =
      ttg::MemDescType::get(tile, type.getElementType(), action.storageEncoding,
                            exactAlloc.getType().getMemorySpace(), true, tile);
  int64_t tilesPerStage = (rows / tile[0]) * (cols / tile[1]);
  for (int64_t stage = 0; stage < action.logicalShape[0]; ++stage) {
    Value payload = broadcastPayload;
    if (!payload) {
      Value source = createStaticExtractTile(builder, loc, initializer, stage,
                                             sourceStageShape);
      payload =
          triton::ReshapeOp::create(builder, loc, payloadType, source, false);
    }
    for (int64_t r = 0; r < rows / tile[0]; ++r) {
      for (int64_t c = 0; c < cols / tile[1]; ++c) {
        int64_t ordinal =
            fragmentRows ? c * (rows / tile[0]) + r : r * (cols / tile[1]) + c;
        Value index = arith::ConstantIntOp::create(
            builder, loc, stage * tilesPerStage + ordinal, 32);
        auto destination = ttg::MemDescIndexOp::create(builder, loc, tileType,
                                                       exactAlloc, index);
        destination->setAttr(kExactSMEMTileAttr,
                             builder.getI32IntegerAttr(ordinal));
        int64_t sourceTile = r * (payloadShape[1] / tile[1]) + c;
        Value value =
            createStaticExtractTile(builder, loc, payload, sourceTile, tile);
        // Let upstream local_store derive vectorization from the actual
        // shared encoding, including transposition and swizzle. Splitting
        // this into generic pointer stores loses that layout contract.
        ttg::LocalStoreOp::create(builder, loc, value, destination);
      }
    }
  }
}

static void applyLogicalLocalStore(ttg::LocalStoreOp store) {
  OpBuilder builder(store);
  Location loc = store.getLoc();
  ExactSMEMStage stage = getExactSMEMStage(store.getDst());
  assert(stage && !stage.viewTransposed &&
         "validated local store must target a stage");
  auto tileType = stage.atom.getType();
  auto tile = tileType.getShape();
  auto sourceType = cast<RankedTensorType>(store.getSrc().getType());
  for (int64_t r = 0; r < stage.root.getRowTiles(); ++r) {
    for (int64_t c = 0; c < stage.root.getColTiles(); ++c) {
      int64_t ordinal = stage.root.getLinearTile(r, c);
      Value index = addOffset(builder, loc, stage.atom.getIndex(), ordinal);
      auto destination = ttg::MemDescIndexOp::create(
          builder, loc, tileType, stage.atom.getSrc(), index);
      destination->setAttr(kExactSMEMTileAttr,
                           builder.getI32IntegerAttr(ordinal));
      int64_t sourceTile = r * (sourceType.getShape()[1] / tile[1]) + c;
      Value value = createStaticExtractTile(builder, loc, store.getSrc(),
                                            sourceTile, tile);
      ttg::LocalStoreOp::create(builder, loc, value, destination);
    }
  }
  store.erase();
}

static void applyRootRewrite(LogicalRootRewriteAction &action) {
  ttg::LocalAllocOp oldAlloc = action.alloc;
  auto oldType = oldAlloc.getType();
  int64_t capacity = action.logicalShape[0];
  int64_t rows = action.logicalShape[1];
  int64_t cols = action.logicalShape[2];
  assert(action.storageTileShape.size() == 2 &&
         "validated root must select a storage tile");
  int64_t storageTileRows = action.storageTileShape[0];
  int64_t storageTileCols = action.storageTileShape[1];
  unsigned fragmentAxis = !llvm::isPowerOf2_64(rows) ? 0u : 1u;
  int64_t rowTiles = rows / storageTileRows;
  int64_t colTiles = cols / storageTileCols;
  int64_t tilesPerStage = rowTiles * colTiles;
  SmallVector<int64_t> exactShape{capacity * tilesPerStage, storageTileRows,
                                  storageTileCols};
  SmallVector<int64_t> storageTileShape{storageTileRows, storageTileCols};
  assert(!action.stages.empty() &&
         "validated root must have at least one stage view");
  auto storageTileEncoding = action.storageEncoding;
  assert(storageTileEncoding && "shared encoding must be planned");

  OpBuilder builder(oldAlloc);
  auto ctaLayout = ttg::CTAEncodingAttr::getDefault(oldAlloc.getContext(), 2);
  auto exactEncoding = ttg::SwizzledSharedEncodingAttr::get(
      oldAlloc.getContext(), 8, 1, 1, {1, 0}, ctaLayout);
  auto exactType = ttg::MemDescType::get(
      exactShape, oldType.getElementType(), exactEncoding,
      oldType.getMemorySpace(), oldType.getMutableMemory(), exactShape);
  auto exactAlloc =
      ttg::LocalAllocOp::create(builder, oldAlloc.getLoc(), exactType, Value());
  exactAlloc->setAttr(kExactSMEMShapeAttr,
                      builder.getDenseI64ArrayAttr(action.logicalShape));
  SMEMLayoutPlan storagePlan({rows, cols}, storageTileShape,
                             storageTileEncoding);
  exactAlloc->setAttr(kSMEMPlanAttr, storagePlan.getAttr());
  int64_t alignment =
      std::max<int64_t>(16, storageTileEncoding.getSwizzlingByteWidth() * 8);
  if (!action.copies.empty())
    alignment = std::max<int64_t>(128, alignment);
  if (IntegerAttr oldAlignment = oldAlloc.getAlignmentAttr())
    alignment = std::max(alignment, oldAlignment.getInt());
  exactAlloc.setAlignmentAttr(builder.getI32IntegerAttr(alignment));

  for (LogicalMemDescUseAction &memdescUse : action.memdescUses) {
    OpOperand *use = memdescUse.use;
    unsigned fieldIndex = use->getOperandNumber();
    if (use->get() == oldAlloc.getResult()) {
      use->set(exactAlloc);
      if (auto warpSpecialize =
              dyn_cast<ttg::WarpSpecializeOp>(use->getOwner()))
        for (Region *partition : warpSpecialize.getPartitionRegions())
          partition->getArgument(fieldIndex).setType(exactType);
    }
  }
  for (LogicalMemDescUseAction &memdescUse : action.memdescUses) {
    OpOperand *use = memdescUse.use;
    unsigned fieldIndex = use->getOperandNumber();
    assert(use->get().getType() == exactType &&
           "forwarded exact-SMEM use must have rewritten storage type");
    if (memdescUse.markTiledPipeField)
      addTiledPipeField(use->getOwner(), fieldIndex, builder);
  }

  for (ttg::MemDescIndexOp stage : action.stages) {
    OpBuilder stageBuilder(stage);
    Value tileCount = arith::ConstantIntOp::create(
        stageBuilder, stage.getLoc(), stageBuilder.getI32Type(), tilesPerStage);
    Value stageBase = arith::MulIOp::create(stageBuilder, stage.getLoc(),
                                            stage.getIndex(), tileCount);
    auto storageTileType = ttg::MemDescType::get(
        storageTileShape, oldType.getElementType(), storageTileEncoding,
        oldType.getMemorySpace(), oldType.getMutableMemory(), storageTileShape);
    Value storage = stage.getSrc();
    if (storage == oldAlloc.getResult())
      storage = exactAlloc;
    assert(storage.getType() == exactType &&
           "exact-SMEM stage must index its rewritten storage value");
    Value storageTile = ttg::MemDescIndexOp::create(
        stageBuilder, stage.getLoc(), storageTileType, storage, stageBase);
    auto carrier = ttg::MemDescReinterpretOp::create(
        stageBuilder, stage.getLoc(), stage.getType(), storageTile);
    carrier->setAttr(kExactSMEMStageAttr, stageBuilder.getUnitAttr());
    stage.getResult().replaceAllUsesWith(carrier);
  }

  // The frontend emits ordinary carrier transposes. Only a view proven to
  // belong to this compact allocation needs the WGMMA descriptor operation.
  for (ttg::MemDescTransOp transpose : action.transposes) {
    OpBuilder viewBuilder(transpose);
    auto view = MemDescWGMMAViewOp::create(
        viewBuilder, transpose.getLoc(), transpose.getType(),
        transpose.getSrc(), transpose.getOrder());
    transpose.getResult().replaceAllUsesWith(view.getResult());
    transpose.erase();
  }

  applyLogicalInitializer(action, exactAlloc);
  for (ttg::LocalStoreOp store : action.localStores)
    applyLogicalLocalStore(store);

  for (LogicalPointerCopyAction &copy : action.pointerCopies)
    applyLogicalPointerCopy(copy, storageTileShape,
                            ArrayRef<int64_t>(action.logicalShape).drop_front(),
                            fragmentAxis);

  for (ttg::TMACopyOp copy : action.copies) {
    OpBuilder copyBuilder(copy);
    ExactSMEMStage stage = getExactSMEMStage(copy.getDst());
    assert(stage && "validated logical TMA destination must be a stage view");
    auto stageType = stage.getType();
#ifndef __HCU__
    auto expectBytes = copy.getExpectBytesAttr();
#endif
    auto storageTileType = ttg::MemDescType::get(
        storageTileShape, stageType.getElementType(),
        stage.atom.getType().getEncoding(), stageType.getMemorySpace(),
        stageType.getMutableMemory(), storageTileShape);
    for (int64_t rowTile = 0; rowTile < rowTiles; ++rowTile) {
      for (int64_t colTile = 0; colTile < colTiles; ++colTile) {
        int64_t tile = fragmentAxis == 0 ? colTile * rowTiles + rowTile
                                         : rowTile * colTiles + colTile;
        Value flatIndex =
            addOffset(copyBuilder, copy.getLoc(), stage.atom.getIndex(), tile);
        auto storageTile = ttg::MemDescIndexOp::create(
            copyBuilder, copy.getLoc(), storageTileType, stage.atom.getSrc(),
            flatIndex);
        storageTile->setAttr(kExactSMEMTileAttr,
                             copyBuilder.getI32IntegerAttr(tile));
        SmallVector<Value> indices(copy.getIndices().begin(),
                                   copy.getIndices().end());
        unsigned rowCoordinate = indices.size() - 2;
        unsigned colCoordinate = indices.size() - 1;
        indices[rowCoordinate] =
            addOffset(copyBuilder, copy.getLoc(), indices[rowCoordinate],
                      rowTile * storageTileRows);
        indices[colCoordinate] =
            addOffset(copyBuilder, copy.getLoc(), indices[colCoordinate],
                      colTile * storageTileCols);
#ifdef __HCU__
        auto tiledCopy = ttg::TMACopyOp::create(
            copyBuilder, copy.getLoc(), copy.getSrc(), storageTile, indices);
#else
        auto tiledCopy = ttg::TMACopyOp::create(
            copyBuilder, copy.getLoc(), copy.getSrc(), storageTile, indices,
            copy.getBarrier(), expectBytes);
        // One logical copy contributes one arrival and its full byte count.
        // Continuation tiles complete the same barrier without another arrival.
        if (copy.getBarrier())
          expectBytes = copyBuilder.getI32IntegerAttr(0);
#endif
        int64_t elementBytes =
            oldType.getElementType().getIntOrFloatBitWidth() / 8;
        tiledCopy->setAttr(
            kLogicalTMACopyBytesAttr,
            copyBuilder.getI64IntegerAttr(storageTileRows * storageTileCols *
                                          elementBytes));
      }
    }
  }

  for (ttg::TMACopyOp copy : action.copies)
    copy.erase();
  for (ttg::MemDescIndexOp stage : action.stages)
    stage.erase();
  oldAlloc.erase();
}

struct TritonTlePlanLogicalDomains
    : public impl::TritonTlePlanLogicalDomainsBase<
          TritonTlePlanLogicalDomains> {
  void runOnOperation() override {
    FailureOr<LogicalDomainPlan> plan = analyzeLogicalDomains(getOperation());
    if (failed(plan)) {
      signalPassFailure();
      return;
    }
    applyLogicalDomainPlan(std::move(*plan));
  }
};
} // namespace

void applyLogicalDomainPlan(LogicalDomainPlan &&plan) {
  markPlannedInputTiles(plan);
  for (const auto &[value, descriptor] : plan.descriptors) {
    const auto &action =
        plan.descriptorRewrites.find(descriptor.provenance.primaryRoot())
            ->second;
    auto oldType = cast<triton::TensorDescType>(value.getType());
    auto oldBlock = oldType.getBlockType();
    auto block = RankedTensorType::get(
        action.blockShape, oldBlock.getElementType(), oldBlock.getEncoding());
    Value rewritten = value;
    rewritten.setType(triton::TensorDescType::get(value.getContext(), block));
  }
  for (const auto &[source, action] : plan.descriptorRewrites) {
    OpBuilder builder(source);
    source->removeAttr("tle.logical_descriptor_shape");
    source->setAttr("tle.tma_box_shape",
                    builder.getDenseI64ArrayAttr(action.boxShape));
  }
  for (LogicalRootRewriteAction &root : plan.roots)
    applyRootRewrite(root);
  applyLogicalTensorActions(plan);
}

} // namespace mlir::triton::tle
