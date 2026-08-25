#include "mlir/IR/AffineExpr.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include <cctype>
#include <limits>
#include <optional>
#include <set>
#include <string>

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

namespace mlir::triton::tle {

namespace {
// Triton shared-memory pointers map to LLVM address space 3 (NVVM shared).
constexpr int kSharedMemoryAddressSpace = 3;
// Cluster-shared pointers map to LLVM address space 7 (NVVM shared::cluster).
constexpr int kClusterSharedMemoryAddressSpace = 7;
} // namespace

// ============================================================================
// ExtractTileOp Builder
// ============================================================================
void ExtractTileOp::build(OpBuilder &builder, OperationState &state, Value src,
                          Value index, ArrayRef<int64_t> tileShape) {
  auto srcType = cast<RankedTensorType>(src.getType());
  auto resultType = RankedTensorType::get(tileShape, srcType.getElementType(),
                                          srcType.getEncoding());
  state.addOperands(src);
  state.addOperands(index);
  state.addAttribute("tile_shape", builder.getDenseI64ArrayAttr(tileShape));
  state.addTypes(resultType);
}

// ============================================================================
// ExtractTileOp Verification
//
// For dynamic index (index operand is not arith.constant):
//   - Only check constraints that are known at compile time: tile_shape
//   positivity, divisibility, element type, rank match
//   - Skip out-of-bounds and CTA tile alignment checks (only known at runtime)
//
// For static index: perform full checks (same as original implementation)
// ============================================================================
LogicalResult ExtractTileOp::verify() {
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto dstTy = cast<RankedTensorType>(getResult().getType());
  auto srcShape = srcTy.getShape();
  auto dstShape = dstTy.getShape();

  // ---- Get tile_shape attribute ----
  auto tileShapeRawAttr = getOperation()->getAttr("tile_shape");
  SmallVector<int64_t> tileShape;
  if (auto denseArray64 =
          mlir::dyn_cast<mlir::DenseI64ArrayAttr>(tileShapeRawAttr)) {
    for (auto v : denseArray64.asArrayRef())
      tileShape.push_back(v);
  }

  // ---- Basic checks required for both static and dynamic index ----

  // Check 1: element types must match
  if (srcTy.getElementType() != dstTy.getElementType())
    return emitError("result element type must match source element type");

  // Check 2: rank must match
  if (srcTy.getRank() != dstTy.getRank())
    return emitError("result rank must equal source rank");

  // Check 3: tile_shape rank must match source rank
  if (tileShape.size() != srcShape.size())
    return emitOpError("tile_shape rank must match source rank");

  // Check 4: tile_shape must be positive in each dimension, divisible, and dst
  // shape must equal tile_shape
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (tileShape[i] <= 0)
      return emitOpError("tile_shape must be positive at dimension ") << i;
    if (srcShape[i] % tileShape[i] != 0)
      return emitOpError(
                 "source shape must be divisible by tile_shape at dimension ")
             << i << " (source=" << srcShape[i] << ", tile=" << tileShape[i]
             << ")";
    if (dstShape[i] != tileShape[i])
      return emitOpError("result shape must equal tile_shape at dimension ")
             << i;
  }

  // ---- Determine if index is a static constant ----
  // getDefiningOp<arith::ConstantOp>() returns nullptr for dynamic Value
  auto indexConstOp =
      getOperation()->getOperand(1).getDefiningOp<arith::ConstantOp>();

  if (!indexConstOp) {
    // Dynamic index: skip out-of-bounds and offset alignment checks, handled at
    // lowering stage
    return success();
  }

  // ---- Full checks for static index ----
  int64_t index =
      mlir::cast<mlir::IntegerAttr>(indexConstOp.getValue()).getInt();

  // Compute logical grid shape
  SmallVector<int64_t> logicalGridShape(srcShape.size(), 0);
  int64_t totalTiles = 1;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    logicalGridShape[i] = srcShape[i] / tileShape[i];
    totalTiles *= logicalGridShape[i];
  }

  // Out-of-bounds check
  if (index < 0 || index >= totalTiles)
    return emitOpError("index out of bounds for tile grid: index=")
           << index << ", total_tiles=" << totalTiles;

  // Delinearize to per-dimension tile indices (row-major order)
  SmallVector<int64_t> tileIndices(srcShape.size(), 0);
  int64_t remain = index;
  for (int i = static_cast<int>(srcShape.size()) - 1; i >= 0; --i) {
    tileIndices[i] = remain % logicalGridShape[i];
    remain /= logicalGridShape[i];
  }

  // tile indices -> coordinate-level offsets
  SmallVector<int64_t> offsets(srcShape.size(), 0);
  for (size_t i = 0; i < srcShape.size(); ++i)
    offsets[i] = tileIndices[i] * tileShape[i];

  // Boundary check
  if (offsets.size() != static_cast<size_t>(srcTy.getRank()))
    return emitError("offsets size must match tensor rank");

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (dstShape[i] > srcShape[i])
      return emitOpError(
                 "result shape cannot exceed source shape at dimension ")
             << i;
    if (offsets[i] + dstShape[i] > srcShape[i])
      return emitOpError("invalid offset at dimension ")
             << i << ": offset(" << offsets[i] << ") + shape(" << dstShape[i]
             << ") > source(" << srcShape[i] << ")";
    if (offsets[i] < 0)
      return emitOpError("offset must be non-negative at dimension ") << i;
  }

  auto encoding = srcTy.getEncoding();
  if (!encoding)
    return success();
  return success();
}

LogicalResult MemDescWGMMAViewOp::verify() {
  auto srcType = getSrc().getType();
  auto resultType = getType();
  auto order = getOrder();

  if (srcType.getRank() != 2 || resultType.getRank() != 2)
    return emitOpError("expects rank-2 source and result memdesc types");
  if (order.size() != static_cast<size_t>(srcType.getRank()))
    return emitOpError("expects order rank to match source rank");
  llvm::SmallBitVector seen(order.size());
  for (int32_t dim : order) {
    if (dim < 0 || dim >= static_cast<int32_t>(order.size()) || seen[dim])
      return emitOpError("expects order to be a permutation");
    seen.set(dim);
  }
  if (srcType.getElementType() != resultType.getElementType())
    return emitOpError("expects source and result element types to match");
  if (srcType.getMemorySpace() != resultType.getMemorySpace())
    return emitOpError("expects source and result memory spaces to match");
  if (!isa<triton::gpu::SharedMemorySpaceAttr>(srcType.getMemorySpace()))
    return emitOpError("expects shared memory descriptors");
  if (mlir::product(srcType.getShape()) != mlir::product(resultType.getShape()))
    return emitOpError("expects source and result to cover the same number of "
                       "elements");
  if (!isa<triton::gpu::NVMMASharedEncodingAttr,
           triton::gpu::SharedLinearEncodingAttr>(resultType.getEncoding()))
    return emitOpError("expects result to use a WGMMA-compatible shared "
                       "encoding");
  return success();
}

static std::optional<int64_t>
getStaticMemDescByteSize(triton::gpu::MemDescType type) {
  int64_t numElements = 1;
  for (int64_t dim : type.getShape()) {
    if (ShapedType::isDynamic(dim) || dim < 0)
      return std::nullopt;
    if (dim != 0 && numElements > std::numeric_limits<int64_t>::max() / dim)
      return std::nullopt;
    numElements *= dim;
  }

  int64_t elementBits = type.getElementTypeBitWidth();
  int64_t elementBytes = (elementBits + 7) / 8;
  if (elementBytes <= 0)
    return std::nullopt;
  if (numElements > std::numeric_limits<int64_t>::max() / elementBytes)
    return std::nullopt;
  return numElements * elementBytes;
}

LogicalResult MemDescAliasOp::verify() {
  auto srcType = getSrc().getType();
  auto resultType = getType();
  int64_t offsetBytes = getOffsetBytesAttr().getInt();

  if (srcType.getMemorySpace() != resultType.getMemorySpace())
    return emitOpError("expects source and result memory spaces to match");
  if (!isa<triton::gpu::SharedMemorySpaceAttr>(srcType.getMemorySpace()))
    return emitOpError("expects shared memory descriptors");
  if (resultType.getMutableMemory() && !srcType.getMutableMemory())
    return emitOpError("cannot create a mutable alias from an immutable source");
  if (offsetBytes < 0)
    return emitOpError("expects non-negative offset_bytes");
  if (offsetBytes > std::numeric_limits<int32_t>::max())
    return emitOpError("expects offset_bytes to fit in i32 for shared memory "
                       "lowering");

  int64_t resultElementBytes = (resultType.getElementTypeBitWidth() + 7) / 8;
  if (resultElementBytes <= 0)
    return emitOpError("expects byte-addressable result element type");
  if (offsetBytes % resultElementBytes != 0)
    return emitOpError("expects offset_bytes to be aligned to the result "
                       "element byte width");

  std::optional<int64_t> srcBytes = getStaticMemDescByteSize(srcType);
  std::optional<int64_t> resultBytes = getStaticMemDescByteSize(resultType);
  if (!srcBytes || !resultBytes)
    return emitOpError("expects static source and result memdesc byte sizes");
  if (*resultBytes > *srcBytes ||
      offsetBytes > *srcBytes - *resultBytes)
    return emitOpError("result byte range must fit within the source view");

  return success();
}

LogicalResult WGMMASharedOperandFenceOp::verify() {
  if (getDeps().empty())
    return emitOpError("expects at least one shared-memory operand");

  for (Value dep : getDeps()) {
    auto type = cast<triton::gpu::MemDescType>(dep.getType());
    if (!isa<triton::gpu::SharedMemorySpaceAttr>(type.getMemorySpace()))
      return emitOpError("expects only shared-memory operands");
  }
  return success();
}

static bool isAsciiIdentStart(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool isAsciiIdentChar(char c) {
  return isAsciiIdentStart(c) || (c >= '0' && c <= '9') || c == '_';
}

static bool isValidPublicPipeName(StringRef name) {
  if (name.empty() || name == "fields" || name == "readers" ||
      name.starts_with("_") || !isAsciiIdentStart(name.front()))
    return false;
  return llvm::all_of(name.drop_front(), isAsciiIdentChar);
}

static LogicalResult verifyPipeNameArray(Operation *op, ArrayAttr namesAttr,
                                         StringRef attrName, bool allowEmpty) {
  if (!allowEmpty && namesAttr.empty())
    return op->emitOpError("expects ")
           << attrName << " to contain at least one name";

  llvm::SmallSet<StringRef, 8> seenNames;
  for (Attribute attr : namesAttr) {
    auto nameAttr = dyn_cast<StringAttr>(attr);
    if (!nameAttr)
      return op->emitOpError("expects ")
             << attrName << " to contain only strings";
    StringRef name = nameAttr.getValue();
    if (!isValidPublicPipeName(name))
      return op->emitOpError("expects valid public pipe ")
             << attrName << " names";
    if (!seenNames.insert(name).second)
      return op->emitOpError("expects unique pipe ") << attrName << " names";
  }

  return success();
}

static LogicalResult verifyPipeAttrs(Operation *op, OperandRange fields) {
  auto capacityAttr = op->getAttrOfType<IntegerAttr>("capacity");
  if (!capacityAttr)
    return op->emitOpError("requires capacity attribute");
  int64_t capacity = capacityAttr.getInt();
  if (capacity <= 0)
    return op->emitOpError("requires positive capacity");

  auto scopeAttr = op->getAttrOfType<StringAttr>("scope");
  if (!scopeAttr)
    return op->emitOpError("requires scope attribute");
  if (scopeAttr.getValue() != "cta")
    return op->emitOpError("MVP supports only scope = \"cta\"");

  auto fieldNamesAttr = op->getAttrOfType<ArrayAttr>("field_names");
  if (!fieldNamesAttr)
    return op->emitOpError("requires field_names attribute");
  if (fieldNamesAttr.size() != fields.size())
    return op->emitOpError("expects field_names size to match field operands");

  if (failed(
          verifyPipeNameArray(op, fieldNamesAttr, "field", fields.empty())))
    return failure();

  if (auto readersAttr = op->getAttrOfType<ArrayAttr>("readers")) {
    if (failed(verifyPipeNameArray(op, readersAttr, "reader", false)))
      return failure();
  }

  if (auto readerNameAttr = op->getAttrOfType<StringAttr>("reader_name")) {
    if (!isValidPublicPipeName(readerNameAttr.getValue()))
      return op->emitOpError("expects valid public pipe reader_name");
  }

  if (fields.empty()) {
    auto pipeNameAttr = op->getAttrOfType<StringAttr>("pipe_name");
    if (!pipeNameAttr || !isValidPublicPipeName(pipeNameAttr.getValue()))
      return op->emitOpError(
          "without payload fields requires a valid explicit pipe_name");
    return success();
  }
  for (Value field : fields) {
    auto type = cast<triton::gpu::MemDescType>(field.getType());
    if (!isa<triton::gpu::SharedMemorySpaceAttr>(type.getMemorySpace()))
      return op->emitOpError("expects only shared-memory pipe fields");
    if (type.getRank() < 2)
      return op->emitOpError("expects pipe fields to have rank >= 2");
    if (type.getShape()[0] != capacity)
      return op->emitOpError("expects field leading dimension to equal "
                             "pipe capacity");
  }
  return success();
}

static LogicalResult verifyPipeStagePhase(Operation *op, Value stage,
                                          Value phase) {
  if (!stage.getType().isInteger(32))
    return op->emitOpError("expects stage to be i32");
  if (!phase.getType().isInteger(1))
    return op->emitOpError("expects phase to be i1");
  return success();
}

static LogicalResult verifyPipeStage(Operation *op, Value stage) {
  if (!stage.getType().isInteger(32))
    return op->emitOpError("expects stage to be i32");
  return success();
}

LogicalResult PipeCreateOp::verify() {
  if (failed(verifyPipeAttrs(getOperation(), getFields())))
    return failure();
  if (getFields().empty()) {
    auto oneShotAttr = getOperation()->getAttrOfType<BoolAttr>("one_shot");
    if (!oneShotAttr || !oneShotAttr.getValue())
      return emitOpError(
          "without payload fields requires one_shot = true");
  }
  return success();
}

LogicalResult PipeCallBeginOp::verify() {
  if (getArguments().size() != getAliases().size())
    return emitOpError("expects one alias result for every call argument");
  for (auto [index, argument, alias] :
       llvm::enumerate(getArguments(), getAliases())) {
    if (argument.getType() != alias.getType())
      return emitOpError("expects call argument and alias result ")
             << index << " to have the same type";
  }
  return success();
}

LogicalResult PipeWriterAcquireOp::verify() {
  if (failed(verifyPipeAttrs(getOperation(), getFields())))
    return failure();
  return verifyPipeStagePhase(getOperation(), getStage(), getPhase());
}

LogicalResult PipeWriterCommitOp::verify() {
  if (failed(verifyPipeAttrs(getOperation(), getFields())))
    return failure();
  return verifyPipeStage(getOperation(), getStage());
}

LogicalResult PipeWriterCloseOp::verify() {
  if (failed(verifyPipeAttrs(getOperation(), getFields())))
    return failure();
  return verifyPipeStagePhase(getOperation(), getStage(), getPhase());
}

LogicalResult PipeReaderWaitOp::verify() {
  if (failed(verifyPipeAttrs(getOperation(), getFields())))
    return failure();
  if (failed(verifyPipeStagePhase(getOperation(), getStage(), getPhase())))
    return failure();
  if (!getIsClosed().getType().isInteger(1))
    return emitOpError("expects is_closed result to be i1");
  return success();
}

LogicalResult PipeReaderReleaseOp::verify() {
  if (failed(verifyPipeAttrs(getOperation(), getFields())))
    return failure();
  return verifyPipeStage(getOperation(), getStage());
}

LogicalResult PipeDrainOp::verify() {
  return verifyPipeAttrs(getOperation(), getFields());
}

void PipeReaderReleaseOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
  MutableOperandRange fields = getFieldsMutable();
  for (unsigned i = 0, e = fields.size(); i < e; ++i)
    effects.emplace_back(MemoryEffects::Free::get(), &fields[i],
                         triton::gpu::SharedMemory::get());
}

void PipeDrainOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
  MutableOperandRange fields = getFieldsMutable();
  for (unsigned i = 0, e = fields.size(); i < e; ++i) {
    effects.emplace_back(MemoryEffects::Read::get(), &fields[i],
                         triton::gpu::SharedMemory::get());
    effects.emplace_back(MemoryEffects::Write::get(), &fields[i],
                         triton::gpu::SharedMemory::get());
  }
}

static bool isValidPublicTaskGridName(StringRef name) {
  if (name.empty() || name == "commit" || name == "epoch" ||
      name == "fields" || name == "name" || name == "scope" ||
      name == "shape" || name == "tile_id" || name == "type" ||
      name.starts_with("_") || !isAsciiIdentStart(name.front()))
    return false;
  return llvm::all_of(name.drop_front(), isAsciiIdentChar);
}

static LogicalResult verifyTaskGridNameArray(Operation *op,
                                             ArrayAttr namesAttr,
                                             StringRef attrName,
                                             bool allowEmpty) {
  if (!allowEmpty && namesAttr.empty())
    return op->emitOpError("expects ")
           << attrName << " to contain at least one name";

  llvm::SmallSet<StringRef, 8> seenNames;
  for (Attribute attr : namesAttr) {
    auto nameAttr = dyn_cast<StringAttr>(attr);
    if (!nameAttr)
      return op->emitOpError("expects ")
             << attrName << " to contain only strings";
    StringRef name = nameAttr.getValue();
    if (!isValidPublicTaskGridName(name))
      return op->emitOpError("expects valid public task_grid ")
             << attrName << " names";
    if (!seenNames.insert(name).second)
      return op->emitOpError("expects unique task_grid ")
             << attrName << " names";
  }

  return success();
}

static LogicalResult verifyTaskGridAttrs(Operation *op, OperandRange fields) {
  auto scopeAttr = op->getAttrOfType<StringAttr>("scope");
  if (!scopeAttr)
    return op->emitOpError("requires scope attribute");
  if (scopeAttr.getValue() != "device" && scopeAttr.getValue() != "cta")
    return op->emitOpError("supports only scope = \"device\" or \"cta\"");

  auto gridNameAttr = op->getAttrOfType<StringAttr>("grid_name");
  if (!gridNameAttr)
    return op->emitOpError("requires grid_name attribute");
  if (!isValidPublicTaskGridName(gridNameAttr.getValue()))
    return op->emitOpError("expects valid public task_grid grid_name");

  auto fieldNamesAttr = op->getAttrOfType<ArrayAttr>("field_names");
  if (!fieldNamesAttr)
    return op->emitOpError("requires field_names attribute");
  if (fieldNamesAttr.size() != fields.size())
    return op->emitOpError("expects field_names size to match field operands");
  if (failed(
          verifyTaskGridNameArray(op, fieldNamesAttr, "field", false)))
    return failure();

  if (fields.empty())
    return op->emitOpError("expects at least one task_grid field");

  if (auto shapeAttr = op->getAttrOfType<DenseI64ArrayAttr>("shape")) {
    if (shapeAttr.empty())
      return op->emitOpError("expects non-empty task_grid shape");
    for (int64_t dim : shapeAttr.asArrayRef()) {
      if (dim <= 0)
        return op->emitOpError("expects positive task_grid shape dimensions");
    }
  }

  return success();
}

LogicalResult TaskGridCreateOp::verify() {
  return verifyTaskGridAttrs(getOperation(), getFields());
}

LogicalResult TaskGridTileIdOp::verify() {
  if (failed(verifyTaskGridAttrs(getOperation(), getFields())))
    return failure();
  auto shapeAttr = getOperation()->getAttrOfType<DenseI64ArrayAttr>("shape");
  if (shapeAttr && getIds().size() != shapeAttr.size())
    return emitOpError("expects tile_id result count to match shape rank");
  if (!shapeAttr && !getIds().empty())
    return emitOpError("requires shape when tile_id has results");
  return success();
}

LogicalResult TaskGridCommitOp::verify() {
  return verifyTaskGridAttrs(getOperation(), getFields());
}

static LogicalResult verifyTaskDomainShape(Operation *op,
                                           ArrayRef<int64_t> domainShape) {
  if (domainShape.empty())
    return op->emitOpError("expects non-empty task domain_shape");
  for (int64_t dim : domainShape) {
    if (dim <= 0)
      return op->emitOpError("expects positive task domain_shape dimensions");
  }
  return success();
}

static LogicalResult verifyProjectionTaskMap(Operation *op, AffineMap map,
                                             int64_t domainRank,
                                             StringRef mapKind) {
  if (static_cast<int64_t>(map.getNumDims()) != domainRank)
    return op->emitOpError()
           << "expects task " << mapKind
           << " map input rank to match domain rank";
  if (map.getNumSymbols() != 0)
    return op->emitOpError()
           << "expects task " << mapKind << " map to have no symbols";
  for (AffineExpr expr : map.getResults()) {
    auto dim = dyn_cast<AffineDimExpr>(expr);
    if (!dim)
      return op->emitOpError()
             << "expects task " << mapKind
             << " map results to be dimension projections";
    if (static_cast<int64_t>(dim.getPosition()) >= domainRank)
      return op->emitOpError()
             << "expects task " << mapKind
             << " map result dimensions to reference the task domain";
  }
  return success();
}

static LogicalResult verifyTaskMapEntryKeys(Operation *op, DictionaryAttr dict,
                                            StringRef mapKind) {
  for (NamedAttribute named : dict) {
    StringRef key = named.getName().getValue();
    if (key != "grid" && key != "map" && key != "wildcard_dims")
      return op->emitOpError()
             << "expects task " << mapKind << " map entry key " << key
             << " to be one of grid, map, or wildcard_dims";
  }
  return success();
}

static LogicalResult verifyWildcardDims(Operation *op,
                                        DenseI64ArrayAttr wildcardDims,
                                        StringRef mapKind) {
  llvm::SmallSet<int64_t, 8> seenDims;
  for (int64_t dim : wildcardDims.asArrayRef()) {
    if (dim < 0)
      return op->emitOpError()
             << "expects task " << mapKind
             << " wildcard_dims to be non-negative";
    if (!seenDims.insert(dim).second)
      return op->emitOpError()
             << "expects task " << mapKind
             << " wildcard_dims to be unique";
  }
  return success();
}

static LogicalResult verifyTaskMapEntries(Operation *op, ArrayAttr entries,
                                          int64_t domainRank,
                                          StringRef mapKind,
                                          bool allowWildcards) {
  for (Attribute entryAttr : entries) {
    auto entry = dyn_cast<DictionaryAttr>(entryAttr);
    if (!entry)
      return op->emitOpError()
             << "expects task " << mapKind
             << " maps to contain only dictionaries";
    if (failed(verifyTaskMapEntryKeys(op, entry, mapKind)))
      return failure();

    auto gridAttr = dyn_cast_or_null<StringAttr>(entry.get("grid"));
    if (!gridAttr)
      return op->emitOpError()
             << "expects task " << mapKind << " map entry to contain grid";
    if (!isValidPublicTaskGridName(gridAttr.getValue()))
      return op->emitOpError()
             << "expects valid public task_grid name in task " << mapKind
             << " map";

    auto mapAttr = dyn_cast_or_null<AffineMapAttr>(entry.get("map"));
    if (!mapAttr)
      return op->emitOpError()
             << "expects task " << mapKind
             << " map entry to contain affine map";
    AffineMap map = mapAttr.getValue();
    if (failed(verifyProjectionTaskMap(op, map, domainRank, mapKind)))
      return failure();

    int64_t wildcardCount = 0;
    if (Attribute wildcardAttr = entry.get("wildcard_dims")) {
      if (!allowWildcards)
        return op->emitOpError()
               << "task writes must not use wildcard_dims";
      auto wildcardDims = dyn_cast<DenseI64ArrayAttr>(wildcardAttr);
      if (!wildcardDims)
        return op->emitOpError()
               << "expects task " << mapKind
               << " wildcard_dims to be an i64 dense array";
      wildcardCount = wildcardDims.size();
      if (failed(verifyWildcardDims(op, wildcardDims, mapKind)))
        return failure();
    }

    if (map.getNumResults() + wildcardCount == 0)
      return op->emitOpError()
             << "expects task " << mapKind
             << " map entry to target at least one grid dimension";
  }

  return success();
}

LogicalResult TaskDeclareOp::verify() {
  if (!isValidPublicTaskGridName(getTaskName()))
    return emitOpError("expects valid public task name");
  ArrayRef<int64_t> domainShape = getDomainShape();
  if (failed(verifyTaskDomainShape(getOperation(), domainShape)))
    return failure();
  int64_t domainRank = domainShape.size();

  if (failed(verifyTaskMapEntries(getOperation(), getReads(), domainRank,
                                  "read", /*allowWildcards=*/true)))
    return failure();
  if (getWrites().empty())
    return emitOpError("expects task writes to contain at least one map");
  if (failed(verifyTaskMapEntries(getOperation(), getWrites(), domainRank,
                                  "write", /*allowWildcards=*/false)))
    return failure();
  return success();
}

// ============================================================================
// InsertTileOp Type Inference + Verification
// ============================================================================
LogicalResult InsertTileOp::inferReturnTypes(
    [[maybe_unused]] MLIRContext *context,
    [[maybe_unused]] std::optional<Location> location, ValueRange operands,
    [[maybe_unused]] DictionaryAttr attributes,
    [[maybe_unused]] OpaqueProperties properties,
    [[maybe_unused]] RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  // insert_tile(src, tile, index) -> result has the same type as src.
  if (operands.size() < 3)
    return failure();

  auto srcTy = dyn_cast<RankedTensorType>(operands[0].getType());
  auto tileTy = dyn_cast<RankedTensorType>(operands[1].getType());
  if (!srcTy || !tileTy)
    return failure();

  // Keep conservative checks here; full diagnostics are handled in verify().
  if (srcTy.getElementType() != tileTy.getElementType() ||
      srcTy.getRank() != tileTy.getRank())
    return failure();

  inferredReturnTypes.clear();
  inferredReturnTypes.push_back(srcTy);
  return success();
}

// ============================================================================
// InsertTileOp Verification
//
// For dynamic index (index operand is not arith.constant):
//   - Only check constraints that are known at compile time: tile_shape
//   positivity, divisibility, element type, rank/result shape match
//   - Skip out-of-bounds and insertion region boundary checks (only known at
//   runtime)
//
// For static index: perform full checks (same as original implementation)
// ============================================================================
LogicalResult InsertTileOp::verify() {
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto tileTy = cast<RankedTensorType>(getTile().getType());
  auto dstTy = cast<RankedTensorType>(getResult().getType());

  auto srcShape = srcTy.getShape();
  auto tileShape = tileTy.getShape();
  auto dstShape = dstTy.getShape();

  // --- Basic checks required for both static and dynamic index ---

  // Check 1: element types must match
  if (srcTy.getElementType() != tileTy.getElementType())
    return emitOpError("tile element type must match source element type");
  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError("result element type must match source element type");

  // Check 2: rank must match
  if (srcTy.getRank() != tileTy.getRank())
    return emitOpError("tile rank must equal source rank");
  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("result rank must equal source rank");

  // Check 3: result shape must equal source shape
  if (dstShape != srcShape)
    return emitOpError("result shape must equal source shape");

  // Check 4: tile_shape must be positive in each dimension and divide source
  // shape
  SmallVector<int64_t> logicalGridShape(srcShape.size(), 0);
  int64_t totalTiles = 1;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (tileShape[i] <= 0)
      return emitOpError("tile shape must be positive at dimension ") << i;
    if (srcShape[i] % tileShape[i] != 0)
      return emitOpError(
                 "source shape must be divisible by tile shape at dimension ")
             << i << " (source=" << srcShape[i] << ", tile=" << tileShape[i]
             << ")";
    logicalGridShape[i] = srcShape[i] / tileShape[i];
    totalTiles *= logicalGridShape[i];
  }

  // Check 5: insert_tile updates values but does not change global layout,
  // result encoding must match source encoding
  auto srcEnc = srcTy.getEncoding();
  auto dstEnc = dstTy.getEncoding();
  if (srcEnc && dstEnc && srcEnc != dstEnc)
    return emitOpError("result encoding must match source encoding");

  // --- Determine if index is a static constant ---
  // insert_tile index is the 3rd operand: (src, tile, index).
  auto idxDef =
      getOperation()->getOperand(2).getDefiningOp<arith::ConstantOp>();
  if (!idxDef) {
    // Dynamic index: skip out-of-bounds and insertion region boundary checks,
    // handled at lowering stage
    return success();
  }

  // --- Full checks for static index ---
  int64_t index = mlir::cast<mlir::IntegerAttr>(idxDef.getValue()).getInt();
  if (index < 0 || index >= totalTiles)
    return emitOpError("index out of bounds for tile grid: index=")
           << index << ", total_tiles=" << totalTiles;

  // Delinearize to per-dimension tile indices (row-major order)
  SmallVector<int64_t> tileIndices(srcShape.size(), 0);
  int64_t remain = index;
  for (int i = static_cast<int>(srcShape.size()) - 1; i >= 0; --i) {
    tileIndices[i] = remain % logicalGridShape[i];
    remain /= logicalGridShape[i];
  }

  // tile indices -> coordinate-level offsets
  SmallVector<int64_t> offsets(srcShape.size(), 0);
  for (size_t i = 0; i < srcShape.size(); ++i)
    offsets[i] = tileIndices[i] * tileShape[i];

  // Boundary check: the full insertion region must be within the source
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (offsets[i] < 0)
      return emitOpError("offset must be non-negative at dimension ") << i;
    else if (offsets[i] + tileShape[i] > srcShape[i])
      return emitOpError("invalid insertion region at dimension ")
             << i << ": offset(" << offsets[i] << ") + tile(" << tileShape[i]
             << ") > source(" << srcShape[i] << ")";
  }

  return success();
}

LogicalResult DSLRegionOp::verify() {
  Region &body = getBody();
  const uint32_t numArguments = body.getNumArguments(),
                 numOperands = getNumOperands();
  if (numArguments != numOperands) {
    return emitOpError() << "expects number of operands (" << numArguments
                         << ") to match number of region arguments ("
                         << numOperands << ")";
  }
  for (auto [arg, operand] : llvm::zip(body.getArguments(), getOperands())) {
    if (arg.getType() != operand.getType()) {
      return emitOpError() << "expects region argument type (" << arg.getType()
                           << ") to match operand type (" << operand.getType()
                           << ")";
    }
  }
  return success();
}

void ExtractSizesOp::build(::mlir::OpBuilder &odsBuilder,
                           ::mlir::OperationState &odsState, size_t num,
                           Value tensor) {
  SmallVector<Type> tys(num, odsBuilder.getI64Type());
  build(odsBuilder, odsState, tys, tensor);
}

void ExtractStridesOp::build(::mlir::OpBuilder &odsBuilder,
                             ::mlir::OperationState &odsState, size_t num,
                             Value tensor) {
  SmallVector<Type> tys(num, odsBuilder.getI64Type());
  build(odsBuilder, odsState, tys, tensor);
}

LogicalResult PackOp::verify() {
  TypedValue<LLVM::LLVMStructType> input = getInput();
  ArrayRef<Type> body = input.getType().getBody();
  if (body.size() < 3 || body.size() % 2 != 1 ||
      !isa<LLVM::LLVMPointerType>(body[0]) ||
      !isa<LLVM::LLVMPointerType>(body[1])) {
    return emitOpError() << "expects input struct to have at least 3 elements, "
                            "with the first two being pointer types.";
  }
  return success();
}

LogicalResult LocalPointersOp::verify() {
  auto memDescTy = dyn_cast<triton::gpu::MemDescType>(getSrc().getType());
  if (!memDescTy)
    return emitOpError() << "expects src operand to be a ttg.memdesc";

  auto resultTensorTy = dyn_cast<RankedTensorType>(getResult().getType());
  auto resultPtrTy = dyn_cast<triton::PointerType>(getResult().getType());
  if (!resultTensorTy && !resultPtrTy)
    return emitOpError()
           << "expects result to be either tensor<tt.ptr<...>> or tt.ptr";

  auto ptrTy =
      resultTensorTy
          ? dyn_cast<triton::PointerType>(resultTensorTy.getElementType())
          : resultPtrTy;
  if (!ptrTy)
    return emitOpError() << "expects result element type to be tt.ptr";

  if (ptrTy.getPointeeType() != memDescTy.getElementType())
    return emitOpError() << "expects pointer pointee type "
                         << ptrTy.getPointeeType()
                         << " to match memdesc element type "
                         << memDescTy.getElementType();

  if (ptrTy.getAddressSpace() != kSharedMemoryAddressSpace)
    return emitOpError() << "expects pointers to live in shared memory";

  auto indices = getIndices();
  if (indices.empty()) {
    if (resultTensorTy) {
      if (resultTensorTy.getShape() != memDescTy.getShape())
        return emitOpError()
               << "zero-index local_pointers expects tensor result shape to "
                  "match buffer shape";
      return success();
    }
    if (!memDescTy.getShape().empty())
      return emitOpError()
             << "zero-index scalar local_pointers is only valid for rank-0 "
                "buffers";
    return success();
  }

  if (indices.size() != memDescTy.getShape().size())
    return emitOpError() << "expects indices count to match buffer rank";

  if (resultTensorTy) {
    auto resultShape = resultTensorTy.getShape();
    Attribute resultEncoding = resultTensorTy.getEncoding();

    ArrayRef<int64_t> indexShape;
    for (Value val : indices) {
      auto indexTy = dyn_cast<RankedTensorType>(val.getType());
      if (!indexTy)
        return emitOpError()
               << "tensor result expects indices to be ranked tensors";
      if (!indexTy.getElementType().isInteger())
        return emitOpError() << "expects indices return tensors to have "
                                "integer element types";
      if (indexShape.empty())
        indexShape = indexTy.getShape();
      else if (indexTy.getShape() != indexShape)
        return emitOpError()
               << "expects indices return tensors to have identical shapes";
      if (resultEncoding && indexTy.getEncoding() &&
          resultEncoding != indexTy.getEncoding())
        return emitOpError()
               << "expects indices return tensors to match result encoding; "
               << "result encoding is " << resultEncoding
               << ", index encoding is " << indexTy.getEncoding();
    }

    if (indexShape != resultShape)
      return emitOpError()
             << "expects indices return tensor shape to match result shape";
    return success();
  }

  for (Value val : indices) {
    if (auto indexTy = dyn_cast<IntegerType>(val.getType())) {
      if (!indexTy.isSignlessInteger())
        return emitOpError()
               << "expects scalar indices to be signless integers";
      continue;
    }
    return emitOpError() << "scalar result expects scalar integer indices";
  }

  return success();
}

LogicalResult ExclusiveCumsumOp::verify() {
  auto srcTy = dyn_cast<RankedTensorType>(getSrc().getType());
  if (!srcTy)
    return emitOpError() << "expects src to be a ranked tensor";

  auto exclusiveTy = dyn_cast<RankedTensorType>(getExclusive().getType());
  if (!exclusiveTy)
    return emitOpError() << "expects exclusive result to be a ranked tensor";
  if (exclusiveTy != srcTy)
    return emitOpError() << "expects exclusive result type to match src type";

  // Keep semantics aligned with current DeepSeek topk use: scan over a single
  // per-block histogram vector.
  if (srcTy.getRank() != 1)
    return emitOpError() << "currently only rank-1 tensors are supported";
  int64_t axisExtent = srcTy.getShape()[0];
  if (ShapedType::isDynamic(axisExtent) || axisExtent <= 0)
    return emitOpError() << "currently only static, positive axis extent is "
                            "supported";
  if (axisExtent > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
    return emitOpError() << "axis extent is too large";

  const int64_t rank = srcTy.getRank();
  int64_t axis = static_cast<int64_t>(getAxis());
  if (axis < 0)
    axis += rank;
  if (axis != 0)
    return emitOpError() << "currently only axis=0 is supported";

  if (getTotal().getType() != srcTy.getElementType())
    return emitOpError() << "expects total result type to match src element "
                            "type";

  return success();
}

LogicalResult InitBarrierGroupOp::verify() {
  ArrayRef<int32_t> offsets = getOffsets();
  ArrayRef<int32_t> counts = getCounts();
  if (offsets.empty())
    return emitOpError() << "expects at least one barrier";
  if (offsets.size() != counts.size())
    return emitOpError() << "expects offsets and counts to have equal length";
  if (getWorkerCount() <= 0)
    return emitOpError() << "expects worker_count to be positive";

  llvm::SmallSet<int32_t, 16> uniqueOffsets;
  for (auto [offset, count] : llvm::zip_equal(offsets, counts)) {
    if (offset < 0 || offset % 8 != 0)
      return emitOpError() << "expects non-negative, 8-byte-aligned offsets; "
                           << "got " << offset;
    if (!uniqueOffsets.insert(offset).second)
      return emitOpError() << "contains duplicate offset " << offset;
    if (count <= 0)
      return emitOpError() << "expects positive participant counts; got "
                           << count << " for offset " << offset;
  }
  return success();
}

LogicalResult DistributedBarrierOp::verify() {
  auto *op = getOperation();
  auto kindAttr = op->getAttrOfType<StringAttr>("group_kind");
  auto rankAttr = op->getAttrOfType<IntegerAttr>("group_rank");
  auto shapeAttr = op->getAttrOfType<DenseI32ArrayAttr>("group_shape");
  auto axesAttr = op->getAttrOfType<DenseI32ArrayAttr>("group_axes");
  auto maskAttr = op->getAttrOfType<DenseI32ArrayAttr>("group_mask");
  auto domainShapeAttr =
      op->getAttrOfType<DenseI32ArrayAttr>("group_domain_shape");

  const bool hasAnyGroupMeta =
      rankAttr || shapeAttr || axesAttr || maskAttr || domainShapeAttr ||
      kindAttr;
  if (!hasAnyGroupMeta)
    return success();

  if (!kindAttr) {
    return emitOpError()
           << "group_kind is required when distributed barrier group metadata "
              "is provided";
  }

  StringRef kind = kindAttr.getValue();
  if (kind != "cluster" && kind != "submesh" && kind != "grid" &&
      kind != "grid_axis_group") {
    return emitOpError()
           << "group_kind must be 'cluster', 'submesh', 'grid', or "
              "'grid_axis_group', got '"
           << kind << "'";
  }

  if (kind == "cluster" || kind == "grid") {
    if (rankAttr || shapeAttr || axesAttr || maskAttr || domainShapeAttr) {
      return emitOpError()
             << kind
             << " group_kind does not accept "
                "group_rank/group_shape/group_axes/group_mask/"
                "group_domain_shape attrs";
    }
    return success();
  }

  if (!rankAttr || !shapeAttr || !axesAttr) {
    return emitOpError()
           << kind
           << " group_kind requires "
              "group_rank/group_shape/group_axes";
  }
  const bool isGridAxisGroup = kind == "grid_axis_group";
  if (isGridAxisGroup) {
    if (maskAttr || !domainShapeAttr) {
      return emitOpError()
             << "grid_axis_group requires group_domain_shape and does not accept "
                "group_mask";
    }
  } else if (!maskAttr || domainShapeAttr) {
    return emitOpError()
           << kind
           << " requires group_mask and does not accept group_domain_shape";
  }
  if (!rankAttr.getType().isInteger(32)) {
    return emitOpError() << "group_rank must be i32";
  }

  int32_t rank = static_cast<int32_t>(rankAttr.getInt());
  if (rank <= 0) {
    return emitOpError() << "group_rank must be > 0";
  }
  if (static_cast<int32_t>(shapeAttr.size()) != rank) {
    return emitOpError() << "group_shape length (" << shapeAttr.size()
                         << ") must match group_rank (" << rank << ")";
  }
  if (static_cast<int32_t>(axesAttr.size()) != rank) {
    return emitOpError() << "group_axes length (" << axesAttr.size()
                         << ") must match group_rank (" << rank << ")";
  }

  llvm::SmallSet<int32_t, 8> seenAxes;
  for (int32_t dim : shapeAttr.asArrayRef()) {
    if (dim <= 0)
      return emitOpError() << "group_shape entries must be > 0";
  }
  for (int32_t axis : axesAttr.asArrayRef()) {
    if (axis < 0)
      return emitOpError() << "group_axes entries must be >= 0";
    if (!seenAxes.insert(axis).second) {
      return emitOpError() << "group_axes entries must be unique";
    }
  }

  if (isGridAxisGroup) {
    ArrayRef<int32_t> domainShape = domainShapeAttr.asArrayRef();
    if (domainShape.size() < static_cast<size_t>(rank)) {
      return emitOpError()
             << "grid_axis_group rank cannot exceed its launch domain rank";
    }
    for (int32_t dim : domainShape) {
      if (dim <= 0)
        return emitOpError() << "group_domain_shape entries must be > 0";
    }

    int64_t domainSize = 1;
    int64_t groupCount = 1;
    int64_t axisGroupSize = 1;
    for (int32_t dim : domainShape) {
      if (domainSize > std::numeric_limits<int32_t>::max() / dim) {
        return emitOpError()
               << "grid_axis_group domain size exceeds i32 range";
      }
      domainSize *= dim;
    }
    for (auto [index, axis] : llvm::enumerate(axesAttr.asArrayRef())) {
      if (axis >= static_cast<int32_t>(domainShape.size())) {
        return emitOpError()
               << "group_axes entry " << axis
               << " exceeds group_domain_shape rank " << domainShape.size();
      }
      if (shapeAttr.asArrayRef()[index] != domainShape[axis]) {
        return emitOpError()
               << "group_shape entry " << shapeAttr.asArrayRef()[index]
               << " must equal group_domain_shape[" << axis
               << "] (" << domainShape[axis] << ")";
      }
      int32_t dim = domainShape[axis];
      if (axisGroupSize > std::numeric_limits<int32_t>::max() / dim) {
        return emitOpError()
               << "grid_axis_group size exceeds i32 range";
      }
      axisGroupSize *= dim;
    }
    for (auto [axis, dim] : llvm::enumerate(domainShape)) {
      if (seenAxes.contains(static_cast<int32_t>(axis)))
        continue;
      if (groupCount > std::numeric_limits<int32_t>::max() / dim) {
        return emitOpError() << "grid_axis_group count exceeds i32 range";
      }
      groupCount *= dim;
    }
    if (groupCount > std::numeric_limits<int32_t>::max() / 4) {
      return emitOpError()
             << "grid_axis_group scratch allocation exceeds i32 byte range";
    }
  } else {
    if (maskAttr.asArrayRef().empty())
      return emitOpError() << "group_mask cannot be empty";
    for (int32_t id : maskAttr.asArrayRef()) {
      if (id < 0)
        return emitOpError() << "group_mask entries must be >= 0";
    }
  }

  return success();
}

LogicalResult RemotePointersOp::verify() {
  Type srcTy = getSrc().getType();
  Type resultTy = getResult().getType();
  auto getPtrInfo = [&](Type ty, triton::PointerType &ptr, bool &isTensor,
                        ArrayRef<int64_t> &shape,
                        Attribute &encoding) -> LogicalResult {
    if (auto tensorTy = dyn_cast<RankedTensorType>(ty)) {
      ptr = dyn_cast<triton::PointerType>(tensorTy.getElementType());
      if (!ptr)
        return emitOpError()
               << "expects tensor src/result element type to be tt.ptr";
      isTensor = true;
      shape = tensorTy.getShape();
      encoding = tensorTy.getEncoding();
      return success();
    }
    if (auto ptrTy = dyn_cast<triton::PointerType>(ty)) {
      ptr = ptrTy;
      isTensor = false;
      shape = ArrayRef<int64_t>();
      encoding = Attribute();
      return success();
    }
    return emitOpError() << "expects src/result to be tensor<tt.ptr<...>> or "
                            "tt.ptr";
  };

  triton::PointerType srcPtrTy;
  triton::PointerType resultPtrTy;
  bool srcIsTensor = false;
  bool resultIsTensor = false;
  ArrayRef<int64_t> srcShape;
  ArrayRef<int64_t> resultShape;
  Attribute srcEncoding;
  Attribute resultEncoding;
  if (failed(getPtrInfo(srcTy, srcPtrTy, srcIsTensor, srcShape, srcEncoding)) ||
      failed(getPtrInfo(resultTy, resultPtrTy, resultIsTensor, resultShape,
                        resultEncoding)))
    return failure();

  if (srcIsTensor != resultIsTensor)
    return emitOpError() << "expects src/result to both be scalar pointers or "
                            "both be pointer tensors";
  if (srcIsTensor) {
    if (srcShape != resultShape)
      return emitOpError() << "expects src/result pointer tensor shapes to "
                              "match";
    if (srcEncoding && resultEncoding && srcEncoding != resultEncoding)
      return emitOpError() << "expects src/result pointer tensor encodings to "
                              "match";
  }
  if (srcPtrTy.getPointeeType() != resultPtrTy.getPointeeType())
    return emitOpError() << "expects src/result pointer pointee types to "
                            "match";

  if (srcPtrTy.getAddressSpace() != kSharedMemoryAddressSpace)
    return emitOpError()
           << "expects src pointers to live in shared memory (addrspace=3)";
  if (resultPtrTy.getAddressSpace() != kClusterSharedMemoryAddressSpace)
    return emitOpError()
           << "expects result pointers to live in cluster shared memory "
              "(addrspace=7)";

  if (!getShardId().getType().isInteger(32))
    return emitOpError() << "expects shard_id to be i32";

  return success();
}

} // namespace mlir::triton::tle
