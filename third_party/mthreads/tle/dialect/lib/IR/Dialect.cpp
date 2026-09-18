#ifdef __TLE__

#include "Dialect/MUSATLE/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include <cctype>
#include <cstdint>
#include <limits>
#include <optional>

// clang-format off
#include "Dialect/MUSATLE/IR/Dialect.cpp.inc"
// clang-format on

using namespace mlir;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::musa_tle {

void MUSATLEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/MUSATLE/IR/Ops.cpp.inc"
      >();
}

} // namespace mlir::triton::musa_tle

#define GET_OP_CLASSES
#include "Dialect/MUSATLE/IR/Ops.cpp.inc"

namespace mlir::triton::musa_tle {
namespace {
constexpr int kSharedMemoryAddressSpace = 3;

static bool isRank0BackingMemDesc(ttg::MemDescType memDescTy) {
  return memDescTy.getShape().size() == 1 && memDescTy.getShape().front() == 1;
}

static SmallVector<int64_t> getDenseI64Array(Attribute attr) {
  SmallVector<int64_t> values;
  if (auto dense = dyn_cast_or_null<DenseI64ArrayAttr>(attr))
    llvm::append_range(values, dense.asArrayRef());
  return values;
}

static LogicalResult verifyTileShape(Operation *op, ArrayRef<int64_t> srcShape,
                                     ArrayRef<int64_t> tileShape,
                                     StringRef attrName) {
  if (tileShape.size() != srcShape.size())
    return op->emitOpError() << attrName << " rank must match source rank";
  for (size_t idx = 0; idx < srcShape.size(); ++idx) {
    int64_t srcDim = srcShape[idx];
    int64_t tileDim = tileShape[idx];
    if (tileDim <= 0)
      return op->emitOpError()
             << attrName << " must be positive at dimension " << idx;
    if (srcDim % tileDim != 0)
      return op->emitOpError()
             << "source shape must be divisible by " << attrName
             << " at dimension " << idx << " (source=" << srcDim
             << ", tile=" << tileDim << ")";
  }
  return success();
}

static std::optional<int64_t> getConstantIndex(Value index) {
  auto constOp = index.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;
  auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
  if (!intAttr)
    return std::nullopt;
  return intAttr.getInt();
}

static LogicalResult verifyStaticTileIndex(Operation *op, Value index,
                                           ArrayRef<int64_t> srcShape,
                                           ArrayRef<int64_t> tileShape) {
  std::optional<int64_t> staticIndex = getConstantIndex(index);
  if (!staticIndex)
    return success();

  int64_t totalTiles = 1;
  for (auto [srcDim, tileDim] : llvm::zip_equal(srcShape, tileShape))
    totalTiles *= srcDim / tileDim;
  if (*staticIndex < 0 || *staticIndex >= totalTiles)
    return op->emitOpError("index out of bounds for tile grid: index=")
           << *staticIndex << ", total_tiles=" << totalTiles;
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

static LogicalResult verifyPipeNameArray(Operation *op, ArrayAttr names,
                                         StringRef description,
                                         bool allowEmpty) {
  if (!allowEmpty && names.empty())
    return op->emitOpError("expects ")
           << description << " to contain at least one name";
  llvm::SmallSet<StringRef, 8> seen;
  for (Attribute attr : names) {
    auto name = dyn_cast<StringAttr>(attr);
    if (!name)
      return op->emitOpError("expects ")
             << description << " to contain only strings";
    if (!isValidPublicPipeName(name.getValue()))
      return op->emitOpError("expects valid public MUSA TLE pipe ")
             << description << " names";
    if (!seen.insert(name.getValue()).second)
      return op->emitOpError("expects unique MUSA TLE pipe ")
             << description << " names";
  }
  return success();
}

static LogicalResult verifyPipeAttrs(Operation *op, OperandRange fields) {
  auto capacity = op->getAttrOfType<IntegerAttr>("capacity");
  if (!capacity || capacity.getInt() <= 0)
    return op->emitOpError("requires positive capacity");
  auto scope = op->getAttrOfType<StringAttr>("scope");
  if (!scope || scope.getValue() != "cta")
    return op->emitOpError("supports only scope = \"cta\"");

  auto fieldNames = op->getAttrOfType<ArrayAttr>("field_names");
  if (!fieldNames || fieldNames.size() != fields.size())
    return op->emitOpError("expects field_names size to match field operands");
  if (failed(verifyPipeNameArray(op, fieldNames, "field", false)))
    return failure();
  if (fields.empty())
    return op->emitOpError("expects at least one pipe field");
  for (Value field : fields) {
    auto type = cast<ttg::MemDescType>(field.getType());
    if (!isa<ttg::SharedMemorySpaceAttr>(type.getMemorySpace()))
      return op->emitOpError("expects only shared-memory pipe fields");
    if (type.getRank() < 2 || type.getShape().front() != capacity.getInt())
      return op->emitOpError(
          "expects field leading dimension to equal pipe capacity");
  }

  if (auto readers = op->getAttrOfType<ArrayAttr>("readers")) {
    if (!isa<PipeCreateOp>(op))
      return op->emitOpError("readers is only valid on musa_tle.pipe.create");
    if (failed(verifyPipeNameArray(op, readers, "reader", false)))
      return failure();
  }
  if (auto readerName = op->getAttrOfType<StringAttr>("reader_name")) {
    if (!isa<PipeReaderWaitOp, PipeReaderReleaseOp>(op))
      return op->emitOpError(
          "reader_name is only valid on MUSA TLE pipe reader operations");
    if (!isValidPublicPipeName(readerName.getValue()))
      return op->emitOpError("expects valid public MUSA TLE pipe reader_name");
  }
  if (auto readerFields = op->getAttrOfType<ArrayAttr>("reader_fields")) {
    if (!isa<PipeReaderWaitOp, PipeReaderReleaseOp>(op))
      return op->emitOpError(
          "reader_fields is only valid on MUSA TLE pipe reader operations");
    if (failed(verifyPipeNameArray(op, readerFields, "reader field", false)))
      return failure();
    for (Attribute readerField : readerFields) {
      if (!llvm::is_contained(fieldNames, readerField))
        return op->emitOpError(
            "expects reader_fields to reference payload field_names");
    }
  }
  return success();
}

static LogicalResult verifyPipeStage(Operation *op, Value stage) {
  if (!stage.getType().isInteger(32))
    return op->emitOpError("expects stage to be i32");
  return success();
}

static LogicalResult verifyPipeStagePhase(Operation *op, Value stage,
                                          Value phase) {
  if (failed(verifyPipeStage(op, stage)))
    return failure();
  if (!phase.getType().isInteger(1))
    return op->emitOpError("expects phase to be i1");
  return success();
}
} // namespace

LogicalResult PipeCreateOp::verify() {
  return verifyPipeAttrs(getOperation(), getFields());
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
  return verifyPipeStagePhase(getOperation(), getStage(), getPhase());
}

LogicalResult PipeReaderReleaseOp::verify() {
  if (failed(verifyPipeAttrs(getOperation(), getFields())))
    return failure();
  return verifyPipeStage(getOperation(), getStage());
}

void PipeReaderReleaseOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
  MutableOperandRange fields = getFieldsMutable();
  for (unsigned index = 0; index < fields.size(); ++index)
    effects.emplace_back(MemoryEffects::Free::get(), &fields[index],
                         ttg::SharedMemory::get());
}

void ExtractTileOp::build(OpBuilder &builder, OperationState &state, Value src,
                          Value index, ArrayRef<int64_t> tileShape) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  auto resultTy = RankedTensorType::get(tileShape, srcTy.getElementType(),
                                        srcTy.getEncoding());
  state.addOperands({src, index});
  state.addAttribute("tile_shape", builder.getDenseI64ArrayAttr(tileShape));
  state.addTypes(resultTy);
}

LogicalResult ExtractTileOp::verify() {
  auto srcTy = dyn_cast<RankedTensorType>(getSrc().getType());
  auto resultTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!srcTy || !resultTy)
    return emitOpError("expects source and result to be ranked tensors");

  SmallVector<int64_t> tileShape =
      getDenseI64Array(getOperation()->getAttr("tile_shape"));
  if (failed(verifyTileShape(getOperation(), srcTy.getShape(), tileShape,
                             "tile_shape")))
    return failure();
  if (srcTy.getElementType() != resultTy.getElementType())
    return emitOpError("result element type must match source element type");
  if (srcTy.getRank() != resultTy.getRank())
    return emitOpError("result rank must match source rank");
  if (resultTy.getShape() != ArrayRef<int64_t>(tileShape))
    return emitOpError("result shape must equal tile_shape");
  if (srcTy.getEncoding() && resultTy.getEncoding() &&
      srcTy.getEncoding() != resultTy.getEncoding())
    return emitOpError("result encoding must match source encoding");
  return verifyStaticTileIndex(getOperation(), getIndex(), srcTy.getShape(),
                               tileShape);
}

void InsertTileOp::build(OpBuilder &builder, OperationState &state, Value src,
                         Value tile, Value index) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  auto tileTy = cast<RankedTensorType>(tile.getType());
  SmallVector<int64_t> tileShape(tileTy.getShape());
  state.addOperands({src, tile, index});
  state.addAttribute("tile_shape", builder.getDenseI64ArrayAttr(tileShape));
  state.addTypes(srcTy);
}

LogicalResult InsertTileOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands.size() < 3)
    return failure();
  auto srcTy = dyn_cast<RankedTensorType>(operands[0].getType());
  auto tileTy = dyn_cast<RankedTensorType>(operands[1].getType());
  if (!srcTy || !tileTy)
    return failure();
  if (srcTy.getElementType() != tileTy.getElementType() ||
      srcTy.getRank() != tileTy.getRank())
    return failure();
  inferredReturnTypes.clear();
  inferredReturnTypes.push_back(srcTy);
  return success();
}

LogicalResult InsertTileOp::verify() {
  auto srcTy = dyn_cast<RankedTensorType>(getSrc().getType());
  auto tileTy = dyn_cast<RankedTensorType>(getTile().getType());
  auto resultTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!srcTy || !tileTy || !resultTy)
    return emitOpError("expects source, tile, and result to be ranked tensors");

  SmallVector<int64_t> tileShape =
      getDenseI64Array(getOperation()->getAttr("tile_shape"));
  if (failed(verifyTileShape(getOperation(), srcTy.getShape(), tileShape,
                             "tile_shape")))
    return failure();
  if (tileTy.getShape() != ArrayRef<int64_t>(tileShape))
    return emitOpError("tile shape must equal tile_shape");
  if (srcTy.getElementType() != tileTy.getElementType())
    return emitOpError("tile element type must match source element type");
  if (srcTy.getElementType() != resultTy.getElementType())
    return emitOpError("result element type must match source element type");
  if (srcTy.getRank() != tileTy.getRank())
    return emitOpError("tile rank must match source rank");
  if (srcTy.getRank() != resultTy.getRank())
    return emitOpError("result rank must match source rank");
  if (resultTy.getShape() != srcTy.getShape())
    return emitOpError("result shape must equal source shape");
  if (srcTy.getEncoding() && resultTy.getEncoding() &&
      srcTy.getEncoding() != resultTy.getEncoding())
    return emitOpError("result encoding must match source encoding");
  return verifyStaticTileIndex(getOperation(), getIndex(), srcTy.getShape(),
                               tileShape);
}

LogicalResult LocalPointersOp::verify() {
  auto memDescTy = dyn_cast<ttg::MemDescType>(getSrc().getType());
  if (!memDescTy)
    return emitOpError() << "expects src operand to be a ttg.memdesc";
  if (!isa<ttg::SharedMemorySpaceAttr>(memDescTy.getMemorySpace()))
    return emitOpError() << "expects src memdesc to live in shared memory";
  if (!isa<ttg::SharedEncodingTrait>(memDescTy.getEncoding()))
    return emitOpError() << "expects src memdesc to use a shared encoding";

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
    if (!memDescTy.getShape().empty() && !isRank0BackingMemDesc(memDescTy))
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
               << "expects indices return tensors to match result encoding";
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

LogicalResult BarrierAllocOp::verify() {
  if (getNumBarriers() <= 0)
    return emitOpError("num_barriers must be positive");
  if (getNumBarriers() > 63)
    return emitOpError("num_barriers exceeds the 63 mthreads hardware "
                       "barrier id limit");
  if (getArriveCount() <= 0)
    return emitOpError("arrive_count must be positive");
  if (getInitPolarity() != 0 && getInitPolarity() != 1)
    return emitOpError("init_polarity must be 0 or 1");
  if (auto expectBytes =
          getOperation()->getAttrOfType<IntegerAttr>("expect_bytes")) {
    if (expectBytes.getInt() <= 0)
      return emitOpError("expect_bytes must be positive when present");
  }
  return success();
}

LogicalResult BarrierIndexOp::verify() {
  APInt constantIndex;
  if (!matchPattern(getIndex(), m_ConstantInt(&constantIndex)))
    return success();

  int64_t index = constantIndex.getSExtValue();
  if (index < 0)
    return emitOpError("barrier index must be non-negative when constant");

  if (auto alloc = getBaseId().getDefiningOp<BarrierAllocOp>()) {
    if (index >= alloc.getNumBarriers())
      return emitOpError("barrier index ")
             << index << " out of bounds for " << alloc.getNumBarriers()
             << " barriers";
  }
  return success();
}

LogicalResult BarrierWaitOp::verify() { return success(); }

LogicalResult BarrierArriveOp::verify() {
  if (getArriveCount() != 1)
    return emitOpError(
        "mthreads hardware barrier arrive requires arrive_count = 1");
  return success();
}

LogicalResult SetLayoutOp::verify() {
  auto srcTy = dyn_cast<RankedTensorType>(getSrc().getType());
  auto resultTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!srcTy || !resultTy)
    return emitOpError("expects source and result to be ranked tensors");

  Attribute targetEncoding = getTargetEncoding();
  if (!dyn_cast<ttg::DistributedEncodingTrait>(targetEncoding))
    return emitOpError("target_encoding must be a distributed encoding");

  auto layoutEncoding = dyn_cast<ttg::LayoutEncodingTrait>(targetEncoding);
  if (!layoutEncoding)
    return emitOpError("distributed target_encoding must expose a layout rank");

  unsigned targetRank = layoutEncoding.getRank();
  if (targetRank != static_cast<unsigned>(srcTy.getRank()))
    return emitOpError("target encoding rank ")
           << targetRank << " must match source tensor rank "
           << srcTy.getRank();
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

LogicalResult SqmmaOp::verify() {
  auto aTy = dyn_cast<ttg::MemDescType>(getA().getType());
  auto bTy = dyn_cast<ttg::MemDescType>(getB().getType());
  auto cTy = dyn_cast<RankedTensorType>(getC().getType());
  auto dTy = dyn_cast<RankedTensorType>(getD().getType());
  if (!aTy || !bTy || !cTy || !dTy)
    return emitOpError("expects memdesc A/B and ranked tensor accumulator");
  if (aTy.getRank() != 2 || bTy.getRank() != 2 || cTy.getRank() != 2)
    return emitOpError("expects rank-2 A, B, and accumulator operands");
  if (!isa<ttg::SharedMemorySpaceAttr>(aTy.getMemorySpace()) ||
      !isa<ttg::SharedMemorySpaceAttr>(bTy.getMemorySpace()))
    return emitOpError("expects A and B in shared memory");
  Type aElemTy = aTy.getElementType();
  Type bElemTy = bTy.getElementType();
  bool supportedInput =
      aElemTy.isF16() || aElemTy.isBF16() || isa<Float8E4M3FNType>(aElemTy);
  if (!supportedInput || aElemTy != bElemTy)
    return emitOpError(
        "mthreads TLE SQMMA requires matching f16, bf16, or fp8e4nv A/B");
  if (!cTy.getElementType().isF32() || !dTy.getElementType().isF32())
    return emitOpError(
        "initial mthreads TLE SQMMA requires an f32 accumulator/result");

  ArrayRef<int64_t> aShape = aTy.getShape();
  ArrayRef<int64_t> bShape = bTy.getShape();
  ArrayRef<int64_t> cShape = cTy.getShape();
  if (aShape[1] != bShape[0] || cShape[0] != aShape[0] ||
      cShape[1] != bShape[1])
    return emitOpError("expects A[M,K] * B[K,N] and accumulator[M,N]");
  if (!getIsAsync())
    return emitOpError("requires isAsync=true");
  if (getMaxNumImpreciseAcc() != 0)
    return emitOpError(
        "initial mthreads TLE SQMMA requires maxNumImpreciseAcc=0");
  return success();
}

LogicalResult SqmmaWaitOp::verify() {
  auto pendings = getOperation()->getAttrOfType<IntegerAttr>("pendings");
  if (!pendings || pendings.getInt() != 0)
    return emitOpError(
        "mthreads TLE wgmma_wait currently requires pendings=0; non-zero "
        "pending groups are not supported");
  return success();
}

} // namespace mlir::triton::musa_tle

#endif // __TLE__
