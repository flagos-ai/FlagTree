#ifdef __ILUVATAR_TLE__

#include "IR/Dialect.h"
#include "IR/VerfiyUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"

#include <limits>

using namespace mlir;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::iluvatar_tle {
namespace {
constexpr int kSharedMemoryAddressSpace = 3;
constexpr int kClusterSharedMemoryAddressSpace = 7;
} // namespace

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

LogicalResult ExtractTileOp::verify() {
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto dstTy = cast<RankedTensorType>(getResult().getType());
  auto srcShape = srcTy.getShape();
  auto dstShape = dstTy.getShape();

  SmallVector<int64_t> tileShape;
  if (auto denseArray64 =
          dyn_cast<DenseI64ArrayAttr>(getOperation()->getAttr("tile_shape"))) {
    for (auto v : denseArray64.asArrayRef())
      tileShape.push_back(v);
  }

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitError("result element type must match source element type");
  if (srcTy.getRank() != dstTy.getRank())
    return emitError("result rank must equal source rank");
  if (tileShape.size() != srcShape.size())
    return emitOpError("tile_shape rank must match source rank");

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

  auto indexConstOp =
      getOperation()->getOperand(1).getDefiningOp<arith::ConstantOp>();
  if (!indexConstOp)
    return success();

  int64_t index = cast<IntegerAttr>(indexConstOp.getValue()).getInt();
  SmallVector<int64_t> logicalGridShape(srcShape.size(), 0);
  int64_t totalTiles = 1;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    logicalGridShape[i] = srcShape[i] / tileShape[i];
    totalTiles *= logicalGridShape[i];
  }

  if (index < 0 || index >= totalTiles)
    return emitOpError("index out of bounds for tile grid: index=")
           << index << ", total_tiles=" << totalTiles;

  SmallVector<int64_t> tileIndices(srcShape.size(), 0);
  int64_t remain = index;
  for (int i = static_cast<int>(srcShape.size()) - 1; i >= 0; --i) {
    tileIndices[i] = remain % logicalGridShape[i];
    remain /= logicalGridShape[i];
  }

  SmallVector<int64_t> offsets(srcShape.size(), 0);
  for (size_t i = 0; i < srcShape.size(); ++i)
    offsets[i] = tileIndices[i] * tileShape[i];

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

  return success();
}

LogicalResult InsertTileOp::inferReturnTypes(
    [[maybe_unused]] MLIRContext *context,
    [[maybe_unused]] std::optional<Location> location, ValueRange operands,
    [[maybe_unused]] DictionaryAttr attributes,
    [[maybe_unused]] OpaqueProperties properties,
    [[maybe_unused]] RegionRange regions,
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
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto tileTy = cast<RankedTensorType>(getTile().getType());
  auto dstTy = cast<RankedTensorType>(getResult().getType());

  auto srcShape = srcTy.getShape();
  auto tileShape = tileTy.getShape();
  auto dstShape = dstTy.getShape();

  if (srcTy.getElementType() != tileTy.getElementType())
    return emitOpError("tile element type must match source element type");
  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError("result element type must match source element type");
  if (srcTy.getRank() != tileTy.getRank())
    return emitOpError("tile rank must equal source rank");
  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("result rank must equal source rank");
  if (dstShape != srcShape)
    return emitOpError("result shape must equal source shape");

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

  auto srcEnc = srcTy.getEncoding();
  auto dstEnc = dstTy.getEncoding();
  if (srcEnc && dstEnc && srcEnc != dstEnc)
    return emitOpError("result encoding must match source encoding");

  auto idxDef =
      getOperation()->getOperand(2).getDefiningOp<arith::ConstantOp>();
  if (!idxDef)
    return success();

  int64_t index = cast<IntegerAttr>(idxDef.getValue()).getInt();
  if (index < 0 || index >= totalTiles)
    return emitOpError("index out of bounds for tile grid: index=")
           << index << ", total_tiles=" << totalTiles;

  SmallVector<int64_t> tileIndices(srcShape.size(), 0);
  int64_t remain = index;
  for (int i = static_cast<int>(srcShape.size()) - 1; i >= 0; --i) {
    tileIndices[i] = remain % logicalGridShape[i];
    remain /= logicalGridShape[i];
  }

  SmallVector<int64_t> offsets(srcShape.size(), 0);
  for (size_t i = 0; i < srcShape.size(); ++i)
    offsets[i] = tileIndices[i] * tileShape[i];

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (offsets[i] < 0)
      return emitOpError("offset must be non-negative at dimension ") << i;
    if (offsets[i] + tileShape[i] > srcShape[i])
      return emitOpError("invalid insertion region at dimension ")
             << i << ": offset(" << offsets[i] << ") + tile(" << tileShape[i]
             << ") > source(" << srcShape[i] << ")";
  }

  return success();
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

LogicalResult GetDeviceIdOp::verify() {
  auto resultTy = getResult().getType();

  if (!resultTy.isInteger(32))
    return emitOpError("result type must be i32");

  return success();
}

LogicalResult GetNumPesOp::verify() {
  auto resultTy = getResult().getType();

  if (!resultTy.isInteger(32))
    return emitOpError("result type must be i32");

  return success();
}

LogicalResult DistributedBarrierOp::verify() {
  auto *op = getOperation();
  auto spaceAttr = op->getAttrOfType<StringAttr>("space");

  if (spaceAttr && spaceAttr.getValue() == "device")
    return DistributedBarrier::verifyDeviceSpace(op, getSrc());

  auto kindAttr = op->getAttrOfType<StringAttr>("group_kind");
  auto rankAttr = op->getAttrOfType<IntegerAttr>("group_rank");
  auto shapeAttr = op->getAttrOfType<DenseI32ArrayAttr>("group_shape");
  auto axesAttr = op->getAttrOfType<DenseI32ArrayAttr>("group_axes");
  auto maskAttr = op->getAttrOfType<DenseI32ArrayAttr>("group_mask");

  const bool hasAnyGroupMeta =
      rankAttr || shapeAttr || axesAttr || maskAttr || kindAttr;
  if (!hasAnyGroupMeta)
    return success();

  if (!kindAttr) {
    return emitOpError()
           << "group_kind is required when distributed barrier group metadata "
              "is provided";
  }

  StringRef kind = kindAttr.getValue();
  if (kind != "cluster" && kind != "submesh" && kind != "grid") {
    return emitOpError()
           << "group_kind must be 'cluster', 'submesh', or 'grid', got '"
           << kind << "'";
  }

  if (kind == "cluster" || kind == "grid") {
    if (rankAttr || shapeAttr || axesAttr || maskAttr) {
      return emitOpError()
             << kind
             << " group_kind does not accept "
                "group_rank/group_shape/group_axes/group_mask attrs";
    }
    return success();
  }

  if (!rankAttr || !shapeAttr || !axesAttr) {
    return emitOpError()
           << "submesh group_kind requires group_rank/group_shape/group_axes";
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
  if (maskAttr) {
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
  auto spaceAttr = getSpace();
  if (spaceAttr == "device") {
    if (failed(RemotePointers::verifyDeviceSpace(getSrc(), getResult())))
      return failure();
  } else {
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
    if (failed(
            getPtrInfo(srcTy, srcPtrTy, srcIsTensor, srcShape, srcEncoding)) ||
        failed(getPtrInfo(resultTy, resultPtrTy, resultIsTensor, resultShape,
                          resultEncoding)))
      return failure();

    if (srcIsTensor != resultIsTensor)
      return emitOpError()
             << "expects src/result to both be scalar pointers or "
                "both be pointer tensors";
    if (srcIsTensor) {
      if (srcShape != resultShape)
        return emitOpError() << "expects src/result pointer tensor shapes to "
                                "match";
      if (srcEncoding && resultEncoding && srcEncoding != resultEncoding)
        return emitOpError()
               << "expects src/result pointer tensor encodings to "
                  "match";
    }
    if (srcPtrTy.getPointeeType() != resultPtrTy.getPointeeType())
      return emitOpError() << "expects src/result pointer pointee types to "
                              "match";

    if (spaceAttr == "cluster" &&
        srcPtrTy.getAddressSpace() != kSharedMemoryAddressSpace)
      return emitOpError()
             << "expects src pointers to live in shared memory (addrspace=3)";
    if (spaceAttr == "cluster" &&
        resultPtrTy.getAddressSpace() != kClusterSharedMemoryAddressSpace)
      return emitOpError()
             << "expects result pointers to live in cluster shared memory "
                "(addrspace=7)";
  }

  if (!getShardId().getType().isInteger(32))
    return emitOpError() << "expects shard_id to be i32";

  bool hasOffset = getOffset() != nullptr;
  if (spaceAttr == "device") {
    if (!hasOffset)
      return emitOpError() << "device space remote pointers require an offset";
  } else {
    if (hasOffset)
      return emitOpError()
             << "offset is only supported for device space remote pointers";
  }

  if (hasOffset) {
    Type offsetTy = getOffset().getType();
    if (auto tensorTy = dyn_cast<RankedTensorType>(offsetTy)) {
      if (!tensorTy.getShape().empty())
        return emitOpError() << "expects offset to be a scalar";
      offsetTy = tensorTy.getElementType();
    }
    if (!offsetTy.isSignlessInteger(64))
      return emitOpError() << "expects offset to be i64";
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

  // Keep semantics aligned with the current single per-block histogram scan use
  // and the tt.scan/tt.reduce lowering: rank-1, static, axis 0.
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

namespace {

LogicalResult verifyBarrierType(Operation *op, ttg::MemDescType barrierType) {
  if (!barrierType.getElementType().isInteger(64) ||
      barrierType.getShape() != ArrayRef<int64_t>({1}))
    return op->emitOpError(
        "barrier allocation must be a descriptor of 1xi64 type");
  return success();
}

} // namespace

// -- InitBarrierOp --
LogicalResult InitBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- WaitBarrierOp --
LogicalResult WaitBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- ArriveBarrierOp --
LogicalResult ArriveBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  if (getCount() < 1)
    return emitOpError("count must be greater than or equal to 1");
  return success();
}

namespace {

bool isValidPublicPipeName(StringRef name) {
  auto isAsciiIdentStart = [](char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
  };
  auto isAsciiIdentChar = [&](char c) {
    return isAsciiIdentStart(c) || (c >= '0' && c <= '9') || c == '_';
  };
  if (name.empty() || name == "fields" || name == "readers" ||
      name.starts_with("_") || !isAsciiIdentStart(name.front()))
    return false;
  return llvm::all_of(name.drop_front(), isAsciiIdentChar);
}

LogicalResult verifyPipeNameArray(Operation *op, ArrayAttr names,
                                  StringRef attrName, bool allowEmpty) {
  if (!names)
    return op->emitOpError("requires ") << attrName << " attribute";
  if (names.empty() && !allowEmpty)
    return op->emitOpError("expects non-empty pipe ") << attrName << " names";
  DenseSet<StringRef> seenNames;
  for (Attribute attr : names) {
    auto strAttr = dyn_cast<StringAttr>(attr);
    if (!strAttr || !isValidPublicPipeName(strAttr.getValue()))
      return op->emitOpError("expects valid public pipe ")
             << attrName << " names";
    if (!seenNames.insert(strAttr.getValue()).second)
      return op->emitOpError("expects unique pipe ") << attrName << " names";
  }
  return success();
}

LogicalResult verifyPipeAttrs(Operation *op, OperandRange fields) {
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
  if (static_cast<int64_t>(fieldNamesAttr.size()) !=
      static_cast<int64_t>(fields.size()))
    return op->emitOpError("expects field_names size to match field operands");

  if (failed(verifyPipeNameArray(op, fieldNamesAttr, "field", false)))
    return failure();

  if (auto readersAttr = op->getAttrOfType<ArrayAttr>("readers")) {
    if (failed(verifyPipeNameArray(op, readersAttr, "reader", false)))
      return failure();
  }

  if (auto readerNameAttr = op->getAttrOfType<StringAttr>("reader_name")) {
    if (!isValidPublicPipeName(readerNameAttr.getValue()))
      return op->emitOpError("expects valid public pipe reader_name");
  }

  if (fields.empty())
    return op->emitOpError("expects at least one pipe field");
  for (Value field : fields) {
    auto type = cast<ttg::MemDescType>(field.getType());
    if (!isa<ttg::SharedMemorySpaceAttr>(type.getMemorySpace()))
      return op->emitOpError("expects only shared-memory pipe fields");
    if (type.getRank() < 2)
      return op->emitOpError("expects pipe fields to have rank >= 2");
    if (type.getShape()[0] != capacity)
      return op->emitOpError("expects field leading dimension to equal "
                             "pipe capacity");
  }
  return success();
}

LogicalResult verifyPipeStagePhase(Operation *op, Value stage, Value phase) {
  if (!stage.getType().isInteger(32))
    return op->emitOpError("expects stage to be i32");
  if (!phase.getType().isInteger(1))
    return op->emitOpError("expects phase to be i1");
  return success();
}

LogicalResult verifyPipeStage(Operation *op, Value stage) {
  if (!stage.getType().isInteger(32))
    return op->emitOpError("expects stage to be i32");
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

void PipeReaderReleaseOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
  MutableOperandRange fields = getFieldsMutable();
  for (unsigned i = 0, e = fields.size(); i < e; ++i)
    effects.emplace_back(MemoryEffects::Free::get(), &fields[i],
                         ttg::SharedMemory::get());
}

void DSLRegionOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // TODO: Conservative memory effects on all dsl_region ops so empty deferred
  // bodies are not DCE'd before materialize lowering exists. Narrow this to
  // alias operands or deferred-only once materialize is implemented.
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
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

} // namespace mlir::triton::iluvatar_tle

#endif // __ILUVATAR_TLE__
