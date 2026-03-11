#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"

#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::tle {

namespace {
// Triton shared-memory pointers map to LLVM address space 3 (NVVM shared).
constexpr int kSharedMemoryAddressSpace = 3;
} // namespace
/*void ExtractTileOp::build(::mlir::OpBuilder &odsBuilder,
                          ::mlir::OperationState &odsState, Value input,
                          Value index, ArrayRef<int64_t> tileShape) {
  auto inputType = cast<RankedTensorType>(input.getType());
  SmallVector<Type> tys = {
      RankedTensorType::get(tileShape, inputType.getElementType())};
  build(odsBuilder, odsState, tys, input, index);
}*/

//============================================================================
// 辅助函数：获取 CTA tile shape
// ============================================================================
static SmallVector<int64_t> getShapePerCTATile(RankedTensorType type) {
  auto encoding = type.getEncoding();
  auto shape = type.getShape();

  if (auto blocked = dyn_cast<gpu::BlockedEncodingAttr>(encoding)) {
    auto sizePerThread = blocked.getSizePerThread();
    auto threadsPerWarp = blocked.getThreadsPerWarp();
    auto warpsPerCTA = blocked.getWarpsPerCTA();

    SmallVector<int64_t> ctaTileShape;
    for (size_t i = 0; i < shape.size(); ++i) {
      ctaTileShape.push_back(
          static_cast<int64_t>(sizePerThread[i]) *
          static_cast<int64_t>(threadsPerWarp[i]) *
          static_cast<int64_t>(warpsPerCTA[i])
      );
    }
    return ctaTileShape;
  }

  // 其他编码类型的支持
  if (auto linear = dyn_cast<gpu::LinearEncodingAttr>(encoding)) {
    auto sizePerThread = linear.getSizePerThread();
    auto threadsPerWarp = linear.getThreadsPerWarp();
    auto warpsPerCTA = linear.getWarpsPerCTA();

    SmallVector<int64_t> ctaTileShape;
    for (size_t i = 0; i < shape.size(); ++i) {
      ctaTileShape.push_back(
          static_cast<int64_t>(sizePerThread[i]) *
          static_cast<int64_t>(threadsPerWarp[i]) *
          static_cast<int64_t>(warpsPerCTA[i])
      );
    }
    return ctaTileShape;
  }

  llvm_unreachable("Unsupported encoding for extract_tile");
}

// ============================================================================
// ExtractTileOp Builder
// ============================================================================
void ExtractTileOp::build(
    OpBuilder &builder,
    OperationState &state,
    Value src,
    Value index,
    ArrayRef<int64_t> tileShape) {
  auto srcType = cast<RankedTensorType>(src.getType());
  auto resultType = RankedTensorType::get(
      tileShape,
      srcType.getElementType(),
      srcType.getEncoding()
  );
  state.addOperands(src);
  state.addOperands(index);
  state.addAttribute("tile_shape", builder.getDenseI64ArrayAttr(tileShape));
  state.addTypes(resultType);
}

// ============================================================================
// ExtractTileOp Verification
//
// 动态 index（index 操作数不是 arith.constant）时：
//   - 只做编译期可知的约束：tile_shape 正数、整除性、元素类型、rank 匹配
//   - 跳过越界检查和 CTA tile 对齐检查（运行时才知道值）
//
// 静态 index 时：执行完整检查（与原实现等价）
// ============================================================================
LogicalResult ExtractTileOp::verify() {
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto dstTy = cast<RankedTensorType>(getResult().getType());
  auto srcShape = srcTy.getShape();
  auto dstShape = dstTy.getShape();

  // ── 获取 tile_shape 属性 ────────────────────────────────────────────────
  auto tileShapeRawAttr = getOperation()->getAttr("tile_shape");
  SmallVector<int64_t> tileShape;
  if (auto denseArray64 = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(tileShapeRawAttr)) {
    for (auto v : denseArray64.asArrayRef())
      tileShape.push_back(v);
  }

  // ── 无论静态/动态都必须通过的基本检查 ─────────────────────────────────

  // 检查1：元素类型必须匹配
  if (srcTy.getElementType() != dstTy.getElementType())
    return emitError("result element type must match source element type");

  // 检查2：rank 必须匹配
  if (srcTy.getRank() != dstTy.getRank())
    return emitError("result rank must equal source rank");

  // 检查3：tile_shape rank 与 source rank 匹配
  if (tileShape.size() != srcShape.size())
    return emitOpError("tile_shape rank must match source rank");

  // 检查4：tile_shape 每维正数 + 整除性 + dst shape 与 tile_shape 一致
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (tileShape[i] <= 0)
      return emitOpError("tile_shape must be positive at dimension ") << i;
    if (srcShape[i] % tileShape[i] != 0)
      return emitOpError("source shape must be divisible by tile_shape at dimension ")
             << i << " (source=" << srcShape[i] << ", tile=" << tileShape[i] << ")";
    if (dstShape[i] != tileShape[i])
      return emitOpError("result shape must equal tile_shape at dimension ") << i;
  }

  // ── 判断 index 是否为静态常量 ────────────────────────────────────────────
  // getDefiningOp<arith::ConstantOp>() 对动态 Value 返回 nullptr
  auto indexConstOp =
      getOperation()->getOperand(1).getDefiningOp<arith::ConstantOp>();

  if (!indexConstOp) {
    // 动态 index：跳过越界和偏移对齐检查，lowering 阶段再处理
    return success();
  }

  // ── 静态 index 的完整检查 ────────────────────────────────────────────────
  int64_t index =
      mlir::cast<mlir::IntegerAttr>(indexConstOp.getValue()).getInt();

  // 计算逻辑网格形状
  SmallVector<int64_t> logicalGridShape(srcShape.size(), 0);
  int64_t totalTiles = 1;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    logicalGridShape[i] = srcShape[i] / tileShape[i];
    totalTiles *= logicalGridShape[i];
  }

  // 越界检查
  if (index < 0 || index >= totalTiles)
    return emitOpError("index out of bounds for tile grid: index=")
           << index << ", total_tiles=" << totalTiles;

  // 反线性化为每一维 tile 索引（行主序）
  SmallVector<int64_t> tileIndices(srcShape.size(), 0);
  int64_t remain = index;
  for (int i = static_cast<int>(srcShape.size()) - 1; i >= 0; --i) {
    tileIndices[i] = remain % logicalGridShape[i];
    remain /= logicalGridShape[i];
  }

  // tile 索引 -> 坐标级 offsets
  SmallVector<int64_t> offsets(srcShape.size(), 0);
  for (size_t i = 0; i < srcShape.size(); ++i)
    offsets[i] = tileIndices[i] * tileShape[i];

  // 边界检查
  if (offsets.size() != static_cast<size_t>(srcTy.getRank()))
    return emitError("offsets size must match tensor rank");

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (dstShape[i] > srcShape[i])
      return emitOpError("result shape cannot exceed source shape at dimension ") << i;
    if (offsets[i] + dstShape[i] > srcShape[i])
      return emitOpError("invalid offset at dimension ") << i
             << ": offset(" << offsets[i] << ") + shape(" << dstShape[i]
             << ") > source(" << srcShape[i] << ")";
    if (offsets[i] < 0)
      return emitOpError("offset must be non-negative at dimension ") << i;
  }

  // ── CTA tile 对齐检查（仅有 encoding 时执行）────────────────────────────
  //
  // 在 Triton IR 阶段，tensor 还没有 encoding，这是正常的；
  // 只在 TritonGPU IR 阶段才有 encoding。
  // 注意：不对齐时此处不报错，lowering 阶段会自动选择 SMEM 中转路径。
  auto encoding = srcTy.getEncoding();
  if (!encoding)
    return success();

  if (auto blocked = dyn_cast_or_null<gpu::BlockedEncodingAttr>(encoding)) {
    auto sizePerThread = blocked.getSizePerThread();
    auto threadsPerWarp = blocked.getThreadsPerWarp();
    auto warpsPerCTA = blocked.getWarpsPerCTA();
    SmallVector<int64_t> ctaTileShape;
    for (size_t i = 0; i < srcShape.size(); ++i) {
      ctaTileShape.push_back(
          static_cast<int64_t>(sizePerThread[i]) *
          static_cast<int64_t>(threadsPerWarp[i]) *
          static_cast<int64_t>(warpsPerCTA[i])
      );
    }
    // CTA tile 对齐检查框架（注释保留，不对齐时由 lowering 选择 SMEM 路径）：
    // for (size_t i = 0; i < srcShape.size(); ++i) {
    //   if (offsets[i] % ctaTileShape[i] != 0)
    //     return emitOpError("offset must be multiple of CTA tile size at dimension ") << i;
    //   if (dstShape[i] % ctaTileShape[i] != 0)
    //     return emitOpError("result shape must be multiple of CTA tile size at dimension ") << i;
    // }
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

LogicalResult DistributedBarrierOp::verify() {
  auto *op = getOperation();
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
  auto srcTy = dyn_cast<RankedTensorType>(getSrc().getType());
  if (!srcTy)
    return emitOpError() << "expects src operand to be a ranked tensor";
  auto resultTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!resultTy)
    return emitOpError() << "expects result to be a ranked tensor";
  if (srcTy != resultTy)
    return emitOpError() << "expects result type to match src type";

  auto ptrTy = dyn_cast<triton::PointerType>(srcTy.getElementType());
  if (!ptrTy)
    return emitOpError() << "expects src/result element type to be tt.ptr";
  if (ptrTy.getAddressSpace() != kSharedMemoryAddressSpace)
    return emitOpError() << "expects pointers to live in shared memory";

  if (!getShardId().getType().isInteger(32))
    return emitOpError() << "expects shard_id to be i32";

  return success();
}

} // namespace mlir::triton::tle
