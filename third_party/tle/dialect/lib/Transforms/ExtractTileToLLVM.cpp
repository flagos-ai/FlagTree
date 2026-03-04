#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/Builders.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/PatternTleToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

namespace ttg = mlir::triton::gpu;
using namespace mlir::triton::tle;

// ============================================================================
// 辅助函数：多维逐元素操作
// ============================================================================
template <typename T1, typename T2, typename BinaryOp>
static SmallVector<T2> multiDimElementwise(ArrayRef<T1> lhs, ArrayRef<T2> rhs,
                                           BinaryOp op) {
  assert(lhs.size() == rhs.size() && "Dimensions must match");
  SmallVector<T2> result;
  result.reserve(lhs.size());

  for (size_t i = 0; i < lhs.size(); ++i) {
    result.push_back(static_cast<T2>(op(lhs[i], rhs[i])));
  }

  return result;
}

// ============================================================================
// 辅助函数：从 BlockedEncoding 提取 CTA tile 遍历顺序
// ============================================================================
static SmallVector<unsigned> getCTATileOrder(RankedTensorType type) {
  if (auto blockedLayout = dyn_cast<ttg::BlockedEncodingAttr>(type.getEncoding())) {
    auto order = blockedLayout.getOrder();
    return SmallVector<unsigned>(order.begin(), order.end());
  }

  unsigned rank = type.getRank();
  SmallVector<unsigned> order;
  order.reserve(rank);
  for (unsigned i = 0; i < rank; ++i)
    order.push_back(rank - 1 - i);
  return order;
}



// ============================================================================
// 辅助函数：反线性化（线性索引 → 多维坐标）
// 核心修复：Triton 的 order[0] 是最快变化的维度 (Stride=1)
// ============================================================================
static SmallVector<unsigned> delinearize(unsigned linearIndex,
                                         ArrayRef<unsigned> shape,
                                         ArrayRef<unsigned> order) {
  SmallVector<unsigned> result(shape.size(), 0);
  unsigned idx = linearIndex;
  // 从 order[0] 开始分配
  for (size_t i = 0; i < order.size(); ++i) {
    unsigned dim = order[i];
    result[dim] = idx % shape[dim];
    idx /= shape[dim];
  }
  return result;
}

// ============================================================================
// 辅助函数：线性化（多维坐标 → 线性索引）
// 核心修复：Triton 的 order[0] 是最快变化的维度 (Stride=1)
// ============================================================================
static unsigned linearize(ArrayRef<unsigned> coords,
                          ArrayRef<unsigned> shape,
                          ArrayRef<unsigned> order) {
  unsigned result = 0;
  unsigned stride = 1;
  // 从 order[0] 开始累加
  for (size_t i = 0; i < order.size(); ++i) {
    unsigned dim = order[i];
    result += coords[dim] * stride;
    stride *= shape[dim];
  }
  return result;
}

// ============================================================================
// 辅助函数：获取 CTA tile shape（保持原实现）
// ============================================================================
static SmallVector<unsigned> getShapePerCTATile(RankedTensorType type) {
  auto encoding = type.getEncoding();

  if (!encoding) {
    llvm_unreachable("extract_tile requires tensor with encoding");
  }

  auto shape = type.getShape();

  if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(encoding)) {
    auto sizePerThread = blocked.getSizePerThread();
    auto threadsPerWarp = blocked.getThreadsPerWarp();
    auto warpsPerCTA = blocked.getWarpsPerCTA();

    SmallVector<unsigned> result;
    for (size_t i = 0; i < shape.size(); ++i) {
      result.push_back(
          static_cast<unsigned>(sizePerThread[i]) *
          static_cast<unsigned>(threadsPerWarp[i]) *
          static_cast<unsigned>(warpsPerCTA[i])
      );
    }
    return result;
  }

  llvm_unreachable("extract_tile only supports BlockedEncoding");
}

// ============================================================================
// ExtractTileOp → LLVM 转换（AMD 风格）
// ============================================================================
struct ExtractTileOpConversion 
    : public ConvertOpToLLVMPattern<ExtractTileOp> {

  using ConvertOpToLLVMPattern<ExtractTileOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      ExtractTileOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();

    // ═══════════════════════════════════════════════════════════
    // Step 1: 类型检查
    // ═══════════════════════════════════════════════════════════
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = dyn_cast<RankedTensorType>(op.getType());

    if (!srcTy || !dstTy) {
      return op.emitError("extract_tile operands must be ranked tensors");
    }

    auto srcEnc = srcTy.getEncoding();
    auto dstEnc = dstTy.getEncoding();

    if (!srcEnc || !dstEnc) {
      return op.emitError("extract_tile requires tensors with encoding");
    }

    if (!isa<ttg::BlockedEncodingAttr>(srcEnc)) {
      return op.emitError("extract_tile only supports BlockedEncodingAttr");
    }

    // ═══════════════════════════════════════════════════════════
    // Step 2: 解包输入寄存器值
    // ═══════════════════════════════════════════════════════════
    auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);

    // ═══════════════════════════════════════════════════════════
    // Step 3 & 4: 逻辑网格与物理网格解耦映射
    // ═══════════════════════════════════════════════════════════
    auto srcShape = srcTy.getShape();
    auto dstShape = dstTy.getShape();

    // 1. 获取物理 CTA tile 的像素形状 (例如 [8, 16])
    auto shapePerCTATile = getShapePerCTATile(srcTy);

    // 2. 计算物理 CTA 网格形状 (例如 [32/8, 16/16] = [4, 1])
    auto srcCTAShape = multiDimElementwise<int64_t, unsigned>(
        srcShape, shapePerCTATile, std::divides<unsigned>());
    auto dstCTAShape = multiDimElementwise<int64_t, unsigned>(
        dstShape, shapePerCTATile, std::divides<unsigned>());

    // 3. 获取前端传进来的 index 标量
    int64_t index = 0;
    if (auto constOp = op->getOperand(1).getDefiningOp<mlir::arith::ConstantOp>()) {
        index = mlir::cast<mlir::IntegerAttr>(constOp.getValue()).getInt();
    }

    // 4. 从属性获取用户定义的 逻辑 Tile 形状 (例如 [16, 16])
    SmallVector<int64_t> logicalTileShape;
    auto tileShapeRawAttr = op->getAttr("tile_shape");
    if (auto denseArray64 = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(tileShapeRawAttr)) {
        for (auto v : denseArray64.asArrayRef()) logicalTileShape.push_back(v);
    }

    // 5. 计算 逻辑网格 形状 (例如 [32/16, 16/16] = [2, 1])
    SmallVector<int64_t> logicalGridShape(srcShape.size(), 0);
    for (size_t i = 0; i < srcShape.size(); ++i) {
        logicalGridShape[i] = srcShape[i] / logicalTileShape[i];
    }

    // 6. 用逻辑网格解包 index 为 逻辑坐标 (index 1 -> [1, 0])
    SmallVector<int64_t> logicalCoords(srcShape.size(), 0);
    int64_t remain = index;
    for (int i = srcShape.size() - 1; i >= 0; --i) {
        logicalCoords[i] = remain % logicalGridShape[i];
        remain /= logicalGridShape[i];
    }

    // 7. 计算起始元素的 绝对像素坐标 (例如 [1*16, 0*16] = [16, 0])
    SmallVector<int64_t> elementCoords(srcShape.size(), 0);
    for (size_t i = 0; i < srcShape.size(); ++i) {
        elementCoords[i] = logicalCoords[i] * logicalTileShape[i];
    }

    //8. 计算提取起点所对应的 起始物理 CTA 坐标 (例如 [16/8, 0/16] = [2, 0])
    auto firstTileCoordinate = multiDimElementwise<int64_t, unsigned>(
            elementCoords, shapePerCTATile, std::divides<unsigned>());


    // 计算需要提取的 CTA tile 总数
    auto numCTATiles = std::accumulate(dstCTAShape.begin(), dstCTAShape.end(),
                                      1, std::multiplies<>());
                                      
    auto srcCTAOrder = getCTATileOrder(srcTy);
    auto dstCTAOrder = getCTATileOrder(dstTy);

    // ═══════════════════════════════════════════════════════════
    // Step 6: 计算每个 CTA tile 的元素数
    // ═══════════════════════════════════════════════════════════
    unsigned totalSrcCTAs = std::accumulate(srcCTAShape.begin(), 
                                           srcCTAShape.end(),
                                           1, std::multiplies<>());

    unsigned elemsPerThreadPerCTA = 
        ttg::getTotalElemsPerThread(srcTy) / totalSrcCTAs;

    // ═══════════════════════════════════════════════════════════
    // Step 7: 提取目标 tiles 的寄存器值（核心循环）
    // ═══════════════════════════════════════════════════════════
    SmallVector<Value> resultVals;
    resultVals.reserve(ttg::getTotalElemsPerThread(dstTy));

    for (size_t i = 0; i < numCTATiles; i++) {
      // 7.1 反线性化：计算当前 tile 在目标张量中的坐标
      auto coordInDstTensor = delinearize(i, dstCTAShape, dstCTAOrder);

      // 7.2 映射到源张量坐标
      // coordInDstTensor + firstTileCoordinate
      auto coordInSrcTensor = multiDimElementwise<unsigned, unsigned>(
          coordInDstTensor, firstTileCoordinate, std::plus<unsigned>());

      // 7.3 线性化：转换为源张量中的线性索引
      auto linearIdxInSrcTensor = linearize(
          coordInSrcTensor, srcCTAShape, srcCTAOrder);

      // 7.4 计算起始元素位置
      size_t startIdx = linearIdxInSrcTensor * elemsPerThreadPerCTA;

      // 7.5 边界检查
      if (startIdx + elemsPerThreadPerCTA > vals.size()) {
        return op.emitError("Internal error: register index out of bounds")
               << " startIdx=" << startIdx 
               << " elemsPerThreadPerCTA=" << elemsPerThreadPerCTA
               << " vals.size()=" << vals.size();
      }

      // 7.6 复制这个 CTA tile 的所有元素
      llvm::append_range(resultVals, 
          llvm::ArrayRef(vals).slice(startIdx, elemsPerThreadPerCTA));
    }

    // ═══════════════════════════════════════════════════════════
    // Step 8: 打包结果
    // ═══════════════════════════════════════════════════════════
    Value ret = packLLElements(
        loc, this->getTypeConverter(), resultVals, rewriter, dstTy
    );

    rewriter.replaceOp(op, ret);
    return success();
  }
};

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================
namespace mlir::triton::tle {

void populateExtractTileOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns,
    unsigned benefit) {
  patterns.add<ExtractTileOpConversion>(typeConverter, benefit);
}

} // namespace mlir::triton::tle