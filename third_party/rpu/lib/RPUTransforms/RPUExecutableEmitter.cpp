#include "RPUExecutableEmitter.h"

#include "RPU/IR/Dialect.h"
#include "RPU/IR/ExecutableKind.h"
#include "RPUDSLSourceBuilder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>

namespace mlir {
namespace rpu {
namespace {

static std::optional<int64_t> getTileVectorExtentF16(Type type) {
  auto tile = dyn_cast<exec::TileType>(type);
  if (!tile || tile.getRank() != 1 || !tile.getElementType().isF16())
    return std::nullopt;
  return tile.getExtent();
}

static bool isTileVectorF16(Type type, int64_t extent) {
  std::optional<int64_t> actual = getTileVectorExtentF16(type);
  return actual && *actual == extent;
}

struct MatrixTileShape {
  int64_t rows;
  int64_t cols;
};

static std::optional<MatrixTileShape> getTileMatrixShapeF16(Type type) {
  auto tile = dyn_cast<exec::TileType>(type);
  if (!tile || tile.getRank() != 2 || !tile.getElementType().isF16())
    return std::nullopt;
  ArrayRef<int64_t> shape = tile.getShape();
  return MatrixTileShape{shape[0], shape[1]};
}

static bool isTileMatrixF16(Type type, int64_t rows, int64_t cols) {
  std::optional<MatrixTileShape> actual = getTileMatrixShapeF16(type);
  return actual && actual->rows == rows && actual->cols == cols;
}

static bool isTile16x16F16(Type type) { return isTileMatrixF16(type, 16, 16); }

static std::optional<int64_t>
getConvKxKExecutableKernelSizeFromOpCount(size_t opCount) {
  if (opCount < 6 || (opCount - 3) % 3 != 0)
    return std::nullopt;
  int64_t windows = static_cast<int64_t>((opCount - 3) / 3);
  for (int64_t kernelSize : {3, 5, 7, 9}) {
    if (exec::isSupportedExecutableConvKxKKernelSize(kernelSize) &&
        windows == kernelSize * kernelSize)
      return kernelSize;
  }
  return std::nullopt;
}

enum class ConvKxKPrimitiveBodyLayout {
  None,
  Canonical,
  ExpandedDotSeed,
};

static ConvKxKPrimitiveBodyLayout
getConvKxKPrimitiveBodyLayout(ArrayRef<Operation *> ops) {
  std::optional<int64_t> kernelSize =
      getConvKxKExecutableKernelSizeFromOpCount(ops.size());
  if (!kernelSize)
    return ConvKxKPrimitiveBodyLayout::None;

  int64_t windows = *kernelSize * *kernelSize;
  auto hasStoreAndReturn = [&](size_t storeIndex) {
    return storeIndex + 1 < ops.size() &&
           isa<exec::StoreMatrixOp>(ops[storeIndex]) &&
           isa<exec::ReturnOp>(ops[storeIndex + 1]);
  };

  if (isa<exec::ZeroOp>(ops[0])) {
    bool valid = true;
    for (int64_t i = 0; i < windows; ++i) {
      size_t base = static_cast<size_t>(1 + i * 3);
      valid &= isa<exec::LoadMatrixOp>(ops[base]) &&
               isa<exec::LoadMatrixOp>(ops[base + 1]) &&
               isa<exec::MmaOp>(ops[base + 2]);
    }
    if (valid && hasStoreAndReturn(static_cast<size_t>(1 + windows * 3)))
      return ConvKxKPrimitiveBodyLayout::Canonical;
  }

  if (isa<exec::LoadMatrixOp>(ops[0]) && isa<exec::LoadMatrixOp>(ops[1]) &&
      isa<exec::ZeroOp>(ops[2]) && isa<exec::MmaOp>(ops[3])) {
    bool valid = true;
    for (int64_t i = 1; i < windows; ++i) {
      size_t base = static_cast<size_t>(4 + (i - 1) * 3);
      valid &= isa<exec::LoadMatrixOp>(ops[base]) &&
               isa<exec::LoadMatrixOp>(ops[base + 1]) &&
               isa<exec::MmaOp>(ops[base + 2]);
    }
    if (valid && hasStoreAndReturn(static_cast<size_t>(3 * windows + 1)))
      return ConvKxKPrimitiveBodyLayout::ExpandedDotSeed;
  }

  return ConvKxKPrimitiveBodyLayout::None;
}

static int64_t getOptionalI32AttrOrZero(Operation *op, llvm::StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr)
    return 0;
  return attr.getInt();
}

static std::string matrixArgName(unsigned argIndex) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "arg" << argIndex << "_matrix";
  return os.str();
}

static std::string pointerExpr(unsigned argIndex, int64_t vecOffset) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "arg" << argIndex;
  if (vecOffset != 0)
    os << " + " << vecOffset * 16;
  return os.str();
}

struct ExecutableMemoryBinding {
  Value ptr;
  unsigned argIndex;
  std::string name;
  std::optional<MatrixTileShape> matrixShape;
  bool physicalB = false;
};

struct ExecutableValueNameHint {
  Value value;
  std::string name;
};

class ExecutableOpSequenceRenderState {
public:
  ExecutableOpSequenceRenderState(
      exec::KernelOp kernel, llvm::raw_ostream &os,
      ArrayRef<ExecutableMemoryBinding> bindings, llvm::StringRef diagnostic,
      ArrayRef<ExecutableValueNameHint> valueNameHints)
      : kernel_(kernel), os_(os), bindings_(bindings), diagnostic_(diagnostic),
        valueNameHints_(valueNameHints) {}

  LogicalResult emit(Operation *op) {
    if (auto load = dyn_cast<exec::LoadContigOp>(op))
      return emitLoadContig(load);
    if (auto store = dyn_cast<exec::StoreContigOp>(op))
      return emitStoreContig(store);
    if (auto load = dyn_cast<exec::LoadMatrixOp>(op))
      return emitLoadMatrix(load);
    if (auto store = dyn_cast<exec::StoreMatrixOp>(op))
      return emitStoreMatrix(store);
    if (auto zero = dyn_cast<exec::ZeroOp>(op))
      return emitZero(zero);
    if (auto mma = dyn_cast<exec::MmaOp>(op))
      return emitMma(mma);
    if (auto add = dyn_cast<exec::AddOp>(op))
      return emitAdd(add);
    if (auto mul = dyn_cast<exec::MulOp>(op))
      return emitMul(mul);
    if (auto max = dyn_cast<exec::MaxOp>(op))
      return emitMax(max);
    if (auto reduceMax = dyn_cast<exec::ReduceMaxAllOp>(op))
      return emitReduceMaxAll(reduceMax);
    if (auto sub = dyn_cast<exec::SubScalarOp>(op))
      return emitSubScalar(sub);
    if (auto exp = dyn_cast<exec::ExpOp>(op))
      return emitExp(exp);
    if (auto sqrt = dyn_cast<exec::SqrtOp>(op))
      return emitSqrt(sqrt);
    if (auto full = dyn_cast<exec::FullOp>(op))
      return emitFull(full);
    if (auto reduceSum = dyn_cast<exec::ReduceSumAllOp>(op))
      return emitReduceSumAll(reduceSum);
    if (auto reciprocal = dyn_cast<exec::ReciprocalOp>(op))
      return emitReciprocal(reciprocal);
    if (auto mul = dyn_cast<exec::MulScalarOp>(op))
      return emitMulScalar(mul);
    if (auto relu = dyn_cast<exec::ReluOp>(op))
      return emitRelu(relu);
    if (auto reduceAxis = dyn_cast<exec::ReduceSumAxisOp>(op))
      return emitReduceSumAxis(reduceAxis);
    if (auto bcast = dyn_cast<exec::BroadcastAddOp>(op))
      return emitBroadcastAdd(bcast);
    if (isa<exec::ReturnOp>(op))
      return success();
    op->emitError(diagnostic_);
    return failure();
  }

private:
  const ExecutableMemoryBinding *lookupBinding(Value ptr) const {
    for (const ExecutableMemoryBinding &binding : bindings_) {
      if (binding.ptr == ptr)
        return &binding;
    }
    return nullptr;
  }

  FailureOr<const ExecutableMemoryBinding *> requireBinding(Value ptr) {
    if (const ExecutableMemoryBinding *binding = lookupBinding(ptr))
      return binding;
    kernel_.emitError(diagnostic_);
    return failure();
  }

  FailureOr<std::string> requireValueName(Value value) {
    std::string name = valueNames_.lookup(value);
    if (!name.empty())
      return name;
    kernel_.emitError(diagnostic_);
    return failure();
  }

  std::optional<std::string> nameHint(Value value) const {
    for (const ExecutableValueNameHint &hint : valueNameHints_) {
      if (hint.value == value)
        return hint.name;
    }
    return std::nullopt;
  }

  std::string nextTileName(Value value) {
    if (std::optional<std::string> hint = nameHint(value))
      return *hint;
    std::string name;
    llvm::raw_string_ostream os(name);
    os << "tile" << tileIndex_++;
    return os.str();
  }

  std::string nextScalarName(Value value) {
    if (std::optional<std::string> hint = nameHint(value))
      return *hint;
    std::string name;
    llvm::raw_string_ostream os(name);
    os << "scalar" << scalarIndex_++;
    return os.str();
  }

  static bool sameMaskInfo(const exec::ExecutableVectorMaskInfo &lhs,
                           const exec::ExecutableVectorMaskInfo &rhs) {
    return lhs.masked == rhs.masked && lhs.logicalN == rhs.logicalN &&
           lhs.blockN == rhs.blockN;
  }

  std::optional<exec::ExecutableVectorMaskInfo> valueMask(Value value) const {
    auto found = valueMasks_.find(value);
    if (found == valueMasks_.end())
      return std::nullopt;
    return found->second;
  }

  void bind(Value value, llvm::StringRef name,
            std::optional<exec::ExecutableVectorMaskInfo> mask = std::nullopt) {
    valueNames_[value] = name.str();
    if (mask && mask->masked)
      valueMasks_[value] = *mask;
  }

  FailureOr<std::string>
  requireMaskedTensorName(const ExecutableMemoryBinding &binding,
                          const exec::ExecutableVectorMaskInfo &mask) {
    auto found = maskedTensorNames_.find(binding.ptr);
    if (found != maskedTensorNames_.end())
      return found->second;
    std::string name;
    llvm::raw_string_ostream nameOs(name);
    nameOs << "arg" << binding.argIndex << "_tensor";
    nameOs.flush();
    os_ << "    auto " << name
        << " = rpu::make_tensor<half, 2, rpu::MemScope::Local>(" << binding.name
        << ", rpu::make_shape(1, " << mask.logicalN << "), rpu::make_stride("
        << mask.blockN << ", 1));\n";
    maskedTensorNames_[binding.ptr] = name;
    return name;
  }

  LogicalResult emitLoadContig(exec::LoadContigOp op) {
    FailureOr<const ExecutableMemoryBinding *> binding =
        requireBinding(op.getPtr());
    if (failed(binding))
      return failure();
    std::string resultName = nextTileName(op.getResult());
    FailureOr<exec::ExecutableVectorMaskInfo> maskInfo =
        exec::getExecutableVectorMaskInfo(op.getOperation(), op.getN(),
                                          "generic vector load emission");
    if (failed(maskInfo))
      return failure();
    if (maskInfo->masked) {
      FailureOr<std::string> tensorName =
          requireMaskedTensorName(**binding, *maskInfo);
      if (failed(tensorName))
        return failure();
      os_ << "    auto " << resultName << " = static_cast<rpu::Tile<half, 16, "
          << maskInfo->blockN << ">>(ctx.load<half, 16, " << maskInfo->blockN
          << ">(rpu::local_tile<16, " << maskInfo->blockN << ">(" << *tensorName
          << ", rpu::make_coord(0, 0))));\n";
      bind(op.getResult(), resultName, *maskInfo);
      return success();
    }
    // Opt-in board fallback: rpu_tile.h `load_contig<NVEC>` covers
    // NVEC × 256 elements on real hardware (one V256 vector = 256
    // elements). Default emission preserves
    // the legacy `op.getN()` element-count path. When
    // RPU_BOARD_LOAD_CONTIG_NVEC=1, emit ceil(op.getN()/256) so a single
    // V256 vector covers any plan kernel with up to 256 elements. The two
    // branches keep the legacy literal `<< op.getN()` emission visible to
    // the gate-script source scan.
    if (const char *env = std::getenv("RPU_BOARD_LOAD_CONTIG_NVEC");
        env && env[0] == '1' && env[1] == '\0') {
      int64_t nvec = (op.getN() + 255) / 256;
      if (nvec < 1)
        nvec = 1;
      os_ << "    auto " << resultName << " = ctx.load_contig<" << nvec << ">("
          << (*binding)->name << ");\n";
    } else {
      os_ << "    auto " << resultName << " = ctx.load_contig<" << op.getN()
          << ">(" << (*binding)->name << ");\n";
    }
    bind(op.getResult(), resultName);
    return success();
  }

  LogicalResult emitStoreContig(exec::StoreContigOp op) {
    FailureOr<const ExecutableMemoryBinding *> binding =
        requireBinding(op.getPtr());
    FailureOr<std::string> valueName = requireValueName(op.getValue());
    if (failed(binding) || failed(valueName))
      return failure();
    FailureOr<exec::ExecutableVectorMaskInfo> maskInfo =
        exec::getExecutableVectorMaskInfo(op.getOperation(), op.getN(),
                                          "generic vector store emission");
    if (failed(maskInfo))
      return failure();
    std::optional<exec::ExecutableVectorMaskInfo> storedMask =
        valueMask(op.getValue());
    if (maskInfo->masked) {
      if (!storedMask || !sameMaskInfo(*storedMask, *maskInfo)) {
        kernel_.emitError(diagnostic_);
        return failure();
      }
      FailureOr<std::string> tensorName =
          requireMaskedTensorName(**binding, *maskInfo);
      if (failed(tensorName))
        return failure();
      os_ << "    ctx.store(rpu::local_tile<16, " << maskInfo->blockN << ">("
          << *tensorName << ", rpu::make_coord(0, 0)), " << *valueName << ");";
      return success();
    }
    if (storedMask) {
      kernel_.emitError(diagnostic_);
      return failure();
    }
    if (const char *env = std::getenv("RPU_BOARD_LOAD_CONTIG_NVEC");
        env && env[0] == '1' && env[1] == '\0') {
      int64_t nvec = (op.getN() + 255) / 256;
      if (nvec < 1)
        nvec = 1;
      os_ << "    ctx.store_contig<" << nvec << ">(" << (*binding)->name << ", "
          << *valueName << ");";
    } else {
      os_ << "    ctx.store_contig<" << op.getN() << ">(" << (*binding)->name
          << ", " << *valueName << ");";
    }
    return success();
  }

  LogicalResult emitLoadMatrix(exec::LoadMatrixOp op) {
    FailureOr<const ExecutableMemoryBinding *> binding =
        requireBinding(op.getPtr());
    if (failed(binding))
      return failure();
    if (!(*binding)->matrixShape) {
      kernel_.emitError(diagnostic_);
      return failure();
    }
    std::string resultName = nextTileName(op.getResult());
    int64_t row = getOptionalI32AttrOrZero(op.getOperation(), "row_offset");
    int64_t col = getOptionalI32AttrOrZero(op.getOperation(), "col_offset");
    if ((*binding)->physicalB) {
      std::string storageName = resultName + "_storage";
      os_ << "    auto " << storageName << " = ctx.load<half, " << op.getCols()
          << ", " << op.getRows() << ">(" << (*binding)->name
          << ", rpu::IndexList{" << col << ", " << row << "});\n";
      os_ << "    rpu::LayoutTile<half, " << op.getRows() << ", "
          << op.getCols() << ", rpu::layout::physical_b> " << resultName << "{"
          << storageName << ".offset, " << storageName << ".top};\n";
      bind(op.getResult(), resultName);
      return success();
    }
    os_ << "    auto " << resultName << " = ctx.load<half, " << op.getRows()
        << ", " << op.getCols() << ">(" << (*binding)->name
        << ", rpu::IndexList{" << row << ", " << col << "});\n";
    bind(op.getResult(), resultName);
    return success();
  }

  LogicalResult emitStoreMatrix(exec::StoreMatrixOp op) {
    FailureOr<const ExecutableMemoryBinding *> binding =
        requireBinding(op.getPtr());
    FailureOr<std::string> valueName = requireValueName(op.getValue());
    if (failed(binding) || failed(valueName))
      return failure();
    if (!(*binding)->matrixShape) {
      kernel_.emitError(diagnostic_);
      return failure();
    }
    int64_t row = getOptionalI32AttrOrZero(op.getOperation(), "row_offset");
    int64_t col = getOptionalI32AttrOrZero(op.getOperation(), "col_offset");
    os_ << "    ctx.store<half, " << op.getRows() << ", " << op.getCols()
        << ">(" << (*binding)->name << ", rpu::IndexList{" << row << ", " << col
        << "}, " << *valueName << ");";
    return success();
  }

  LogicalResult emitZero(exec::ZeroOp op) {
    std::string resultName = nextTileName(op.getResult());
    os_ << "    auto " << resultName << " = ctx.zeros<half, " << op.getRows()
        << ", " << op.getCols() << ">();\n";
    bind(op.getResult(), resultName);
    return success();
  }

  LogicalResult emitMma(exec::MmaOp op) {
    FailureOr<std::string> lhsName = requireValueName(op.getLhs());
    FailureOr<std::string> rhsName = requireValueName(op.getRhs());
    FailureOr<std::string> accName = requireValueName(op.getAcc());
    if (failed(lhsName) || failed(rhsName) || failed(accName))
      return failure();
    std::optional<MatrixTileShape> lhsShape =
        getTileMatrixShapeF16(op.getLhs().getType());
    std::optional<MatrixTileShape> rhsShape =
        getTileMatrixShapeF16(op.getRhs().getType());
    if (!lhsShape || !rhsShape) {
      kernel_.emitError(diagnostic_);
      return failure();
    }
    os_ << "    ctx.mma<" << lhsShape->rows << ", " << lhsShape->cols << ", "
        << rhsShape->cols << ">(" << *lhsName << ", " << *rhsName << ", "
        << *accName << ");\n";
    bind(op.getResult(), *accName);
    return success();
  }

  LogicalResult emitBinaryElementwise(Value lhs, Value rhs, Value result,
                                      llvm::StringRef opToken) {
    FailureOr<std::string> lhsName = requireValueName(lhs);
    FailureOr<std::string> rhsName = requireValueName(rhs);
    if (failed(lhsName) || failed(rhsName))
      return failure();
    std::optional<exec::ExecutableVectorMaskInfo> lhsMask = valueMask(lhs);
    std::optional<exec::ExecutableVectorMaskInfo> rhsMask = valueMask(rhs);
    std::optional<exec::ExecutableVectorMaskInfo> resultMask;
    if (lhsMask || rhsMask) {
      if (!lhsMask || !rhsMask || !sameMaskInfo(*lhsMask, *rhsMask)) {
        kernel_.emitError(diagnostic_);
        return failure();
      }
      resultMask = *lhsMask;
    }
    std::string resultName = nextTileName(result);
    os_ << "    auto " << resultName << " = " << *lhsName << " " << opToken
        << " " << *rhsName << ";\n";
    bind(result, resultName, resultMask);
    return success();
  }

  LogicalResult emitAdd(exec::AddOp op) {
    return emitBinaryElementwise(op.getLhs(), op.getRhs(), op.getResult(), "+");
  }

  LogicalResult emitMul(exec::MulOp op) {
    return emitBinaryElementwise(op.getLhs(), op.getRhs(), op.getResult(), "*");
  }

  LogicalResult emitMax(exec::MaxOp op) {
    FailureOr<std::string> lhsName = requireValueName(op.getLhs());
    FailureOr<std::string> rhsName = requireValueName(op.getRhs());
    if (failed(lhsName) || failed(rhsName))
      return failure();
    std::string resultName = nextTileName(op.getResult());
    auto tileType = dyn_cast<exec::TileType>(op.getResult().getType());
    if (tileType && tileType.getRank() == 1) {
      os_ << "    auto " << resultName << " = rpu::maximum(" << *lhsName << ", "
          << *rhsName << ");\n";
    } else {
      os_ << "    auto " << resultName << " = rpu::max_binop(" << *lhsName
          << ", " << *rhsName << ");\n";
    }
    bind(op.getResult(), resultName);
    return success();
  }

  LogicalResult emitReduceMaxAll(exec::ReduceMaxAllOp op) {
    FailureOr<std::string> inputName = requireValueName(op.getInput());
    if (failed(inputName))
      return failure();
    std::string resultName = nextScalarName(op.getResult());
    os_ << "    auto " << resultName << " = ctx.reduce_max_all(" << *inputName
        << ");\n";
    bind(op.getResult(), resultName);
    return success();
  }

  LogicalResult emitSubScalar(exec::SubScalarOp op) {
    FailureOr<std::string> lhsName = requireValueName(op.getLhs());
    FailureOr<std::string> rhsName = requireValueName(op.getRhs());
    if (failed(lhsName) || failed(rhsName))
      return failure();
    std::string resultName = nextTileName(op.getResult());
    os_ << "    auto " << resultName << " = " << *lhsName << " - " << *rhsName
        << ";\n";
    bind(op.getResult(), resultName);
    return success();
  }

  LogicalResult emitExp(exec::ExpOp op) {
    FailureOr<std::string> inputName = requireValueName(op.getInput());
    if (failed(inputName))
      return failure();
    std::string resultName = nextTileName(op.getResult());
    os_ << "    auto " << resultName << " = rpu::exp(" << *inputName << ");\n";
    bind(op.getResult(), resultName);
    return success();
  }

  LogicalResult emitSqrt(exec::SqrtOp op) {
    FailureOr<std::string> inputName = requireValueName(op.getInput());
    if (failed(inputName))
      return failure();
    std::string resultName = nextTileName(op.getResult());
    os_ << "    auto " << resultName << " = rpu::sqrt(" << *inputName << ");\n";
    bind(op.getResult(), resultName);
    return success();
  }

  LogicalResult emitRelu(exec::ReluOp op) {
    FailureOr<std::string> inputName = requireValueName(op.getInput());
    if (failed(inputName))
      return failure();
    std::string resultName = nextTileName(op.getResult());
    // Mirror emitLoadContig/emitFull's env-gated NVEC scaling: on the board
    // path the 1D Tile has NVEC = ceil(n/256); otherwise NVEC = n.
    auto tileType = cast<exec::TileType>(op.getResult().getType());
    int64_t n = tileType.getShape()[0];
    int64_t nvec = n;
    if (const char *env = std::getenv("RPU_BOARD_LOAD_CONTIG_NVEC");
        env && env[0] == '1' && env[1] == '\0') {
      nvec = (n + 255) / 256;
      if (nvec < 1)
        nvec = 1;
    }
    os_ << "    auto " << resultName << " = rpu::maximum(" << *inputName
        << ", ctx.full<half, " << nvec << ">(half(0)));\n";
    bind(op.getResult(), resultName);
    return success();
  }

  LogicalResult emitReduceSumAxis(exec::ReduceSumAxisOp op) {
    FailureOr<std::string> inputName = requireValueName(op.getInput());
    if (failed(inputName))
      return failure();
    std::string resultName = nextTileName(op.getResult());
    auto outTile = dyn_cast<exec::TileType>(op.getResult().getType());
    if (!outTile || outTile.getRank() != 1) {
      kernel_.emitError(diagnostic_);
      return failure();
    }
    int64_t outDim = outTile.getShape()[0];
    // Fallback used only when the per-op generic dispatcher consumes the
    // IR-level rank-1 result (e.g. plan-side mirror tests). The
    // reduce_sum_axis executable sequence body emitter (rpu.kind ==
    // reduce_sum_axis0/1) overrides the entire body with a rank-2
    // broadcast + ctx.store<half, M, N> pattern, so it does not call this
    // per-op emitter — see emitReduceSumAxisOpSequenceBody.
    os_ << "    auto " << resultName << " = ctx.reshape<" << outDim
        << ">(ctx.reduce_sum<" << op.getAxis() << ">(" << *inputName << "));\n";
    bind(op.getResult(), resultName);
    return success();
  }

  LogicalResult emitBroadcastAdd(exec::BroadcastAddOp op) {
    FailureOr<std::string> lhsName = requireValueName(op.getLhs());
    FailureOr<std::string> rhsName = requireValueName(op.getRhs());
    if (failed(lhsName) || failed(rhsName))
      return failure();
    std::string resultName = nextTileName(op.getResult());
    auto lhsType = cast<exec::TileType>(op.getLhs().getType());
    int64_t rows = lhsType.getShape()[0];
    // rhs is a rank-1 Tile<half, NVEC> produced by ctx.load_contig. To engage
    // the per-row rank-2 broadcast operator+ (`Tile<half, M, N> +
    // Tile<half, M, 1>` at rpu_tile.h:5599), reinterpret the same VLM-backed
    // POD as a `Tile<half, rows, 1>` view via the public `from_raw` helper.
    // POD layout (offset+top) matches across the rank-1 NVEC tile and the
    // rank-2 [rows, 1] tile when rows == NVEC*16; load_contig with
    // NVEC == rows/16 is what the body builder emits.
    os_ << "    auto " << resultName << "_col = rpu::Tile<half, " << rows
        << ", 1>::from_raw(" << *rhsName << ".ptr(), " << *rhsName
        << ".top);\n";
    os_ << "    auto " << resultName << " = " << *lhsName << " + " << resultName
        << "_col;\n";
    bind(op.getResult(), resultName);
    return success();
  }

  LogicalResult emitFull(exec::FullOp op) {
    FailureOr<std::string> valueName = requireValueName(op.getValue());
    if (failed(valueName))
      return failure();
    std::string resultName = nextTileName(op.getResult());
    // Match the env-gated NVEC scaling that emitLoadContig/emitStoreContig
    // apply, so the broadcast Tile shape lines up with what store_contig
    // expects (Tile<half, ceil(n/256)> on board, Tile<half, n> otherwise).
    int64_t nvec = op.getN();
    if (const char *env = std::getenv("RPU_BOARD_LOAD_CONTIG_NVEC");
        env && env[0] == '1' && env[1] == '\0') {
      nvec = (op.getN() + 255) / 256;
      if (nvec < 1)
        nvec = 1;
    }
    // The IR-level F16 scalar maps to rpu::Tile<half, 1> at the DSL level
    // (ctx.reduce_*_all returns Tile<half, 1>), so we broadcast it to a
    // Tile<half, NVEC> by adding a zero-tile of the target shape — that
    // selects the rank-1 broadcast operator in rpu_tile.h.
    os_ << "    auto " << resultName << " = " << *valueName
        << " + ctx.full<half, " << nvec << ">(half(0));\n";
    bind(op.getResult(), resultName);
    return success();
  }

  LogicalResult emitReduceSumAll(exec::ReduceSumAllOp op) {
    FailureOr<std::string> inputName = requireValueName(op.getInput());
    if (failed(inputName))
      return failure();
    std::string resultName = nextScalarName(op.getResult());
    os_ << "    auto " << resultName << " = ctx.reduce_sum_all(" << *inputName
        << ");\n";
    bind(op.getResult(), resultName);
    return success();
  }

  LogicalResult emitReciprocal(exec::ReciprocalOp op) {
    FailureOr<std::string> inputName = requireValueName(op.getInput());
    if (failed(inputName))
      return failure();
    std::string resultName = nextScalarName(op.getResult());
    os_ << "    auto " << resultName << " = rpu::reciprocal(" << *inputName
        << ");\n";
    bind(op.getResult(), resultName);
    return success();
  }

  LogicalResult emitMulScalar(exec::MulScalarOp op) {
    FailureOr<std::string> lhsName = requireValueName(op.getLhs());
    FailureOr<std::string> rhsName = requireValueName(op.getRhs());
    if (failed(lhsName) || failed(rhsName))
      return failure();
    std::string resultName = nextTileName(op.getResult());
    os_ << "    auto " << resultName << " = " << *lhsName << " * " << *rhsName
        << ";\n";
    bind(op.getResult(), resultName);
    return success();
  }

  exec::KernelOp kernel_;
  llvm::raw_ostream &os_;
  ArrayRef<ExecutableMemoryBinding> bindings_;
  llvm::StringRef diagnostic_;
  ArrayRef<ExecutableValueNameHint> valueNameHints_;
  llvm::DenseMap<Value, std::string> valueNames_;
  llvm::DenseMap<Value, exec::ExecutableVectorMaskInfo> valueMasks_;
  llvm::DenseMap<Value, std::string> maskedTensorNames_;
  int64_t tileIndex_ = 0;
  int64_t scalarIndex_ = 0;
};

static FailureOr<std::string> emitGenericExecutableOpSequenceBody(
    exec::KernelOp kernel, ArrayRef<Operation *> ops,
    ArrayRef<ExecutableMemoryBinding> bindings, llvm::StringRef diagnostic,
    llvm::StringRef prelude = "",
    ArrayRef<ExecutableValueNameHint> valueNameHints = {}) {
  if (failed(exec::verifyGenericRenderableExecutableOpSequence(kernel, ops,
                                                               diagnostic)))
    return failure();
  std::string body;
  llvm::raw_string_ostream os(body);
  os << prelude;
  ExecutableOpSequenceRenderState state(kernel, os, bindings, diagnostic,
                                        valueNameHints);
  for (Operation *op : ops) {
    if (failed(state.emit(op)))
      return failure();
  }
  return os.str();
}

static FailureOr<std::string> buildKernelArgumentPrototype(
    ArrayRef<exec::ExecutableKernelArgumentInfo> args) {
  std::string prototype;
  llvm::raw_string_ostream os(prototype);
  for (const exec::ExecutableKernelArgumentInfo &arg : args) {
    unsigned i = arg.index;
    if (i != 0)
      os << ", ";
    os << "half* " << arg.name;
  }
  return os.str();
}

static FailureOr<std::string>
buildKernelArgumentPrototype(exec::KernelOp kernel,
                             llvm::StringRef diagnostic) {
  FailureOr<llvm::SmallVector<exec::ExecutableKernelArgumentInfo, 4>> args =
      exec::getExecutableKernelArguments(kernel, diagnostic);
  if (failed(args))
    return failure();

  return buildKernelArgumentPrototype(*args);
}

static FailureOr<std::string> emitGenericSimpleAddContigOpSequenceBody(
    exec::KernelOp kernel, ArrayRef<Operation *> ops,
    exec::ExecutableAddOpSequenceInfo info, unsigned outArg, unsigned lhsArg,
    unsigned rhsArg) {
  llvm::SmallVector<ExecutableMemoryBinding, 3> bindings;
  bindings.push_back(ExecutableMemoryBinding{
      info.lhsLoad.getPtr(), lhsArg, pointerExpr(lhsArg, 0), std::nullopt});
  bindings.push_back(ExecutableMemoryBinding{
      info.rhsLoad.getPtr(), rhsArg, pointerExpr(rhsArg, 0), std::nullopt});
  bindings.push_back(ExecutableMemoryBinding{
      info.store.getPtr(), outArg, pointerExpr(outArg, 0), std::nullopt});
  return emitGenericExecutableOpSequenceBody(
      kernel, ops, bindings,
      "generic compact Add emission requires bound executable operands");
}

static FailureOr<std::string> emitAddOpSequenceBody(exec::KernelOp kernel,
                                                    ArrayRef<Operation *> ops) {
  FailureOr<exec::ExecutableAddOpSequenceInfo> maybeInfo =
      exec::getExecutableAddOpSequenceInfo(
          kernel, ops,
          "executable add sequence emission requires verified IR info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableAddOpSequenceInfo info = *maybeInfo;

  int64_t n = info.n;
  FailureOr<llvm::SmallVector<unsigned, 4>> args =
      exec::getExecutableKernelArgumentIndices(
          kernel,
          {info.store.getPtr(), info.lhsLoad.getPtr(), info.rhsLoad.getPtr()},
          "executable add op-sequence emission requires block argument "
          "pointers");
  if (failed(args))
    return failure();

  unsigned outArg = (*args)[0];
  unsigned lhsArg = (*args)[1];
  unsigned rhsArg = (*args)[2];

  bool masked = info.masked;
  int64_t logicalN = info.logicalN;

  if (!masked && n <= 128)
    return emitGenericSimpleAddContigOpSequenceBody(kernel, ops, info, outArg,
                                                    lhsArg, rhsArg);

  std::string body;
  llvm::raw_string_ostream os(body);
  if (masked) {
    os << llvm::formatv("    auto arg{0}_tensor = rpu::make_tensor<half, 2, "
                        "rpu::MemScope::Local>(arg{0}, rpu::make_shape(1, "
                        "{1}), rpu::make_stride({2}, 1));\n"
                        "    auto arg{3}_tensor = rpu::make_tensor<half, 2, "
                        "rpu::MemScope::Local>(arg{3}, rpu::make_shape(1, "
                        "{1}), rpu::make_stride({2}, 1));\n"
                        "    auto arg{4}_tensor = rpu::make_tensor<half, 2, "
                        "rpu::MemScope::Local>(arg{4}, rpu::make_shape(1, "
                        "{1}), rpu::make_stride({2}, 1));\n"
                        "    auto tile0 = static_cast<rpu::Tile<half, 16, "
                        "{2}>>(ctx.load<half, 16, {2}>(rpu::local_tile<16, "
                        "{2}>(arg{0}_tensor, rpu::make_coord(0, 0))));\n"
                        "    auto tile1 = static_cast<rpu::Tile<half, 16, "
                        "{2}>>(ctx.load<half, 16, {2}>(rpu::local_tile<16, "
                        "{2}>(arg{3}_tensor, rpu::make_coord(0, 0))));\n"
                        "    auto tile2 = tile0 + tile1;\n"
                        "    ctx.store(rpu::local_tile<16, {2}>(arg{4}_tensor, "
                        "rpu::make_coord(0, 0)), tile2);",
                        lhsArg, logicalN, n, rhsArg, outArg);
    return os.str();
  }

  // Opt-in board ABI: when RPU_BOARD_LOAD_CONTIG_NVEC=1, route the
  // contig load/store path up to one full V256 vector (256 fp16) per NVEC,
  // and convert element count to NVEC via ceil-divide-by-256. Default keeps
  // the legacy n ≤ 128 threshold and element-count NVEC.
  bool useBoardNvec = false;
  if (const char *env = std::getenv("RPU_BOARD_LOAD_CONTIG_NVEC");
      env && env[0] == '1' && env[1] == '\0') {
    useBoardNvec = true;
  }
  int64_t contigThreshold = useBoardNvec ? 256 : 128;
  if (n <= contigThreshold) {
    if (useBoardNvec) {
      n = (n + 255) / 256;
      if (n < 1)
        n = 1;
    }
    os << "    auto tile0 = ctx.load_contig<" << n << ">("
       << pointerExpr(lhsArg, 0) << ");\n";
    os << "    auto tile1 = ctx.load_contig<" << n << ">("
       << pointerExpr(rhsArg, 0) << ");\n";
    os << "    auto tile2 = tile0 + tile1;\n";
    os << "    ctx.store_contig<" << n << ">(" << pointerExpr(outArg, 0)
       << ", tile2);";
    return os.str();
  }

  int64_t totalRows = n * 16;
  os << llvm::formatv("    auto arg{0}_tensor = rpu::make_tensor<half, 2, "
                      "rpu::MemScope::Local>(arg{0}, rpu::make_shape({1}, 16), "
                      "rpu::make_stride(16, 1));\n"
                      "    auto arg{2}_tensor = rpu::make_tensor<half, 2, "
                      "rpu::MemScope::Local>(arg{2}, rpu::make_shape({1}, 16), "
                      "rpu::make_stride(16, 1));\n"
                      "    auto arg{3}_tensor = rpu::make_tensor<half, 2, "
                      "rpu::MemScope::Local>(arg{3}, rpu::make_shape({1}, 16), "
                      "rpu::make_stride(16, 1));\n",
                      lhsArg, totalRows, rhsArg, outArg);
  constexpr int64_t kMaxRowsPerFrame = 1024;
  for (int64_t rowOffset = 0; rowOffset < totalRows;
       rowOffset += kMaxRowsPerFrame) {
    int64_t chunkRows =
        std::min<int64_t>(kMaxRowsPerFrame, totalRows - rowOffset);
    os << "    {\n";
    os << "        auto frame = ctx.tile_frame();\n";
    os << llvm::formatv(
        "        auto tile0 = static_cast<rpu::Tile<half, {0}, "
        "16>>(ctx.load<half, {0}, 16>(rpu::local_tile<{0}, 16>(arg{1}_tensor, "
        "rpu::make_coord({2}, 0))));\n"
        "        auto tile1 = static_cast<rpu::Tile<half, {0}, "
        "16>>(ctx.load<half, {0}, 16>(rpu::local_tile<{0}, 16>(arg{3}_tensor, "
        "rpu::make_coord({2}, 0))));\n"
        "        auto tile2 = tile0 + tile1;\n"
        "        ctx.store(rpu::local_tile<{0}, 16>(arg{4}_tensor, "
        "rpu::make_coord({2}, 0)), tile2);\n",
        chunkRows, lhsArg, rowOffset, rhsArg, outArg);
    os << "    }\n";
  }
  return os.str();
}

static FailureOr<RPUExecutableEmissionResult> emitAddFromExecutableContract(
    const exec::ExecutableKernelContractInfo &contract) {
  exec::KernelOp kernel = contract.kernel;
  ArrayRef<Operation *> ops = contract.bodyOps;

  FailureOr<exec::ExecutableAddOpSequenceInfo> maybeInfo =
      exec::getExecutableAddOpSequenceInfo(
          kernel, ops,
          "executable add contract-view emission requires verified IR info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableAddOpSequenceInfo info = *maybeInfo;

  int64_t n = info.n;
  if (n <= 0 || info.rhsLoad.getN() != n || info.store.getN() != n ||
      !isTileVectorF16(info.lhsLoad.getResult().getType(), n) ||
      !isTileVectorF16(info.rhsLoad.getResult().getType(), n) ||
      !isTileVectorF16(info.add.getResult().getType(), n) ||
      !isTileVectorF16(info.store.getValue().getType(), n)) {
    kernel.emitError("executable add contract-view emission requires positive "
                     "!rpu.tile<Nxf16>");
    return failure();
  }

  FailureOr<std::string> prototype =
      buildKernelArgumentPrototype(contract.arguments);
  if (failed(prototype))
    return failure();

  FailureOr<std::string> body = emitAddOpSequenceBody(kernel, ops);
  if (failed(body))
    return failure();

  std::string source =
      buildRPUDSLProgram(contract.kernelName, *prototype, *body);
  return RPUExecutableEmissionResult{contract.kernelName, "rpu_executable",
                                     source};
}

static FailureOr<std::string>
emitGemmOpSequenceBody(exec::KernelOp kernel, ArrayRef<Operation *> ops) {
  FailureOr<exec::ExecutableGemmOpSequenceInfo> maybeInfo =
      exec::getExecutableGemmOpSequenceInfo(
          kernel, ops,
          "executable gemm sequence emission requires verified IR info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableGemmOpSequenceInfo info = *maybeInfo;

  int64_t m = info.m;
  int64_t n = info.n;
  int64_t k = info.k;

  FailureOr<llvm::SmallVector<unsigned, 4>> args =
      exec::getExecutableKernelArgumentIndices(
          kernel,
          {info.store.getPtr(), info.lhsLoad.getPtr(), info.rhsLoad.getPtr()},
          "executable gemm op-sequence emission requires block argument "
          "pointers");
  if (failed(args))
    return failure();

  unsigned outArg = (*args)[0];
  unsigned lhsArg = (*args)[1];
  unsigned rhsArg = (*args)[2];

  std::string prelude;
  llvm::raw_string_ostream preludeOs(prelude);
  preludeOs << "    rpu::Array<half, 2> " << matrixArgName(lhsArg) << "{arg"
            << lhsArg << ", " << m << ", " << k << "};\n";
  preludeOs << "    rpu::Array<half, 2> " << matrixArgName(rhsArg) << "{arg"
            << rhsArg << ", " << n << ", " << k << "};\n";
  preludeOs << "    rpu::Array<half, 2> " << matrixArgName(outArg) << "{arg"
            << outArg << ", " << m << ", " << n << "};\n";
  preludeOs.flush();

  llvm::SmallVector<ExecutableMemoryBinding, 3> bindings;
  bindings.push_back(ExecutableMemoryBinding{info.lhsLoad.getPtr(), lhsArg,
                                             matrixArgName(lhsArg),
                                             MatrixTileShape{m, k}});
  bindings.push_back(ExecutableMemoryBinding{info.rhsLoad.getPtr(), rhsArg,
                                             matrixArgName(rhsArg),
                                             MatrixTileShape{n, k}, true});
  bindings.push_back(ExecutableMemoryBinding{info.store.getPtr(), outArg,
                                             matrixArgName(outArg),
                                             MatrixTileShape{m, n}});

  llvm::SmallVector<Operation *, 8> orderedOps;
  for (Operation *op : ops) {
    if (isa<exec::ZeroOp>(op))
      orderedOps.push_back(op);
  }
  for (Operation *op : ops) {
    if (!isa<exec::ZeroOp>(op))
      orderedOps.push_back(op);
  }

  llvm::SmallVector<ExecutableValueNameHint, 3> valueNameHints;
  valueNameHints.push_back(
      ExecutableValueNameHint{info.lhsLoad.getResult(), "tile0"});
  valueNameHints.push_back(
      ExecutableValueNameHint{info.rhsLoad.getResult(), "tile1"});
  valueNameHints.push_back(
      ExecutableValueNameHint{info.zero.getResult(), "tile2"});

  return emitGenericExecutableOpSequenceBody(
      kernel, orderedOps, bindings,
      "generic compact GEMM emission requires bound executable operands",
      prelude, valueNameHints);
}

static FailureOr<RPUExecutableEmissionResult> emitGemmFromExecutableContract(
    const exec::ExecutableKernelContractInfo &contract) {
  exec::KernelOp kernel = contract.kernel;
  ArrayRef<Operation *> ops = contract.bodyOps;

  FailureOr<exec::ExecutableGemmOpSequenceInfo> maybeInfo =
      exec::getExecutableGemmOpSequenceInfo(
          kernel, ops,
          "executable gemm contract-view emission requires verified IR info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableGemmOpSequenceInfo info = *maybeInfo;

  int64_t m = info.m;
  int64_t n = info.n;
  int64_t k = info.k;
  if (info.rhsLoad.getRows() != k || info.zero.getRows() != m ||
      info.zero.getCols() != n || info.store.getRows() != m ||
      info.store.getCols() != n ||
      !isTileMatrixF16(info.lhsLoad.getResult().getType(), m, k) ||
      !isTileMatrixF16(info.rhsLoad.getResult().getType(), k, n) ||
      !isTileMatrixF16(info.zero.getResult().getType(), m, n) ||
      !isTileMatrixF16(info.mma.getResult().getType(), m, n) ||
      !isTileMatrixF16(info.store.getValue().getType(), m, n)) {
    kernel.emitError("executable gemm contract-view emission requires "
                     "canonical matrix tile shapes");
    return failure();
  }

  FailureOr<std::string> prototype =
      buildKernelArgumentPrototype(contract.arguments);
  if (failed(prototype))
    return failure();

  FailureOr<std::string> body = emitGemmOpSequenceBody(kernel, ops);
  if (failed(body))
    return failure();
  std::string source =
      buildRPUDSLProgram(contract.kernelName, *prototype, *body);
  return RPUExecutableEmissionResult{contract.kernelName, "rpu_executable",
                                     source};
}

static FailureOr<std::string>
emitSoftmaxOpSequenceBody(exec::KernelOp kernel, ArrayRef<Operation *> ops) {
  FailureOr<exec::ExecutableSoftmaxOpSequenceInfo> maybeInfo =
      exec::getExecutableSoftmaxOpSequenceInfo(
          kernel, ops,
          "executable softmax sequence emission requires verified IR info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableSoftmaxOpSequenceInfo info = *maybeInfo;

  FailureOr<llvm::SmallVector<unsigned, 4>> args =
      exec::getExecutableKernelArgumentIndices(
          kernel, {info.load.getPtr(), info.store.getPtr()},
          "executable softmax op-sequence emission requires block argument "
          "pointers");
  if (failed(args))
    return failure();

  unsigned inputArg = (*args)[0];
  unsigned outArg = (*args)[1];

  llvm::SmallVector<ExecutableMemoryBinding, 2> bindings;
  bindings.push_back(ExecutableMemoryBinding{
      info.load.getPtr(), inputArg, pointerExpr(inputArg, 0), std::nullopt});
  bindings.push_back(ExecutableMemoryBinding{
      info.store.getPtr(), outArg, pointerExpr(outArg, 0), std::nullopt});
  return emitGenericExecutableOpSequenceBody(
      kernel, ops, bindings,
      "generic compact Softmax emission requires bound executable operands");
}

static FailureOr<RPUExecutableEmissionResult> emitSoftmaxFromExecutableContract(
    const exec::ExecutableKernelContractInfo &contract) {
  exec::KernelOp kernel = contract.kernel;
  ArrayRef<Operation *> ops = contract.bodyOps;

  FailureOr<exec::ExecutableSoftmaxOpSequenceInfo> maybeInfo =
      exec::getExecutableSoftmaxOpSequenceInfo(
          kernel, ops,
          "executable softmax contract-view emission requires verified IR "
          "info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableSoftmaxOpSequenceInfo info = *maybeInfo;

  int64_t nvec = info.nvec;
  if (nvec <= 0 || info.store.getN() != nvec ||
      !isTileVectorF16(info.load.getResult().getType(), nvec) ||
      !isTileVectorF16(info.reduceMax.getInput().getType(), nvec) ||
      !isTileVectorF16(info.sub.getResult().getType(), nvec) ||
      !isTileVectorF16(info.exp.getResult().getType(), nvec) ||
      !isTileVectorF16(info.reduceSum.getInput().getType(), nvec) ||
      !isTileVectorF16(info.mul.getResult().getType(), nvec) ||
      !isTileVectorF16(info.store.getValue().getType(), nvec)) {
    kernel.emitError("executable softmax contract-view emission requires "
                     "positive !rpu.tile<Nxf16>");
    return failure();
  }

  FailureOr<std::string> prototype =
      buildKernelArgumentPrototype(contract.arguments);
  if (failed(prototype))
    return failure();

  FailureOr<std::string> body = emitSoftmaxOpSequenceBody(kernel, ops);
  if (failed(body))
    return failure();

  std::string source =
      buildRPUDSLProgram(contract.kernelName, *prototype, *body);
  return RPUExecutableEmissionResult{contract.kernelName, "rpu_executable",
                                     source};
}

static FailureOr<std::string>
emitConvKxKOpSequenceBody(exec::KernelOp kernel, ArrayRef<Operation *> ops) {
  std::optional<int64_t> kernelSize =
      getConvKxKExecutableKernelSizeFromOpCount(ops.size());
  ConvKxKPrimitiveBodyLayout layout = getConvKxKPrimitiveBodyLayout(ops);
  if (!kernelSize || layout == ConvKxKPrimitiveBodyLayout::None) {
    kernel.emitError(
        "executable .rc emission requires supported convkxk op count");
    return failure();
  }

  int64_t windows = *kernelSize * *kernelSize;
  size_t zeroIndex = layout == ConvKxKPrimitiveBodyLayout::Canonical ? 0 : 2;
  size_t firstXIndex = layout == ConvKxKPrimitiveBodyLayout::Canonical ? 1 : 0;
  size_t firstWIndex = layout == ConvKxKPrimitiveBodyLayout::Canonical ? 2 : 1;
  size_t storeIndex = layout == ConvKxKPrimitiveBodyLayout::Canonical
                          ? static_cast<size_t>(1 + windows * 3)
                          : static_cast<size_t>(3 * windows + 1);
  auto zero = cast<exec::ZeroOp>(ops[zeroIndex]);
  auto firstX = cast<exec::LoadMatrixOp>(ops[firstXIndex]);
  auto firstW = cast<exec::LoadMatrixOp>(ops[firstWIndex]);
  auto store = cast<exec::StoreMatrixOp>(ops[storeIndex]);

  FailureOr<llvm::SmallVector<unsigned, 4>> args =
      exec::getExecutableKernelArgumentIndices(
          kernel, {firstX.getPtr(), firstW.getPtr(), store.getPtr()},
          "executable convkxk op-sequence emission requires block argument "
          "pointers");
  if (failed(args))
    return failure();

  unsigned inputArg = (*args)[0];
  unsigned weightArg = (*args)[1];
  unsigned outArg = (*args)[2];
  int64_t m = zero.getRows();
  int64_t n = zero.getCols();
  int64_t k = firstX.getCols();
  int64_t inputRows = 0;
  int64_t weightRows = 0;

  for (int64_t window = 0; window < windows; ++window) {
    size_t loadBase =
        layout == ConvKxKPrimitiveBodyLayout::Canonical
            ? static_cast<size_t>(1 + window * 3)
            : (window == 0 ? 0 : static_cast<size_t>(4 + (window - 1) * 3));
    auto loadX = cast<exec::LoadMatrixOp>(ops[loadBase]);
    auto loadW = cast<exec::LoadMatrixOp>(ops[loadBase + 1]);
    int64_t xRow = getOptionalI32AttrOrZero(loadX.getOperation(), "row_offset");
    int64_t wRow = getOptionalI32AttrOrZero(loadW.getOperation(), "row_offset");
    inputRows = std::max<int64_t>(inputRows, xRow + loadX.getRows());
    weightRows = std::max<int64_t>(weightRows, wRow + loadW.getRows());
  }

  std::string prelude;
  llvm::raw_string_ostream os(prelude);
  os << "    rpu::Array<half, 2> x_arr{arg" << inputArg << ", " << inputRows
     << ", " << k << "};\n";
  os << "    rpu::Array<half, 2> w_arr{arg" << weightArg << ", " << weightRows
     << ", " << n << "};\n";
  os << "    rpu::Array<half, 2> out_arr{arg" << outArg << ", "
     << store.getRows() << ", " << store.getCols() << "};\n";
  os.flush();

  llvm::SmallVector<ExecutableMemoryBinding, 3> bindings;
  bindings.push_back(ExecutableMemoryBinding{firstX.getPtr(), inputArg, "x_arr",
                                             MatrixTileShape{inputRows, k}});
  bindings.push_back(ExecutableMemoryBinding{
      firstW.getPtr(), weightArg, "w_arr", MatrixTileShape{weightRows, n}});
  bindings.push_back(ExecutableMemoryBinding{
      store.getPtr(), outArg, "out_arr",
      MatrixTileShape{store.getRows(), store.getCols()}});

  llvm::SmallVector<ExecutableValueNameHint, 1> valueNameHints;
  valueNameHints.push_back(ExecutableValueNameHint{zero.getResult(), "acc"});

  return emitGenericExecutableOpSequenceBody(
      kernel, ops, bindings,
      "generic ConvKxK emission requires bound executable operands", prelude,
      valueNameHints);
}

static FailureOr<RPUExecutableEmissionResult> emitConvKxKFromExecutableContract(
    const exec::ExecutableKernelContractInfo &contract) {
  exec::KernelOp kernel = contract.kernel;
  ArrayRef<Operation *> ops = contract.bodyOps;

  std::optional<int64_t> kernelSize =
      getConvKxKExecutableKernelSizeFromOpCount(ops.size());
  if (!kernelSize ||
      getConvKxKPrimitiveBodyLayout(ops) == ConvKxKPrimitiveBodyLayout::None) {
    kernel.emitError("executable convkxk contract-view emission requires "
                     "supported convkxk op count");
    return failure();
  }

  FailureOr<std::string> prototype =
      buildKernelArgumentPrototype(contract.arguments);
  if (failed(prototype))
    return failure();

  FailureOr<std::string> body = emitConvKxKOpSequenceBody(kernel, ops);
  if (failed(body))
    return failure();

  std::string source =
      buildRPUDSLProgram(contract.kernelName, *prototype, *body);
  return RPUExecutableEmissionResult{contract.kernelName, "rpu_executable",
                                     source};
}

static void updateMatrixExtent(exec::LoadMatrixOp load, int64_t &rows,
                               int64_t &cols) {
  int64_t row = getOptionalI32AttrOrZero(load.getOperation(), "row_offset");
  int64_t col = getOptionalI32AttrOrZero(load.getOperation(), "col_offset");
  rows = std::max<int64_t>(rows, row + load.getRows());
  cols = std::max<int64_t>(cols, col + load.getCols());
}

static FailureOr<std::string>
emitResNetBlockOpSequenceBody(exec::KernelOp kernel,
                              ArrayRef<Operation *> ops) {
  bool hidden32 = ops.size() == 21;
  auto x = cast<exec::LoadMatrixOp>(ops[0]);
  auto firstW1 = cast<exec::LoadMatrixOp>(ops[1]);
  auto firstW2 = cast<exec::LoadMatrixOp>(hidden32 ? ops[11] : ops[6]);
  auto store = cast<exec::StoreMatrixOp>(hidden32 ? ops[19] : ops[12]);

  FailureOr<llvm::SmallVector<unsigned, 4>> args =
      exec::getExecutableKernelArgumentIndices(
          kernel,
          {store.getPtr(), x.getPtr(), firstW1.getPtr(), firstW2.getPtr()},
          "executable residual op-sequence emission requires block argument "
          "pointers");
  if (failed(args))
    return failure();

  unsigned outArg = (*args)[0];
  unsigned xArg = (*args)[1];
  unsigned w1Arg = (*args)[2];
  unsigned w2Arg = (*args)[3];

  int64_t xRows = 0;
  int64_t xCols = 0;
  int64_t w1Rows = 0;
  int64_t w1Cols = 0;
  int64_t w2Rows = 0;
  int64_t w2Cols = 0;
  updateMatrixExtent(x, xRows, xCols);
  if (hidden32) {
    updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[1]), w1Rows, w1Cols);
    updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[6]), w1Rows, w1Cols);
    updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[11]), w2Rows, w2Cols);
    updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[14]), w2Rows, w2Cols);
  } else {
    updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[1]), w1Rows, w1Cols);
    updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[6]), w2Rows, w2Cols);
  }

  int64_t outRows =
      getOptionalI32AttrOrZero(store.getOperation(), "row_offset") +
      store.getRows();
  int64_t outCols =
      getOptionalI32AttrOrZero(store.getOperation(), "col_offset") +
      store.getCols();

  std::string prelude;
  llvm::raw_string_ostream os(prelude);
  os << "    rpu::Array<half, 2> x_arr{arg" << xArg << ", " << xRows << ", "
     << xCols << "};\n";
  os << "    rpu::Array<half, 2> w1_arr{arg" << w1Arg << ", " << w1Rows << ", "
     << w1Cols << "};\n";
  os << "    rpu::Array<half, 2> w2_arr{arg" << w2Arg << ", " << w2Rows << ", "
     << w2Cols << "};\n";
  os << "    rpu::Array<half, 2> out_arr{arg" << outArg << ", " << outRows
     << ", " << outCols << "};\n";
  os.flush();

  llvm::SmallVector<ExecutableMemoryBinding, 4> bindings;
  bindings.push_back(ExecutableMemoryBinding{x.getPtr(), xArg, "x_arr",
                                             MatrixTileShape{xRows, xCols}});
  bindings.push_back(ExecutableMemoryBinding{firstW1.getPtr(), w1Arg, "w1_arr",
                                             MatrixTileShape{w1Rows, w1Cols}});
  bindings.push_back(ExecutableMemoryBinding{firstW2.getPtr(), w2Arg, "w2_arr",
                                             MatrixTileShape{w2Rows, w2Cols}});
  bindings.push_back(ExecutableMemoryBinding{
      store.getPtr(), outArg, "out_arr", MatrixTileShape{outRows, outCols}});

  llvm::SmallVector<ExecutableValueNameHint, 20> valueNameHints;
  auto addHint = [&](Value value, llvm::StringRef name) {
    valueNameHints.push_back(ExecutableValueNameHint{value, name.str()});
  };
  addHint(x.getResult(), "x");

  if (hidden32) {
    addHint(cast<exec::LoadMatrixOp>(ops[1]).getResult(), "w1_0");
    addHint(cast<exec::ZeroOp>(ops[2]).getResult(), "conv1_0");
    addHint(cast<exec::ZeroOp>(ops[4]).getResult(), "zero1_0");
    addHint(cast<exec::MaxOp>(ops[5]).getResult(), "relu1_0");
    addHint(cast<exec::LoadMatrixOp>(ops[6]).getResult(), "w1_1");
    addHint(cast<exec::ZeroOp>(ops[7]).getResult(), "conv1_1");
    addHint(cast<exec::ZeroOp>(ops[9]).getResult(), "zero1_1");
    addHint(cast<exec::MaxOp>(ops[10]).getResult(), "relu1_1");
    addHint(cast<exec::LoadMatrixOp>(ops[11]).getResult(), "w2_0");
    addHint(cast<exec::ZeroOp>(ops[12]).getResult(), "conv2");
    addHint(cast<exec::LoadMatrixOp>(ops[14]).getResult(), "w2_1");
    addHint(cast<exec::AddOp>(ops[16]).getResult(), "residual");
    addHint(cast<exec::ZeroOp>(ops[17]).getResult(), "zero2");
    addHint(cast<exec::MaxOp>(ops[18]).getResult(), "out_tile");
    return emitGenericExecutableOpSequenceBody(
        kernel, ops, bindings,
        "generic Residual emission requires bound executable operands", prelude,
        valueNameHints);
  }

  addHint(cast<exec::LoadMatrixOp>(ops[1]).getResult(), "w1");
  addHint(cast<exec::ZeroOp>(ops[2]).getResult(), "conv1");
  addHint(cast<exec::ZeroOp>(ops[4]).getResult(), "zero1");
  addHint(cast<exec::MaxOp>(ops[5]).getResult(), "relu1");
  addHint(cast<exec::LoadMatrixOp>(ops[6]).getResult(), "w2");
  addHint(cast<exec::ZeroOp>(ops[7]).getResult(), "conv2");
  addHint(cast<exec::AddOp>(ops[9]).getResult(), "residual");
  addHint(cast<exec::ZeroOp>(ops[10]).getResult(), "zero2");
  addHint(cast<exec::MaxOp>(ops[11]).getResult(), "out_tile");
  return emitGenericExecutableOpSequenceBody(
      kernel, ops, bindings,
      "generic Residual emission requires bound executable operands", prelude,
      valueNameHints);
}

static FailureOr<RPUExecutableEmissionResult>
emitResNetBlock16FromExecutableContract(
    const exec::ExecutableKernelContractInfo &contract) {
  exec::KernelOp kernel = contract.kernel;
  ArrayRef<Operation *> ops = contract.bodyOps;

  FailureOr<std::string> prototype =
      buildKernelArgumentPrototype(contract.arguments);
  if (failed(prototype))
    return failure();

  FailureOr<std::string> body = emitResNetBlockOpSequenceBody(kernel, ops);
  if (failed(body))
    return failure();
  std::string source =
      buildRPUDSLProgram(contract.kernelName, *prototype, *body);
  return RPUExecutableEmissionResult{contract.kernelName, "rpu_executable",
                                     source};
}

static FailureOr<std::string>
emitResNet50BottleneckOpSequenceBody(exec::KernelOp kernel,
                                     ArrayRef<Operation *> ops) {
  bool bottleneck32 = ops.size() == 172;
  int64_t tail = bottleneck32 ? 5 + 9 * 17 : 3 + 9 * 7;
  auto xSkip = cast<exec::LoadMatrixOp>(ops[0]);
  auto firstW1 = cast<exec::LoadMatrixOp>(ops[1]);
  auto firstW2 = cast<exec::LoadMatrixOp>(bottleneck32 ? ops[14] : ops[8]);
  auto firstW3 =
      cast<exec::LoadMatrixOp>(ops[bottleneck32 ? tail + 4 : tail + 2]);
  auto store =
      cast<exec::StoreMatrixOp>(ops[bottleneck32 ? tail + 12 : tail + 8]);

  FailureOr<llvm::SmallVector<unsigned, 4>> args =
      exec::getExecutableKernelArgumentIndices(
          kernel,
          {store.getPtr(), xSkip.getPtr(), firstW1.getPtr(), firstW2.getPtr(),
           firstW3.getPtr()},
          "executable bottleneck op-sequence emission requires block argument "
          "pointers");
  if (failed(args))
    return failure();

  unsigned outArg = (*args)[0];
  unsigned xArg = (*args)[1];
  unsigned w1Arg = (*args)[2];
  unsigned w2Arg = (*args)[3];
  unsigned w3Arg = (*args)[4];

  int64_t xRows = 0;
  int64_t xCols = 0;
  int64_t w1Rows = 0;
  int64_t w1Cols = 0;
  int64_t w2Rows = 0;
  int64_t w2Cols = 0;
  int64_t w3Rows = 0;
  int64_t w3Cols = 0;
  updateMatrixExtent(xSkip, xRows, xCols);
  if (bottleneck32) {
    updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[1]), w1Rows, w1Cols);
    updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[2]), w1Rows, w1Cols);
    for (int64_t window = 0; window < 9; ++window) {
      int64_t base = 5 + window * 17;
      updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[base]), xRows, xCols);
      updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[base + 9]), w2Rows,
                         w2Cols);
      updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[base + 11]), w2Rows,
                         w2Cols);
      updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[base + 13]), w2Rows,
                         w2Cols);
      updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[base + 15]), w2Rows,
                         w2Cols);
    }
    updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[tail + 4]), w3Rows, w3Cols);
    updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[tail + 7]), w3Rows, w3Cols);
  } else {
    updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[1]), w1Rows, w1Cols);
    for (int64_t window = 0; window < 9; ++window) {
      int64_t base = 3 + window * 7;
      updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[base]), xRows, xCols);
      updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[base + 5]), w2Rows,
                         w2Cols);
    }
    updateMatrixExtent(cast<exec::LoadMatrixOp>(ops[tail + 2]), w3Rows, w3Cols);
  }

  int64_t outRows =
      getOptionalI32AttrOrZero(store.getOperation(), "row_offset") +
      store.getRows();
  int64_t outCols =
      getOptionalI32AttrOrZero(store.getOperation(), "col_offset") +
      store.getCols();

  std::string prelude;
  llvm::raw_string_ostream os(prelude);
  os << "    rpu::Array<half, 2> x_arr{arg" << xArg << ", " << xRows << ", "
     << xCols << "};\n";
  os << "    rpu::Array<half, 2> w1_arr{arg" << w1Arg << ", " << w1Rows << ", "
     << w1Cols << "};\n";
  os << "    rpu::Array<half, 2> w2_arr{arg" << w2Arg << ", " << w2Rows << ", "
     << w2Cols << "};\n";
  os << "    rpu::Array<half, 2> w3_arr{arg" << w3Arg << ", " << w3Rows << ", "
     << w3Cols << "};\n";
  os << "    rpu::Array<half, 2> out_arr{arg" << outArg << ", " << outRows
     << ", " << outCols << "};\n";
  os.flush();

  llvm::SmallVector<ExecutableMemoryBinding, 5> bindings;
  bindings.push_back(ExecutableMemoryBinding{xSkip.getPtr(), xArg, "x_arr",
                                             MatrixTileShape{xRows, xCols}});
  bindings.push_back(ExecutableMemoryBinding{firstW1.getPtr(), w1Arg, "w1_arr",
                                             MatrixTileShape{w1Rows, w1Cols}});
  bindings.push_back(ExecutableMemoryBinding{firstW2.getPtr(), w2Arg, "w2_arr",
                                             MatrixTileShape{w2Rows, w2Cols}});
  bindings.push_back(ExecutableMemoryBinding{firstW3.getPtr(), w3Arg, "w3_arr",
                                             MatrixTileShape{w3Rows, w3Cols}});
  bindings.push_back(ExecutableMemoryBinding{
      store.getPtr(), outArg, "out_arr", MatrixTileShape{outRows, outCols}});

  llvm::SmallVector<ExecutableValueNameHint, 128> valueNameHints;
  auto addHint = [&](Value value, llvm::StringRef name) {
    valueNameHints.push_back(ExecutableValueNameHint{value, name.str()});
  };
  addHint(xSkip.getResult(), "x_skip");

  if (bottleneck32) {
    addHint(cast<exec::LoadMatrixOp>(ops[1]).getResult(), "w1_0");
    addHint(cast<exec::LoadMatrixOp>(ops[2]).getResult(), "w1_1");
    addHint(cast<exec::ZeroOp>(ops[3]).getResult(), "conv2_acc_0");
    addHint(cast<exec::ZeroOp>(ops[4]).getResult(), "conv2_acc_1");

    for (int64_t ky = 0; ky < 3; ++ky) {
      for (int64_t kx = 0; kx < 3; ++kx) {
        int64_t window = ky * 3 + kx;
        int64_t base = 5 + window * 17;
        std::string xName = llvm::formatv("x_{0}_{1}", ky, kx).str();
        std::string conv10 = llvm::formatv("conv1_{0}_{1}_0", ky, kx).str();
        std::string zero10 = llvm::formatv("zero1_{0}_{1}_0", ky, kx).str();
        std::string relu10 = llvm::formatv("relu1_{0}_{1}_0", ky, kx).str();
        std::string conv11 = llvm::formatv("conv1_{0}_{1}_1", ky, kx).str();
        std::string zero11 = llvm::formatv("zero1_{0}_{1}_1", ky, kx).str();
        std::string relu11 = llvm::formatv("relu1_{0}_{1}_1", ky, kx).str();
        std::string w200 = llvm::formatv("w2_{0}_{1}_00", ky, kx).str();
        std::string w210 = llvm::formatv("w2_{0}_{1}_10", ky, kx).str();
        std::string w201 = llvm::formatv("w2_{0}_{1}_01", ky, kx).str();
        std::string w211 = llvm::formatv("w2_{0}_{1}_11", ky, kx).str();
        addHint(cast<exec::LoadMatrixOp>(ops[base]).getResult(), xName);
        addHint(cast<exec::ZeroOp>(ops[base + 1]).getResult(), conv10);
        addHint(cast<exec::ZeroOp>(ops[base + 3]).getResult(), zero10);
        addHint(cast<exec::MaxOp>(ops[base + 4]).getResult(), relu10);
        addHint(cast<exec::ZeroOp>(ops[base + 5]).getResult(), conv11);
        addHint(cast<exec::ZeroOp>(ops[base + 7]).getResult(), zero11);
        addHint(cast<exec::MaxOp>(ops[base + 8]).getResult(), relu11);
        addHint(cast<exec::LoadMatrixOp>(ops[base + 9]).getResult(), w200);
        addHint(cast<exec::LoadMatrixOp>(ops[base + 11]).getResult(), w210);
        addHint(cast<exec::LoadMatrixOp>(ops[base + 13]).getResult(), w201);
        addHint(cast<exec::LoadMatrixOp>(ops[base + 15]).getResult(), w211);
      }
    }

    addHint(cast<exec::ZeroOp>(ops[tail]).getResult(), "zero2_0");
    addHint(cast<exec::MaxOp>(ops[tail + 1]).getResult(), "relu2_0");
    addHint(cast<exec::ZeroOp>(ops[tail + 2]).getResult(), "zero2_1");
    addHint(cast<exec::MaxOp>(ops[tail + 3]).getResult(), "relu2_1");
    addHint(cast<exec::LoadMatrixOp>(ops[tail + 4]).getResult(), "w3_0");
    addHint(cast<exec::ZeroOp>(ops[tail + 5]).getResult(), "conv3");
    addHint(cast<exec::LoadMatrixOp>(ops[tail + 7]).getResult(), "w3_1");
    addHint(cast<exec::AddOp>(ops[tail + 9]).getResult(), "residual");
    addHint(cast<exec::ZeroOp>(ops[tail + 10]).getResult(), "zero3");
    addHint(cast<exec::MaxOp>(ops[tail + 11]).getResult(), "out_tile");
    return emitGenericExecutableOpSequenceBody(
        kernel, ops, bindings,
        "generic Bottleneck emission requires bound executable operands",
        prelude, valueNameHints);
  }

  addHint(cast<exec::LoadMatrixOp>(ops[1]).getResult(), "w1");
  addHint(cast<exec::ZeroOp>(ops[2]).getResult(), "conv2_acc");

  for (int64_t ky = 0; ky < 3; ++ky) {
    for (int64_t kx = 0; kx < 3; ++kx) {
      int64_t window = ky * 3 + kx;
      int64_t base = 3 + window * 7;
      std::string xName = llvm::formatv("x_{0}_{1}", ky, kx).str();
      std::string conv1 = llvm::formatv("conv1_{0}_{1}", ky, kx).str();
      std::string zero1 = llvm::formatv("zero1_{0}_{1}", ky, kx).str();
      std::string relu1 = llvm::formatv("relu1_{0}_{1}", ky, kx).str();
      std::string w2 = llvm::formatv("w2_{0}_{1}", ky, kx).str();
      addHint(cast<exec::LoadMatrixOp>(ops[base]).getResult(), xName);
      addHint(cast<exec::ZeroOp>(ops[base + 1]).getResult(), conv1);
      addHint(cast<exec::ZeroOp>(ops[base + 3]).getResult(), zero1);
      addHint(cast<exec::MaxOp>(ops[base + 4]).getResult(), relu1);
      addHint(cast<exec::LoadMatrixOp>(ops[base + 5]).getResult(), w2);
    }
  }

  addHint(cast<exec::ZeroOp>(ops[tail]).getResult(), "zero2");
  addHint(cast<exec::MaxOp>(ops[tail + 1]).getResult(), "relu2");
  addHint(cast<exec::LoadMatrixOp>(ops[tail + 2]).getResult(), "w3");
  addHint(cast<exec::ZeroOp>(ops[tail + 3]).getResult(), "conv3");
  addHint(cast<exec::AddOp>(ops[tail + 5]).getResult(), "residual");
  addHint(cast<exec::ZeroOp>(ops[tail + 6]).getResult(), "zero3");
  addHint(cast<exec::MaxOp>(ops[tail + 7]).getResult(), "out_tile");
  return emitGenericExecutableOpSequenceBody(
      kernel, ops, bindings,
      "generic Bottleneck emission requires bound executable operands", prelude,
      valueNameHints);
}

static FailureOr<RPUExecutableEmissionResult>
emitResNet50Bottleneck16FromExecutableContract(
    const exec::ExecutableKernelContractInfo &contract) {
  exec::KernelOp kernel = contract.kernel;
  ArrayRef<Operation *> ops = contract.bodyOps;

  FailureOr<std::string> prototype =
      buildKernelArgumentPrototype(contract.arguments);
  if (failed(prototype))
    return failure();

  FailureOr<std::string> body =
      emitResNet50BottleneckOpSequenceBody(kernel, ops);
  if (failed(body))
    return failure();
  std::string source =
      buildRPUDSLProgram(contract.kernelName, *prototype, *body);
  return RPUExecutableEmissionResult{contract.kernelName, "rpu_executable",
                                     source};
}

static FailureOr<std::string>
emitSqrtOpSequenceBody(exec::KernelOp kernel, ArrayRef<Operation *> ops) {
  FailureOr<exec::ExecutableSqrtOpSequenceInfo> maybeInfo =
      exec::getExecutableSqrtOpSequenceInfo(
          kernel, ops,
          "executable sqrt sequence emission requires verified IR info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableSqrtOpSequenceInfo info = *maybeInfo;

  FailureOr<llvm::SmallVector<unsigned, 4>> args =
      exec::getExecutableKernelArgumentIndices(
          kernel, {info.load.getPtr(), info.store.getPtr()},
          "executable sqrt op-sequence emission requires block argument "
          "pointers");
  if (failed(args))
    return failure();

  unsigned inputArg = (*args)[0];
  unsigned outArg = (*args)[1];

  llvm::SmallVector<ExecutableMemoryBinding, 2> bindings;
  bindings.push_back(ExecutableMemoryBinding{
      info.load.getPtr(), inputArg, pointerExpr(inputArg, 0), std::nullopt});
  bindings.push_back(ExecutableMemoryBinding{
      info.store.getPtr(), outArg, pointerExpr(outArg, 0), std::nullopt});
  return emitGenericExecutableOpSequenceBody(
      kernel, ops, bindings,
      "generic compact Sqrt emission requires bound executable operands");
}

static FailureOr<RPUExecutableEmissionResult> emitSqrtFromExecutableContract(
    const exec::ExecutableKernelContractInfo &contract) {
  exec::KernelOp kernel = contract.kernel;
  ArrayRef<Operation *> ops = contract.bodyOps;

  FailureOr<exec::ExecutableSqrtOpSequenceInfo> maybeInfo =
      exec::getExecutableSqrtOpSequenceInfo(
          kernel, ops,
          "executable sqrt contract-view emission requires verified IR info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableSqrtOpSequenceInfo info = *maybeInfo;

  int64_t nvec = info.nvec;
  if (nvec <= 0 || info.store.getN() != nvec ||
      !isTileVectorF16(info.load.getResult().getType(), nvec) ||
      !isTileVectorF16(info.sqrt.getResult().getType(), nvec) ||
      !isTileVectorF16(info.store.getValue().getType(), nvec)) {
    kernel.emitError("executable sqrt contract-view emission requires positive "
                     "!rpu.tile<Nxf16>");
    return failure();
  }

  FailureOr<std::string> prototype =
      buildKernelArgumentPrototype(contract.arguments);
  if (failed(prototype))
    return failure();

  FailureOr<std::string> body = emitSqrtOpSequenceBody(kernel, ops);
  if (failed(body))
    return failure();

  std::string source =
      buildRPUDSLProgram(contract.kernelName, *prototype, *body);
  return RPUExecutableEmissionResult{contract.kernelName, "rpu_executable",
                                     source};
}

static FailureOr<std::string>
emitReduceSumAllOpSequenceBody(exec::KernelOp kernel,
                               ArrayRef<Operation *> ops) {
  FailureOr<exec::ExecutableReduceSumAllOpSequenceInfo> maybeInfo =
      exec::getExecutableReduceSumAllOpSequenceInfo(
          kernel, ops,
          "executable reduce_sum_all sequence emission requires verified IR "
          "info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableReduceSumAllOpSequenceInfo info = *maybeInfo;

  FailureOr<llvm::SmallVector<unsigned, 4>> args =
      exec::getExecutableKernelArgumentIndices(
          kernel, {info.load.getPtr(), info.store.getPtr()},
          "executable reduce_sum_all op-sequence emission requires block "
          "argument pointers");
  if (failed(args))
    return failure();

  unsigned inputArg = (*args)[0];
  unsigned outArg = (*args)[1];

  llvm::SmallVector<ExecutableMemoryBinding, 2> bindings;
  bindings.push_back(ExecutableMemoryBinding{
      info.load.getPtr(), inputArg, pointerExpr(inputArg, 0), std::nullopt});
  bindings.push_back(ExecutableMemoryBinding{
      info.store.getPtr(), outArg, pointerExpr(outArg, 0), std::nullopt});
  return emitGenericExecutableOpSequenceBody(
      kernel, ops, bindings,
      "generic compact ReduceSumAll emission requires bound executable "
      "operands");
}

static FailureOr<RPUExecutableEmissionResult>
emitReduceSumAllFromExecutableContract(
    const exec::ExecutableKernelContractInfo &contract) {
  exec::KernelOp kernel = contract.kernel;
  ArrayRef<Operation *> ops = contract.bodyOps;

  FailureOr<exec::ExecutableReduceSumAllOpSequenceInfo> maybeInfo =
      exec::getExecutableReduceSumAllOpSequenceInfo(
          kernel, ops,
          "executable reduce_sum_all contract-view emission requires verified "
          "IR info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableReduceSumAllOpSequenceInfo info = *maybeInfo;

  int64_t nvec = info.nvec;
  if (nvec <= 0 || info.store.getN() != nvec ||
      !isTileVectorF16(info.load.getResult().getType(), nvec) ||
      !isTileVectorF16(info.full.getResult().getType(), nvec) ||
      !isTileVectorF16(info.store.getValue().getType(), nvec)) {
    kernel.emitError("executable reduce_sum_all contract-view emission "
                     "requires positive !rpu.tile<Nxf16>");
    return failure();
  }

  FailureOr<std::string> prototype =
      buildKernelArgumentPrototype(contract.arguments);
  if (failed(prototype))
    return failure();

  FailureOr<std::string> body = emitReduceSumAllOpSequenceBody(kernel, ops);
  if (failed(body))
    return failure();

  std::string source =
      buildRPUDSLProgram(contract.kernelName, *prototype, *body);
  return RPUExecutableEmissionResult{contract.kernelName, "rpu_executable",
                                     source};
}

static FailureOr<std::string>
emitReluOpSequenceBody(exec::KernelOp kernel, ArrayRef<Operation *> ops) {
  FailureOr<exec::ExecutableReluOpSequenceInfo> maybeInfo =
      exec::getExecutableReluOpSequenceInfo(
          kernel, ops,
          "executable relu sequence emission requires verified IR info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableReluOpSequenceInfo info = *maybeInfo;
  FailureOr<llvm::SmallVector<unsigned, 4>> args =
      exec::getExecutableKernelArgumentIndices(
          kernel, {info.load.getPtr(), info.store.getPtr()},
          "executable relu op-sequence emission requires block argument "
          "pointers");
  if (failed(args))
    return failure();
  unsigned inputArg = (*args)[0];
  unsigned outArg = (*args)[1];
  llvm::SmallVector<ExecutableMemoryBinding, 2> bindings;
  bindings.push_back(ExecutableMemoryBinding{
      info.load.getPtr(), inputArg, pointerExpr(inputArg, 0), std::nullopt});
  bindings.push_back(ExecutableMemoryBinding{
      info.store.getPtr(), outArg, pointerExpr(outArg, 0), std::nullopt});
  return emitGenericExecutableOpSequenceBody(
      kernel, ops, bindings,
      "generic compact Relu emission requires bound executable operands");
}

static FailureOr<RPUExecutableEmissionResult> emitReluFromExecutableContract(
    const exec::ExecutableKernelContractInfo &contract) {
  exec::KernelOp kernel = contract.kernel;
  ArrayRef<Operation *> ops = contract.bodyOps;
  FailureOr<exec::ExecutableReluOpSequenceInfo> maybeInfo =
      exec::getExecutableReluOpSequenceInfo(
          kernel, ops,
          "executable relu contract-view emission requires verified IR info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableReluOpSequenceInfo info = *maybeInfo;

  int64_t nvec = info.nvec;
  if (nvec <= 0 || info.store.getN() != nvec ||
      !isTileVectorF16(info.load.getResult().getType(), nvec) ||
      !isTileVectorF16(info.relu.getResult().getType(), nvec) ||
      !isTileVectorF16(info.store.getValue().getType(), nvec)) {
    kernel.emitError("executable relu contract-view emission requires positive "
                     "!rpu.tile<Nxf16>");
    return failure();
  }

  FailureOr<std::string> prototype =
      buildKernelArgumentPrototype(contract.arguments);
  if (failed(prototype))
    return failure();
  FailureOr<std::string> body = emitReluOpSequenceBody(kernel, ops);
  if (failed(body))
    return failure();
  std::string source =
      buildRPUDSLProgram(contract.kernelName, *prototype, *body);
  return RPUExecutableEmissionResult{contract.kernelName, "rpu_executable",
                                     source};
}

static FailureOr<std::string>
emitMaximumOpSequenceBody(exec::KernelOp kernel, ArrayRef<Operation *> ops) {
  FailureOr<exec::ExecutableMaximumOpSequenceInfo> maybeInfo =
      exec::getExecutableMaximumOpSequenceInfo(
          kernel, ops,
          "executable maximum sequence emission requires verified IR info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableMaximumOpSequenceInfo info = *maybeInfo;
  FailureOr<llvm::SmallVector<unsigned, 4>> args =
      exec::getExecutableKernelArgumentIndices(
          kernel,
          {info.lhsLoad.getPtr(), info.rhsLoad.getPtr(), info.store.getPtr()},
          "executable maximum op-sequence emission requires block argument "
          "pointers");
  if (failed(args))
    return failure();
  unsigned lhsArg = (*args)[0];
  unsigned rhsArg = (*args)[1];
  unsigned outArg = (*args)[2];
  llvm::SmallVector<ExecutableMemoryBinding, 3> bindings;
  bindings.push_back(ExecutableMemoryBinding{
      info.lhsLoad.getPtr(), lhsArg, pointerExpr(lhsArg, 0), std::nullopt});
  bindings.push_back(ExecutableMemoryBinding{
      info.rhsLoad.getPtr(), rhsArg, pointerExpr(rhsArg, 0), std::nullopt});
  bindings.push_back(ExecutableMemoryBinding{
      info.store.getPtr(), outArg, pointerExpr(outArg, 0), std::nullopt});
  return emitGenericExecutableOpSequenceBody(
      kernel, ops, bindings,
      "generic compact Maximum emission requires bound executable operands");
}

static FailureOr<RPUExecutableEmissionResult> emitMaximumFromExecutableContract(
    const exec::ExecutableKernelContractInfo &contract) {
  exec::KernelOp kernel = contract.kernel;
  ArrayRef<Operation *> ops = contract.bodyOps;
  FailureOr<exec::ExecutableMaximumOpSequenceInfo> maybeInfo =
      exec::getExecutableMaximumOpSequenceInfo(
          kernel, ops,
          "executable maximum contract-view emission requires verified IR "
          "info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableMaximumOpSequenceInfo info = *maybeInfo;

  int64_t n = info.n;
  if (n <= 0 || info.lhsLoad.getN() != n || info.rhsLoad.getN() != n ||
      info.store.getN() != n ||
      !isTileVectorF16(info.lhsLoad.getResult().getType(), n) ||
      !isTileVectorF16(info.rhsLoad.getResult().getType(), n) ||
      !isTileVectorF16(info.max.getResult().getType(), n) ||
      !isTileVectorF16(info.store.getValue().getType(), n)) {
    kernel.emitError("executable maximum contract-view emission requires "
                     "positive !rpu.tile<Nxf16>");
    return failure();
  }

  FailureOr<std::string> prototype =
      buildKernelArgumentPrototype(contract.arguments);
  if (failed(prototype))
    return failure();
  FailureOr<std::string> body = emitMaximumOpSequenceBody(kernel, ops);
  if (failed(body))
    return failure();
  std::string source =
      buildRPUDSLProgram(contract.kernelName, *prototype, *body);
  return RPUExecutableEmissionResult{contract.kernelName, "rpu_executable",
                                     source};
}

static FailureOr<std::string>
emitReduceSumAxisOpSequenceBody(exec::KernelOp kernel,
                                ArrayRef<Operation *> ops) {
  FailureOr<exec::ExecutableReduceSumAxisOpSequenceInfo> maybeInfo =
      exec::getExecutableReduceSumAxisOpSequenceInfo(
          kernel, ops,
          "executable reduce_sum_axis sequence emission requires verified IR "
          "info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableReduceSumAxisOpSequenceInfo info = *maybeInfo;
  FailureOr<llvm::SmallVector<unsigned, 4>> args =
      exec::getExecutableKernelArgumentIndices(
          kernel, {info.load.getPtr(), info.store.getPtr()},
          "executable reduce_sum_axis op-sequence emission requires block "
          "argument pointers");
  if (failed(args))
    return failure();
  unsigned inputArg = (*args)[0];
  unsigned outArg = (*args)[1];
  int64_t M = info.rows;
  int64_t N = info.cols;
  int64_t axis = info.axis;

  // Hand-rolled body so we can keep V256 thread/lane semantics consistent
  // end-to-end. The DSL `ctx.store<T, M, N>` static_asserts on
  // M%16==N%16==0, so we cannot store the rank-2 reduce result Tile<1, N> /
  // Tile<M, 1> directly. We compose the reduction in 16-row chunks because
  // (a) rpu_tile.h reduce_sum<0>'s singleton path only accumulates the
  // first M_TILES iteration for M > 16, and (b) zeros<M, N>() + Tile<1, N>
  // / Tile<M, 1> via RPU_TILE_BCASTOP_R2_{N,M} also leaves slot 1 of result
  // uninitialised for M_TILES > 1. By chunking both reduce and bcast into
  // Tile<16, N> / Tile<M, 16> units we stay in the single-V256-slot regime
  // where the DSL primitives are byte-exact verified.
  std::string body;
  llvm::raw_string_ostream os(body);
  os << "    rpu::Array<half, 2> arg" << inputArg << "_arr{arg" << inputArg
     << ", " << M << ", " << N << "};\n";
  os << "    rpu::Array<half, 2> arg" << outArg << "_arr{arg" << outArg << ", "
     << M << ", " << N << "};\n";
  int64_t mChunks = M / 16;
  if (mChunks == 1) {
    // Fast path: M == 16, no chunking needed. ctx.reduce_sum<axis> on a
    // single-V256-slot input is byte-exact verified on board (16x16 case).
    os << "    auto tile0 = ctx.load<half, " << M << ", " << N << ">(arg"
       << inputArg << "_arr, rpu::IndexList{0, 0});\n";
    os << "    auto tile1 = ctx.reduce_sum<" << axis << ">(tile0);\n";
    os << "    auto tile2 = ctx.zeros<half, " << M << ", " << N
       << ">() + tile1;\n";
    os << "    ctx.store<half, " << M << ", " << N << ">(arg" << outArg
       << "_arr, rpu::IndexList{0, 0}, tile2);";
  } else if (axis == 0) {
    // Accumulate col-sums across chunks into a Tile<16, N> via chained
    // BCASTOP_R2_N (zeros + Tile<1, N>, then + Tile<1, N>, ...). This keeps
    // every value in DSL-native Tile shapes — no from_raw rank-1 round-trip
    // (the rank-1 round-trip does not preserve the data on the target).
    for (int64_t c = 0; c < mChunks; ++c) {
      int64_t rowOff = c * 16;
      os << "    auto tile_x" << c << " = ctx.load<half, 16, " << N << ">(arg"
         << inputArg << "_arr, rpu::IndexList{" << rowOff << ", 0});\n";
      os << "    auto tile_r" << c << " = ctx.reduce_sum<0>(tile_x" << c
         << ");\n";
    }
    os << "    auto acc = ctx.zeros<half, 16, " << N << ">() + tile_r0;\n";
    for (int64_t c = 1; c < mChunks; ++c) {
      os << "    acc = acc + tile_r" << c << ";\n";
    }
    // Store the same accumulator to every M-chunk slice of the output region.
    for (int64_t c = 0; c < mChunks; ++c) {
      int64_t rowOff = c * 16;
      os << "    ctx.store<half, 16, " << N << ">(arg" << outArg
         << "_arr, rpu::IndexList{" << rowOff << ", 0}, acc);\n";
    }
  } else {
    // axis == 1. Per-chunk row reductions (Tile<16, 1>) are independent —
    // no cross-chunk accumulation needed. Bcast + store each chunk to its
    // own row slice.
    for (int64_t c = 0; c < mChunks; ++c) {
      int64_t rowOff = c * 16;
      os << "    {\n";
      os << "        auto tile_x = ctx.load<half, 16, " << N << ">(arg"
         << inputArg << "_arr, rpu::IndexList{" << rowOff << ", 0});\n";
      os << "        auto tile_r = ctx.reduce_sum<1>(tile_x);\n";
      os << "        auto full = ctx.zeros<half, 16, " << N
         << ">() + tile_r;\n";
      os << "        ctx.store<half, 16, " << N << ">(arg" << outArg
         << "_arr, rpu::IndexList{" << rowOff << ", 0}, full);\n";
      os << "    }\n";
    }
  }
  if (!body.empty() && body.back() == '\n')
    body.pop_back();
  return body;
}

static FailureOr<RPUExecutableEmissionResult>
emitReduceSumAxisFromExecutableContract(
    const exec::ExecutableKernelContractInfo &contract) {
  exec::KernelOp kernel = contract.kernel;
  ArrayRef<Operation *> ops = contract.bodyOps;
  FailureOr<exec::ExecutableReduceSumAxisOpSequenceInfo> maybeInfo =
      exec::getExecutableReduceSumAxisOpSequenceInfo(
          kernel, ops,
          "executable reduce_sum_axis contract-view emission requires verified "
          "IR info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableReduceSumAxisOpSequenceInfo info = *maybeInfo;
  int64_t rows = info.rows;
  int64_t cols = info.cols;
  int64_t outDim = info.axis == 0 ? cols : rows;
  if (rows <= 0 || cols <= 0 || info.store.getN() != outDim ||
      !isTileMatrixF16(info.load.getResult().getType(), rows, cols) ||
      !isTileVectorF16(info.reduce.getResult().getType(), outDim) ||
      !isTileVectorF16(info.store.getValue().getType(), outDim)) {
    kernel.emitError(
        "executable reduce_sum_axis contract-view emission requires positive "
        "!rpu.tile<RxCxf16> and matching output extent");
    return failure();
  }
  FailureOr<std::string> prototype =
      buildKernelArgumentPrototype(contract.arguments);
  if (failed(prototype))
    return failure();
  FailureOr<std::string> body = emitReduceSumAxisOpSequenceBody(kernel, ops);
  if (failed(body))
    return failure();
  std::string source =
      buildRPUDSLProgram(contract.kernelName, *prototype, *body);
  return RPUExecutableEmissionResult{contract.kernelName, "rpu_executable",
                                     source};
}

static FailureOr<std::string>
emitBroadcastAddOpSequenceBody(exec::KernelOp kernel,
                               ArrayRef<Operation *> ops) {
  FailureOr<exec::ExecutableBroadcastAddOpSequenceInfo> maybeInfo =
      exec::getExecutableBroadcastAddOpSequenceInfo(
          kernel, ops,
          "executable broadcast_add sequence emission requires verified IR "
          "info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableBroadcastAddOpSequenceInfo info = *maybeInfo;
  FailureOr<llvm::SmallVector<unsigned, 4>> args =
      exec::getExecutableKernelArgumentIndices(
          kernel,
          {info.lhsLoad.getPtr(), info.rhsLoad.getPtr(), info.store.getPtr()},
          "executable broadcast_add op-sequence emission requires block "
          "argument pointers");
  if (failed(args))
    return failure();
  unsigned lhsArg = (*args)[0];
  unsigned rhsArg = (*args)[1];
  unsigned outArg = (*args)[2];
  int64_t totalElems = info.rows * info.cols;
  int64_t nChunks = (totalElems + 255) / 256;
  if (nChunks < 1)
    nChunks = 1;

  // Hand-rolled body that bypasses RPU_TILE_BCASTOP_R2_M. The harness preloads
  // b in a V256 broadcast layout (see make_bcast_nc_plus_n_case docstring).
  // With that layout, load_contig<1>(b_ptr + 256*chunk) and the matching load
  // of a each yield one V256; a rank-1 add produces the
  // desired broadcast result. ctx.store_contig writes back in the row-major
  // layout the kernel expects. We chunk into NVEC=1 V256 slots (256 elements
  // each) sequentially: NVEC=2+ load/store via load_contig<NVEC> doesn't
  // produce the expected per-thread layout on this target, so we keep to
  // single-slot loads here.
  std::string body;
  llvm::raw_string_ostream os(body);
  for (int64_t chunk = 0; chunk < nChunks; ++chunk) {
    int64_t off = chunk * 256;
    os << "    {\n";
    os << "        auto tile_a = ctx.load_contig<1>(arg" << lhsArg << " + "
       << off << ");\n";
    os << "        auto tile_b = ctx.load_contig<1>(arg" << rhsArg << " + "
       << off << ");\n";
    os << "        auto tile_r = tile_a + tile_b;\n";
    os << "        ctx.store_contig<1>(arg" << outArg << " + " << off
       << ", tile_r);\n";
    os << "    }\n";
  }
  // Trim trailing newline so the program builder's brace placement stays
  // consistent with the other hand-rolled emitters.
  if (!body.empty() && body.back() == '\n')
    body.pop_back();
  return body;
}

static FailureOr<RPUExecutableEmissionResult>
emitBroadcastAddFromExecutableContract(
    const exec::ExecutableKernelContractInfo &contract) {
  exec::KernelOp kernel = contract.kernel;
  ArrayRef<Operation *> ops = contract.bodyOps;
  FailureOr<exec::ExecutableBroadcastAddOpSequenceInfo> maybeInfo =
      exec::getExecutableBroadcastAddOpSequenceInfo(
          kernel, ops,
          "executable broadcast_add contract-view emission requires verified "
          "IR info");
  if (failed(maybeInfo))
    return failure();
  exec::ExecutableBroadcastAddOpSequenceInfo info = *maybeInfo;
  int64_t rows = info.rows;
  int64_t cols = info.cols;
  if (rows <= 0 || cols <= 0 ||
      !isTileMatrixF16(info.lhsLoad.getResult().getType(), rows, cols) ||
      !isTileVectorF16(info.rhsLoad.getResult().getType(), rows) ||
      !isTileMatrixF16(info.broadcastAdd.getResult().getType(), rows, cols) ||
      !isTileMatrixF16(info.store.getValue().getType(), rows, cols)) {
    kernel.emitError(
        "executable broadcast_add contract-view emission requires "
        "matching !rpu.tile<RxCxf16> matrix and !rpu.tile<Rxf16> vector");
    return failure();
  }
  FailureOr<std::string> prototype =
      buildKernelArgumentPrototype(contract.arguments);
  if (failed(prototype))
    return failure();
  FailureOr<std::string> body = emitBroadcastAddOpSequenceBody(kernel, ops);
  if (failed(body))
    return failure();
  std::string source =
      buildRPUDSLProgram(contract.kernelName, *prototype, *body);
  return RPUExecutableEmissionResult{contract.kernelName, "rpu_executable",
                                     source};
}

static FailureOr<std::string> emitGenericVectorOpSequenceBody(
    const exec::ExecutableKernelContractInfo &contract) {
  for (Operation *op : contract.bodyOps) {
    if (isa<exec::LoadMatrixOp, exec::StoreMatrixOp>(op)) {
      op->emitError(
          "generic executable emission supports only vector memory ops");
      return failure();
    }
  }

  if (failed(exec::verifyExecutableElementwise1DOpSequence(
          contract.kernel, contract.bodyOps,
          "generic Elementwise1D emission requires verifier-only executable IR "
          "contract")))
    return failure();

  llvm::SmallVector<ExecutableMemoryBinding, 4> bindings;
  for (const exec::ExecutableKernelArgumentInfo &arg : contract.arguments) {
    bindings.push_back(
        ExecutableMemoryBinding{arg.value, arg.index, arg.name, std::nullopt});
  }

  return emitGenericExecutableOpSequenceBody(
      contract.kernel, contract.bodyOps, bindings,
      "generic executable emission requires vector-compatible operands");
}

static FailureOr<RPUExecutableEmissionResult> emitGenericFromExecutableContract(
    const exec::ExecutableKernelContractInfo &contract) {
  FailureOr<std::string> prototype =
      buildKernelArgumentPrototype(contract.arguments);
  if (failed(prototype))
    return failure();

  FailureOr<std::string> body = emitGenericVectorOpSequenceBody(contract);
  if (failed(body))
    return failure();

  std::string source =
      buildRPUDSLProgram(contract.kernelName, *prototype, *body);
  return RPUExecutableEmissionResult{contract.kernelName, "rpu_executable",
                                     source};
}

using EmitFn = FailureOr<RPUExecutableEmissionResult> (*)(exec::KernelOp);

using ContractEmitFn = FailureOr<RPUExecutableEmissionResult> (*)(
    const exec::ExecutableKernelContractInfo &);

struct ExecutableEmitterEntry {
  llvm::StringLiteral kind;
  ContractEmitFn emit;
};

static constexpr ExecutableEmitterEntry kExecutableEmitterEntries[] = {
    {"add", emitAddFromExecutableContract},
    {"gemm", emitGemmFromExecutableContract},
    {"softmax", emitSoftmaxFromExecutableContract},
    {"sqrt", emitSqrtFromExecutableContract},
    {"reduce_sum_all", emitReduceSumAllFromExecutableContract},
    {"relu", emitReluFromExecutableContract},
    {"maximum", emitMaximumFromExecutableContract},
    {"reduce_sum_axis0", emitReduceSumAxisFromExecutableContract},
    {"reduce_sum_axis1", emitReduceSumAxisFromExecutableContract},
    {"broadcast_add", emitBroadcastAddFromExecutableContract},
    {"convkxk", emitConvKxKFromExecutableContract},
    {"resnet_block", emitResNetBlock16FromExecutableContract},
    {"resnet50_bottleneck", emitResNet50Bottleneck16FromExecutableContract},
    {"generic", emitGenericFromExecutableContract},
};

static std::optional<const ExecutableEmitterEntry *>
lookupExecutableEmitterEntry(llvm::StringRef kind) {
  for (const ExecutableEmitterEntry &entry : kExecutableEmitterEntries) {
    if (entry.kind == kind)
      return &entry;
  }
  return std::nullopt;
}

} // namespace

FailureOr<RPUExecutableEmissionResult>
emitRPUDSLFromExecutableModule(ModuleOp module) {
  FailureOr<exec::ExecutableKernelContractInfo> contract =
      exec::getVerifiedExecutableKernelContractFromModule(
          module, "RPU executable .rc emission");
  if (failed(contract))
    return failure();

  std::optional<const ExecutableEmitterEntry *> entry =
      lookupExecutableEmitterEntry(contract->kind);
  if (entry)
    return (*entry)->emit(*contract);

  contract->kernel.emitError("executable .rc emission unsupported kernel kind");
  return failure();
}

} // namespace rpu
} // namespace mlir
