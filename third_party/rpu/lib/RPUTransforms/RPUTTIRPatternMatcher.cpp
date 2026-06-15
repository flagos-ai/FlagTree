#include "RPUTTIRPatternMatcher.h"

#include "RPUExecutableDirectElementwise1D.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace mlir {
namespace rpu {

struct LoadRole {
  int64_t arg = -1;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t rowBase = 0;
  int64_t rowStride = 0;
};

struct DotShape {
  int64_t m = 0;
  int64_t k = 0;
  int64_t n = 0;
};

enum class ReductionKind { Max, Sum };

struct CompactVectorMemoryMatch {
  int64_t nElements = 0;
  int64_t logicalN = 0;
  bool masked = false;
  llvm::DenseMap<Value, int64_t> ptrToArg;
  llvm::DenseMap<Value, int64_t> loadToArg;
};

struct Elementwise1DValueGraphMatch {
  TraceAnchor anchor;
  int64_t n = 0;
  int64_t logicalN = 0;
  bool masked = false;
  int64_t outputArgIndex = -1;
  SmallVector<unsigned, 4> inputArgIndices;
  SmallVector<exec::ExecutableCompactVectorBinaryBuildOp, 4> ops;
};

std::string stringifyLocation(Location loc) {
  std::string output;
  llvm::raw_string_ostream os(output);
  loc.print(os);
  return output;
}

TraceAnchor moduleAnchor(ModuleOp module) {
  return TraceAnchor{"module", "builtin.module",
                     stringifyLocation(module.getLoc())};
}

TraceAnchor functionAnchor(triton::FuncOp func) {
  return TraceAnchor{"function", "tt.func", stringifyLocation(func.getLoc())};
}

TraceAnchor opAnchor(Operation *op) {
  return TraceAnchor{"op", op->getName().getStringRef().str(),
                     stringifyLocation(op->getLoc())};
}

TraceAttempt failedAttempt(StringRef pattern, StringRef reason,
                           const TraceAnchor &anchor) {
  return TraceAttempt{pattern.str(), "failed", reason.str(), anchor};
}

TraceAttempt matchedAttempt(StringRef pattern, const TraceAnchor &anchor) {
  return TraceAttempt{pattern.str(), "matched", "selected", anchor};
}

void appendAllFailedAttempts(SmallVectorImpl<TraceAttempt> &attempts,
                             StringRef reason, const TraceAnchor &anchor) {
  attempts.push_back(failedAttempt(kAddPattern, reason, anchor));
  attempts.push_back(failedAttempt(kGemmPattern, reason, anchor));
  attempts.push_back(failedAttempt(kSoftmaxPattern, reason, anchor));
  attempts.push_back(failedAttempt(kSqrtPattern, reason, anchor));
  attempts.push_back(failedAttempt(kReduceSumAllPattern, reason, anchor));
  attempts.push_back(failedAttempt(kResNetBlockPattern, reason, anchor));
  attempts.push_back(failedAttempt(kConvKxKPattern, reason, anchor));
  attempts.push_back(failedAttempt(kResNet50BottleneckPattern, reason, anchor));
}

bool isAllowedNNTopLevelOp(Operation &op);

bool isAllowedAddTopLevelOp(Operation &op) {
  return isa<triton::MakeRangeOp, triton::LoadOp, arith::AddFOp,
             triton::StoreOp, triton::SplatOp, triton::AddPtrOp,
             triton::ReturnOp, arith::ConstantOp, arith::CmpIOp,
             arith::TruncFOp>(&op);
}

bool isAllowedGemmTopLevelOp(Operation &op) {
  return isa<triton::LoadOp, triton::DotOp, triton::StoreOp,
             triton::MakeRangeOp, triton::ExpandDimsOp, triton::SplatOp,
             triton::AddPtrOp, triton::BroadcastOp, triton::ReturnOp,
             arith::ConstantOp, arith::MulIOp, arith::TruncFOp>(&op);
}

bool isAllowedSoftmaxTopLevelOp(Operation &op) {
  return isa<triton::MakeRangeOp, triton::LoadOp, triton::ReduceOp,
             arith::ExtFOp, arith::SubFOp, arith::DivFOp, arith::TruncFOp,
             triton::StoreOp, triton::SplatOp, triton::AddPtrOp,
             triton::ReturnOp>(&op) ||
         isa<math::ExpOp>(&op) || op.getName().getStringRef() == "math.exp";
}

bool isAllowedSqrtTopLevelOp(Operation &op) {
  return isa<triton::MakeRangeOp, triton::LoadOp, arith::ExtFOp,
             arith::TruncFOp, triton::StoreOp, triton::SplatOp,
             triton::AddPtrOp, triton::ReturnOp, arith::ConstantOp,
             arith::CmpIOp>(&op) ||
         isa<math::SqrtOp>(&op) || op.getName().getStringRef() == "math.sqrt";
}

bool isAllowedReduceSumAllTopLevelOp(Operation &op) {
  return isa<triton::MakeRangeOp, triton::LoadOp, triton::ReduceOp,
             triton::StoreOp, triton::SplatOp, triton::AddPtrOp,
             triton::ReturnOp, arith::ConstantOp, arith::CmpIOp,
             arith::TruncFOp>(&op);
}

bool isAllowedReluTopLevelOp(Operation &op) {
  return isa<triton::MakeRangeOp, triton::LoadOp, arith::ExtFOp,
             arith::TruncFOp, arith::MaxNumFOp, arith::MaximumFOp,
             triton::StoreOp, triton::SplatOp, triton::AddPtrOp,
             triton::ReturnOp, arith::ConstantOp, arith::CmpIOp>(&op);
}

bool isAllowedMaximumTopLevelOp(Operation &op) {
  return isa<triton::MakeRangeOp, triton::LoadOp, arith::ExtFOp,
             arith::TruncFOp, arith::MaxNumFOp, arith::MaximumFOp,
             triton::StoreOp, triton::SplatOp, triton::AddPtrOp,
             triton::ReturnOp, arith::ConstantOp, arith::CmpIOp>(&op);
}

bool isAllowedReduceSumAxisTopLevelOp(Operation &op) {
  return isa<triton::MakeRangeOp, triton::ExpandDimsOp, triton::BroadcastOp,
             triton::LoadOp, triton::ReduceOp, triton::StoreOp, triton::SplatOp,
             triton::AddPtrOp, triton::ReturnOp, arith::ConstantOp,
             arith::MulIOp, arith::AddIOp, arith::TruncFOp>(&op);
}

bool isAllowedBroadcastAddTopLevelOp(Operation &op) {
  return isa<triton::MakeRangeOp, triton::ExpandDimsOp, triton::BroadcastOp,
             triton::LoadOp, arith::AddFOp, triton::StoreOp, triton::SplatOp,
             triton::AddPtrOp, triton::ReturnOp, arith::ConstantOp,
             arith::MulIOp, arith::AddIOp, arith::TruncFOp>(&op);
}

TraceAnchor
rankedFailureAnchor(triton::FuncOp func,
                    llvm::function_ref<bool(Operation &)> isAllowed,
                    llvm::function_ref<int(Operation &)> priorityFor) {
  if (func.getBody().empty() || !func.getBody().hasOneBlock())
    return functionAnchor(func);
  Block &entry = func.getBody().front();
  Operation *best = nullptr;
  int bestPriority = 1000000;
  for (Operation &op : entry.getOperations()) {
    if (!isAllowed(op))
      return opAnchor(&op);
    int priority = priorityFor(op);
    if (priority >= 0 && priority < bestPriority) {
      best = &op;
      bestPriority = priority;
    }
  }
  if (best)
    return opAnchor(best);
  return functionAnchor(func);
}

TraceAnchor addFailureAnchor(triton::FuncOp func) {
  return rankedFailureAnchor(func, isAllowedAddTopLevelOp,
                             [](Operation &op) -> int {
                               if (isa<triton::StoreOp>(&op))
                                 return 0;
                               if (isa<arith::AddFOp>(&op))
                                 return 1;
                               if (isa<triton::LoadOp>(&op))
                                 return 2;
                               return -1;
                             });
}

TraceAnchor gemmFailureAnchor(triton::FuncOp func) {
  return rankedFailureAnchor(func, isAllowedGemmTopLevelOp,
                             [](Operation &op) -> int {
                               if (isa<triton::DotOp>(&op))
                                 return 0;
                               if (isa<triton::StoreOp>(&op))
                                 return 1;
                               if (isa<triton::LoadOp>(&op))
                                 return 2;
                               return -1;
                             });
}

TraceAnchor softmaxFailureAnchor(triton::FuncOp func) {
  return rankedFailureAnchor(
      func, isAllowedSoftmaxTopLevelOp, [](Operation &op) -> int {
        if (isa<triton::StoreOp>(&op))
          return 0;
        if (isa<triton::ReduceOp>(&op))
          return 1;
        if (isa<math::ExpOp>(&op) || op.getName().getStringRef() == "math.exp")
          return 2;
        if (isa<triton::LoadOp>(&op))
          return 3;
        return -1;
      });
}

TraceAnchor sqrtFailureAnchor(triton::FuncOp func) {
  return rankedFailureAnchor(func, isAllowedSqrtTopLevelOp,
                             [](Operation &op) -> int {
                               if (isa<triton::StoreOp>(&op))
                                 return 0;
                               if (isa<math::SqrtOp>(&op) ||
                                   op.getName().getStringRef() == "math.sqrt")
                                 return 1;
                               if (isa<triton::LoadOp>(&op))
                                 return 2;
                               return -1;
                             });
}

TraceAnchor reduceSumAllFailureAnchor(triton::FuncOp func) {
  return rankedFailureAnchor(func, isAllowedReduceSumAllTopLevelOp,
                             [](Operation &op) -> int {
                               if (isa<triton::StoreOp>(&op))
                                 return 0;
                               if (isa<triton::ReduceOp>(&op))
                                 return 1;
                               if (isa<triton::LoadOp>(&op))
                                 return 2;
                               return -1;
                             });
}

TraceAnchor reluFailureAnchor(triton::FuncOp func) {
  return rankedFailureAnchor(
      func, isAllowedReluTopLevelOp, [](Operation &op) -> int {
        if (isa<triton::StoreOp>(&op))
          return 0;
        if (isa<arith::MaxNumFOp>(&op) || isa<arith::MaximumFOp>(&op))
          return 1;
        if (isa<triton::LoadOp>(&op))
          return 2;
        return -1;
      });
}

TraceAnchor maximumFailureAnchor(triton::FuncOp func) {
  return rankedFailureAnchor(
      func, isAllowedMaximumTopLevelOp, [](Operation &op) -> int {
        if (isa<triton::StoreOp>(&op))
          return 0;
        if (isa<arith::MaxNumFOp>(&op) || isa<arith::MaximumFOp>(&op))
          return 1;
        if (isa<triton::LoadOp>(&op))
          return 2;
        return -1;
      });
}

TraceAnchor reduceSumAxisFailureAnchor(triton::FuncOp func) {
  return rankedFailureAnchor(func, isAllowedReduceSumAxisTopLevelOp,
                             [](Operation &op) -> int {
                               if (isa<triton::StoreOp>(&op))
                                 return 0;
                               if (isa<triton::ReduceOp>(&op))
                                 return 1;
                               if (isa<triton::LoadOp>(&op))
                                 return 2;
                               return -1;
                             });
}

TraceAnchor broadcastAddFailureAnchor(triton::FuncOp func) {
  return rankedFailureAnchor(func, isAllowedBroadcastAddTopLevelOp,
                             [](Operation &op) -> int {
                               if (isa<triton::StoreOp>(&op))
                                 return 0;
                               if (isa<arith::AddFOp>(&op))
                                 return 1;
                               if (isa<triton::LoadOp>(&op))
                                 return 2;
                               return -1;
                             });
}

TraceAnchor convKxKFailureAnchor(triton::FuncOp func) {
  return rankedFailureAnchor(func, isAllowedNNTopLevelOp,
                             [](Operation &op) -> int {
                               if (isa<triton::StoreOp>(&op))
                                 return 0;
                               if (isa<triton::DotOp>(&op))
                                 return 1;
                               if (isa<triton::LoadOp>(&op))
                                 return 2;
                               return -1;
                             });
}

TraceAnchor resNetBlockFailureAnchor(triton::FuncOp func) {
  return rankedFailureAnchor(func, isAllowedNNTopLevelOp,
                             [](Operation &op) -> int {
                               if (isa<triton::StoreOp>(&op))
                                 return 0;
                               if (isa<arith::AddFOp>(&op))
                                 return 1;
                               if (isa<triton::DotOp>(&op))
                                 return 2;
                               if (isa<triton::LoadOp>(&op))
                                 return 3;
                               return -1;
                             });
}

TraceAnchor resNet50BottleneckFailureAnchor(triton::FuncOp func) {
  return rankedFailureAnchor(func, isAllowedNNTopLevelOp,
                             [](Operation &op) -> int {
                               if (isa<triton::StoreOp>(&op))
                                 return 0;
                               if (isa<triton::DotOp>(&op))
                                 return 1;
                               if (isa<triton::LoadOp>(&op))
                                 return 2;
                               return -1;
                             });
}

FailureOr<triton::FuncOp> getSinglePublicFunc(ModuleOp module) {
  SmallVector<triton::FuncOp> publicFuncs;
  module.walk([&](triton::FuncOp func) {
    std::optional<StringRef> visibility = func.getSymVisibility();
    if (!visibility || *visibility == "public")
      publicFuncs.push_back(func);
  });

  if (publicFuncs.size() != 1)
    return failure();
  return publicFuncs.front();
}

bool isF16Ptr(Type type) {
  auto pointerType = dyn_cast<triton::PointerType>(type);
  return pointerType && pointerType.getPointeeType().isF16();
}

bool isRankedVectorOf(Value value, Type elementType, int64_t n) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getRank() == 1 && type.getShape()[0] == n &&
         type.getElementType() == elementType;
}

bool isRankedTensorOf(Value value, Type elementType, ArrayRef<int64_t> shape) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getShape() == shape &&
         type.getElementType() == elementType;
}

bool isRankedTensorOfF16Ptr(Value value, ArrayRef<int64_t> shape) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getShape() == shape && isF16Ptr(type.getElementType());
}

std::optional<int64_t> getEntryBlockArgNumber(Value value, Block &entry) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != &entry)
    return std::nullopt;
  return blockArg.getArgNumber();
}

bool isRankedI32TensorOf(Value value, ArrayRef<int64_t> shape) {
  return isRankedTensorOf(value, IntegerType::get(value.getContext(), 32),
                          shape);
}

bool isZeroF32Tensor(Value value, ArrayRef<int64_t> shape) {
  if (!isRankedTensorOf(value, Float32Type::get(value.getContext()), shape))
    return false;

  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;

  auto dense = dyn_cast<DenseFPElementsAttr>(constant.getValue());
  return dense && dense.isSplat() && dense.getSplatValue<APFloat>().isZero();
}

bool isMakeRange(Value value, int64_t nElements) {
  auto range = value.getDefiningOp<triton::MakeRangeOp>();
  return range && range.getStartAttr().getInt() == 0 &&
         range.getEndAttr().getInt() == nElements &&
         isRankedVectorOf(range.getResult(),
                          IntegerType::get(value.getContext(), 32), nElements);
}

bool isExpandedRange(Value value, int64_t nElements, uint32_t axis,
                     ArrayRef<int64_t> shape) {
  auto expand = value.getDefiningOp<triton::ExpandDimsOp>();
  return expand && expand.getAxis() == axis &&
         isRankedI32TensorOf(expand.getResult(), shape) &&
         isMakeRange(expand.getSrc(), nElements);
}

std::optional<int64_t> getSplatI32Constant(Value value);

std::optional<int64_t> getRowRangeBase(Value value, int64_t rows) {
  if (isExpandedRange(value, rows, /*axis=*/1, {rows, 1}))
    return 0;

  auto add = value.getDefiningOp<arith::AddIOp>();
  if (!add || !isRankedI32TensorOf(add.getResult(), {rows, 1}))
    return std::nullopt;

  std::optional<int64_t> lhsConstant = getSplatI32Constant(add.getLhs());
  std::optional<int64_t> rhsConstant = getSplatI32Constant(add.getRhs());
  if (lhsConstant) {
    std::optional<int64_t> base = getRowRangeBase(add.getRhs(), rows);
    if (base)
      return *lhsConstant + *base;
  }
  if (rhsConstant) {
    std::optional<int64_t> base = getRowRangeBase(add.getLhs(), rows);
    if (base)
      return *base + *rhsConstant;
  }
  return std::nullopt;
}

std::optional<int64_t> matchScaledRowOffsetBase(Value value, int64_t rows,
                                                int64_t stride) {
  if (!isRankedI32TensorOf(value, {rows, 1}))
    return std::nullopt;

  auto mul = value.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return std::nullopt;

  std::optional<int64_t> lhsConstant = getSplatI32Constant(mul.getLhs());
  std::optional<int64_t> rhsConstant = getSplatI32Constant(mul.getRhs());
  Value rangeSide;
  if (lhsConstant && *lhsConstant == stride) {
    rangeSide = mul.getRhs();
  } else if (rhsConstant && *rhsConstant == stride) {
    rangeSide = mul.getLhs();
  } else {
    return std::nullopt;
  }

  return getRowRangeBase(rangeSide, rows);
}

bool isScaledRowOffset(Value value, int64_t rows, int64_t stride) {
  std::optional<int64_t> rowBase =
      matchScaledRowOffsetBase(value, rows, stride);
  return rowBase && *rowBase == 0;
}

std::optional<LoadRole>
matchRowMajorF16PointerAccess(Value pointer, Block &entry, int64_t rows,
                              int64_t cols, int64_t rowStride) {
  if (!isRankedTensorOfF16Ptr(pointer, {rows, cols}))
    return std::nullopt;

  auto finalAdd = pointer.getDefiningOp<triton::AddPtrOp>();
  if (!finalAdd)
    return std::nullopt;

  auto rowBroadcast = finalAdd.getPtr().getDefiningOp<triton::BroadcastOp>();
  auto colBroadcast = finalAdd.getOffset().getDefiningOp<triton::BroadcastOp>();
  if (!rowBroadcast || !colBroadcast ||
      !isRankedTensorOfF16Ptr(rowBroadcast.getResult(), {rows, cols}) ||
      !isRankedI32TensorOf(colBroadcast.getResult(), {rows, cols}) ||
      !isExpandedRange(colBroadcast.getSrc(), cols, /*axis=*/0, {1, cols}))
    return std::nullopt;

  auto rowAdd = rowBroadcast.getSrc().getDefiningOp<triton::AddPtrOp>();
  if (!rowAdd || !isRankedTensorOfF16Ptr(rowAdd.getResult(), {rows, 1}) ||
      !isRankedI32TensorOf(rowAdd.getOffset(), {rows, 1}))
    return std::nullopt;
  std::optional<int64_t> rowBase =
      matchScaledRowOffsetBase(rowAdd.getOffset(), rows, rowStride);
  if (!rowBase || *rowBase < 0)
    return std::nullopt;

  auto splat = rowAdd.getPtr().getDefiningOp<triton::SplatOp>();
  if (!splat || !isRankedTensorOfF16Ptr(splat.getResult(), {rows, 1}))
    return std::nullopt;
  std::optional<int64_t> arg = getEntryBlockArgNumber(splat.getSrc(), entry);
  if (!arg)
    return std::nullopt;
  return LoadRole{*arg, rows, cols, *rowBase, rowStride};
}

std::optional<int64_t> matchRowMajorF16Pointer(Value pointer, Block &entry,
                                               int64_t rows, int64_t cols,
                                               int64_t rowStride) {
  std::optional<LoadRole> access =
      matchRowMajorF16PointerAccess(pointer, entry, rows, cols, rowStride);
  if (!access || access->rowBase != 0)
    return std::nullopt;
  return access->arg;
}

std::optional<LoadRole> getLoadedF16MatrixRole(Value value, Block &entry,
                                               int64_t rowStride) {
  auto load = value.getDefiningOp<triton::LoadOp>();
  if (!load || load.getMask() || load.getOther())
    return std::nullopt;

  auto type = dyn_cast<RankedTensorType>(load.getResult().getType());
  if (!type || type.getRank() != 2 ||
      type.getElementType() != Float16Type::get(value.getContext()))
    return std::nullopt;

  return matchRowMajorF16PointerAccess(load.getPtr(), entry, type.getShape()[0],
                                       type.getShape()[1], rowStride);
}

std::optional<int64_t> deriveWindowInputWidth(ArrayRef<int64_t> rowBases,
                                              int64_t kernelSize) {
  int64_t kernelArea = kernelSize * kernelSize;
  if (kernelSize <= 1 || static_cast<int64_t>(rowBases.size()) != kernelArea ||
      rowBases.empty() || rowBases.front() != 0)
    return std::nullopt;

  int64_t inputWidth = rowBases[kernelSize];
  if (inputWidth < kernelSize)
    return std::nullopt;

  for (int64_t index = 0; index < kernelArea; ++index) {
    int64_t ky = index / kernelSize;
    int64_t kx = index % kernelSize;
    if (rowBases[index] != ky * inputWidth + kx)
      return std::nullopt;
  }
  return inputWidth;
}

std::optional<DotShape> getF16DotShape(triton::DotOp dot) {
  auto lhsType = dyn_cast<RankedTensorType>(dot.getA().getType());
  auto rhsType = dyn_cast<RankedTensorType>(dot.getB().getType());
  auto resultType = dyn_cast<RankedTensorType>(dot.getD().getType());
  if (!lhsType || !rhsType || !resultType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || resultType.getRank() != 2 ||
      lhsType.getElementType() != Float16Type::get(dot.getContext()) ||
      rhsType.getElementType() != Float16Type::get(dot.getContext()) ||
      resultType.getElementType() != Float32Type::get(dot.getContext()))
    return std::nullopt;

  int64_t m = lhsType.getShape()[0];
  int64_t k = lhsType.getShape()[1];
  int64_t rhsK = rhsType.getShape()[0];
  int64_t n = rhsType.getShape()[1];
  if (m <= 0 || k <= 0 || n <= 0 || rhsK != k ||
      resultType.getShape()[0] != m || resultType.getShape()[1] != n)
    return std::nullopt;

  return DotShape{m, k, n};
}

bool isZeroLike(Value value, ArrayRef<int64_t> shape) {
  return isZeroF32Tensor(value, shape);
}

std::optional<Value> getReluResult(Value source, ArrayRef<int64_t> shape) {
  Value found;
  for (Operation *user : source.getUsers()) {
    Value lhs;
    Value rhs;
    Value result;
    if (auto max = dyn_cast<arith::MaxNumFOp>(user)) {
      lhs = max.getLhs();
      rhs = max.getRhs();
      result = max.getResult();
    } else if (auto max = dyn_cast<arith::MaximumFOp>(user)) {
      lhs = max.getLhs();
      rhs = max.getRhs();
      result = max.getResult();
    } else {
      continue;
    }

    if (!isRankedTensorOf(result, Float32Type::get(source.getContext()), shape))
      return std::nullopt;
    bool matches = (lhs == source && isZeroLike(rhs, shape)) ||
                   (rhs == source && isZeroLike(lhs, shape));
    if (!matches || found)
      return std::nullopt;
    found = result;
  }
  if (!found)
    return std::nullopt;
  return found;
}

std::optional<Value> getTruncF16Result(Value source, ArrayRef<int64_t> shape) {
  Value found;
  for (Operation *user : source.getUsers()) {
    auto trunc = dyn_cast<arith::TruncFOp>(user);
    if (!trunc || trunc.getIn() != source)
      continue;
    if (!isRankedTensorOf(trunc.getResult(),
                          Float16Type::get(source.getContext()), shape) ||
        found)
      return std::nullopt;
    found = trunc.getResult();
  }
  if (!found)
    return std::nullopt;
  return found;
}

std::optional<Value> getReluTruncF16Result(Value source,
                                           ArrayRef<int64_t> shape) {
  std::optional<Value> relu = getReluResult(source, shape);
  if (!relu)
    return std::nullopt;
  return getTruncF16Result(*relu, shape);
}

bool isExtOf(Value value, Value source, ArrayRef<int64_t> shape) {
  auto ext = value.getDefiningOp<arith::ExtFOp>();
  return ext && ext.getIn() == source &&
         isRankedTensorOf(ext.getResult(), Float32Type::get(value.getContext()),
                          shape);
}

bool isAllowedNNTopLevelOp(Operation &op) {
  if (op.getNumRegions() != 0)
    return false;
  return isa<triton::MakeRangeOp, triton::ExpandDimsOp, triton::SplatOp,
             triton::AddPtrOp, triton::BroadcastOp, triton::LoadOp,
             triton::DotOp, triton::StoreOp, triton::ReturnOp,
             arith::ConstantOp, arith::MulIOp, arith::AddIOp, arith::AddFOp,
             arith::MaxNumFOp, arith::MaximumFOp, arith::ExtFOp,
             arith::TruncFOp>(&op);
}

FailureOr<triton::SplatOp> getSingleSplatUser(Value scalar, Type elementType,
                                              int64_t nElements) {
  triton::SplatOp found;
  for (Operation *user : scalar.getUsers()) {
    auto splat = dyn_cast<triton::SplatOp>(user);
    if (!splat || found ||
        !isRankedVectorOf(splat.getResult(), elementType, nElements))
      return failure();
    found = splat;
  }
  if (!found)
    return failure();
  return found;
}

bool isReduceBlockArgPair(Value lhs, Value rhs, Block &block) {
  auto lhsArg = dyn_cast<BlockArgument>(lhs);
  auto rhsArg = dyn_cast<BlockArgument>(rhs);
  if (!lhsArg || !rhsArg || lhsArg.getOwner() != &block ||
      rhsArg.getOwner() != &block)
    return false;
  return ((lhsArg.getArgNumber() == 0 && rhsArg.getArgNumber() == 1) ||
          (lhsArg.getArgNumber() == 1 && rhsArg.getArgNumber() == 0));
}

FailureOr<ReductionKind> classifyScalarReduction(triton::ReduceOp reduce) {
  if (reduce.getAxis() != 0 || reduce->getNumOperands() != 1 ||
      reduce->getNumResults() != 1 ||
      reduce->getResult(0).getType() != Float32Type::get(reduce.getContext()))
    return failure();

  Region &combine = reduce.getCombineOp();
  if (!combine.hasOneBlock())
    return failure();
  Block &block = combine.front();
  if (block.getNumArguments() != 2)
    return failure();
  for (BlockArgument arg : block.getArguments()) {
    if (arg.getType() != Float32Type::get(reduce.getContext()))
      return failure();
  }

  auto ret = dyn_cast<triton::ReduceReturnOp>(block.getTerminator());
  if (!ret || ret->getNumOperands() != 1)
    return failure();

  Operation *producer = ret->getOperand(0).getDefiningOp();
  if (!producer || producer->getBlock() != &block ||
      &block.front() != producer || producer->getNextNode() != ret)
    return failure();

  if (auto max = dyn_cast<arith::MaxNumFOp>(producer)) {
    if (isReduceBlockArgPair(max.getLhs(), max.getRhs(), block))
      return ReductionKind::Max;
  } else if (auto max = dyn_cast<arith::MaximumFOp>(producer)) {
    if (isReduceBlockArgPair(max.getLhs(), max.getRhs(), block))
      return ReductionKind::Max;
  } else if (auto add = dyn_cast<arith::AddFOp>(producer)) {
    if (isReduceBlockArgPair(add.getLhs(), add.getRhs(), block))
      return ReductionKind::Sum;
  }
  return failure();
}

std::optional<int64_t> getSplatI32Constant(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;

  Attribute attr = constant.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getInt();

  auto dense = dyn_cast<DenseIntElementsAttr>(attr);
  if (!dense || !dense.isSplat())
    return std::nullopt;
  return dense.getSplatValue<APInt>().getSExtValue();
}

std::optional<int64_t> getMaskLogicalN(Value mask, Value range,
                                       int64_t nElements) {
  auto cmp = mask.getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt ||
      cmp.getLhs() != range)
    return std::nullopt;

  auto logicalN = getSplatI32Constant(cmp.getRhs());
  if (!logicalN || *logicalN <= 0 || *logicalN > nElements)
    return std::nullopt;
  return logicalN;
}

static FailureOr<CompactVectorMemoryMatch>
recognizeCompactVectorMemory(triton::FuncOp func, Block &entry,
                             ArrayRef<triton::MakeRangeOp> ranges,
                             ArrayRef<triton::LoadOp> loads,
                             triton::StoreOp store, int64_t expectedInputs) {
  FunctionType functionType = func.getFunctionType();
  if (functionType.getNumInputs() != expectedInputs ||
      functionType.getNumResults() != 0)
    return failure();

  for (Type input : functionType.getInputs()) {
    if (!isF16Ptr(input))
      return failure();
  }

  if (ranges.size() != 1)
    return failure();

  triton::MakeRangeOp range = ranges.front();
  int64_t nElements = range.getEndAttr().getInt();
  if (range.getStartAttr().getInt() != 0 || nElements <= 0 ||
      !isRankedVectorOf(range.getResult(),
                        IntegerType::get(func.getContext(), 32), nElements))
    return failure();

  CompactVectorMemoryMatch match;
  match.nElements = nElements;
  match.logicalN = nElements;

  llvm::DenseMap<Value, int64_t> splatToArg;
  for (Operation &op : entry.getOperations()) {
    if (auto splat = dyn_cast<triton::SplatOp>(&op)) {
      Value src = splat.getSrc();
      if (auto blockArg = dyn_cast<BlockArgument>(src)) {
        if (blockArg.getOwner() == &entry)
          splatToArg[splat.getResult()] = blockArg.getArgNumber();
      }
    }
  }

  for (Operation &op : entry.getOperations()) {
    if (auto addPtr = dyn_cast<triton::AddPtrOp>(&op)) {
      auto found = splatToArg.find(addPtr.getPtr());
      if (found != splatToArg.end() && addPtr.getOffset() == range.getResult())
        match.ptrToArg[addPtr.getResult()] = found->second;
    }
  }

  Value mask = store.getMask();
  match.masked = static_cast<bool>(mask);
  if (match.masked) {
    auto maybeLogicalN = getMaskLogicalN(mask, range.getResult(), nElements);
    if (!maybeLogicalN || nElements % 16 != 0)
      return failure();
    match.logicalN = *maybeLogicalN;
  } else if (nElements % 16 != 0) {
    return failure();
  }

  for (triton::LoadOp load : loads) {
    auto found = match.ptrToArg.find(load.getPtr());
    if (found == match.ptrToArg.end())
      return failure();
    if (load.getMask() != mask)
      return failure();
    match.loadToArg[load.getResult()] = found->second;
  }

  return match;
}

static bool matchAddOp(Operation &op, Value &lhs, Value &rhs, Value &result) {
  auto add = dyn_cast<arith::AddFOp>(&op);
  if (!add)
    return false;
  lhs = add.getLhs();
  rhs = add.getRhs();
  result = add.getResult();
  return true;
}

static std::optional<exec::ExecutableCompactVectorBinaryOpcode>
classifyElementwise1DBinaryOp(Operation &op, Value &lhs, Value &rhs,
                              Value &result) {
  if (auto add = dyn_cast<arith::AddFOp>(&op)) {
    lhs = add.getLhs();
    rhs = add.getRhs();
    result = add.getResult();
    return exec::ExecutableCompactVectorBinaryOpcode::Add;
  }
  if (auto mul = dyn_cast<arith::MulFOp>(&op)) {
    lhs = mul.getLhs();
    rhs = mul.getRhs();
    result = mul.getResult();
    return exec::ExecutableCompactVectorBinaryOpcode::Mul;
  }
  return std::nullopt;
}

FailureOr<AddOperands> recognizeAdd(triton::FuncOp func) {
  if (func.getBody().empty() || !func.getBody().hasOneBlock())
    return failure();

  Block &entry = func.getBody().front();

  SmallVector<triton::MakeRangeOp> ranges;
  SmallVector<triton::LoadOp> loads;
  SmallVector<Operation *> binaryOps;
  SmallVector<triton::StoreOp> stores;
  for (Operation &op : entry.getOperations()) {
    if (op.getNumRegions() != 0)
      return failure();
    if (auto range = dyn_cast<triton::MakeRangeOp>(&op))
      ranges.push_back(range);
    else if (auto load = dyn_cast<triton::LoadOp>(&op))
      loads.push_back(load);
    else {
      Value lhs;
      Value rhs;
      Value result;
      if (matchAddOp(op, lhs, rhs, result))
        binaryOps.push_back(&op);
      else if (auto store = dyn_cast<triton::StoreOp>(&op))
        stores.push_back(store);
      else if (isa<triton::SplatOp, triton::AddPtrOp, triton::ReturnOp,
                   arith::ConstantOp, arith::CmpIOp, arith::TruncFOp>(&op))
        continue;
      else
        return failure();
    }
  }

  if (ranges.size() != 1 || loads.size() != 2 || binaryOps.size() != 1 ||
      stores.size() != 1)
    return failure();

  triton::StoreOp store = stores.front();
  FailureOr<CompactVectorMemoryMatch> memory =
      recognizeCompactVectorMemory(func, entry, ranges, loads, store,
                                   /*expectedInputs=*/3);
  if (failed(memory))
    return failure();
  int64_t nElements = memory->nElements;

  Value binaryLhs;
  Value binaryRhs;
  Value binaryResult;
  if (!matchAddOp(*binaryOps.front(), binaryLhs, binaryRhs, binaryResult))
    return failure();
  if (!isRankedVectorOf(binaryResult, Float16Type::get(func.getContext()),
                        nElements))
    return failure();

  if (store.getValue() != binaryResult)
    return failure();

  auto lhs = memory->loadToArg.find(binaryLhs);
  auto rhs = memory->loadToArg.find(binaryRhs);
  auto out = memory->ptrToArg.find(store.getPtr());
  if (lhs == memory->loadToArg.end() || rhs == memory->loadToArg.end() ||
      out == memory->ptrToArg.end())
    return failure();

  AddOperands operands;
  operands.anchor = opAnchor(store.getOperation());
  operands.n = nElements;
  operands.logicalN = memory->logicalN;
  operands.masked = memory->masked;
  operands.out = out->second;
  operands.lhs = lhs->second;
  operands.rhs = rhs->second;
  return operands;
}

static FailureOr<Elementwise1DValueGraphMatch>
recognizeElementwise1DValueGraph(triton::FuncOp func) {
  if (func.getBody().empty() || !func.getBody().hasOneBlock())
    return failure();

  Block &entry = func.getBody().front();

  SmallVector<triton::MakeRangeOp> ranges;
  SmallVector<triton::LoadOp> loads;
  SmallVector<Operation *> binaryOps;
  SmallVector<triton::StoreOp> stores;
  for (Operation &op : entry.getOperations()) {
    if (op.getNumRegions() != 0)
      return failure();
    if (auto range = dyn_cast<triton::MakeRangeOp>(&op)) {
      ranges.push_back(range);
    } else if (auto load = dyn_cast<triton::LoadOp>(&op)) {
      loads.push_back(load);
    } else {
      Value lhs;
      Value rhs;
      Value result;
      if (classifyElementwise1DBinaryOp(op, lhs, rhs, result))
        binaryOps.push_back(&op);
      else if (auto store = dyn_cast<triton::StoreOp>(&op))
        stores.push_back(store);
      else if (isa<triton::SplatOp, triton::AddPtrOp, triton::ReturnOp,
                   arith::ConstantOp, arith::CmpIOp, arith::TruncFOp>(&op))
        continue;
      else
        return failure();
    }
  }

  if (loads.size() < 2 || loads.size() > 4 || binaryOps.empty() ||
      binaryOps.size() > 3 || stores.size() != 1)
    return failure();

  triton::StoreOp store = stores.front();
  FailureOr<CompactVectorMemoryMatch> memory =
      recognizeCompactVectorMemory(func, entry, ranges, loads, store,
                                   /*expectedInputs=*/loads.size() + 1);
  // 256 = one full V256 (the hardware vector width), the largest contig
  // load_contig<1> on real hardware via the board NVEC fallback (see
  // RPU_BOARD_LOAD_CONTIG_NVEC gate in RPUExecutableEmitter::emitLoadContig).
  // The default legacy path emits load_contig<n> with n in element count, so
  // 256 still fits in that path's tile shape.
  if (failed(memory) || memory->nElements > 256)
    return failure();

  Elementwise1DValueGraphMatch graph;
  graph.anchor = opAnchor(store.getOperation());
  graph.n = memory->nElements;
  graph.logicalN = memory->logicalN;
  graph.masked = memory->masked;

  llvm::DenseMap<Value, int64_t> valueToSlot;
  for (triton::LoadOp load : loads) {
    auto found = memory->loadToArg.find(load.getResult());
    if (found == memory->loadToArg.end())
      return failure();
    valueToSlot[load.getResult()] = graph.inputArgIndices.size();
    graph.inputArgIndices.push_back(static_cast<unsigned>(found->second));
  }

  Value finalResult;
  for (Operation *op : binaryOps) {
    Value lhsValue;
    Value rhsValue;
    Value resultValue;
    std::optional<exec::ExecutableCompactVectorBinaryOpcode> opcode =
        classifyElementwise1DBinaryOp(*op, lhsValue, rhsValue, resultValue);
    if (!opcode ||
        !isRankedVectorOf(resultValue, Float16Type::get(func.getContext()),
                          memory->nElements))
      return failure();

    auto lhs = valueToSlot.find(lhsValue);
    auto rhs = valueToSlot.find(rhsValue);
    if (lhs == valueToSlot.end() || rhs == valueToSlot.end())
      return failure();

    graph.ops.push_back(exec::ExecutableCompactVectorBinaryBuildOp{
        *opcode, lhs->second, rhs->second});
    valueToSlot[resultValue] =
        graph.inputArgIndices.size() + graph.ops.size() - 1;
    finalResult = resultValue;
  }

  if (!finalResult || store.getValue() != finalResult)
    return failure();

  auto out = memory->ptrToArg.find(store.getPtr());
  if (out == memory->ptrToArg.end())
    return failure();
  graph.outputArgIndex = out->second;
  return graph;
}

FailureOr<Elementwise16ValueMapLoweringRequest>
recognizeElementwise16ValueMapRequest(triton::FuncOp func) {
  FailureOr<Elementwise1DValueGraphMatch> graph =
      recognizeElementwise1DValueGraph(func);
  if (failed(graph))
    return failure();

  Elementwise16ValueMapLoweringRequest request;
  request.n = graph->n;
  request.logicalN = graph->logicalN;
  request.masked = graph->masked;
  request.outputArgIndex = static_cast<unsigned>(graph->outputArgIndex);
  request.inputArgIndices = graph->inputArgIndices;
  request.ops.append(graph->ops.begin(), graph->ops.end());
  return request;
}

FailureOr<Elementwise1DExecutableRequest>
recognizeElementwise1DExecutableRequest(triton::FuncOp func) {
  FailureOr<Elementwise1DValueGraphMatch> graph =
      recognizeElementwise1DValueGraph(func);
  if (failed(graph))
    return failure();

  Elementwise1DExecutableRequest request;
  request.anchor = graph->anchor;
  request.n = graph->n;
  request.logicalN = graph->logicalN;
  request.masked = graph->masked;
  request.outputArgIndex = static_cast<unsigned>(graph->outputArgIndex);
  request.inputArgIndices = graph->inputArgIndices;
  request.ops.append(graph->ops.begin(), graph->ops.end());
  return request;
}

FailureOr<GemmOperands> recognizeGemm(triton::FuncOp func) {
  FunctionType functionType = func.getFunctionType();
  if (functionType.getNumInputs() != 3 || functionType.getNumResults() != 0)
    return failure();

  for (Type input : functionType.getInputs()) {
    if (!isF16Ptr(input))
      return failure();
  }
  if (func.getBody().empty() || !func.getBody().hasOneBlock())
    return failure();

  Block &entry = func.getBody().front();

  SmallVector<triton::LoadOp> loads;
  SmallVector<triton::DotOp> dots;
  SmallVector<triton::StoreOp> stores;
  for (Operation &op : entry.getOperations()) {
    if (op.getNumRegions() != 0)
      return failure();
    if (auto load = dyn_cast<triton::LoadOp>(&op))
      loads.push_back(load);
    else if (auto dot = dyn_cast<triton::DotOp>(&op))
      dots.push_back(dot);
    else if (auto store = dyn_cast<triton::StoreOp>(&op))
      stores.push_back(store);
    else if (isa<triton::MakeRangeOp, triton::ExpandDimsOp, triton::SplatOp,
                 triton::AddPtrOp, triton::BroadcastOp, triton::ReturnOp,
                 arith::ConstantOp, arith::MulIOp, arith::TruncFOp>(&op))
      continue;
    else
      return failure();
  }

  if (loads.size() != 2 || dots.size() != 1 || stores.size() != 1)
    return failure();

  triton::DotOp dot = dots.front();
  auto lhsType = dyn_cast<RankedTensorType>(dot.getA().getType());
  auto rhsType = dyn_cast<RankedTensorType>(dot.getB().getType());
  auto resultType = dyn_cast<RankedTensorType>(dot.getD().getType());
  if (!lhsType || !rhsType || !resultType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || resultType.getRank() != 2 ||
      lhsType.getElementType() != Float16Type::get(func.getContext()) ||
      rhsType.getElementType() != Float16Type::get(func.getContext()) ||
      resultType.getElementType() != Float32Type::get(func.getContext()))
    return failure();

  int64_t m = lhsType.getShape()[0];
  int64_t k = lhsType.getShape()[1];
  int64_t rhsK = rhsType.getShape()[0];
  int64_t n = rhsType.getShape()[1];
  if (m <= 0 || k <= 0 || n <= 0 || rhsK != k ||
      resultType.getShape()[0] != m || resultType.getShape()[1] != n)
    return failure();
  if (!isZeroF32Tensor(dot.getC(), {m, n}))
    return failure();

  llvm::DenseMap<Value, triton::LoadOp> loadByResult;
  for (triton::LoadOp load : loads) {
    if (load.getMask())
      return failure();
    loadByResult[load.getResult()] = load;
  }

  auto lhsLoad = loadByResult.find(dot.getA());
  auto rhsLoad = loadByResult.find(dot.getB());
  if (lhsLoad == loadByResult.end() || rhsLoad == loadByResult.end())
    return failure();
  if (!isRankedTensorOf(lhsLoad->second.getResult(),
                        Float16Type::get(func.getContext()), {m, k}) ||
      !isRankedTensorOf(rhsLoad->second.getResult(),
                        Float16Type::get(func.getContext()), {k, n}))
    return failure();

  triton::StoreOp store = stores.front();
  if (store.getMask())
    return failure();

  Value storeValue = store.getValue();
  if (storeValue != dot.getD()) {
    auto trunc = storeValue.getDefiningOp<arith::TruncFOp>();
    if (!trunc || trunc.getIn() != dot.getD() ||
        !isRankedTensorOf(trunc.getResult(),
                          Float16Type::get(func.getContext()), {m, n}))
      return failure();
  }
  if (!isRankedTensorOfF16Ptr(store.getPtr(), {m, n}))
    return failure();

  std::optional<int64_t> lhsArg =
      matchRowMajorF16Pointer(lhsLoad->second.getPtr(), entry, m, k,
                              /*rowStride=*/k);
  std::optional<int64_t> rhsArg =
      matchRowMajorF16Pointer(rhsLoad->second.getPtr(), entry, k, n,
                              /*rowStride=*/n);
  std::optional<int64_t> outArg =
      matchRowMajorF16Pointer(store.getPtr(), entry, m, n, /*rowStride=*/n);
  if (!lhsArg || !rhsArg || !outArg)
    return failure();

  GemmOperands operands;
  operands.anchor = opAnchor(store.getOperation());
  operands.m = m;
  operands.n = n;
  operands.k = k;
  operands.out = *outArg;
  operands.lhs = *lhsArg;
  operands.rhs = *rhsArg;
  return operands;
}

FailureOr<SoftmaxOperands> recognizeSoftmax(triton::FuncOp func) {
  FunctionType functionType = func.getFunctionType();
  if (functionType.getNumInputs() != 2 || functionType.getNumResults() != 0)
    return failure();

  for (Type input : functionType.getInputs()) {
    if (!isF16Ptr(input))
      return failure();
  }
  if (func.getBody().empty() || !func.getBody().hasOneBlock())
    return failure();

  Block &entry = func.getBody().front();

  SmallVector<triton::MakeRangeOp> ranges;
  SmallVector<triton::LoadOp> loads;
  SmallVector<triton::ReduceOp> reductions;
  SmallVector<arith::ExtFOp> exts;
  SmallVector<arith::SubFOp> subs;
  SmallVector<Operation *> exps;
  SmallVector<arith::DivFOp> divs;
  SmallVector<arith::TruncFOp> truncs;
  SmallVector<triton::StoreOp> stores;

  for (Operation &op : entry.getOperations()) {
    if (auto range = dyn_cast<triton::MakeRangeOp>(&op))
      ranges.push_back(range);
    else if (auto load = dyn_cast<triton::LoadOp>(&op))
      loads.push_back(load);
    else if (auto reduce = dyn_cast<triton::ReduceOp>(&op))
      reductions.push_back(reduce);
    else if (op.getNumRegions() != 0)
      return failure();
    else if (auto ext = dyn_cast<arith::ExtFOp>(&op))
      exts.push_back(ext);
    else if (auto sub = dyn_cast<arith::SubFOp>(&op))
      subs.push_back(sub);
    else if (isa<math::ExpOp>(&op) || op.getName().getStringRef() == "math.exp")
      exps.push_back(&op);
    else if (auto div = dyn_cast<arith::DivFOp>(&op))
      divs.push_back(div);
    else if (auto trunc = dyn_cast<arith::TruncFOp>(&op))
      truncs.push_back(trunc);
    else if (auto store = dyn_cast<triton::StoreOp>(&op))
      stores.push_back(store);
    else if (isa<triton::SplatOp, triton::AddPtrOp, triton::ReturnOp>(&op))
      continue;
    else
      return failure();
  }

  if (ranges.size() != 1 || loads.size() != 1 || reductions.size() != 2 ||
      exts.size() != 1 || subs.size() != 1 || exps.size() != 1 ||
      truncs.size() != 1 || stores.size() != 1)
    return failure();

  triton::MakeRangeOp range = ranges.front();
  int64_t nElements = range.getEndAttr().getInt();
  if (range.getStartAttr().getInt() != 0 || nElements <= 0 ||
      nElements % 16 != 0 ||
      !isRankedVectorOf(range.getResult(),
                        IntegerType::get(func.getContext(), 32), nElements))
    return failure();

  triton::LoadOp load = loads.front();
  if (load.getMask() || load.getOther() ||
      !isRankedVectorOf(load.getResult(), Float16Type::get(func.getContext()),
                        nElements))
    return failure();

  arith::ExtFOp ext = exts.front();
  if (ext.getIn() != load.getResult() ||
      !isRankedVectorOf(ext.getResult(), Float32Type::get(func.getContext()),
                        nElements))
    return failure();

  Value maxScalar;
  Value sumScalar;
  for (triton::ReduceOp reduce : reductions) {
    FailureOr<ReductionKind> kind = classifyScalarReduction(reduce);
    if (failed(kind))
      return failure();
    if (*kind == ReductionKind::Max) {
      if (maxScalar || reduce->getOperand(0) != ext.getResult())
        return failure();
      maxScalar = reduce->getResult(0);
    }
  }
  if (!maxScalar)
    return failure();

  FailureOr<triton::SplatOp> maybeMaxSplat = getSingleSplatUser(
      maxScalar, Float32Type::get(func.getContext()), nElements);
  if (failed(maybeMaxSplat))
    return failure();
  triton::SplatOp maxSplat = *maybeMaxSplat;

  arith::SubFOp sub = subs.front();
  if (sub.getLhs() != ext.getResult() || sub.getRhs() != maxSplat.getResult() ||
      !isRankedVectorOf(sub.getResult(), Float32Type::get(func.getContext()),
                        nElements))
    return failure();

  Operation *exp = exps.front();
  if (exp->getNumOperands() != 1 || exp->getNumResults() != 1 ||
      exp->getOperand(0) != sub.getResult() ||
      !isRankedVectorOf(exp->getResult(0), Float32Type::get(func.getContext()),
                        nElements))
    return failure();

  for (triton::ReduceOp reduce : reductions) {
    FailureOr<ReductionKind> kind = classifyScalarReduction(reduce);
    if (failed(kind))
      return failure();
    if (*kind == ReductionKind::Sum) {
      if (sumScalar || reduce->getOperand(0) != exp->getResult(0))
        return failure();
      sumScalar = reduce->getResult(0);
    }
  }
  if (!sumScalar)
    return failure();

  FailureOr<triton::SplatOp> maybeSumSplat = getSingleSplatUser(
      sumScalar, Float32Type::get(func.getContext()), nElements);
  if (failed(maybeSumSplat))
    return failure();
  triton::SplatOp sumSplat = *maybeSumSplat;

  if (divs.size() != 1)
    return failure();
  arith::DivFOp div = divs.front();
  if (div.getLhs() != exp->getResult(0) || div.getRhs() != sumSplat.getResult())
    return failure();
  Value normalized = div.getResult();
  if (!isRankedVectorOf(normalized, Float32Type::get(func.getContext()),
                        nElements))
    return failure();

  arith::TruncFOp trunc = truncs.front();
  if (trunc.getIn() != normalized ||
      !isRankedVectorOf(trunc.getResult(), Float16Type::get(func.getContext()),
                        nElements))
    return failure();

  triton::StoreOp store = stores.front();
  if (store.getMask() || store.getValue() != trunc.getResult() ||
      !isRankedTensorOfF16Ptr(store.getPtr(), {nElements}))
    return failure();

  llvm::DenseMap<Value, int64_t> ptrToArg;
  for (Operation &op : entry.getOperations()) {
    if (auto splat = dyn_cast<triton::SplatOp>(&op)) {
      if (std::optional<int64_t> arg =
              getEntryBlockArgNumber(splat.getSrc(), entry))
        ptrToArg[splat.getResult()] = *arg;
    } else if (auto addPtr = dyn_cast<triton::AddPtrOp>(&op)) {
      auto found = ptrToArg.find(addPtr.getPtr());
      if (found != ptrToArg.end() && addPtr.getOffset() == range.getResult())
        ptrToArg[addPtr.getResult()] = found->second;
    }
  }

  auto inputArg = ptrToArg.find(load.getPtr());
  auto outArg = ptrToArg.find(store.getPtr());
  if (inputArg == ptrToArg.end() || outArg == ptrToArg.end())
    return failure();

  SoftmaxOperands operands;
  operands.anchor = opAnchor(store.getOperation());
  operands.n = nElements;
  operands.out = outArg->second;
  operands.input = inputArg->second;
  return operands;
}

FailureOr<ConvKxKOperands> recognizeConvKxK(triton::FuncOp func) {
  FunctionType functionType = func.getFunctionType();
  if (functionType.getNumInputs() != 3 || functionType.getNumResults() != 0)
    return failure();
  for (Type input : functionType.getInputs()) {
    if (!isF16Ptr(input))
      return failure();
  }
  if (func.getBody().empty() || !func.getBody().hasOneBlock())
    return failure();

  Block &entry = func.getBody().front();
  SmallVector<triton::DotOp> dots;
  SmallVector<triton::StoreOp> stores;
  for (Operation &op : entry.getOperations()) {
    if (!isAllowedNNTopLevelOp(op))
      return failure();
    if (auto dot = dyn_cast<triton::DotOp>(&op))
      dots.push_back(dot);
    else if (auto store = dyn_cast<triton::StoreOp>(&op))
      stores.push_back(store);
  }

  int64_t kernelArea = dots.size();
  int64_t kernelSize = 0;
  while (kernelSize * kernelSize < kernelArea)
    ++kernelSize;
  if (kernelArea <= 1 || kernelSize * kernelSize != kernelArea ||
      stores.size() != 1)
    return failure();

  std::optional<DotShape> firstShape = getF16DotShape(dots.front());
  if (!firstShape)
    return failure();
  int64_t m = firstShape->m;
  int64_t inChannels = firstShape->k;
  int64_t outChannels = firstShape->n;

  std::optional<LoadRole> firstInput =
      getLoadedF16MatrixRole(dots.front().getA(), entry, inChannels);
  std::optional<LoadRole> firstWeight =
      getLoadedF16MatrixRole(dots.front().getB(), entry, outChannels);
  if (!firstInput || !firstWeight || firstInput->rows != m ||
      firstInput->cols != inChannels || firstInput->rowBase != 0 ||
      firstWeight->rows != inChannels || firstWeight->cols != outChannels ||
      firstWeight->rowBase != 0)
    return failure();

  std::optional<Value> acc;
  SmallVector<int64_t> inputRowBases;
  for (auto [dotIndex, dot] : llvm::enumerate(dots)) {
    std::optional<DotShape> shape = getF16DotShape(dot);
    if (!shape || shape->m != m || shape->k != inChannels ||
        shape->n != outChannels)
      return failure();

    std::optional<LoadRole> input =
        getLoadedF16MatrixRole(dot.getA(), entry, inChannels);
    std::optional<LoadRole> weight =
        getLoadedF16MatrixRole(dot.getB(), entry, outChannels);
    if (!input || !weight || input->arg != firstInput->arg ||
        input->rows != m || input->cols != inChannels ||
        weight->arg != firstWeight->arg || weight->rows != inChannels ||
        weight->cols != outChannels ||
        weight->rowBase != static_cast<int64_t>(dotIndex) * inChannels)
      return failure();
    inputRowBases.push_back(input->rowBase);

    if (acc) {
      if (dot.getC() != *acc)
        return failure();
    } else if (!isZeroLike(dot.getC(), {m, outChannels})) {
      return failure();
    }
    acc = dot.getD();
  }
  std::optional<int64_t> inputWidth =
      deriveWindowInputWidth(inputRowBases, kernelSize);
  if (!inputWidth)
    return failure();

  triton::StoreOp store = stores.front();
  if (store.getMask())
    return failure();
  auto trunc = store.getValue().getDefiningOp<arith::TruncFOp>();
  if (!trunc || trunc.getIn() != *acc ||
      !isRankedTensorOf(trunc.getResult(), Float16Type::get(func.getContext()),
                        {m, outChannels}))
    return failure();

  std::optional<LoadRole> outAccess = matchRowMajorF16PointerAccess(
      store.getPtr(), entry, m, outChannels, outChannels);
  if (!outAccess || outAccess->rowBase != 0)
    return failure();

  ConvKxKOperands operands;
  operands.anchor = opAnchor(store.getOperation());
  operands.kernelSize = kernelSize;
  operands.m = m;
  operands.inChannels = inChannels;
  operands.outChannels = outChannels;
  operands.inputWidth = *inputWidth;
  operands.out = outAccess->arg;
  operands.input = firstInput->arg;
  operands.weight = firstWeight->arg;
  return operands;
}

FailureOr<ResNetBlockOperands> recognizeResNetBlock(triton::FuncOp func) {
  FunctionType functionType = func.getFunctionType();
  if (functionType.getNumInputs() != 4 || functionType.getNumResults() != 0)
    return failure();
  for (Type input : functionType.getInputs()) {
    if (!isF16Ptr(input))
      return failure();
  }
  if (func.getBody().empty() || !func.getBody().hasOneBlock())
    return failure();

  Block &entry = func.getBody().front();
  SmallVector<triton::DotOp> dots;
  SmallVector<triton::StoreOp> stores;
  for (Operation &op : entry.getOperations()) {
    if (!isAllowedNNTopLevelOp(op))
      return failure();
    if (auto dot = dyn_cast<triton::DotOp>(&op))
      dots.push_back(dot);
    else if (auto store = dyn_cast<triton::StoreOp>(&op))
      stores.push_back(store);
  }
  if (dots.size() != 2 || stores.size() != 1)
    return failure();

  std::optional<DotShape> dot1Shape = getF16DotShape(dots[0]);
  std::optional<DotShape> dot2Shape = getF16DotShape(dots[1]);
  if (!dot1Shape || !dot2Shape)
    return failure();
  int64_t m = dot1Shape->m;
  int64_t channels = dot1Shape->k;
  int64_t hidden = dot1Shape->n;
  if (dot2Shape->m != m || dot2Shape->k != hidden || dot2Shape->n != channels)
    return failure();

  std::optional<LoadRole> x =
      getLoadedF16MatrixRole(dots[0].getA(), entry, channels);
  std::optional<LoadRole> w1 =
      getLoadedF16MatrixRole(dots[0].getB(), entry, hidden);
  std::optional<LoadRole> w2 =
      getLoadedF16MatrixRole(dots[1].getB(), entry, channels);
  if (!x || !w1 || !w2 || x->rows != m || x->cols != channels ||
      x->rowBase != 0 || w1->rows != channels || w1->cols != hidden ||
      w1->rowBase != 0 || w2->rows != hidden || w2->cols != channels ||
      w2->rowBase != 0)
    return failure();

  std::optional<Value> relu1Fp16 =
      getReluTruncF16Result(dots[0].getD(), {m, hidden});
  if (!relu1Fp16 || dots[1].getA() != *relu1Fp16)
    return failure();

  Value skipF32;
  for (Operation *user : dots[0].getA().getUsers()) {
    if (auto ext = dyn_cast<arith::ExtFOp>(user)) {
      if (isExtOf(ext.getResult(), dots[0].getA(), {m, channels})) {
        if (skipF32)
          return failure();
        skipF32 = ext.getResult();
      }
    }
  }
  if (!skipF32)
    return failure();

  if (!isZeroLike(dots[0].getC(), {m, hidden}) || dots[1].getC() != skipF32)
    return failure();

  std::optional<Value> relu2 = getReluResult(dots[1].getD(), {m, channels});
  if (!relu2)
    return failure();
  std::optional<Value> storeValue = getTruncF16Result(*relu2, {m, channels});
  if (!storeValue)
    return failure();

  triton::StoreOp store = stores.front();
  if (store.getMask() || store.getValue() != *storeValue)
    return failure();
  std::optional<LoadRole> outAccess = matchRowMajorF16PointerAccess(
      store.getPtr(), entry, m, channels, channels);
  if (!outAccess || outAccess->rowBase != 0)
    return failure();

  ResNetBlockOperands operands;
  operands.anchor = opAnchor(store.getOperation());
  operands.m = m;
  operands.channels = channels;
  operands.hidden = hidden;
  operands.out = outAccess->arg;
  operands.x = x->arg;
  operands.w1 = w1->arg;
  operands.w2 = w2->arg;
  return operands;
}

FailureOr<ResNet50BottleneckOperands>
recognizeResNet50Bottleneck(triton::FuncOp func) {
  FunctionType functionType = func.getFunctionType();
  if (functionType.getNumInputs() != 5 || functionType.getNumResults() != 0)
    return failure();
  for (Type input : functionType.getInputs()) {
    if (!isF16Ptr(input))
      return failure();
  }
  if (func.getBody().empty() || !func.getBody().hasOneBlock())
    return failure();

  Block &entry = func.getBody().front();
  SmallVector<triton::DotOp> dots;
  SmallVector<triton::StoreOp> stores;
  for (Operation &op : entry.getOperations()) {
    if (!isAllowedNNTopLevelOp(op))
      return failure();
    if (auto dot = dyn_cast<triton::DotOp>(&op))
      dots.push_back(dot);
    else if (auto store = dyn_cast<triton::StoreOp>(&op))
      stores.push_back(store);
  }
  if (dots.size() < 3 || (dots.size() - 1) % 2 != 0 || stores.size() != 1)
    return failure();

  int64_t kernelArea = (dots.size() - 1) / 2;
  int64_t kernelSize = 0;
  while (kernelSize * kernelSize < kernelArea)
    ++kernelSize;
  if (kernelSize * kernelSize != kernelArea || kernelSize != 3)
    return failure();

  std::optional<DotShape> firstShape = getF16DotShape(dots[0]);
  if (!firstShape)
    return failure();
  int64_t m = firstShape->m;
  int64_t channels = firstShape->k;
  int64_t bottleneck = firstShape->n;

  std::optional<LoadRole> firstInput =
      getLoadedF16MatrixRole(dots[0].getA(), entry, channels);
  std::optional<LoadRole> w1 =
      getLoadedF16MatrixRole(dots[0].getB(), entry, bottleneck);
  if (!firstInput || !w1 || firstInput->rows != m ||
      firstInput->cols != channels || firstInput->rowBase != 0 ||
      w1->rows != channels || w1->cols != bottleneck || w1->rowBase != 0)
    return failure();

  std::optional<LoadRole> skipInput;
  std::optional<Value> conv2Acc;
  SmallVector<int64_t> inputRowBases;
  for (int64_t pairIndex = 0; pairIndex < kernelArea; ++pairIndex) {
    triton::DotOp conv1 = dots[pairIndex * 2];
    triton::DotOp conv2 = dots[pairIndex * 2 + 1];
    std::optional<DotShape> conv1Shape = getF16DotShape(conv1);
    std::optional<DotShape> conv2Shape = getF16DotShape(conv2);
    if (!conv1Shape || !conv2Shape || conv1Shape->m != m ||
        conv1Shape->k != channels || conv1Shape->n != bottleneck ||
        conv2Shape->m != m || conv2Shape->k != bottleneck ||
        conv2Shape->n != bottleneck)
      return failure();

    std::optional<LoadRole> input =
        getLoadedF16MatrixRole(conv1.getA(), entry, channels);
    std::optional<LoadRole> conv1W =
        getLoadedF16MatrixRole(conv1.getB(), entry, bottleneck);
    std::optional<LoadRole> conv2W =
        getLoadedF16MatrixRole(conv2.getB(), entry, bottleneck);
    if (!input || !conv1W || !conv2W || input->arg != firstInput->arg ||
        input->rows != m || input->cols != channels || conv1W->arg != w1->arg ||
        conv1W->rows != channels || conv1W->cols != bottleneck ||
        conv1W->rowBase != 0 || conv2W->rows != bottleneck ||
        conv2W->cols != bottleneck || conv2W->rowBase != pairIndex * bottleneck)
      return failure();
    inputRowBases.push_back(input->rowBase);
    if (pairIndex == 0) {
      skipInput = input;
    }

    std::optional<Value> relu1Fp16 =
        getReluTruncF16Result(conv1.getD(), {m, bottleneck});
    if (!relu1Fp16 || conv2.getA() != *relu1Fp16)
      return failure();

    if (!isZeroLike(conv1.getC(), {m, bottleneck}))
      return failure();
    if (conv2Acc) {
      if (conv2.getC() != *conv2Acc)
        return failure();
    } else if (!isZeroLike(conv2.getC(), {m, bottleneck})) {
      return failure();
    }
    conv2Acc = conv2.getD();
  }
  std::optional<int64_t> inputWidth =
      deriveWindowInputWidth(inputRowBases, kernelSize);
  if (!inputWidth)
    return failure();

  std::optional<LoadRole> w2 =
      getLoadedF16MatrixRole(dots[1].getB(), entry, bottleneck);
  if (!w2)
    return failure();
  for (int64_t pairIndex = 0; pairIndex < kernelArea; ++pairIndex) {
    std::optional<LoadRole> conv2W = getLoadedF16MatrixRole(
        dots[pairIndex * 2 + 1].getB(), entry, bottleneck);
    if (!conv2W || conv2W->arg != w2->arg || conv2W->rows != bottleneck ||
        conv2W->cols != bottleneck || conv2W->rowBase != pairIndex * bottleneck)
      return failure();
  }

  std::optional<Value> relu2Fp16 =
      getReluTruncF16Result(*conv2Acc, {m, bottleneck});
  if (!relu2Fp16)
    return failure();

  triton::DotOp finalDot = dots.back();
  std::optional<DotShape> finalShape = getF16DotShape(finalDot);
  std::optional<LoadRole> w3 =
      getLoadedF16MatrixRole(finalDot.getB(), entry, channels);
  if (!finalShape || finalShape->m != m || finalShape->k != bottleneck ||
      finalShape->n != channels || finalDot.getA() != *relu2Fp16 || !w3 ||
      w3->rows != bottleneck || w3->cols != channels || w3->rowBase != 0)
    return failure();

  Value skipF32;
  for (Operation *user : dots[0].getA().getUsers()) {
    if (auto ext = dyn_cast<arith::ExtFOp>(user)) {
      if (isExtOf(ext.getResult(), dots[0].getA(), {m, channels})) {
        if (skipF32)
          return failure();
        skipF32 = ext.getResult();
      }
    }
  }
  if (!skipF32)
    return failure();

  if (finalDot.getC() != skipF32)
    return failure();

  std::optional<Value> relu3 = getReluResult(finalDot.getD(), {m, channels});
  if (!relu3)
    return failure();
  std::optional<Value> storeValue = getTruncF16Result(*relu3, {m, channels});
  if (!storeValue)
    return failure();

  triton::StoreOp store = stores.front();
  if (store.getMask() || store.getValue() != *storeValue)
    return failure();
  std::optional<LoadRole> outAccess = matchRowMajorF16PointerAccess(
      store.getPtr(), entry, m, channels, channels);
  if (!outAccess || outAccess->rowBase != 0 || !skipInput)
    return failure();

  ResNet50BottleneckOperands operands;
  operands.anchor = opAnchor(store.getOperation());
  operands.kernelSize = kernelSize;
  operands.m = m;
  operands.channels = channels;
  operands.bottleneck = bottleneck;
  operands.inputWidth = *inputWidth;
  operands.out = outAccess->arg;
  operands.input = skipInput->arg;
  operands.w1 = w1->arg;
  operands.w2 = w2->arg;
  operands.w3 = w3->arg;
  return operands;
}

FailureOr<SqrtOperands> recognizeSqrt(triton::FuncOp func) {
  FunctionType functionType = func.getFunctionType();
  if (functionType.getNumInputs() != 2 || functionType.getNumResults() != 0)
    return failure();

  for (Type input : functionType.getInputs()) {
    if (!isF16Ptr(input))
      return failure();
  }
  if (func.getBody().empty() || !func.getBody().hasOneBlock())
    return failure();

  Block &entry = func.getBody().front();

  SmallVector<triton::MakeRangeOp> ranges;
  SmallVector<triton::LoadOp> loads;
  SmallVector<arith::ExtFOp> exts;
  SmallVector<Operation *> sqrts;
  SmallVector<arith::TruncFOp> truncs;
  SmallVector<triton::StoreOp> stores;

  for (Operation &op : entry.getOperations()) {
    if (auto range = dyn_cast<triton::MakeRangeOp>(&op))
      ranges.push_back(range);
    else if (auto load = dyn_cast<triton::LoadOp>(&op))
      loads.push_back(load);
    else if (op.getNumRegions() != 0)
      return failure();
    else if (auto ext = dyn_cast<arith::ExtFOp>(&op))
      exts.push_back(ext);
    else if (isa<math::SqrtOp>(&op) ||
             op.getName().getStringRef() == "math.sqrt")
      sqrts.push_back(&op);
    else if (auto trunc = dyn_cast<arith::TruncFOp>(&op))
      truncs.push_back(trunc);
    else if (auto store = dyn_cast<triton::StoreOp>(&op))
      stores.push_back(store);
    else if (isa<triton::SplatOp, triton::AddPtrOp, triton::ReturnOp,
                 arith::ConstantOp, arith::CmpIOp>(&op))
      continue;
    else
      return failure();
  }

  if (ranges.size() != 1 || loads.size() != 1 || exts.size() != 1 ||
      sqrts.size() != 1 || truncs.size() != 1 || stores.size() != 1)
    return failure();

  triton::StoreOp store = stores.front();
  FailureOr<CompactVectorMemoryMatch> memory =
      recognizeCompactVectorMemory(func, entry, ranges, loads, store,
                                   /*expectedInputs=*/2);
  if (failed(memory))
    return failure();
  int64_t nElements = memory->nElements;
  if (nElements <= 0 || nElements % 16 != 0)
    return failure();

  triton::LoadOp load = loads.front();
  if (load.getMask() || load.getOther() ||
      !isRankedVectorOf(load.getResult(), Float16Type::get(func.getContext()),
                        nElements))
    return failure();

  arith::ExtFOp ext = exts.front();
  if (ext.getIn() != load.getResult() ||
      !isRankedVectorOf(ext.getResult(), Float32Type::get(func.getContext()),
                        nElements))
    return failure();

  Operation *sqrt = sqrts.front();
  if (sqrt->getNumOperands() != 1 || sqrt->getNumResults() != 1 ||
      sqrt->getOperand(0) != ext.getResult() ||
      !isRankedVectorOf(sqrt->getResult(0), Float32Type::get(func.getContext()),
                        nElements))
    return failure();

  arith::TruncFOp trunc = truncs.front();
  if (trunc.getIn() != sqrt->getResult(0) ||
      !isRankedVectorOf(trunc.getResult(), Float16Type::get(func.getContext()),
                        nElements))
    return failure();

  if (store.getMask() || store.getValue() != trunc.getResult())
    return failure();

  auto inputArg = memory->loadToArg.find(load.getResult());
  auto outArg = memory->ptrToArg.find(store.getPtr());
  if (inputArg == memory->loadToArg.end() || outArg == memory->ptrToArg.end())
    return failure();

  SqrtOperands operands;
  operands.anchor = opAnchor(store.getOperation());
  operands.n = nElements;
  operands.out = outArg->second;
  operands.input = inputArg->second;
  return operands;
}

FailureOr<ReduceSumAllOperands> recognizeReduceSumAll(triton::FuncOp func) {
  FunctionType functionType = func.getFunctionType();
  if (functionType.getNumInputs() != 2 || functionType.getNumResults() != 0)
    return failure();

  for (Type input : functionType.getInputs()) {
    if (!isF16Ptr(input))
      return failure();
  }
  if (func.getBody().empty() || !func.getBody().hasOneBlock())
    return failure();

  Block &entry = func.getBody().front();

  SmallVector<triton::MakeRangeOp> ranges;
  SmallVector<triton::LoadOp> loads;
  SmallVector<triton::ReduceOp> reductions;
  SmallVector<triton::StoreOp> stores;

  for (Operation &op : entry.getOperations()) {
    if (auto range = dyn_cast<triton::MakeRangeOp>(&op))
      ranges.push_back(range);
    else if (auto load = dyn_cast<triton::LoadOp>(&op))
      loads.push_back(load);
    else if (auto reduce = dyn_cast<triton::ReduceOp>(&op))
      reductions.push_back(reduce);
    else if (op.getNumRegions() != 0)
      return failure();
    else if (auto store = dyn_cast<triton::StoreOp>(&op))
      stores.push_back(store);
    else if (isa<triton::SplatOp, triton::AddPtrOp, triton::ReturnOp,
                 arith::ConstantOp, arith::CmpIOp, arith::TruncFOp>(&op))
      continue;
    else
      return failure();
  }

  if (ranges.size() != 1 || loads.size() != 1 || reductions.size() != 1 ||
      stores.size() != 1)
    return failure();

  triton::MakeRangeOp range = ranges.front();
  int64_t nElements = range.getEndAttr().getInt();
  if (range.getStartAttr().getInt() != 0 || nElements <= 0 ||
      nElements % 16 != 0 ||
      !isRankedVectorOf(range.getResult(),
                        IntegerType::get(func.getContext(), 32), nElements))
    return failure();

  triton::LoadOp load = loads.front();
  if (load.getMask() || load.getOther() ||
      !isRankedVectorOf(load.getResult(), Float16Type::get(func.getContext()),
                        nElements))
    return failure();

  triton::ReduceOp reduce = reductions.front();
  Type f16 = Float16Type::get(func.getContext());
  if (reduce.getAxis() != 0 || reduce->getNumOperands() != 1 ||
      reduce->getNumResults() != 1 ||
      reduce->getOperand(0) != load.getResult() ||
      reduce->getResult(0).getType() != f16)
    return failure();

  Region &combine = reduce.getCombineOp();
  if (!combine.hasOneBlock())
    return failure();
  Block &block = combine.front();
  if (block.getNumArguments() != 2)
    return failure();
  for (BlockArgument arg : block.getArguments()) {
    if (arg.getType() != f16)
      return failure();
  }
  auto ret = dyn_cast<triton::ReduceReturnOp>(block.getTerminator());
  if (!ret || ret->getNumOperands() != 1)
    return failure();
  Operation *producer = ret->getOperand(0).getDefiningOp();
  if (!producer || producer->getBlock() != &block ||
      &block.front() != producer || producer->getNextNode() != ret)
    return failure();
  auto add = dyn_cast<arith::AddFOp>(producer);
  if (!add || !isReduceBlockArgPair(add.getLhs(), add.getRhs(), block))
    return failure();

  triton::StoreOp store = stores.front();
  if (store.getMask() || store.getValue() != reduce->getResult(0) ||
      store.getValue().getType() != f16)
    return failure();

  // Map load pointer (which goes through MakeRange/Splat/AddPtr) to its
  // source block argument, and require the store pointer to be a raw block
  // argument from the function entry.
  llvm::DenseMap<Value, int64_t> splatBlockArg;
  for (Operation &op : entry.getOperations()) {
    if (auto splat = dyn_cast<triton::SplatOp>(&op)) {
      if (auto blockArg = dyn_cast<BlockArgument>(splat.getSrc())) {
        if (blockArg.getOwner() == &entry)
          splatBlockArg[splat.getResult()] = blockArg.getArgNumber();
      }
    }
  }
  llvm::DenseMap<Value, int64_t> ptrToArg;
  for (Operation &op : entry.getOperations()) {
    if (auto addPtr = dyn_cast<triton::AddPtrOp>(&op)) {
      auto found = splatBlockArg.find(addPtr.getPtr());
      if (found != splatBlockArg.end() &&
          addPtr.getOffset() == range.getResult())
        ptrToArg[addPtr.getResult()] = found->second;
    }
  }

  auto inputArgIt = ptrToArg.find(load.getPtr());
  if (inputArgIt == ptrToArg.end())
    return failure();
  std::optional<int64_t> outArg = getEntryBlockArgNumber(store.getPtr(), entry);
  if (!outArg)
    return failure();

  ReduceSumAllOperands operands;
  operands.anchor = opAnchor(store.getOperation());
  operands.n = nElements;
  operands.out = *outArg;
  operands.input = inputArgIt->second;
  return operands;
}

FailureOr<ReluOperands> recognizeRelu(triton::FuncOp func) {
  FunctionType functionType = func.getFunctionType();
  if (functionType.getNumInputs() != 2 || functionType.getNumResults() != 0)
    return failure();

  for (Type input : functionType.getInputs()) {
    if (!isF16Ptr(input))
      return failure();
  }
  if (func.getBody().empty() || !func.getBody().hasOneBlock())
    return failure();

  Block &entry = func.getBody().front();

  SmallVector<triton::MakeRangeOp> ranges;
  SmallVector<triton::LoadOp> loads;
  SmallVector<arith::ExtFOp> exts;
  SmallVector<Operation *> maxes;
  SmallVector<arith::TruncFOp> truncs;
  SmallVector<triton::StoreOp> stores;

  for (Operation &op : entry.getOperations()) {
    if (auto range = dyn_cast<triton::MakeRangeOp>(&op))
      ranges.push_back(range);
    else if (auto load = dyn_cast<triton::LoadOp>(&op))
      loads.push_back(load);
    else if (op.getNumRegions() != 0)
      return failure();
    else if (auto ext = dyn_cast<arith::ExtFOp>(&op))
      exts.push_back(ext);
    else if (isa<arith::MaxNumFOp>(&op) || isa<arith::MaximumFOp>(&op))
      maxes.push_back(&op);
    else if (auto trunc = dyn_cast<arith::TruncFOp>(&op))
      truncs.push_back(trunc);
    else if (auto store = dyn_cast<triton::StoreOp>(&op))
      stores.push_back(store);
    else if (isa<triton::SplatOp, triton::AddPtrOp, triton::ReturnOp,
                 arith::ConstantOp, arith::CmpIOp>(&op))
      continue;
    else
      return failure();
  }

  if (ranges.size() != 1 || loads.size() != 1 || exts.size() != 1 ||
      maxes.size() != 1 || truncs.size() != 1 || stores.size() != 1)
    return failure();

  triton::StoreOp store = stores.front();
  FailureOr<CompactVectorMemoryMatch> memory =
      recognizeCompactVectorMemory(func, entry, ranges, loads, store,
                                   /*expectedInputs=*/2);
  if (failed(memory))
    return failure();
  int64_t nElements = memory->nElements;
  if (nElements <= 0 || nElements % 16 != 0)
    return failure();

  triton::LoadOp load = loads.front();
  Type f16 = Float16Type::get(func.getContext());
  Type f32 = Float32Type::get(func.getContext());
  if (load.getMask() || load.getOther() ||
      !isRankedVectorOf(load.getResult(), f16, nElements))
    return failure();

  arith::ExtFOp ext = exts.front();
  if (ext.getIn() != load.getResult() ||
      !isRankedVectorOf(ext.getResult(), f32, nElements))
    return failure();

  std::optional<Value> reluResult = getReluResult(ext.getResult(), {nElements});
  if (!reluResult || *reluResult != maxes.front()->getResult(0))
    return failure();

  arith::TruncFOp trunc = truncs.front();
  if (trunc.getIn() != *reluResult ||
      !isRankedVectorOf(trunc.getResult(), f16, nElements))
    return failure();

  if (store.getMask() || store.getValue() != trunc.getResult())
    return failure();

  auto inputArg = memory->loadToArg.find(load.getResult());
  auto outArg = memory->ptrToArg.find(store.getPtr());
  if (inputArg == memory->loadToArg.end() || outArg == memory->ptrToArg.end())
    return failure();

  ReluOperands operands;
  operands.anchor = opAnchor(store.getOperation());
  operands.n = nElements;
  operands.out = outArg->second;
  operands.input = inputArg->second;
  return operands;
}

FailureOr<MaximumOperands> recognizeMaximum(triton::FuncOp func) {
  FunctionType functionType = func.getFunctionType();
  if (functionType.getNumInputs() != 3 || functionType.getNumResults() != 0)
    return failure();

  for (Type input : functionType.getInputs()) {
    if (!isF16Ptr(input))
      return failure();
  }
  if (func.getBody().empty() || !func.getBody().hasOneBlock())
    return failure();

  Block &entry = func.getBody().front();

  SmallVector<triton::MakeRangeOp> ranges;
  SmallVector<triton::LoadOp> loads;
  SmallVector<Operation *> maxes;
  SmallVector<triton::StoreOp> stores;

  for (Operation &op : entry.getOperations()) {
    if (auto range = dyn_cast<triton::MakeRangeOp>(&op))
      ranges.push_back(range);
    else if (auto load = dyn_cast<triton::LoadOp>(&op))
      loads.push_back(load);
    else if (op.getNumRegions() != 0)
      return failure();
    else if (isa<arith::MaxNumFOp>(&op) || isa<arith::MaximumFOp>(&op))
      maxes.push_back(&op);
    else if (auto store = dyn_cast<triton::StoreOp>(&op))
      stores.push_back(store);
    else if (isa<triton::SplatOp, triton::AddPtrOp, triton::ReturnOp,
                 arith::ConstantOp, arith::CmpIOp>(&op))
      continue;
    else
      return failure();
  }

  if (ranges.size() != 1 || loads.size() != 2 || maxes.size() != 1 ||
      stores.size() != 1)
    return failure();

  triton::StoreOp store = stores.front();
  FailureOr<CompactVectorMemoryMatch> memory =
      recognizeCompactVectorMemory(func, entry, ranges, loads, store,
                                   /*expectedInputs=*/3);
  if (failed(memory))
    return failure();
  int64_t nElements = memory->nElements;
  if (nElements <= 0 || nElements % 16 != 0)
    return failure();

  Type f16 = Float16Type::get(func.getContext());
  for (triton::LoadOp load : loads) {
    if (load.getMask() || load.getOther() ||
        !isRankedVectorOf(load.getResult(), f16, nElements))
      return failure();
  }

  Operation *max = maxes.front();
  if (max->getNumOperands() != 2 || max->getNumResults() != 1 ||
      !isRankedVectorOf(max->getResult(0), f16, nElements))
    return failure();

  Value lhsV = max->getOperand(0);
  Value rhsV = max->getOperand(1);
  if (lhsV == rhsV)
    return failure();
  if (max->getResult(0).getType() != lhsV.getType() ||
      max->getResult(0).getType() != rhsV.getType())
    return failure();

  auto lhsLoadIt = memory->loadToArg.find(lhsV);
  auto rhsLoadIt = memory->loadToArg.find(rhsV);
  if (lhsLoadIt == memory->loadToArg.end() ||
      rhsLoadIt == memory->loadToArg.end())
    return failure();

  if (store.getMask() || store.getValue() != max->getResult(0))
    return failure();

  auto outArg = memory->ptrToArg.find(store.getPtr());
  if (outArg == memory->ptrToArg.end())
    return failure();

  MaximumOperands operands;
  operands.anchor = opAnchor(store.getOperation());
  operands.n = nElements;
  operands.out = outArg->second;
  operands.lhs = lhsLoadIt->second;
  operands.rhs = rhsLoadIt->second;
  return operands;
}

// Verify the combine region of a tt.reduce: must be a 2-arg block with
// arith.addf(arg0, arg1) -> tt.reduce.return.
static bool isReduceAddCombine(triton::ReduceOp reduce, Type elementType) {
  Region &combine = reduce.getCombineOp();
  if (!combine.hasOneBlock())
    return false;
  Block &block = combine.front();
  if (block.getNumArguments() != 2)
    return false;
  for (BlockArgument arg : block.getArguments()) {
    if (arg.getType() != elementType)
      return false;
  }
  auto ret = dyn_cast<triton::ReduceReturnOp>(block.getTerminator());
  if (!ret || ret->getNumOperands() != 1)
    return false;
  Operation *producer = ret->getOperand(0).getDefiningOp();
  if (!producer || producer->getBlock() != &block ||
      &block.front() != producer || producer->getNextNode() != ret)
    return false;
  auto add = dyn_cast<arith::AddFOp>(producer);
  if (!add || !isReduceBlockArgPair(add.getLhs(), add.getRhs(), block))
    return false;
  return true;
}

FailureOr<ReduceSumAxisOperands> recognizeReduceSumAxis(triton::FuncOp func,
                                                        int64_t axis) {
  if (axis != 0 && axis != 1)
    return failure();
  FunctionType functionType = func.getFunctionType();
  if (functionType.getNumInputs() != 2 || functionType.getNumResults() != 0)
    return failure();
  for (Type input : functionType.getInputs()) {
    if (!isF16Ptr(input))
      return failure();
  }
  if (func.getBody().empty() || !func.getBody().hasOneBlock())
    return failure();
  Block &entry = func.getBody().front();

  triton::LoadOp load;
  triton::ReduceOp reduce;
  triton::StoreOp store;
  for (Operation &op : entry.getOperations()) {
    if (auto ld = dyn_cast<triton::LoadOp>(&op)) {
      if (load)
        return failure();
      load = ld;
    } else if (auto rd = dyn_cast<triton::ReduceOp>(&op)) {
      if (reduce)
        return failure();
      reduce = rd;
    } else if (auto st = dyn_cast<triton::StoreOp>(&op)) {
      if (store)
        return failure();
      store = st;
    } else if (!isAllowedReduceSumAxisTopLevelOp(op)) {
      return failure();
    }
  }
  if (!load || !reduce || !store)
    return failure();
  if (load.getMask() || load.getOther())
    return failure();

  auto loadType = dyn_cast<RankedTensorType>(load.getResult().getType());
  if (!loadType || loadType.getRank() != 2 ||
      !loadType.getElementType().isF16())
    return failure();
  int64_t rows = loadType.getShape()[0];
  int64_t cols = loadType.getShape()[1];
  if (rows <= 0 || cols <= 0)
    return failure();

  if (reduce.getAxis() != static_cast<uint32_t>(axis) ||
      reduce->getNumOperands() != 1 || reduce->getNumResults() != 1 ||
      reduce->getOperand(0) != load.getResult())
    return failure();
  int64_t outDim = axis == 0 ? cols : rows;
  Type f16 = Float16Type::get(func.getContext());
  if (!isRankedVectorOf(reduce->getResult(0), f16, outDim))
    return failure();
  if (!isReduceAddCombine(reduce, f16))
    return failure();

  if (store.getMask() || store.getValue() != reduce->getResult(0))
    return failure();

  // The store pointer is `out + tl.arange(0, outDim)` — an AddPtrOp whose
  // base is splat of arg0 and offset is a make_range of outDim.
  auto addPtr = store.getPtr().getDefiningOp<triton::AddPtrOp>();
  if (!addPtr)
    return failure();
  auto splat = addPtr.getPtr().getDefiningOp<triton::SplatOp>();
  if (!splat)
    return failure();
  auto blockArg = dyn_cast<BlockArgument>(splat.getSrc());
  if (!blockArg || blockArg.getOwner() != &entry)
    return failure();
  int64_t outArg = blockArg.getArgNumber();

  // The load pointer should resolve to arg1. Walk back through addptr and
  // broadcast ops to find the underlying splat of the block argument.
  std::optional<int64_t> inputArg;
  {
    Value ptr = load.getPtr();
    bool progress = true;
    while (progress) {
      progress = false;
      if (auto add = ptr.getDefiningOp<triton::AddPtrOp>()) {
        ptr = add.getPtr();
        progress = true;
      } else if (auto bcast = ptr.getDefiningOp<triton::BroadcastOp>()) {
        ptr = bcast.getSrc();
        progress = true;
      } else if (auto expand = ptr.getDefiningOp<triton::ExpandDimsOp>()) {
        ptr = expand.getSrc();
        progress = true;
      }
    }
    if (auto splatIn = ptr.getDefiningOp<triton::SplatOp>()) {
      if (auto inArg = dyn_cast<BlockArgument>(splatIn.getSrc())) {
        if (inArg.getOwner() == &entry)
          inputArg = inArg.getArgNumber();
      }
    }
  }
  if (!inputArg)
    return failure();

  ReduceSumAxisOperands operands;
  operands.anchor = opAnchor(store.getOperation());
  operands.rows = rows;
  operands.cols = cols;
  operands.axis = axis;
  operands.out = outArg;
  operands.input = *inputArg;
  return operands;
}

FailureOr<BroadcastAddOperands> recognizeBroadcastAdd(triton::FuncOp func) {
  FunctionType functionType = func.getFunctionType();
  if (functionType.getNumInputs() != 3 || functionType.getNumResults() != 0)
    return failure();
  for (Type input : functionType.getInputs()) {
    if (!isF16Ptr(input))
      return failure();
  }
  if (func.getBody().empty() || !func.getBody().hasOneBlock())
    return failure();
  Block &entry = func.getBody().front();

  SmallVector<triton::LoadOp> loads;
  SmallVector<arith::AddFOp> adds;
  triton::StoreOp store;
  for (Operation &op : entry.getOperations()) {
    if (auto ld = dyn_cast<triton::LoadOp>(&op))
      loads.push_back(ld);
    else if (auto add = dyn_cast<arith::AddFOp>(&op))
      adds.push_back(add);
    else if (auto st = dyn_cast<triton::StoreOp>(&op)) {
      if (store)
        return failure();
      store = st;
    } else if (!isAllowedBroadcastAddTopLevelOp(op))
      return failure();
  }
  if (loads.size() != 2 || adds.size() != 1 || !store)
    return failure();
  Type f16 = Float16Type::get(func.getContext());

  triton::LoadOp matLoad;
  triton::LoadOp vecLoad;
  for (triton::LoadOp ld : loads) {
    if (ld.getMask() || ld.getOther())
      return failure();
    auto type = dyn_cast<RankedTensorType>(ld.getResult().getType());
    if (!type || !type.getElementType().isF16())
      return failure();
    if (type.getRank() == 2) {
      if (matLoad)
        return failure();
      matLoad = ld;
    } else if (type.getRank() == 1) {
      if (vecLoad)
        return failure();
      vecLoad = ld;
    } else {
      return failure();
    }
  }
  if (!matLoad || !vecLoad)
    return failure();

  auto matType = cast<RankedTensorType>(matLoad.getResult().getType());
  auto vecType = cast<RankedTensorType>(vecLoad.getResult().getType());
  int64_t rows = matType.getShape()[0];
  int64_t cols = matType.getShape()[1];
  if (vecType.getShape()[0] != rows || rows <= 0 || cols <= 0)
    return failure();

  // Trace add.rhs back through broadcast/expand_dims to the vector load.
  arith::AddFOp add = adds.front();
  if (add.getLhs() != matLoad.getResult())
    return failure();
  if (!isRankedTensorOf(add.getResult(), f16, ArrayRef<int64_t>{rows, cols}))
    return failure();
  Value rhs = add.getRhs();
  auto broadcast = rhs.getDefiningOp<triton::BroadcastOp>();
  if (!broadcast)
    return failure();
  if (!isRankedTensorOf(broadcast.getResult(), f16,
                        ArrayRef<int64_t>{rows, cols}))
    return failure();
  Value expandSrc = broadcast.getSrc();
  auto expand = expandSrc.getDefiningOp<triton::ExpandDimsOp>();
  if (!expand)
    return failure();
  if (!isRankedTensorOf(expand.getResult(), f16, ArrayRef<int64_t>{rows, 1}))
    return failure();
  if (expand.getSrc() != vecLoad.getResult())
    return failure();

  if (store.getMask() || store.getValue() != add.getResult())
    return failure();

  // Walk through addptr/broadcast/expand_dims to find the base block-arg
  // that's been splatted (the original pointer kernel arg).
  auto tracePtrToBlockArg = [&](Value ptr) -> std::optional<int64_t> {
    bool progress = true;
    while (progress) {
      progress = false;
      if (auto add = ptr.getDefiningOp<triton::AddPtrOp>()) {
        ptr = add.getPtr();
        progress = true;
      } else if (auto bcast = ptr.getDefiningOp<triton::BroadcastOp>()) {
        ptr = bcast.getSrc();
        progress = true;
      } else if (auto expand = ptr.getDefiningOp<triton::ExpandDimsOp>()) {
        ptr = expand.getSrc();
        progress = true;
      }
    }
    if (auto splat = ptr.getDefiningOp<triton::SplatOp>()) {
      if (auto blockArg = dyn_cast<BlockArgument>(splat.getSrc())) {
        if (blockArg.getOwner() == &entry)
          return static_cast<int64_t>(blockArg.getArgNumber());
      }
    }
    return std::nullopt;
  };

  std::optional<int64_t> outArg = tracePtrToBlockArg(store.getPtr());
  if (!outArg)
    return failure();
  std::optional<int64_t> rhsArg = tracePtrToBlockArg(vecLoad.getPtr());
  if (!rhsArg)
    return failure();
  std::optional<int64_t> lhsArg = tracePtrToBlockArg(matLoad.getPtr());
  if (!lhsArg)
    return failure();

  BroadcastAddOperands operands;
  operands.anchor = opAnchor(store.getOperation());
  operands.rows = rows;
  operands.cols = cols;
  operands.out = *outArg;
  operands.lhs = *lhsArg;
  operands.rhs = *rhsArg;
  return operands;
}

} // namespace rpu
} // namespace mlir
