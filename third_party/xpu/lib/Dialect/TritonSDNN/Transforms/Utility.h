#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton/Dialect/Triton/IR/Utility.h" //ceil
#include "xpu/include/Dialect/TritonSDNN/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <type_traits>

// Utility
namespace mlir {
namespace triton {
namespace sdnn {

// Give a value and a operation, if every user of the value is a `T`, return
// all of the users except the given operation. Otherwise, return null.
template <typename T>
SmallVector<T> findAllUsersOfType(Value val, Operation *op,
                                  bool skipFillOp = false) {
  SmallVector<T> find;
  for (auto user : val.getUsers()) {
    if (user == op)
      continue;
    else if (isa<T>(user))
      find.push_back(cast<T>(user));
    else if (skipFillOp && isa<EWOp>(user) &&
             cast<EWOp>(user).getFunct() == EWType::fill)
      continue;
    else
      return SmallVector<T>();
  }
  return find;
}

inline void checkAllUsersExt(Value value, Operation *op, bool skipFillOp,
                             std::function<void(Operation *)> callback) {
  std::queue<Operation *> ops;

  for (auto user : value.getUsers()) {
    if (skipFillOp && isa<EWOp>(user) &&
        cast<EWOp>(user).getFunct() == EWType::fill)
      continue;
    if (user == op)
      continue;
    ops.push(user);
  }
  while (!ops.empty()) {
    auto op = ops.front();
    ops.pop();
    TypeSwitch<Operation *>(op)
        .Case<memref::SubViewOp>([&](auto subviewOp) {
          for (auto u : subviewOp.getResult().getUsers()) {
            ops.push(u);
          }
        })
        .Case<memref::CastOp>([&](auto castOp) {
          for (auto u : castOp.getResult().getUsers()) {
            ops.push(u);
          }
        })
        .Case<memref::ExpandShapeOp>([&](auto expandShapeOp) {
          for (auto u : expandShapeOp.getResult().getUsers()) {
            ops.push(u);
          }
        })
        .Case<triton::sdnn::EWOp>([&](auto ewOp) {
          if (skipFillOp && ewOp.getFunct() == EWType::fill)
            return;
          callback(ewOp);
        })
        .Default([&](auto op) { callback(op); });
  }
}

// Give a value and a operation, if the value has 2 users and one of them is
// the operation, return the other one if it is `T`. Otherwise, return null.
template <typename T>
T findTheOtherUserOfType(Value val, Operation *op, bool skipFillOp = false) {
  SmallVector<T> find = findAllUsersOfType<T>(val, op, skipFillOp);
  return find.size() == 1 ? find[0] : T();
}

inline bool isOutput(Value value, Operation *op) {
  if (auto dmaOp = dyn_cast<triton::sdnn::DMAOp>(op)) {
    return value == dmaOp.getDst();
  } else if (auto dsOp = dyn_cast<triton::sdnn::DSOp>(op)) {
    return value == dsOp.getDst();
  } else if (auto rsOp = dyn_cast<triton::sdnn::RSOp>(op)) {
    return value == rsOp.getDst();
  } else if (auto ewOp = dyn_cast<triton::sdnn::EWOp>(op)) {
    return value == ewOp.getDst();
  } else if (auto mmaOp = dyn_cast<triton::sdnn::MMAOp>(op)) {
    return value == mmaOp.getC();
  } else if (auto dstOp = dyn_cast<DestinationStyleOpInterface>(op)) {
    for (auto dpsInit : dstOp.getDpsInits()) {
      if (value == dpsInit)
        return true;
    }
  } else if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
    return value == copyOp.getTarget();
  }
  return false;
}

std::map<int64_t, unsigned> getNumOccurences(ArrayRef<int64_t> vals);
bool haveCompatibleOffsets(MemRefType t1, MemRefType t2);
bool haveCompatibleStrides(MemRefType t1, MemRefType t2,
                           const llvm::SmallBitVector &droppedDims);
FailureOr<llvm::SmallBitVector>
computeMemRefRankReductionMask(MemRefType originalType, MemRefType reducedType,
                               ArrayRef<OpFoldResult> sizes);
inline std::pair<Block &, Operation &>
getGenericOpComponent(linalg::GenericOp genericOp) {
  auto &region = genericOp.getRegion();
  auto &entryBlock = region.front();
  auto &op = entryBlock.front();
  return {entryBlock, op};
}

inline std::optional<Value> getOperandBroadcast(Value value) {
  if (!value || !isa<RankedTensorType>(value.getType())) {
    return std::nullopt;
  }
  auto rsOp = value.getDefiningOp<triton::sdnn::RSOp>();
  if (!rsOp || !rsOp->hasAttr("format"))
    return std::nullopt;
  auto dsOp = rsOp.getSrc().getDefiningOp<triton::sdnn::DSOp>();
  if (!dsOp || !dsOp->hasAttr("batch"))
    return std::nullopt;
  return dsOp.getSrc();
}

inline bool isShape1x1(ShapedType type) {
  auto shape = type.getShape();
  return shape.size() == 2 && shape[0] == 1 && shape[1] == 1;
}

inline bool isShape1xN(ShapedType type) {
  auto shape = type.getShape();
  return shape.size() == 2 && shape[0] == 1;
}

inline bool isShapeNx1(ShapedType type) {
  auto shape = type.getShape();
  return shape.size() == 2 && shape[1] == 1;
}

inline SmallVector<Value> getDynamicShapes(Value value, RewriterBase &rewriter,
                                           Location loc) {
  SmallVector<Value> dynamicShapes;
  if (auto rankedType = dyn_cast<ShapedType>(value.getType())) {
    for (int64_t i = 0; i < rankedType.getRank(); i++) {
      if (rankedType.isDynamicDim(i)) {
        dynamicShapes.push_back(rewriter.create<tensor::DimOp>(loc, value, i));
      }
    }
    return dynamicShapes;
  }
  return {};
}

inline tensor::EmptyOp createEmptyOpLike(Value value, Type elemTy,
                                         RewriterBase &rewriter, Location loc) {
  auto shapeType = cast<ShapedType>(value.getType());
  auto shapes = shapeType.getShape();
  auto dynamicShapes = getDynamicShapes(value, rewriter, loc);
  return rewriter.create<tensor::EmptyOp>(loc, shapes, elemTy, dynamicShapes);
}

inline bool isMemrefTransposed(MemRefType memrefType) {
  auto [strides, offset] = memrefType.getStridesAndOffset();
  if (strides.size() == 2 && strides[0] == 1 && strides[1] != 1)
    return true;
  return false;
}

inline triton::sdnn::EWType getPredicate(Operation *op) {
  auto predicate = triton::sdnn::EWType::select_le;
  llvm::TypeSwitch<Operation *>(op)
      .Case<>([&](arith::CmpIOp cmpiOp) {
        switch (cmpiOp.getPredicate()) {
        case arith::CmpIPredicate::sle:
        case arith::CmpIPredicate::ule:
          predicate = triton::sdnn::EWType::select_le;
          break;
        case arith::CmpIPredicate::sge:
        case arith::CmpIPredicate::uge:
          predicate = triton::sdnn::EWType::select_ge;
          break;
        case arith::CmpIPredicate::slt:
        case arith::CmpIPredicate::ult:
          predicate = triton::sdnn::EWType::select_lt;
          break;
        case arith::CmpIPredicate::sgt:
        case arith::CmpIPredicate::ugt:
          predicate = triton::sdnn::EWType::select_gt;
          break;
        default:
          llvm_unreachable("Unsupported select predicate.");
        }
      })
      .Case<>([&](arith::CmpFOp cmpfOp) {
        switch (cmpfOp.getPredicate()) {
        case arith::CmpFPredicate::OLE:
        case arith::CmpFPredicate::ULE:
          predicate = triton::sdnn::EWType::select_le;
          break;
        case arith::CmpFPredicate::OGE:
        case arith::CmpFPredicate::UGE:
          predicate = triton::sdnn::EWType::select_ge;
          break;
        case arith::CmpFPredicate::OLT:
        case arith::CmpFPredicate::ULT:
          predicate = triton::sdnn::EWType::select_lt;
          break;
        case arith::CmpFPredicate::OGT:
        case arith::CmpFPredicate::UGT:
          predicate = triton::sdnn::EWType::select_gt;
          break;
        default:
          llvm_unreachable("Unsupported select predicate.");
        }
      })
      .Default([&](Operation *) {
        llvm_unreachable("Unsupported select predicate.");
      });

  return predicate;
}

template <typename T, typename = void> struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<
    T, std::void_t<decltype(std::begin(std::declval<T>())),
                   decltype(std::end(std::declval<T>())),
                   typename std::iterator_traits<decltype(std::begin(
                       std::declval<T>()))>::iterator_category>>
    : std::true_type {};

template <typename T> constexpr bool is_iterable_v = is_iterable<T>::value;

template <typename T>
typename std::enable_if<is_iterable_v<T>, void>::type
printAnyImpl(const char *name, const T &value, const char *file, int line) {
  llvm::dbgs() << "\n//---------- " << name << "\n"
               << file << ":" << line << "\n"
               << name << " (iterable): [\n";
  for (auto [idx, element] : llvm::enumerate(value)) {
    using ValueType = typename std::remove_reference<decltype(element)>::type;
    if constexpr (std::is_pointer_v<ValueType> ||
                  std::is_member_pointer_v<ValueType> ||
                  std::is_member_object_pointer_v<ValueType>)
      llvm::dbgs() << "\t" << idx << "\t" << *element << "\n";
    else
      llvm::dbgs() << "\t" << idx << "\t" << element << "\n";
  }
  llvm::dbgs() << "]\n----------//\n";
}

template <typename T>
typename std::enable_if<!is_iterable_v<T>, void>::type
printAnyImpl(const char *name, const T &value, const char *file, int line) {
  if constexpr (std::is_pointer_v<T> || std::is_member_pointer_v<T> ||
                std::is_member_object_pointer_v<T>)
    llvm::dbgs() << "\n//---------- " << name << "\n"
                 << file << ":" << line << "\n"
                 << name << " " << *value << "\n----------//\n";
  else
    llvm::dbgs() << "\n//---------- " << name << "\n"
                 << file << ":" << line << "\n"
                 << name << " " << value << "\n----------//\n";
}

#define PRINT_ANY_SINGLE(x)                                                    \
  mlir::triton::sdnn::printAnyImpl(#x, (x), __FILE__, __LINE__)
#define PRINT_ANY_1(x) PRINT_ANY_SINGLE(x);
#define PRINT_ANY_2(x, ...)                                                    \
  PRINT_ANY_SINGLE(x);                                                         \
  PRINT_ANY_1(__VA_ARGS__)
#define PRINT_ANY_3(x, ...)                                                    \
  PRINT_ANY_SINGLE(x);                                                         \
  PRINT_ANY_2(__VA_ARGS__)
#define PRINT_ANY_4(x, ...)                                                    \
  PRINT_ANY_SINGLE(x);                                                         \
  PRINT_ANY_3(__VA_ARGS__)
#define PRINT_ANY_5(x, ...)                                                    \
  PRINT_ANY_SINGLE(x);                                                         \
  PRINT_ANY_4(__VA_ARGS__)
#define PRINT_ANY_6(x, ...)                                                    \
  PRINT_ANY_SINGLE(x);                                                         \
  PRINT_ANY_5(__VA_ARGS__)
#define PRINT_ANY_7(x, ...)                                                    \
  PRINT_ANY_SINGLE(x);                                                         \
  PRINT_ANY_6(__VA_ARGS__)
#define PRINT_ANY_8(x, ...)                                                    \
  PRINT_ANY_SINGLE(x);                                                         \
  PRINT_ANY_7(__VA_ARGS__)
#define PRINT_ANY_9(x, ...)                                                    \
  PRINT_ANY_SINGLE(x);                                                         \
  PRINT_ANY_8(__VA_ARGS__)
#define PRINT_ANY_GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, NAME,     \
                            ...)                                               \
  NAME

// 用llvm::dbgs()打印多个参数的文件名，行号，变量名以及值
#define printAny(...)                                                          \
  do {                                                                         \
    PRINT_ANY_GET_MACRO(__VA_ARGS__, PRINT_ANY_10, PRINT_ANY_9, PRINT_ANY_8,   \
                        PRINT_ANY_7, PRINT_ANY_6, PRINT_ANY_5, PRINT_ANY_4,    \
                        PRINT_ANY_3, PRINT_ANY_2, PRINT_ANY_1)                 \
    (__VA_ARGS__)                                                              \
  } while (0)

int64_t getDynamicSizeForTiling(Value v, int idx);

inline int64_t
getDynamicSizeForTiling(const SmallVector<OpFoldResult> &mixedSizes, int idx) {
  if (idx < 0 || static_cast<size_t>(idx) >= mixedSizes.size()) {
    return ShapedType::kDynamic;
  }
  // Get the defining operation of the dynamic size
  if (auto size_val = mixedSizes[idx].dyn_cast<Attribute>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(size_val))
      return intAttr.getInt();
  } else if (auto val = mixedSizes[idx].dyn_cast<Value>()) {
    return getDynamicSizeForTiling(val, idx);
  }
  return ShapedType::kDynamic;
}

inline int64_t getDynamicSizeForTiling(Value v, int idx) {
  if (auto dim = v.getDefiningOp<memref::DimOp>()) {
    if (dim.getConstantIndex())
      return getDynamicSizeForTiling(dim.getSource(),
                                     dim.getConstantIndex().value());
  } else if (auto alloc = v.getDefiningOp<memref::AllocOp>()) {
    return getDynamicSizeForTiling(alloc.getMixedSizes(), idx);
  } else if (auto loop = v.getDefiningOp<LoopLikeOpInterface>()) {
    for (auto r : loop->getResults()) {
      if (r == v)
        getDynamicSizeForTiling(loop.getTiedLoopInit(r)->get(), idx);
    }
  } else if (auto interface =
                 v.getDefiningOp<mlir::OffsetSizeAndStrideOpInterface>()) {
    return getDynamicSizeForTiling(interface.getMixedSizes(), idx);
  } else if (auto min = v.getDefiningOp<arith::MinSIOp>()) {
    Value arg0, arg1, arg2;
    Value upper0, upper1, upper2;
    Value destCaptrueV;
    auto pattern1 = m_Op<arith::MinSIOp>(
        m_Op<arith::SubIOp>(
            m_Op<arith::MaxSIOp>(
                m_Op<arith::MinSIOp>(
                    matchers::m_Any(&upper0),
                    m_Op<arith::AddIOp>(matchers::m_Any(&arg0),
                                        matchers::m_Any(&destCaptrueV))),
                matchers::m_Any(&arg1)),
            matchers::m_Any(&arg2)),
        matchers::m_Any(&upper1));

    auto pattern2 = m_Op<arith::MinSIOp>(
        m_Op<arith::SubIOp>(
            m_Op<arith::MaxSIOp>(
                m_Op<arith::MinSIOp>(
                    m_Op<arith::AddIOp>(matchers::m_Any(&arg0),
                                        matchers::m_Any(&destCaptrueV)),
                    matchers::m_Any(&upper0)),
                matchers::m_Any(&arg1)),
            matchers::m_Any(&arg2)),
        matchers::m_Any(&upper1));

    auto pattern3 = m_Op<arith::MinSIOp>(
        m_Op<arith::MaxSIOp>(
            m_Op<arith::MinSIOp>(matchers::m_Any(&upper0),
                                 matchers::m_Any(&destCaptrueV)),
            matchers::m_Any(&arg0)),
        matchers::m_Any(&upper1));

    if (pattern1.match(min) || pattern2.match(min) || pattern3.match(min)) {
      return getDynamicSizeForTiling(SmallVector<OpFoldResult>{destCaptrueV},
                                     0);
    } else if (min.getLhs().getDefiningOp<arith::MinSIOp>()) {
      auto lhsRes = getDynamicSizeForTiling(min.getLhs(), idx);
      return lhsRes != ShapedType::kDynamic
                 ? lhsRes
                 : getDynamicSizeForTiling(min.getRhs(), idx);
    } else if (min.getRhs().getDefiningOp<arith::MinSIOp>()) {
      auto rhsRes = getDynamicSizeForTiling(min.getRhs(), idx);
      return rhsRes != ShapedType::kDynamic
                 ? rhsRes
                 : getDynamicSizeForTiling(min.getLhs(), idx);
    }
  } else if (auto c = v.getDefiningOp<arith::ConstantIntOp>()) {
    return c.value();
  } else if (auto c = v.getDefiningOp<arith::ConstantIndexOp>()) {
    return c.value();
  } else if (auto min = v.getDefiningOp<affine::AffineMinOp>()) {
    AffineMap affineMap = min.getAffineMap();
    for (mlir::AffineExpr expr : affineMap.getResults()) {
      if (auto constExpr = dyn_cast<mlir::AffineConstantExpr>(expr)) {
        int64_t constantValue = constExpr.getValue();
        // LLVM_DEBUG(llvm::dbgs()
        //            << __func__ << " find Constant value For AffineMap: "
        //            << constantValue << "\n");
        return constantValue;
      }
    }
  } else if (auto ewOp = v.getDefiningOp<sdnn::EWOp>()) {
    if (auto extract = ewOp.getDst().getDefiningOp<tensor::ExtractSliceOp>()) {
      return getDynamicSizeForTiling(extract, idx);
    }
  } else if (auto mmaOp = v.getDefiningOp<sdnn::MMAOp>()) {
    if (auto extract = mmaOp.getC().getDefiningOp<tensor::ExtractSliceOp>()) {
      return getDynamicSizeForTiling(extract, idx);
    }
  }
  return ShapedType::kDynamic;
}

bool isConstOne(const OpFoldResult &ofr);
bool isConstZero(const OpFoldResult &ofr);

} // namespace sdnn
} // namespace triton
} // namespace mlir
