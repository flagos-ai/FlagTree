#include "tle/utils/include/Protocol.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"
#include <exception>

#define COND_CHECK(cond)                                                       \
  if (!(cond)) {                                                               \
    return {};                                                                 \
  }

namespace mlir::triton::tle::protocol {

/* --------------- Definitions --------------- */

/* --------------- ProtocolImpl --------------- */

template <typename T> struct GenericProtocolImpl {
  static SmallVector<Value> apply(TritonOpBuilder &builder, TypeRange &tgts,
                                  TypedValue<T> src);
};

/* --------------- Implementatoins --------------- */

/* --------------- Protocol --------------- */

namespace signature {

SmallVector<Value> RankedTensorPattern::apply(TritonOpBuilder &builder,
                                              TypeRange &tgts,
                                              TypedValue<E> src) {
  const size_t rank = src.getType().getRank();
  SmallVector<Value> rets;
  Type tgt = tgts[0];
  LLVM::LLVMPointerType ty = cast<LLVM::LLVMPointerType>(tgt);
  rets.push_back(builder.create<ExtractAllocatedPtrOp>(ty, src));
  tgt = tgts[1];
  ty = cast<LLVM::LLVMPointerType>(tgt);
  rets.push_back(builder.create<ExtractAlignedPtrOp>(ty, src));
  tgt = tgts[2];
  COND_CHECK(tgt.isInteger(64));
  rets.push_back(builder.create<ExtractOffsetOp>(src));
  for (size_t i = 3; i < 3 + 2 * rank; ++i) {
    tgt = tgts[i];
    COND_CHECK(tgt.isInteger(64));
  }
  ExtractSizesOp sizesOp = builder.create<ExtractSizesOp>(rank, src);
  ExtractStridesOp stridesOp = builder.create<ExtractStridesOp>(rank, src);
  for (const auto &result :
       llvm::concat<OpResult>(sizesOp.getResults(), stridesOp.getResults())) {
    rets.push_back(result);
  }
  tgts = tgts.drop_front(3 + 2 * rank);
  return rets;
}

SmallVector<Value> PointerPattern::apply(TritonOpBuilder &builder,
                                         TypeRange &tgts, TypedValue<E> src) {
  Type tgt = tgts.front();
  LLVM::LLVMPointerType llvmPtrTy = cast<LLVM::LLVMPointerType>(tgt);
  tgts = tgts.drop_front();
  return {builder.create<tle::ExtractPtrOp>(llvmPtrTy, src)};
}

} // namespace signature

namespace ret {

SmallVector<Value> LLVMStructurePattern::apply(TritonOpBuilder &builder,
                                               TypeRange &tgts,
                                               TypedValue<E> src) {
  COND_CHECK(!tgts.empty());
  RankedTensorType tgt = dyn_cast<RankedTensorType>(tgts.front());
  COND_CHECK(tgt);
  const size_t rank = tgt.getRank();
  LLVM::LLVMStructType structTy = src.getType();
  ArrayRef<Type> types = structTy.getBody();
  const size_t size = types.size();
  COND_CHECK(size == 5 &&
             llvm::all_of(types.take_front(2),
                          [](const Type &ty) -> bool {
                            return isa<LLVM::LLVMPointerType>(ty);
                          }) &&
             types[2].isInteger(64) &&
             llvm::all_of(types.take_back(2), [rank](const Type &ty) -> bool {
               LLVM::LLVMArrayType arrayTy = dyn_cast<LLVM::LLVMArrayType>(ty);
               return arrayTy && arrayTy.getElementType().isInteger(64) &&
                      arrayTy.getNumElements() == rank;
             }));
  tgts = tgts.drop_front();
  return {builder.create<tle::PackOp>(tgt, src)};
}

} // namespace ret

SmallVector<Value> IntegerPattern::apply(TritonOpBuilder &builder,
                                         TypeRange &tgts, TypedValue<E> src) {
  return GenericProtocolImpl<E>::apply(builder, tgts, src);
}

SmallVector<Value> FloatPattern::apply(TritonOpBuilder &builder,
                                       TypeRange &tgts, TypedValue<E> src) {
  return GenericProtocolImpl<E>::apply(builder, tgts, src);
}

/* --------------- ProtocolPattern --------------- */

SmallVector<Value> ProtocolPatternT<>::apply(TritonOpBuilder &builder,
                                             TypeRange &tgts, Value src) {
  return {};
}

/* --------------- ProtocolImpl --------------- */

template <typename T>
SmallVector<Value> GenericProtocolImpl<T>::apply(TritonOpBuilder &builder,
                                                 TypeRange &tgts,
                                                 TypedValue<T> src) {
  Type tgt = tgts.front();
  COND_CHECK(tgt == src.getType());
  tgts = tgts.drop_front();
  return {src};
}

} // namespace mlir::triton::tle::protocol
