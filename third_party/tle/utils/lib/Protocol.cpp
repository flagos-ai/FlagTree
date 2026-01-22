#include "tle/utils/include/Protocol.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::triton::tle {

/* --------------- Definitions --------------- */

/* --------------- ProtocolImpl --------------- */

template <typename T> struct GenericProtocolImpl {
  static SmallVector<Value> consume(TritonOpBuilder &builder, ValueRange &tgts,
                                    TypedValue<T> src);
};

/* --------------- Implementatoins --------------- */

/* --------------- Protocol --------------- */

SmallVector<Value>
Protocol<RankedTensorType>::consume(TritonOpBuilder &builder, ValueRange &tgts,
                                    TypedValue<RankedTensorType> src) {
  size_t counter = 0;
  SmallVector<Value> rets;
  Value val = tgts[counter++];
  LLVM::LLVMPointerType ty = cast<LLVM::LLVMPointerType>(val.getType());
  rets.push_back(builder.create<ExtractAllocatedPtrOp>(ty, src));
  val = tgts[counter++];
  ty = cast<LLVM::LLVMPointerType>(val.getType());
  rets.push_back(builder.create<ExtractAlignedPtrOp>(ty, src));
  val = tgts[counter++];
  assert(val.getType().isInteger(64));
  rets.push_back(builder.create<ExtractOffsetOp>(src));
  const size_t rank = src.getType().getRank();
  for (size_t i = counter; i < counter + 2 * rank; ++i) {
    val = tgts[i];
    assert(val.getType().isInteger(64));
  }
  counter += 2 * rank;
  ExtractSizesOp sizesOp = builder.create<ExtractSizesOp>(rank, src);
  ExtractStridesOp stridesOp = builder.create<ExtractStridesOp>(rank, src);
  for (const auto &result :
       llvm::concat<OpResult>(sizesOp.getResults(), stridesOp.getResults())) {
    rets.push_back(result);
  }
  tgts = tgts.drop_front(counter);
  return rets;
}

SmallVector<Value> Protocol<PointerType>::consume(TritonOpBuilder &builder,
                                                  ValueRange &tgts,
                                                  TypedValue<PointerType> src) {
  Value tgt = tgts.front();
  LLVM::LLVMPointerType llvmPtrTy = cast<LLVM::LLVMPointerType>(tgt.getType());
  tgts = tgts.drop_front();
  return {builder.create<tle::ExtractPtrOp>(llvmPtrTy, src)};
}

SmallVector<Value> Protocol<IntegerType>::consume(TritonOpBuilder &builder,
                                                  ValueRange &tgts,
                                                  TypedValue<IntegerType> src) {
  return GenericProtocolImpl<IntegerType>::consume(builder, tgts, src);
}

SmallVector<Value> Protocol<FloatType>::consume(TritonOpBuilder &builder,
                                                ValueRange &tgts,
                                                TypedValue<FloatType> src) {
  return GenericProtocolImpl<FloatType>::consume(builder, tgts, src);
}

/* --------------- ProtocolPattern --------------- */

SmallVector<Value> ProtocolPattern<>::consume(TritonOpBuilder &builder,
                                              ValueRange &tgts, Value src) {
  return {};
}

/* --------------- ProtocolImpl --------------- */

template <typename T>
SmallVector<Value> GenericProtocolImpl<T>::consume(TritonOpBuilder &builder,
                                                   ValueRange &tgts,
                                                   TypedValue<T> src) {
  Value val = tgts.front();
  assert(val.getType() == src.getType());
  tgts = tgts.drop_front();
  return SmallVector<Value>{src};
}

} // namespace mlir::triton::tle
