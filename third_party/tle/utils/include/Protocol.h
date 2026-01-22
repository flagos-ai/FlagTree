#ifndef TLE_UTILS_PROTOCOL_H_
#define TLE_UTILS_PROTOCOL_H_

#include "IR/Dialect.h"
#include "ir.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include <cstddef>
#include <stdexcept>

/* --------------- Definitions --------------- */

namespace mlir::triton::tle {

/* --------------- Protocol  --------------- */

template <typename T> struct Protocol {
  static SmallVector<Value> consume(TritonOpBuilder &builder, ValueRange &tgts,
                                    TypedValue<T> src);
};

template <> struct Protocol<RankedTensorType> {
  static SmallVector<Value> consume(TritonOpBuilder &builder, ValueRange &tgts,
                                    TypedValue<RankedTensorType> src);
};

template <> struct Protocol<PointerType> {
  static SmallVector<Value> consume(TritonOpBuilder &builder, ValueRange &tgts,
                                    TypedValue<PointerType> src);
};

template <> struct Protocol<IntegerType> {
  static SmallVector<Value> consume(TritonOpBuilder &builder, ValueRange &tgts,
                                    TypedValue<IntegerType> src);
};

template <> struct Protocol<FloatType> {
  static SmallVector<Value> consume(TritonOpBuilder &builder, ValueRange &tgts,
                                    TypedValue<FloatType> src);
};

/* --------------- ProtocolPattern --------------- */

template <typename... Ts> struct ProtocolPattern {
  static SmallVector<Value> consume(TritonOpBuilder &builder, ValueRange &tgts,
                                    Value src);
};

template <> struct ProtocolPattern<> {
  static SmallVector<Value> consume(TritonOpBuilder &builder, ValueRange &tgts,
                                    Value src);
};

template <typename T, typename... Ts> struct ProtocolPattern<T, Ts...> {
  static SmallVector<Value> consume(TritonOpBuilder &builder, ValueRange &tgts,
                                    Value src);
};

/* --------------- Implementatoins --------------- */

/* --------------- ProtocolPattern --------------- */

template <typename T, typename... Ts>
SmallVector<Value> ProtocolPattern<T, Ts...>::consume(TritonOpBuilder &builder,
                                                      ValueRange &tgts,
                                                      Value src) {
  if (isa<T>(src.getType())) {
    return Protocol<T>::consume(builder, tgts, cast<TypedValue<T>>(src));
  } else {
    return ProtocolPattern<Ts...>::consume(builder, tgts, src);
  }
}

} // namespace mlir::triton::tle

#endif
