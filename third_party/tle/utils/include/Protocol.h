#ifndef TLE_UTILS_PROTOCOL_H_
#define TLE_UTILS_PROTOCOL_H_

#include "ir.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include <type_traits>

/* --------------- Definitions --------------- */

namespace mlir::triton::tle::protocol {

/* --------------- Protocol  --------------- */

struct Protocol {};

template <typename T> struct ProtocolT : public Protocol {
  using E = T;
  static SmallVector<Value> apply(TritonOpBuilder &builder, TypeRange &tgts,
                                  TypedValue<E> src);
};

namespace signature {
struct RankedTensorPattern final : public ProtocolT<RankedTensorType> {
  static SmallVector<Value> apply(TritonOpBuilder &builder, TypeRange &tgts,
                                  TypedValue<E> src);
};

struct PointerPattern : public ProtocolT<PointerType> {
  static SmallVector<Value> apply(TritonOpBuilder &builder, TypeRange &tgts,
                                  TypedValue<E> src);
};

} // namespace signature

namespace ret {

struct LLVMStructurePattern final : public ProtocolT<LLVM::LLVMStructType> {
  static SmallVector<Value> apply(TritonOpBuilder &builder, TypeRange &tgts,
                                  TypedValue<E> src);
};

} // namespace ret

struct IntegerPattern final : public ProtocolT<IntegerType> {
  static SmallVector<Value> apply(TritonOpBuilder &builder, TypeRange &tgts,
                                  TypedValue<E> src);
};

struct FloatPattern final : public ProtocolT<FloatType> {
  static SmallVector<Value> apply(TritonOpBuilder &builder, TypeRange &tgts,
                                  TypedValue<E> src);
};

/* --------------- ProtocolPattern --------------- */

struct ProtocolPattern {};

template <typename... Ps> struct ProtocolPatternT : public ProtocolPattern {
  static SmallVector<Value> apply(TritonOpBuilder &builder, TypeRange &tgts,
                                  Value src);
};

template <> struct ProtocolPatternT<> {
  static SmallVector<Value> apply(TritonOpBuilder &builder, TypeRange &tgts,
                                  Value src);
};

template <typename P, typename... Ps> struct ProtocolPatternT<P, Ps...> {
  static SmallVector<Value> apply(TritonOpBuilder &builder, TypeRange &tgts,
                                  Value src);
};

using SignaturePattern =
    ProtocolPatternT<signature::RankedTensorPattern, signature::PointerPattern,
                     IntegerPattern, FloatPattern>;
using ReturnPattern =
    ProtocolPatternT<ret::LLVMStructurePattern, IntegerPattern, FloatPattern>;

/* --------------- PatternUtils --------------- */

template <typename P, typename = void> struct ProtocolPatternImpl {
  static SmallVector<Value> apply(TritonOpBuilder &builder, TypeRange &tgts,
                                  Value src);
};

template <typename P>
struct ProtocolPatternImpl<P,
                           std::enable_if_t<std::is_base_of_v<Protocol, P>>> {
  static SmallVector<Value> apply(TritonOpBuilder &builder, TypeRange &tgts,
                                  Value src);
};

template <typename P>
struct ProtocolPatternImpl<
    P, std::enable_if_t<std::is_base_of_v<ProtocolPattern, P>>> {
  static SmallVector<Value> apply(TritonOpBuilder &builder, TypeRange &tgts,
                                  Value src);
};

/* --------------- Implementatoins --------------- */

/* --------------- ProtocolPattern --------------- */

template <typename P, typename... Ps>
SmallVector<Value> ProtocolPatternT<P, Ps...>::apply(TritonOpBuilder &builder,
                                                     TypeRange &tgts,
                                                     Value src) {
  using E = typename P::E;
  SmallVector<Value> rets = ProtocolPatternImpl<P>::apply(builder, tgts, src);
  rets.append(ProtocolPatternT<Ps...>::apply(builder, tgts, src));
  return rets;
}

/* --------------- PatternUtils --------------- */

template <typename P>
SmallVector<Value>
ProtocolPatternImpl<P, std::enable_if_t<std::is_base_of_v<Protocol, P>>>::apply(
    TritonOpBuilder &builder, TypeRange &tgts, Value src) {
  using E = typename P::E;
  if (TypedValue<E> v = dyn_cast<TypedValue<E>>(src)) {
    return P::apply(builder, tgts, v);
  } else {
    return {};
  }
}

template <typename P>
SmallVector<Value>
ProtocolPatternImpl<P,
                    std::enable_if_t<std::is_base_of_v<ProtocolPattern, P>>>::
    apply(TritonOpBuilder &builder, TypeRange &tgts, Value src) {
  return P::apply(builder, tgts, src);
}

} // namespace mlir::triton::tle::protocol

#endif
