/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TLE_UTILS_PROTOCOL_H_
#define TLE_UTILS_PROTOCOL_H_

#include "ir.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <type_traits>

namespace ttg = mlir::triton::gpu;

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

struct MemDescPattern final : public ProtocolT<ttg::MemDescType> {
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
    ProtocolPatternT<signature::RankedTensorPattern, signature::MemDescPattern,
                     signature::PointerPattern, IntegerPattern, FloatPattern>;
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
