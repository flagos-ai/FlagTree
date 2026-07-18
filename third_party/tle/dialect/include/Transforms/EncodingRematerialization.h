#ifndef TLE_DIALECT_TRANSFORMS_ENCODINGREMATERIALIZATION_H
#define TLE_DIALECT_TRANSFORMS_ENCODINGREMATERIALIZATION_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::triton::gpu {

struct EncodingRematerializationCacheEntry {
  Value source;
  Attribute targetEncoding;
  Value rematerialized;
};

using EncodingRematerializationCache =
    SmallVector<EncodingRematerializationCacheEntry, 16>;

class EncodingRematerializer;

class EncodingRematerializationPolicy {
public:
  virtual ~EncodingRematerializationPolicy() = default;

  virtual bool isCustomEncodingPolymorphicOp(Operation *op) const;

  virtual bool hasInterveningSharedMemoryWriteAlias(Operation *from,
                                                    Operation *to,
                                                    Value memdesc) const;

  virtual bool areCustomValuesEquivalent(
      EncodingRematerializer &rematerializer, Value lhs, Value rhs,
      SmallVectorImpl<std::pair<Value, Value>> &active, unsigned depth) const;

  virtual FailureOr<Value> rematerializeCustomValue(
      EncodingRematerializer &rematerializer, Value value,
      RankedTensorType targetType, Attribute targetEncoding,
      llvm::SmallPtrSetImpl<Value> &active, unsigned depth) const;
};

class EncodingRematerializer {
public:
  EncodingRematerializer(RewriterBase &rewriter, Operation *insertBefore,
                         EncodingRematerializationCache &cache,
                         DominanceInfo &dominance,
                         const EncodingRematerializationPolicy &policy);

  RewriterBase &getRewriter() const { return rewriter; }
  Operation *getInsertBefore() const { return insertBefore; }
  DominanceInfo &getDominance() const { return dominance; }
  const EncodingRematerializationPolicy &getPolicy() const { return policy; }

  bool isAvailableAt(Value value) const;

  FailureOr<Value> rematerialize(Value value, Attribute targetEncoding,
                                 llvm::SmallPtrSetImpl<Value> &active,
                                 unsigned depth);

  bool areEquivalentIgnoringEncoding(
      Value lhs, Value rhs, SmallVectorImpl<std::pair<Value, Value>> &active,
      unsigned depth);

private:
  RewriterBase &rewriter;
  Operation *insertBefore;
  EncodingRematerializationCache &cache;
  DominanceInfo &dominance;
  const EncodingRematerializationPolicy &policy;
};

FailureOr<Value> rematerializeWithEncoding(
    RewriterBase &rewriter, Operation *insertBefore, Value value,
    Attribute targetEncoding, EncodingRematerializationCache &cache,
    DominanceInfo &dominance, const EncodingRematerializationPolicy &policy);

void collectAvailableEquivalentNvidiaMmaEncodings(
    Value root, Operation *insertBefore, DominanceInfo &dominance,
    const EncodingRematerializationPolicy &policy,
    SmallVectorImpl<Attribute> &encodings);

} // namespace mlir::triton::gpu

#endif // TLE_DIALECT_TRANSFORMS_ENCODINGREMATERIALIZATION_H
