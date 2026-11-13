#ifndef TRITON_IR_TRAITS_H_
#define TRITON_IR_TRAITS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir {
namespace OpTrait {

// These functions are out-of-line implementations of the methods in the
// corresponding trait classes. This avoids them being template
// instantiated/duplicated.
namespace impl {
// IR construction limit, including temporary tensors describing node transfers.
// HCU runs TLE node transfer fusion before GPU lowering; that pass retains the
// 2^20 limit for surviving tensors to guard register pressure and compile time.
int constexpr maxTensorNumElements = 33554432;

LogicalResult verifyTensorSize(Operation *op);
LogicalResult verifyTensorLayouts(Operation *op);

LogicalResult verifySameOperandsEncoding(Operation *op,
                                         bool allowTensorPointerType = false);
LogicalResult verifyEquivalentType(Type typeA, Type typeB);
LogicalResult
verifySameOperandsAndResultEncoding(Operation *op,
                                    bool allowTensorPointerType = false);

LogicalResult verifySameLoadStoreOperandsShape(Operation *op);

LogicalResult verifySameLoadStoreOperandsAndResultShape(Operation *op);

} // namespace impl

template <class ConcreteType>
class TensorSizeTrait : public TraitBase<ConcreteType, TensorSizeTrait> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyTensorSize(op);
  }
};

// Trait applied to all Triton MLIR ops.  Checks that the layouts of tensors are
// valid.
template <class ConcreteType>
class VerifyTensorLayoutsTrait
    : public TraitBase<ConcreteType, VerifyTensorLayoutsTrait> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyTensorLayouts(op);
  }
};

template <typename ConcreteType>
class SameOperandsAndResultEncoding
    : public TraitBase<ConcreteType, SameOperandsAndResultEncoding> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsAndResultEncoding(op);
  }
};

template <typename ConcreteType>
class SameOperandsEncoding
    : public TraitBase<ConcreteType, SameOperandsEncoding> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsEncoding(op);
  }
};

template <typename ConcreteType>
class SameLoadStoreOperandsShape
    : public TraitBase<ConcreteType, SameLoadStoreOperandsShape> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameLoadStoreOperandsShape(op);
  }
};

template <typename ConcreteType>
class SameLoadStoreOperandsAndResultShape
    : public TraitBase<ConcreteType, SameLoadStoreOperandsAndResultShape> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameLoadStoreOperandsAndResultShape(op);
  }
};

template <typename ConcreteType>
class SameLoadStoreOperandsEncoding
    : public TraitBase<ConcreteType, SameLoadStoreOperandsEncoding> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsEncoding(op,
                                            /*allowTensorPointerType=*/true);
  }
};

template <typename ConcreteType>
class SameLoadStoreOperandsAndResultEncoding
    : public TraitBase<ConcreteType, SameLoadStoreOperandsAndResultEncoding> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsAndResultEncoding(
        op, /*allowTensorPointerType=*/true);
  }
};

// This trait indicates that regions in the op may execute concurrently with
// each other.
template <typename ConcreteType>
struct AsyncRegions : public TraitBase<ConcreteType, AsyncRegions> {};

} // namespace OpTrait
} // namespace mlir

#endif
