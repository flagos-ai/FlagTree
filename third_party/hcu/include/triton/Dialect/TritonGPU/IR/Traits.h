// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITONGPU_IR_TRAITS_H_
#define TRITONGPU_IR_TRAITS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir {
namespace OpTrait {

template <typename ConcreteType>
class MemDescViewTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, MemDescViewTrait> {
  // Optional: Add methods or verification logic here
};

template <typename ConcreteType>
class LocalLoadTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, LocalLoadTrait> {
  // Optional: Add methods or verification logic here
};

} // namespace OpTrait
} // namespace mlir

#endif
