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

#ifndef TRITON_GCU_PATTERNS
#define TRITON_GCU_PATTERNS

include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Math/IR/MathOps.td"
include "triton/Dialect/Triton/IR/TritonOps.td"
include "mlir/IR/PatternBase.td"

def SplatFloatElementsAttr0 : ElementsAttrBase<
  CPred<"::llvm::isa<::mlir::DenseElementsAttr>($_self) &&"
  "::llvm::cast<::mlir::DenseElementsAttr>($_self).isSplat() &&"
  "::llvm::cast<::mlir::DenseElementsAttr>($_self).getElementType().isF32() &&"
  "::llvm::cast<::mlir::DenseElementsAttr>($_self).getSplatValue<APFloat>().isZero()">,
  "the value of the splat elements attribute is a floating-point number 0"> {

  let storageType = [{ ::mlir::DenseElementsAttr }];
  let returnType = [{ ::mlir::DenseElementsAttr }];

  let convertFromStorage = "$_self";
}

def SplatFloatElementsAttr1 : ElementsAttrBase<
  CPred<"::llvm::isa<::mlir::DenseElementsAttr>($_self) &&"
  "::llvm::cast<::mlir::DenseElementsAttr>($_self).isSplat() &&"
  "::llvm::cast<::mlir::DenseElementsAttr>($_self).getElementType().isF32() &&"
  "::llvm::cast<::mlir::DenseElementsAttr>($_self).getSplatValue<float>() == 1.0">,
  "the value of the splat elements attribute is a floating-point number 1"> {

  let storageType = [{ ::mlir::DenseElementsAttr }];
  let returnType = [{ ::mlir::DenseElementsAttr }];

  let convertFromStorage = "$_self";
}

// 1 / (1 + math.exp(-x))
def CombineSigmoidPattern : Pat<
        (Arith_DivFOp:$res (ConstantLikeMatcher SplatFloatElementsAttr1),
                           (Arith_AddFOp (Math_ExpOp (Arith_SubFOp (ConstantLikeMatcher SplatFloatElementsAttr0), F32Tensor:$x, $_), $_),
                                         (ConstantLikeMatcher SplatFloatElementsAttr1), $_), $_),
        (TT_ExternElementwiseOp (variadic:$inputs $x), ConstantStrAttr<StrAttr, "">, ConstantStrAttr<StrAttr, "">,
        ConstantStrAttr<StrAttr, "__gcu_sigmoid">, ConstBoolAttrTrue, (location $res)), [], [], (addBenefit 30)>;

// math.log(1 + math.exp(x))
def CombineSoftplusPattern : Pat<
        (Math_LogOp:$res (Arith_AddFOp (Math_ExpOp F32Tensor:$x, $_), (ConstantLikeMatcher SplatFloatElementsAttr1), $_), $_),
        (TT_ExternElementwiseOp (variadic:$inputs $x), ConstantStrAttr<StrAttr, "">, ConstantStrAttr<StrAttr, "">,
        ConstantStrAttr<StrAttr, "__gcu_softplus">, ConstBoolAttrTrue, (location $res)), [], [], (addBenefit 30)>;
#endif
