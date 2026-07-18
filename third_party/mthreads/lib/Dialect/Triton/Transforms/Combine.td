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

#ifndef TRITON_PATTERNS
#define TRITON_PATTERNS

include "mlir/Dialect/Arith/IR/ArithOps.td"
include "triton/Dialect/Triton/IR/TritonOps.td"
include "mlir/IR/PatternBase.td"

// addptr(addptr(%ptr, %idx0), %idx1) => addptr(%ptr, AddI(%idx0, %idx1))
//   Note: leave (sub %c0, %c0) canceling to ArithDialect
//         (ref: ArithCanonicalization.td)
defvar DefOverflow = ConstantEnumCase<Arith_IntegerOverflowAttr, "none">;

def CopyDiscardableAttrs: NativeCodeCallVoid<
        "$1.getOwner()->setDiscardableAttrs(triton::filterDiscardableAttrs($0.getOwner(), "
        "{\"tt.divisibility\", \"tt.contiguity\", \"tt.constancy\", \"tt.pointee_type\"}))">;

def CombineAddPtrPattern : Pat<
        (TT_AddPtrOp:$src (TT_AddPtrOp $ptr, $idx0), $idx1),
        (TT_AddPtrOp:$dest $ptr, (Arith_AddIOp $idx0, $idx1, DefOverflow)),
        [(Constraint<CPred<"isAddPtrOffsetCombinable($0, $1)">> $idx0, $idx1)],
        [(CopyDiscardableAttrs $src, $dest)]>;

#endif
