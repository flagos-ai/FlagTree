/*
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

#ifndef TLE_RAW_ATTR_DEFS
#define TLE_RAW_ATTR_DEFS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "tle/dialect/include/IR/TleDialect.td"

def Tle_FlagCxWaitKindAttr : I32EnumAttr<
  "FlagCxWaitKind", "",
  [
    I32EnumAttrCase<"SIGNAL", 0, "signal">,
    I32EnumAttrCase<"COUNTER", 1, "counter">,
    I32EnumAttrCase<"SHADOW", 2, "shadow">,
  ]> {
  let cppNamespace = "::mlir::triton::tle";
}

def Tle_FlagCxCoopKindAttr : I32EnumAttr<
  "FlagCxCoopKind", "",
  [
    I32EnumAttrCase<"THREAD", 0, "thread">,
    I32EnumAttrCase<"WARP", 1, "warp">,
    I32EnumAttrCase<"BLOCK", 2, "block">,
  ]> {
  let cppNamespace = "::mlir::triton::tle";
}

#endif
