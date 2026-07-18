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

#ifndef TRITONXPU_ENUMS
#define TRITONXPU_ENUMS

include "mlir/IR/EnumAttr.td"

def TT_DtypeAttr : I32EnumAttr<
    "Dtype", "",
    [
        I32EnumAttrCase<"UNKNOWN", 0, "unknown">,
        I32EnumAttrCase<"FP32", 1, "fp32">,
        I32EnumAttrCase<"FP16", 2, "fp16">,
    ]> {
    let cppNamespace = "::mlir::triton";
}

def TT_OffsetStatePolicyAttr : I32EnumAttr<
    "OffsetStatePolicy", "",
    [
        I32EnumAttrCase<"EMPTY", 0, "empty">,
        I32EnumAttrCase<"UNKNOWN", 1, "unknown">,
        I32EnumAttrCase<"DISCRETE_SAME", 2, "discrete_same">,
        I32EnumAttrCase<"CONTINUOUS", 3, "continuous">,
        I32EnumAttrCase<"DISCRETE", 4, "discrete">,
        I32EnumAttrCase<"LOCALLY_CONTINUOUS", 5, "locally_continuous">
    ]> {
    let cppNamespace = "::mlir::triton";
}

def TT_SyncAttr : I32EnumAttr<
    "MemorySyncMode", "",
    [
        I32EnumAttrCase<"SYNC", 1, "sync">,
        I32EnumAttrCase<"ASYNC", 2, "async">,
    ]> {
    let cppNamespace = "::mlir::triton";
}

#endif // TRITONXPU_ENUMS
