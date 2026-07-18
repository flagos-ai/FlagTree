// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//===----------------------------------------------------------------------===//
// CGA encoding attribute definition emitted early to break interface cycles.
//===----------------------------------------------------------------------===//

#ifndef TRITONGPU_CGAENCODING_ATTR_TD
#define TRITONGPU_CGAENCODING_ATTR_TD

include "triton/Dialect/TritonGPU/IR/TritonGPUAttrBase.td"

//===----------------------------------------------------------------------===//
// CGA Layout
//===----------------------------------------------------------------------===//

def CGAEncodingAttr : TritonGPU_Attr<"CGAEncoding", "cga_encoding"> {
  let parameters = (ins LinearLayoutParam:$linearLayout);

  let description = [{
Describes how blocks (CTAs) in a cooperative thread array (CGA) map onto logical
tensor dimensions. The `LinearLayout` maps from `block` into `dim0`, `dim1`...
  }];

  let extraClassDeclaration = [{
    // Map with empty bases and dims [dim0, dim1, ...]
    static CGAEncodingAttr get1CTALayout(MLIRContext *context, int rank);
    // Map with bases = [[1,], [2,], ..., [numCTAs/2]] into dim0
    static CGAEncodingAttr get1DLayout(MLIRContext *context, int numCTAs);
    // Legacy, we should kill this! Note that it is not true in general that
    // fromSplitParams(enc.getCTAsPerCGA(), enc.getCTASplitNum(), enc.getCTAOrder()) == enc!!
    static CGAEncodingAttr fromSplitParams(MLIRContext *context,
                                           ArrayRef<unsigned> CTAsPerCGA,
                                           ArrayRef<unsigned> CTASplitNum,
                                           ArrayRef<unsigned> CTAOrder);

    unsigned getRank() const { return getLinearLayout().getNumOutDims(); }
    SmallVector<unsigned> getCTAsPerCGA() const;
    SmallVector<unsigned> getCTASplitNum() const;
    SmallVector<unsigned> getCTAOrder() const;
  }];

  let genVerifyDecl = 1;
}

#endif // TRITONGPU_CGAENCODING_ATTR_TD
