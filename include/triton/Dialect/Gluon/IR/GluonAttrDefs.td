// Copyright 2018-2020 Philippe Tillet
// Copyright 2020-2022 OpenAI
// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef GLUON_ATTRDEFS
#define GLUON_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/Gluon/IR/GluonDialect.td"

def Gluon_AutoEncodingAttr : AttrDef<Gluon_Dialect, "AutoEncoding"> {
  let mnemonic = "auto_encoding";
  let attrName = "gluon.auto_encoding";
  let description = [{
    An encoding that is inferred from neighboring ops in the graph.
  }];
}

def Gluon_CoalescedEncodingAttr : AttrDef<Gluon_Dialect, "CoalescedEncoding"> {
  let mnemonic = "coalesced_encoding";
  let attrName = "gluon.coalesced_encoding";
  let description = [{
    An encoding that is optimized for load/store performance.
  }];
}

#endif
