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

#ifndef TRITON_TO_TILEIR_CONVERSION_PASSES_H
#define TRITON_TO_TILEIR_CONVERSION_PASSES_H

#include "TritonToTileIR/TritonToTileIRPass.h"

namespace mlir {
namespace triton {

// Generate the pass class declarations (and options structs).
#define GEN_PASS_DECL
#include "TritonToTileIR/Passes.h.inc"

// Generate the pass registration.
#define GEN_PASS_REGISTRATION
#include "TritonToTileIR/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_TO_TILEIR_CONVERSION_PASSES_H
