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

#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_TARGETUTILS_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_TARGETUTILS_H_

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::AMD {

// A list of ISA families we care about.
enum class ISAFamily {
  Unknown,
  CDNA1,
  CDNA2,
  CDNA3,
  CDNA4,
  RDNA1,
  RDNA2,
  RDNA3,
  RDNA4,
  GFX1250,
};

// Deduces the corresponding ISA family for the given target gfx |arch|.
ISAFamily deduceISAFamily(llvm::StringRef arch);

// Retursn true if given architecture support V_DOT instruction.
bool supportsVDot(llvm::StringRef arch);

bool isCDNA(ISAFamily isaFamily);

bool isRDNA(ISAFamily isaFamily);

// Here is a partial definition of DppCtrl enums. For the complete definition,
// please check:
// https://github.com/llvm/llvm-project/blob/8c75290/llvm/lib/Target/AMDGPU/SIDefines.h#L939
enum class DppCtrl : uint32_t {
  QUAD_PERM_FIRST = 0,
  ROW_SHL0 = 0x100,
  ROW_SHR0 = 0x110,
  BCAST15 = 0x142,
  BCAST31 = 0x143
};

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_TARGETUTILS_H_
