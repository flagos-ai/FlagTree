# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

if(FLAGTREE_ILUVATAR_TLE)
    set(_ILUVATAR_TLE_LIBS IluvatarTleToLLVM)
else()
    set(_ILUVATAR_TLE_LIBS "")
endif()

add_triton_library(TritonILUVATARGPUToLLVM
    AtomicRMWOpsEmitter.cpp
    MaskedOpsToLLVM.cpp
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM/TCU.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    Utility.cpp
    TargetInfo.cpp
    SPMDOpToLLVM.cpp

    DEPENDS
    TritonILUVATARGPUConversionPassIncGen
    LLVMIRIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
    TritonILUVATARGPUIR
    TritonILUVATARUtils
    ${_ILUVATAR_TLE_LIBS}
    MLIRUBToLLVM
    LLVMCore
    LLVMPasses
    LLVMSupport
)
