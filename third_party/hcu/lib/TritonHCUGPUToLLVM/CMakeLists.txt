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

if(FLAGTREE_TLE)
    set(_TLE_LIBS
        TritonTLEAnalysis
        TritonNVIDIAGPUToLLVM
        TleToLLVM # flagtree tle raw
        TritonTLETransforms
    )
else()
    set(_TLE_LIBS "")
endif()

add_triton_library(TritonHCUGPUToLLVM
    AsyncUtility.cpp
    AtomicRMWOpsEmitter.cpp
    AllocateSharedMemory.cpp
    BufferOpsEmitter.cpp
    TensorPtrOpsToLLVM.cpp
    ConvertLayoutOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    MaskedOpsToLLVM.cpp
    DotOpToLLVM/FMA.cpp
    # DotOpToLLVM/MFMA.cpp
    DotOpToLLVM/WMMA.cpp
    DotOpToLLVM.cpp
    # ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    GCNAsmFormat.cpp
    TritonGPUToLLVM.cpp
    BuiltinFuncToLLVM.cpp
    Utility.cpp
    TargetInfo.cpp
    TargetUtils.cpp
    OptimizeLDSUsage.cpp
    OptimizeLDSUtility.cpp
    SPMDOpToLLVM.cpp
    SchedInstructions.cpp
    UpcastMXFPToLLVM.cpp
    Fp4ToFpOpToLLVM.cpp
    MembarUtility.cpp
    WaitCntHCUUtility.cpp
    ScalarizePackedFOps.cpp
    TDMUtility.cpp
    BarrierOpToLLVM.cpp
    # ---- HCU code ---- #
    DotOpToLLVM/MMAC.cpp
    ElementwiseOpHCUToLLVM.cpp
    MlsOpToLLVM.cpp
    ConvertWarpSpecializeToLLVM.cpp
    SetFuncInlineAttrs.cpp

    DEPENDS
    TritonHCUGPUConversionPassIncGen
    LLVMIRIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
    TritonHCUGPUIR
    ${_TLE_LIBS}
    LLVMCore
    LLVMPasses
    LLVMSupport
)
