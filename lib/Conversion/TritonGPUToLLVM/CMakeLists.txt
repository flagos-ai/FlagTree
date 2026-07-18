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

add_triton_library(TritonGPUToLLVM
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM/FMADotUtility.cpp
    AllocateSharedMemory.cpp
    AllocateSharedMemoryUtility.cpp
    AllocateWarpGroups.cpp
    AssertOpToLLVM.cpp
    ControlFlowOpToLLVM.cpp
    ConvertLayoutOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    FuncOpToLLVM.cpp
    GatherOpToLLVM.cpp
    GlobalScratchMemoryAllocation.cpp
    HistogramOpToLLVM.cpp
    MakeRangeOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    PrintOpToLLVM.cpp
    ReduceOpToLLVM.cpp
    ScanOpToLLVM.cpp
    SPMDOpToLLVM.cpp
    TypeConverter.cpp
    Utility.cpp
    ViewOpToLLVM.cpp

    DEPENDS
    TritonGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRGPUDialect
    MLIRGPUToNVVMTransforms
    MLIRGPUToROCDLTransforms
    MLIRGPUTransforms
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)
