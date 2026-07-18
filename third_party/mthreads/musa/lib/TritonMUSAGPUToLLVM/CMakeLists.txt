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

if(FLAGTREE_MTHREADS_TLE)
  set(_TLE_LIBS MUSATLEToLLVM)
else()
  set(_TLE_LIBS "")
endif()

add_triton_library(TritonMUSAGPUToLLVM
  AllocateSharedMemory.cpp
  BarrierOpToLLVM.cpp
  ConvertLayoutOpToLLVM.cpp
  DotOpToLLVM.cpp
  DotOpToLLVM/SQMMA.cpp
  DotOpToLLVM/WMMA.cpp
  ElementwiseOpToLLVM.cpp
  Fp4ToFpOpToLLVM.cpp
  LoadStoreOpToLLVM.cpp
  MUSAOpsToLLVM.cpp
  SPMDOpToLLVM.cpp
  TensorPtrOpsToLLVM.cpp
  TritonGPUToLLVM.cpp
  TargetInfo.cpp
  ThreadIdOpToLLVM.cpp
  Utility.cpp
  WarpIdOpToLLVM.cpp

  DEPENDS
  TritonMUSAGPUConversionPassIncGen
  MUSAAttrDefsIncGen
  MTGPUTableGen
  MTGPUTypesIncGen

  LINK_LIBS PUBLIC
  TritonAnalysis
  TritonGPUToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRUBToLLVM
  MTGPUIR
  MUSAIR
  ${_TLE_LIBS}
  MLIRGPUToMTVMTransforms
)
