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

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_conversion_library(MLIRTritonGCUTransforms_${arch}
  Combine.cpp
  GCUSupportVerifier.cpp
  GCUWarpSpecialization.cpp
  GCUTleLowerAsyncLoad.cpp
  WarpSpecialization/CodePartitionUtility.cpp
  WarpSpecialization/TaskIdPropagation.cpp
  WarpSpecialization/Utility.cpp
  WarpSpecialization/WSBuffer.cpp
  WarpSpecialization/WSCodePartition.cpp
  WarpSpecialization/WSLowerMem.cpp
  WarpSpecialization/WSSpecialize.cpp
  WarpSpecialization/WSTaskIdPropagate.cpp
  WarpSpecialization/WSTaskPartition.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/triton_gcu/Transforms

  DEPENDS
  TritonGCUTableGen_${arch}
  TritonGCUTransformsPassIncGen_${arch}
  TritonGCUCombineIncGen_${arch}
  triton_gcu400

  LINK_LIBS PUBLIC
  MLIRGPUTransforms
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRSideEffectInterfaces
  )


set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(TritonGCUCombine.inc -gen-rewriters)
add_public_tablegen_target(TritonGCUCombineIncGen_${arch})
