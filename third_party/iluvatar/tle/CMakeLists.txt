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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include/IR)
add_subdirectory(include/Transforms)

add_triton_library(IluvatarTleIR
  lib/IR/Dialect.cpp

  DEPENDS
  IluvatarTleTableGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)

add_triton_library(IluvatarTleToLLVM
  lib/Conversion/TleToLLVM.cpp
  lib/Conversion/TleToLLVM/TleTileToLLVMUtils.cpp
  lib/Conversion/TleToLLVM/ExtractTileToLLVM.cpp
  lib/Conversion/TleToLLVM/InsertTileToLLVM.cpp
  lib/Conversion/TleToLLVM/LocalPointersOpToLLVM.cpp

  DEPENDS
  IluvatarTleTableGen
  TritonGPUTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  IluvatarTleIR
  TritonGPUToLLVM
)

add_triton_library(IluvatarTleTransforms
  lib/Transforms/IluvatarTleInsertLocalPointerBarriers.cpp
  lib/Transforms/IluvatarTleOptimizeLocalPointerLoads.cpp
  lib/Transforms/IluvatarTleOptimizeLocalPointerStores.cpp

  DEPENDS
  IluvatarTleTableGen
  IluvatarTleTransformsIncGen

  LINK_LIBS PUBLIC
  IluvatarTleIR
  TritonIR
  TritonGPUIR
)
