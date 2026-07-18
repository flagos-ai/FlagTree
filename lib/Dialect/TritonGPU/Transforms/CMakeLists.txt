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

set(TritonGPUTransformsTleSources)
if(FLAGTREE_TLE)
  list(APPEND TritonGPUTransformsTleSources
    ${CMAKE_SOURCE_DIR}/third_party/tle/dialect/lib/Transforms/EncodingRematerialization.cpp
    Pipeliner/TleWGMMAAnalysis.cpp
    Pipeliner/TleWGMMAUserPromisePipeline.cpp
  )
endif()

get_flagtree_backend_lib("TritonGPUTransforms" _EXTRA_LINK_LIBS)

add_triton_library(TritonGPUTransforms
  AccelerateMatmul.cpp
  ProcessSharedMemoryHint.cpp
  Coalesce.cpp
  F32DotTC.cpp
  FuseNestedLoops.cpp
  CombineTensorSelectAndIf.cpp
  DecomposeScaledBlocked.cpp
  HoistTMEMAlloc.cpp
  ReduceDataDuplication.cpp
  OptimizeAccumulatorInit.cpp
  OptimizeDotOperands.cpp
  OptimizeThreadLocality.cpp
  Pipeliner/AssignLatencies.cpp
  Pipeliner/LowerLoops.cpp
  Pipeliner/MMAv5PipelineUtility.cpp
  Pipeliner/ScheduleLoops.cpp
  ${TritonGPUTransformsTleSources}
  Pipeliner/WGMMAPipeline.cpp
  Pipeliner/PipelineExpander.cpp
  Pipeliner/TestPipelineLowerLoop.cpp
  Pipeliner/SoftwarePipeliner.cpp
  Pipeliner/TMAStoresPipeline.cpp
  Pipeliner/MMAv5PipelineUtility.cpp
  Pipeliner/PipeliningUtility.cpp
  Pipeliner/Schedule.cpp
  Prefetch.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  CoalesceAsyncCopy.cpp
  Utility.cpp
  CoalesceUtils.cpp
  LayoutPropagationUtility.cpp
  WarpSpecialization/AutomaticWarpSpecialization.cpp
  WarpSpecialization/Partition.cpp
  WarpSpecialization/OptimizePartitionWarps.cpp
  WarpSpecialization/PartitionBuilder.cpp
  WarpSpecialization/PartitionLoops.cpp
  WarpSpecialization/PartitionScheduling.cpp

  DEPENDS
  TritonGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonTransforms
  TritonGPUIR
  TritonNvidiaGPUIR
  NVWSIR
  NVWSTransforms
  TritonToTritonGPU
  TritonInstrumentIR
  MLIRTransformUtils
  ${_EXTRA_LINK_LIBS}
)
