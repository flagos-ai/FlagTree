add_triton_library(EVLinalgTransforms
  InsertDeallocOp.cpp
  RegionMemAllocator.cpp
  MemoryAlloc.cpp
  MemoryAllocPass.cpp
  RemoveLoopIterArgsWithMemrefType.cpp
  SetMemRefScopePass.cpp
  RemoveRedundencyCopyPass.cpp
  RewriteDataType.cpp
  DEPENDS
  EVLinalgTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  MLIRBufferizationTransforms
  MLIRLinalgTransforms
  EvasLinalgIR
)

target_compile_options(EVLinalgTransforms PRIVATE -Wno-deprecated-declarations)
