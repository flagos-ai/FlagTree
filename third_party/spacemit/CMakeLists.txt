set(SPINE_TRITON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(SPINE_TRITON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(".")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
include_directories(${Python3_INCLUDE_DIR})
include_directories(${pybind11_INCLUDE_DIR})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(backend)
add_subdirectory(tools/spine-triton-opt)

add_triton_plugin(SpineTriton ${CMAKE_CURRENT_SOURCE_DIR}/triton_shared.cc LINK_LIBS TritonSharedAnalysis TritonToLinalg TritonTilingExtIR XSMTIR TleIR TLEToLinalg ProtonIR MLIRPtrDialect MLIRParser MLIRLinalgDialect MLIRMemRefDialect MLIRVectorDialect MLIRSCFDialect MLIRMathDialect MLIRBufferizationDialect MLIRFuncDialect MLIRFuncInlinerExtension)
target_link_libraries(SpineTriton PRIVATE Python3::Module pybind11::headers ${Python3_LIBRARIES})

# Explicitly declare tablegen dependencies to prevent race conditions in parallel builds.
# TleTableGen (generic third_party/tle dialect) is wired up from the root
# CMakeLists late block: third_party/spacemit is configured BEFORE the tle
# plugin defines its targets, so TleTableGen cannot be referenced here.
add_dependencies(SpineTriton
  XSMTTableGen
  XSMTAsyncTableGen
  TritonStructuredTableGen
  TritonTilingExtOpsIncGen
  TritonTilingExtInterfacesIncGen
  TLEToLinalgConversionPassIncGen
)
