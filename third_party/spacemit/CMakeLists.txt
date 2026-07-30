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
add_subdirectory(test)
add_subdirectory(tools/spine-triton-opt)

add_triton_plugin(SpineTriton ${CMAKE_CURRENT_SOURCE_DIR}/triton_shared.cc LINK_LIBS TritonSharedAnalysis TritonToLinalg TritonTilingExtIR XSMTIR TLEIR TLEToLinalg ProtonIR MLIRPtrDialect)
target_link_libraries(SpineTriton PRIVATE Python3::Module pybind11::headers ${Python3_LIBRARIES})

# Explicitly declare tablegen dependencies to prevent race conditions in parallel builds
add_dependencies(SpineTriton
  XSMTTableGen
  XSMTAsyncTableGen
  TritonStructuredTableGen
  TritonTilingExtOpsIncGen
  TritonTilingExtInterfacesIncGen
  TLETableGen
  TLEToLinalgConversionPassIncGen
)
