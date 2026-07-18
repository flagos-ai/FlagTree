include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils/include)
add_subdirectory(dialect)

if(FLAGTREE_PLUGIN)
  add_triton_plugin(TritonMCTLE
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_mctle.cc
    LINK_LIBS TritonIR
  )
  target_link_libraries(TritonMCTLE PRIVATE
    Python3::Module
    pybind11::headers
    MLIRTargetLLVMIRImport
  )
  add_dependencies(TritonMCTLE
    McTleTableGen
    TritonTLETransformsIncGen
  )

else()
  add_triton_plugin(TritonMCTLE
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_mctle.cc
    LINK_LIBS TritonIR
  )
  target_link_libraries(TritonMCTLE PRIVATE
    ${metaxTritonPluginLib}
    Python3::Module
    pybind11::headers
    MLIRTargetLLVMIRImport
  )
  add_dependencies(TritonMCTLE
    McTleTableGen
    TritonTLETransformsIncGen
  )
endif()
