include_directories(${CMAKE_CURRENT_SOURCE_DIR}/backend/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/backend/include)
if(FLAGTREE_ILUVATAR_TLE)
  # Reuse the backend-neutral distributed-barrier dialect.  Iluvatar supplies
  # its own lowering below, while sharing the TLE Python builder contract.
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/tle/dialect/include
                   ${PROJECT_BINARY_DIR}/third_party/tle/dialect/include)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/tle/dialect/lib/IR
                   ${PROJECT_BINARY_DIR}/third_party/tle/dialect/lib/IR)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tle/dialect/include)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/tle/dialect/include)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tle/utils/include)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/tle/utils/include)
  include_directories(${PROJECT_SOURCE_DIR}/third_party/tle/dialect/include)
  include_directories(${PROJECT_BINARY_DIR})
  add_subdirectory(tle)
  add_subdirectory(tle_raw)
  add_dependencies(TritonAnalysis IluvatarTleTableGen)
endif()
add_subdirectory(backend/include)
add_subdirectory(backend/lib)
if(TRITON_BUILD_PYTHON_MODULE)
  find_package(LLD REQUIRED CONFIG PATHS "${LLD_DIR}")
  include_directories(${LLD_INCLUDE_DIRS})
  message(STATUS "Found LLD distro-package @ ${LLD_DIR} and LLD include dirs @ ${LLD_INCLUDE_DIRS}")
  if(FLAGTREE_ILUVATAR_TLE)
    set(_ILUVATAR_TLE_PLUGIN_SOURCES
      ${CMAKE_CURRENT_SOURCE_DIR}/tle/triton_iluvatar_tle.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/tle/triton_iluvatar_tle_raw.cc)
    set(_ILUVATAR_TLE_PLUGIN_LIBS IluvatarTleIR IluvatarTleTransforms
      TleIR IluvatarTleRawUtils IluvatarTLERawTransforms MLIRTargetLLVMIRImport)
    set(_ILUVATAR_TLE_PLUGIN_DEPS IluvatarTleTableGen IluvatarTleTransformsIncGen
      TleTableGen IluvatarTLERawTransformsIncGen)
  else()
    set(_ILUVATAR_TLE_PLUGIN_SOURCES "")
    set(_ILUVATAR_TLE_PLUGIN_LIBS "")
    set(_ILUVATAR_TLE_PLUGIN_DEPS "")
  endif()
  add_triton_plugin(TritonILUVATAR
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_iluvatar.cc
    ${_ILUVATAR_TLE_PLUGIN_SOURCES}
    LINK_LIBS TritonILUVATARGPUToLLVM ${_ILUVATAR_TLE_PLUGIN_LIBS})
  if(FLAGTREE_ILUVATAR_TLE)
    add_dependencies(TritonILUVATAR ${_ILUVATAR_TLE_PLUGIN_DEPS})
  endif()
  target_link_libraries(TritonILUVATAR PRIVATE Python3::Module pybind11::headers lldCommon lldELF)
endif()
if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
# add_subdirectory(test)
