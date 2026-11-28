include_directories(${CMAKE_CURRENT_SOURCE_DIR}/backend/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/backend/include)
if(FLAGTREE_ILUVATAR_TLE)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tle/dialect/include)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/tle/dialect/include)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tle/utils/include)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/tle/utils/include)
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
      IluvatarTleRawUtils IluvatarTLERawTransforms MLIRTargetLLVMIRImport)
    set(_ILUVATAR_TLE_PLUGIN_DEPS IluvatarTleTableGen IluvatarTleTransformsIncGen
      IluvatarTLERawTransformsIncGen)
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
  if(FLAGTREE_COMMON_IR_ENABLED)
    if(NOT TARGET CommonIRIR)
      message(FATAL_ERROR "FLAGTREE_COMMON_IR requires the FLIR CommonIR dialect")
    endif()
    if(NOT TARGET TleCommonIRToTTGIR)
      add_subdirectory(
        ${PROJECT_SOURCE_DIR}/third_party/tle/include/CommonIRToTTGIR
        ${PROJECT_BINARY_DIR}/third_party/tle/include/CommonIRToTTGIR)
      add_subdirectory(
        ${PROJECT_SOURCE_DIR}/third_party/tle/lib/CommonIRToTTGIR
        ${PROJECT_BINARY_DIR}/third_party/tle/lib/CommonIRToTTGIR)
    endif()
    target_link_libraries(TritonILUVATAR PRIVATE CommonIRIR TleCommonIRToTTGIR)
    target_include_directories(TritonILUVATAR PRIVATE
      ${PROJECT_SOURCE_DIR}/third_party
      ${PROJECT_BINARY_DIR}/third_party)
    add_dependencies(TritonILUVATAR
      CommonIRTableGen
      CommonIRAttrDefsIncGen
      CommonIRTypesIncGen
      TleCommonIRToTTGIRPassIncGen)
  endif()
  target_link_libraries(TritonILUVATAR PRIVATE Python3::Module pybind11::headers lldCommon lldELF)
endif()
if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
# add_subdirectory(test)
