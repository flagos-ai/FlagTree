# The shared Triton core targets are created by FlagTreeOptions.cmake. Keep
# this backend CMake file focused on MetaX-only sources and plugins.
target_sources(TritonTools PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/spec_cpp/lib/Tools/SwizzleDumpUtils.cpp
)

if(TRITON_BUILD_PYTHON_MODULE)
  if(BUILD_MCTLE)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugin)
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/plugin)
    add_subdirectory(plugin/mctle)
  endif()

  include_directories(plugin)
  add_subdirectory(plugin)
  if(BUILD_MCTLE AND NOT FLAGTREE_PLUGIN)
    # The public SDK provides MCTLE passes whose sources are not distributed.
    find_file(MetaxSdkPlugin metaxTritonPlugin.so
      HINTS "${CMAKE_CURRENT_SOURCE_DIR}" "$ENV{FLAGTREE_OFFLINE_BUILD_DIR}"
            "$ENV{HOME}/.flagtree/metax"
      REQUIRED)
    set(packaged_plugin "${TRITON_WHEEL_DIR}/_C/metaxTritonPlugin.so")
    configure_file("${MetaxSdkPlugin}" "${packaged_plugin}" COPYONLY)
    # Standalone MLIR tools also need to resolve the SDK's Python bindings.
    find_package(Python3 REQUIRED COMPONENTS Development.Embed)
    target_link_libraries(TritonMETAXGPUToLLVM PUBLIC "${packaged_plugin}"
      "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:Python3::Python>")
    set(CMAKE_BUILD_RPATH "$ORIGIN;${CMAKE_BUILD_RPATH}" PARENT_SCOPE)
  endif()
endif()

if(BUILD_MCTLE AND TRITON_BUILD_PYTHON_MODULE)
  add_dependencies(TritonAnalysis McTleTableGen)
  add_dependencies(TritonMETAXGPUToLLVM McTleTableGen)
  # FlagPrism：generate MCTLE headers before compiling MetaX TritonGPUIR sources.
  add_dependencies(TritonGPUIR McTleTableGen)
  # FlagPrism end
  target_link_libraries(TritonToTritonGPU PUBLIC McTleIR)
endif()
