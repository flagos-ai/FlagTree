add_subdirectory(include)
add_subdirectory(lib)

if(FLAGTREE_PLUGIN)
  set(PLUGIN_DIR "${Python3_SITELIB}/triton/_C")
elseif(EDITABLE_MODE)
  set(PLUGIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
else()
  set(PLUGIN_DIR "$ENV{HOME}/.flagtree/mthreads")
endif()
if(TRITON_BUILD_PYTHON_MODULE)
  if(FLAGTREE_PLUGIN)
    add_subdirectory(plugin)
    add_triton_plugin(TritonMTHREADS
      SHARED_LIB mthreadsTritonPlugin
    )
  else()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/plugin/include/CMakeLists.txt")
      # Generate plugin TableGen headers for compile-time consumers without
      # pulling plugin implementation sources into the main build.
      include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugin/include)
      include_directories(${CMAKE_CURRENT_BINARY_DIR}/plugin/include)
      include_directories(${CMAKE_SOURCE_DIR}/third_party/mthreads/include)
      include_directories(${CMAKE_BINARY_DIR}/third_party/mthreads/include)
      add_subdirectory(plugin/include plugin/include)
    else()
      message(WARNING
        "plugin/include is missing; plugin TableGen headers will not be "
        "generated in the prebuilt-plugin flow.")
    endif()
    find_library(BackendTritonPluginLib
      NAMES
        mthreadsTritonPlugin.so
      PATHS
        ${PLUGIN_DIR}
      REQUIRED
    )
    add_triton_plugin(TritonMTHREADS
      SHARED_LIB ${BackendTritonPluginLib}
    )
    list(APPEND CMAKE_BUILD_RPATH "$ORIGIN")
    set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" PARENT_SCOPE)
  endif()
endif()

add_subdirectory(bin)
