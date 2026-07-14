add_subdirectory(include)
add_subdirectory(lib)

if(FLAGTREE_PLUGIN)
  set(PLUGIN_DIR "${Python3_SITELIB}/triton/_C")
elseif(EDITABLE_MODE)
  set(PLUGIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
else()
  set(PLUGIN_DIR "$ENV{HOME}/.flagtree/${FLAGTREE_BACKEND}")
endif()

if(TRITON_BUILD_PYTHON_MODULE)
  if(BUILD_MCTLE)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugin)
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/plugin)
    add_subdirectory(plugin/mctle)
  endif()

  if(FLAGTREE_PLUGIN)
    include_directories(plugin)
    add_subdirectory(plugin)
  else()
    find_library(BackendTritonPluginLib
      NAMES
        metaxTritonPlugin.so
      PATHS
        ${PLUGIN_DIR}
      REQUIRED
    )
    add_triton_plugin(TritonMetax
      SHARED_LIB ${BackendTritonPluginLib}
    )
    list(APPEND CMAKE_BUILD_RPATH "$ORIGIN")
    set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" PARENT_SCOPE)
  endif()
endif()
