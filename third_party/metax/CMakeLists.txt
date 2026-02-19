add_subdirectory(include)
add_subdirectory(lib)

option(EDITABLE_MODE "Build in developer (editable) mode" OFF)
if(FLAGTREE_PLUGIN)
  set(METAX_PLUGIN_DIR "${Python3_SITELIB}/triton/_C")
elseif(EDITABLE_MODE)
  set(METAX_PLUGIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
else()
  set(METAX_PLUGIN_DIR "${Python3_SITELIB}/triton/_C")
endif()
if(TRITON_BUILD_PYTHON_MODULE)
  if(FLAGTREE_PLUGIN)
    add_subdirectory(plugin)
    add_triton_plugin(TritonMetax
      SHARED_LIB metaxTritonPlugin
    )
  else()
    find_library(metaxTritonPluginLib
      NAMES
        metaxTritonPlugin.so
      PATHS
        ${METAX_PLUGIN_DIR}
      REQUIRED
    )
    add_triton_plugin(TritonMetax
      SHARED_LIB ${metaxTritonPluginLib}
    )
  endif()
endif()

add_subdirectory(bin)
