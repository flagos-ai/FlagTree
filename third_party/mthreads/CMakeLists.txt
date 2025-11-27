add_subdirectory(include)
add_subdirectory(lib)

if(EDITABLE_MODE)
  set(MTHREADS_PLUGIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
else()
  set (MTHREADS_PLUGIN_DIR "${Python3_SITELIB}/triton/_C")
endif()
if(TRITON_BUILD_PYTHON_MODULE)
  if(FLAGTREE_PLUGIN)
    add_subdirectory(plugin)
    add_triton_plugin(TritonMTHREADS
      SHARED_LIB mthreadsTritonPlugin
    )
  else()
    if(EDITABLE_MODE)
      find_library(mthreadsTritonPluginLib
      NAMES
        mthreadsTritonPlugin.so
      PATHS
      ${CMAKE_CURRENT_SOURCE_DIR}
      REQUIRED
      )
      add_triton_plugin(TritonMTHREADS
        SHARED_LIB ${mthreadsTritonPluginLib}
      )
    else()
      find_library(mthreadsTritonPluginLib
        NAMES
          mthreadsTritonPlugin.so
        PATHS
        ${MTHREADS_PLUGIN_DIR}
        REQUIRED
      )
      add_triton_plugin(TritonMTHREADS
        SHARED_LIB ${mthreadsTritonPluginLib}
      )
    endif()
  endif()
endif()

add_subdirectory(bin)
