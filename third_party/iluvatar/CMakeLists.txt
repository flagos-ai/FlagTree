add_compile_options("-Wno-deprecated-declarations")
add_compile_options("-Wno-error=deprecated-declarations")

option(EDITABLE_MODE "Build in developer (editable) mode" OFF)
if(EDITABLE_MODE)
  set(ILUVATAR_PLUGIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
else()
  set (ILUVATAR_PLUGIN_DIR "${Python3_SITELIB}/triton/_C")
endif()
if(TRITON_BUILD_PYTHON_MODULE)
  if(FLAGTREE_PLUGIN)
    add_subdirectory(plugin)
    add_triton_plugin(TritonILUVATAR
      SHARED_LIB iluvatarTritonPlugin
    )
  else()
    if(EDITABLE_MODE)
      find_library(iluvatarTritonPluginLib
      NAMES
        iluvatarTritonPlugin.so
      PATHS
        ${ILUVATAR_PLUGIN_DIR}
      REQUIRED
      )
      add_triton_plugin(TritonILUVATAR
        SHARED_LIB ${iluvatarTritonPluginLib}
      )
    else()
      find_library(iluvatarTritonPluginLib
        NAMES
          iluvatarTritonPlugin.so
        PATHS
        ${ILUVATAR_PLUGIN_DIR}
        REQUIRED
      )
      add_triton_plugin(TritonILUVATAR
        SHARED_LIB ${iluvatarTritonPluginLib}
      )
    endif()
  endif()
endif()

add_subdirectory(backend/spec/lib)
add_subdirectory(${PROJECT_SOURCE_DIR}/include
                 ${PROJECT_BINARY_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib
                 ${PROJECT_BINARY_DIR}/lib)
add_subdirectory(bin)
