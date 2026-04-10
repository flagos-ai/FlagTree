add_compile_options("-Wno-deprecated-declarations")
add_compile_options("-Wno-error=deprecated-declarations")

option(EDITABLE_MODE "Build in developer (editable) mode" OFF)
if(FLAGTREE_PLUGIN)
  set(PLUGIN_DIR "${Python3_SITELIB}/triton/_C")
elseif(EDITABLE_MODE)
  set(PLUGIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
else()
  set(PLUGIN_DIR "$ENV{HOME}/.flagtree/iluvatar")
endif()
if(TRITON_BUILD_PYTHON_MODULE)
  if(FLAGTREE_PLUGIN)
    add_subdirectory(plugin)
    add_triton_plugin(TritonILUVATAR
      SHARED_LIB iluvatarTritonPlugin
    )
  else()
    find_library(BackendTritonPluginLib
      NAMES
        iluvatarTritonPlugin.so
      PATHS
        ${PLUGIN_DIR}
      REQUIRED
    )
    add_triton_plugin(TritonILUVATAR
      SHARED_LIB ${BackendTritonPluginLib}
    )
    list(APPEND CMAKE_BUILD_RPATH "$ORIGIN")
    set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" PARENT_SCOPE)
  endif()
endif()

add_subdirectory(backend/spec/lib)
add_subdirectory(${PROJECT_SOURCE_DIR}/include
                 ${PROJECT_BINARY_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib
                 ${PROJECT_BINARY_DIR}/lib)
add_subdirectory(bin)
