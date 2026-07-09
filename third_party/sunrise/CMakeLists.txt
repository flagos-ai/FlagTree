add_compile_options("-Wno-deprecated-declarations")
add_compile_options("-Wno-error=deprecated-declarations")


option(EDITABLE_MODE "Build in developer (editable) mode" OFF)
if(EDITABLE_MODE)
  if(DEFINED ENV{FLAGTREE_CACHE_DIR} AND NOT "$ENV{FLAGTREE_CACHE_DIR}" STREQUAL "")
    set(SUNRISE_PLUGIN_DIR "$ENV{FLAGTREE_CACHE_DIR}/sunrise")
  else()
    set(SUNRISE_PLUGIN_DIR "$ENV{HOME}/.flagtree/sunrise")
  endif()
else()
  set(SUNRISE_PLUGIN_DIR "${Python3_SITELIB}/triton/_C")
endif()

if(TRITON_BUILD_PYTHON_MODULE)
  if(FLAGTREE_PLUGIN)
    add_subdirectory(plugin)
    add_triton_plugin(TritonSunrise
      SHARED_LIB sunriseTritonPlugin
    )
  else()
    find_library(sunriseTritonPluginLib
    NAMES
      sunriseTritonPlugin.so
    PATHS
      ${SUNRISE_PLUGIN_DIR}
    REQUIRED
    )
    add_triton_plugin(TritonSunrise
      SHARED_LIB ${sunriseTritonPluginLib}
    )
  endif()
endif()

add_subdirectory(backend/spec/lib)
add_subdirectory(${PROJECT_SOURCE_DIR}/include
                 ${PROJECT_BINARY_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib
                 ${PROJECT_BINARY_DIR}/lib)
