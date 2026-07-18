# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
