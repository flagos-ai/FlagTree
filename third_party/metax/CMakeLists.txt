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
