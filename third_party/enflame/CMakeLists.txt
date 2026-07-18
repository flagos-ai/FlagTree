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

#set(CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)


set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

if (DEFINED ENV{LLVM_SYSPATH})
else()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/includexxx)

# Add cmake module path so that include() can find the corresponding files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/2nd")

# Add build for triton_enflame plugin
add_triton_plugin(TritonEnflame
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_enflame.cc

    DEPENDS
    TritonTableGen
)

# Set include directories
target_include_directories(TritonEnflame PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_gcu/include
)

# Link necessary libraries
target_link_libraries(TritonEnflame PRIVATE
    # MLIR core libraries
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIROptLib

    # Python binding libraries
    Python3::Module
    pybind11::headers

    # If GCU-related libraries need to be linked, they can be added here
    # GCUIRgcu300
    # TritonGCUIR_gcu300
    # MLIRTritonGCUCommon_gcu300
    # MLIRTritonToGCU_gcu300
    # MLIRTritonGCUTransforms_gcu300
)

# Set compilation options
target_compile_definitions(TritonEnflame PRIVATE
    -DTRITON_ENFLAME_VERSION="1.0.0"
)


set(HACK_INCLUDE_DIRS true)
if(HACK_INCLUDE_DIRS)

    get_property(inc_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)

    list(REMOVE_ITEM inc_dirs "${CMAKE_SOURCE_DIR}/third_party")
    list(REMOVE_ITEM inc_dirs "${CMAKE_BINARY_DIR}/third_party")
    list(REMOVE_ITEM inc_dirs "${MLIR_INCLUDE_DIRS}")
    list(REMOVE_ITEM inc_dirs "${LLVM_INCLUDE_DIRS}")

    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES "${include_dirs}")
endif()

# Install python directory to install prefix
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/
    DESTINATION python
    USE_SOURCE_PERMISSIONS
)

# Add triton_gcu subdirectory build
add_subdirectory(triton_gcu)
