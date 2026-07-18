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

set(GPU_INSTRUMENTATION_PASSES
	PrintLoadStoreMemSpaces
    )

set(PrintLoadStoreMemSpaces_SOURCES
    PrintLoadStoreMemSpaces.cpp
    )


foreach( plugin ${GPU_INSTRUMENTATION_PASSES} )
    add_library(
      ${plugin}
      SHARED
      ${${plugin}_SOURCES}
      )

    target_link_libraries(
      ${plugin}
      PRIVATE
      LLVMCore
      LLVMSupport
      LLVMTransformUtils
      "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
      )
    # CMAKE_LIBRARY_OUTPUT_DIRECTORY is only set during the Python
    # build. It is empty if building directly from the root
    # CMakeLists.txt file. Therefore if not building from Python just
    # use the default CMake shared lib path otherwise this causes a hard
    # build error
    if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(${plugin} PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY
      "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../instrumentation")
    endif(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)

    # This is set to -fvisibility=hidden in the top level CMake file
    # which causes the llvmGetPassPluginInfo symbol to be hidden and
    # an "entry point not found" error. Reset it just for this target
    if(NOT MSVC)
      target_compile_options(${plugin} PRIVATE -fvisibility=default -fno-rtti)
    endif()
endforeach()
