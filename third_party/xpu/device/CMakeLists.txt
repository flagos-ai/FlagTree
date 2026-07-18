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

function(add_xpu_libdevice OUTPUT SRC ARCH)
  set(CLANG ${LLVM_TOOLS_BINARY_DIR}/clang)

  add_custom_target(
    libdevice-${ARCH} ALL
    COMMAND ${CLANG} --xpu-arch=${ARCH} ${SRC} -c -emit-llvm --xpu-device-only -O3 -o ${OUTPUT} -std=c++11 -Wno-literal-range
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/liblaunch.a ${XPU3_LIB_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/liblaunch_shared.so ${XPU3_LIB_DIR}/../so
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/libLLVM-15.so ${XPU3_LIB_DIR}/../so
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/libclang-cpp.so.15 ${XPU3_LIB_DIR}/../so
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/libxpujitc.so ${XPU3_LIB_DIR}/../so
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/libdevice-xpu3s.bc ${XPU3_LIB_DIR}
    DEPENDS ${SRC}
    COMMENT "Building libdevice-${ARCH} ..."
    VERBATIM
  )
endfunction()

set(XPU_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/trigonometric.xpu)
set(XPU3_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../backend/xpu3/lib/)
file(MAKE_DIRECTORY ${XPU3_LIB_DIR})
add_xpu_libdevice(${XPU3_LIB_DIR}/libdevice-xpu3.bc ${XPU_LIB_SRC} xpu3)
