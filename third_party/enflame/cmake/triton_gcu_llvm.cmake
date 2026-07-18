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

# 使用预编译的 LLVM
# 检查环境变量中指定的 LLVM 路径
function(setup_llvm_download LLVM_HASH OUT_LLVM_DIR)
    if(DEFINED ENV{KURAMA_LLVM_DIR})
        message(STATUS ": using user provide llvm path $ENV{KURAMA_LLVM_DIR}")
        set(KURAMA_LLVM_DIR "$ENV{KURAMA_LLVM_DIR}")
    elseif(KURAMA_LLVM_DIR AND EXISTS ${KURAMA_LLVM_DIR}/lib/cmake)
        message(STATUS ": using previous exists llvm")
    else()
        message(FATAL_ERROR "KURAMA_LLVM_DIR environment variable is not set or LLVM not found at specified path")
    endif()

    set(LLVM_INCLUDE_DIRS ${KURAMA_LLVM_DIR}/include)
    set(LLVM_LIBRARY_DIR ${KURAMA_LLVM_DIR}/lib)
    set(${OUT_LLVM_DIR} ${KURAMA_LLVM_DIR} PARENT_SCOPE)
endfunction()

if(DEFINED ENV{KURAMA_LLVM_DIR})
    message(STATUS ": using user provide llvm path $ENV{KURAMA_LLVM_DIR}")
    set(KURAMA_LLVM_DIR "$ENV{KURAMA_LLVM_DIR}")
elseif(KURAMA_LLVM_DIR AND EXISTS ${KURAMA_LLVM_DIR}/lib/cmake)
    message(STATUS ": using previous exists llvm")
else()
    message(FATAL_ERROR "KURAMA_LLVM_DIR environment variable is not set or LLVM not found at specified path")
endif()
