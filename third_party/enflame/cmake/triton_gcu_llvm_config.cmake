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

message(STATUS ": set llvm path ${KURAMA_LLVM_DIR}")
message(STATUS ": Use the LLVM version with revision to re-config for the kurama")

set(LLVM_ROOT_DIR ${KURAMA_LLVM_DIR})

if(DEFINED BUILD_CAPS_PATH)
  set(VAR_BUILD_CAPS_PATH "${BUILD_CAPS_PATH}" CACHE PATH "Fallback path to search for BUILD CAPS installs")
elseif(DEFINED ENV{BUILD_CAPS_PATH})
  set(VAR_BUILD_CAPS_PATH "$ENV{BUILD_CAPS_PATH}" CACHE PATH "Fallback path to search for BUILD CAPS installs")
elseif(EXISTS ${CMAKE_BINARY_DIR}/opt/tops)
  set(VAR_BUILD_CAPS_PATH "${CMAKE_BINARY_DIR}/opt/tops" CACHE PATH "search build dir for BUILD CAPS installs")
else()
  set(VAR_BUILD_CAPS_PATH "/opt/tops" CACHE PATH "Fallback path to search for BUILD CAPS installs")
endif()

message(STATUS ": MLIR Default BUILD CAPS toolkit path: ${VAR_BUILD_CAPS_PATH}")

if(DEFINED EXECUTION_CAPS_PATH)
  set(VAR_EXECUTION_CAPS_PATH "${EXECUTION_CAPS_PATH}" CACHE PATH "Fallback path to search for TEST CAPS installs")
elseif(DEFINED ENV{EXECUTION_CAPS_PATH})
  set(VAR_EXECUTION_CAPS_PATH "$ENV{EXECUTION_CAPS_PATH}" CACHE PATH "Fallback path to search for TEST CAPS installs")
else()
  set(VAR_EXECUTION_CAPS_PATH "/opt/tops" CACHE PATH "Fallback path to search for TEST CAPS installs")
endif()

message(STATUS ": MLIR Default TEST CAPS toolkit path: ${VAR_EXECUTION_CAPS_PATH}")

set(LLVM_EXTERNAL_LIT ${VAR_BUILD_CAPS_PATH}/bin/llvm-lit)

set(LLVM_LIBRARY_DIR ${LLVM_ROOT_DIR}/lib)

# LLVM
set(LLVM_DIR ${LLVM_LIBRARY_DIR}/cmake/llvm)
message(STATUS ": llvm found in ${LLVM_DIR}")
find_package(LLVM REQUIRED HINTS ${LLVM_DIR})

# MLIR
set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
message(STATUS ": mlir found in ${MLIR_DIR}")
find_package(MLIR REQUIRED CONFIG HINTS ${MLIR_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(${LLVM_CMAKE_DIR}/TableGen.cmake) # required by AddMLIR
include(${LLVM_CMAKE_DIR}/AddLLVM.cmake)
include(${MLIR_CMAKE_DIR}/AddMLIR.cmake)
