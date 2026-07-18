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

include(FetchContent)

set(GOOGLETEST_DIR "" CACHE STRING "Location of local GoogleTest repo to build against")

if(NOT GOOGLETEST_DIR)
  set(_default_googletest "$ENV{HOME}/.triton/googletest-1.17.0")
  if(EXISTS "${_default_googletest}")
    set(GOOGLETEST_DIR "${_default_googletest}" CACHE STRING "Location of local GoogleTest repo to build against" FORCE)
  else()
    message("Dowloading https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz")
  endif()
  unset(_default_googletest)
endif()

if(GOOGLETEST_DIR)
  set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST ${GOOGLETEST_DIR} CACHE STRING "GoogleTest source directory override")
endif()

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
  )

FetchContent_GetProperties(googletest)

if(NOT googletest_POPULATED)
  FetchContent_MakeAvailable(googletest)
  if (MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  endif()
endif()
