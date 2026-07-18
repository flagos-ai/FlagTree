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

project(Proton LANGUAGES CXX)

set(PROTON_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
set(PROTON_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")

# ============ Check for includes =============
if(NOT CUPTI_INCLUDE_DIR)
  message(FATAL_ERROR "CUPTI include directory not defined")
endif()
if(NOT ROCTRACER_INCLUDE_DIR)
  message(FATAL_ERROR "ROCTRACER include directory not defined")
endif()
if(NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "JSON include directory not defined")
endif()

# ============ Dependencies =============
find_package(Python3 REQUIRED Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

# ============ Define a GLOBAL property to store object-libraries ============
set_property(GLOBAL PROPERTY PROTON_LIBS "")

# ============ Define a function to create object libraries ============
function(add_proton_library name)
  add_library(${name} OBJECT ${ARGN})

  target_link_libraries(${name} PRIVATE Python3::Module pybind11::headers)

  # Use system to skip warnings caused by legacy clang compilers
  target_include_directories(${name}
    SYSTEM PRIVATE
      "${ROCTRACER_INCLUDE_DIR}"
  )

  target_include_directories(${name}
    PRIVATE
      "${CUPTI_INCLUDE_DIR}"
      "${JSON_INCLUDE_DIR}"
      "${PROTON_COMMON_DIR}/include"
      "${PROTON_SRC_DIR}/include"
  )

  # If HIP is AMD-based
  target_compile_definitions(${name} PRIVATE __HIP_PLATFORM_AMD__)

  # Append this library name to the GLOBAL property "PROTON_LIBS"
  set_property(GLOBAL APPEND PROPERTY PROTON_LIBS ${name})
endfunction()

# ============ Add subdirectory with actual code that calls add_proton_library ============
add_subdirectory("${PROTON_COMMON_DIR}")
add_subdirectory("${PROTON_SRC_DIR}")

# ============ Add subdirectory with proton tests ============
add_subdirectory(test)

# ============ Possibly handle macOS specifics ============
if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  # Other platforms build with -flto, but we found that this adds significant overhead to our macos CI without providing a major benefit.
  set(PROTON_PYTHON_LDFLAGS "-undefined dynamic_lookup")
endif()

# ============ Collect all object libraries from property and build final shared lib ============
get_property(_proton_obj_libs GLOBAL PROPERTY PROTON_LIBS)

if(NOT _proton_obj_libs)
  message(WARNING "No object libraries were defined in 'PROTON_LIBS'!")
endif()

set(_proton_obj_sources "")
foreach(_lib IN LISTS _proton_obj_libs)
  list(APPEND _proton_obj_sources $<TARGET_OBJECTS:${_lib}>)
  message(STATUS "Collecting object files from ${_lib}")
endforeach()

add_library(proton SHARED ${_proton_obj_sources})

target_link_libraries(proton PRIVATE Python3::Module)
# Apply any macOS linker flags or extra link options
if(PROTON_PYTHON_LDFLAGS)
  target_link_options(proton PRIVATE ${PROTON_PYTHON_LDFLAGS})
endif()
