# ============================================================================
# XPU SDNN Object File Validation
# ============================================================================
# Similar to LLVM's llvm_check_source_file_list, this ensures all .o files
# in a directory are properly included in the CMakeLists.txt build target.
#
# Usage:
#   - Set XPU_OPTIONAL_OBJECTS to exclude certain .o files from validation
#   - Use PARTIAL_OBJECTS_INTENDED in add_xpu_sdnn_object() to skip validation
# ============================================================================

# White list for optional .o files that may exist but don't need to be included
set(XPU_OPTIONAL_OBJECTS "" CACHE STRING
    "List of .o file names that are optional and don't need to be in CMakeLists.txt")

function(xpu_check_object_file_list target_dir object_files skip_check)
  # Skip check if PARTIAL_OBJECTS_INTENDED is set
  if(skip_check)
    return()
  endif()

  # Collect all .o files passed to add_xpu_sdnn_object (listed in CMakeLists.txt)
  set(listed_files "")
  foreach(obj_file ${object_files})
    get_filename_component(abs_path "${obj_file}" REALPATH)
    list(APPEND listed_files "${abs_path}")
  endforeach()

  # Glob all .o files that actually exist in the directory
  file(GLOB globbed_files "${target_dir}/*.o")

  # Check each globbed file
  set(missing_files "")
  foreach(obj_file ${globbed_files})
    get_filename_component(filename "${obj_file}" NAME)
    get_filename_component(realpath "${obj_file}" REALPATH)

    # Skip hidden files (e.g., editor backups)
    if(NOT "${filename}" MATCHES "^\\.")
      # Check if in optional list
      if(NOT "${filename}" IN_LIST XPU_OPTIONAL_OBJECTS)
        # Check if in listed files
        if(NOT "${realpath}" IN_LIST listed_files)
          list(APPEND missing_files "${filename}")
        endif()
      endif()
    endif()
  endforeach()

  # Report error if any files are missing
  if(missing_files)
    string(REPLACE ";" "\n    " missing_files_formatted "${missing_files}")
    message(FATAL_ERROR
      "Found .o files in ${target_dir} that are NOT included in CMakeLists.txt:\n"
      "    ${missing_files_formatted}\n"
      "\n"
      "XPU's build system enforces that all .o files are added to a build target.\n"
      "This ensures consistency and prevents linking errors.\n"
      "\n"
      "To fix this issue, choose one of the following:\n"
      "  1. Add the missing .o files to add_xpu_sdnn_object() in ${CMAKE_CURRENT_LIST_FILE}\n"
      "  2. Add the filename to XPU_OPTIONAL_OBJECTS if it's conditionally compiled:\n"
      "       set(XPU_OPTIONAL_OBJECTS \"${missing_files}\" CACHE STRING \"\" FORCE)\n"
      "  3. Add PARTIAL_OBJECTS_INTENDED to the target if multiple targets share this directory\n"
      "  4. Remove the .o files from the directory if they're no longer needed\n")
  endif()
endfunction()

function(add_xpu_sdnn_object name)
  set_property(GLOBAL APPEND PROPERTY TRITON_LIBS ${name})
  cmake_parse_arguments(ARG "PARTIAL_OBJECTS_INTENDED" "" "DEPENDS;LINK_LIBS" ${ARGN})
  add_library(${name} OBJECT IMPORTED GLOBAL)
  message(STATUS "DEBUG add_xpu_sdnn_object ARG_UNPARSED_ARGUMENTS [${ARG_UNPARSED_ARGUMENTS}]")

  # Validate object files - check if all .o files in directory are included
  if(ARG_UNPARSED_ARGUMENTS)
    list(GET ARG_UNPARSED_ARGUMENTS 0 first_file)
    if(first_file MATCHES "\\.o$")
      get_filename_component(obj_dir "${first_file}" DIRECTORY)
      xpu_check_object_file_list("${obj_dir}" "${ARG_UNPARSED_ARGUMENTS}" "${ARG_PARTIAL_OBJECTS_INTENDED}")
    endif()
  endif()

  set_target_properties(${name} PROPERTIES
    IMPORTED_OBJECTS "${ARG_UNPARSED_ARGUMENTS}"
  )
  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  if(ARG_LINK_LIBS)
    target_link_libraries(${name} INTERFACE ${ARG_LINK_LIBS})
  endif()
  llvm_update_compile_flags(${name})
endfunction()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include) # For #include "Dialect/TritonXPU/IR/Dialect.h"
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(device)
add_subdirectory(include)
add_subdirectory(lib)

add_library(TritonShareForXPU STATIC IMPORTED)
set_target_properties(TritonShareForXPU PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libTritonSharedForXPU.a
)
target_link_libraries(TritonShareForXPU INTERFACE
    MLIRSCFTransforms
    MLIRLinalgTransforms
    MLIRReconcileUnrealizedCasts
    MLIRSCFTransformOps
    MLIRTransformDialect
    MLIRTransformDialectTransforms
    MLIRLinalgTransformOps
    MLIRTransformDebugExtension
)

if(TRITON_BUILD_PYTHON_MODULE)
    add_triton_plugin(TritonXPU
      ${CMAKE_CURRENT_SOURCE_DIR}/triton_xpu.cc

      LINK_LIBS
      TritonShareForXPU
    ) #LINK_LIBS TritonToTritonXPU TritonXPUToLLVM
    add_xpu_sdnn_object(TritonXPUSDNN
      ${CMAKE_CURRENT_SOURCE_DIR}/triton_xpu_sdnn.cc.o
    )
endif()
