# Conceal the public module string while keeping the private lowering path available.
message(STATUS "[XPU] TRITON_XPU_CONCEAL_IR: target-scoped concealment")

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
  # Public LLVM 22's llvm_update_compile_flags rejects IMPORTED targets; these
  # precompiled objects cannot benefit from compile flags anyway.
  get_target_property(_xpu_sdnn_obj_imported ${name} IMPORTED)
  if(NOT _xpu_sdnn_obj_imported)
    llvm_update_compile_flags(${name})
  endif()
endfunction()

# Put the XPU backend's own include dir BEFORE everything else (including the
# main Triton tree's include dir) so XPU headers shadow the main-tree ones.
# This lets the XPU backend ship its own triton/Tools/Sys/GetEnv.hpp that
# whitelists TRITONXPU_* env vars without modifying the main Triton tree.
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include) # For #include "triton/Dialect/TritonXPU/IR/Dialect.h"
include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(device)
add_subdirectory(include)

# The XPU-owned TritonGPUIR Dialect.cpp replacement
# includes TritonXPU Dialect.h. Keep this backend-specific dependency here so
# the generated XPU attribute-interface header exists before that main target
# compiles, without changing main-tree CMake files.
if(TARGET TritonGPUIR)
  add_dependencies(TritonGPUIR TritonXPUAttrDefsIncGen)
  target_include_directories(TritonGPUIR BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include)
endif()


# The XPU overlay's python/src/gluon_ir.cc includes
# triton/Dialect/TritonXPU/IR/Dialect.h for the `triton_xpu.raw` builders
# (tle.raw on the cluster path), which pulls in the TableGen'd Ops.h.inc /
# TritonXPUAttrDefs.h.inc. `triton` only links TritonXPUIR, so without an
# explicit dependency the generation can race with compiling that source.
#
# The SDNN declarations the package stages into triton/Dialect/TritonSDNN/IR/
# are included by that same source, and their TableGen targets are
# `add_public_tablegen_target` -- which by MLIR convention is **not** part of
# `all`, so nothing else pulls them in either.  An incremental tree hides the
# gap (the .inc files were generated long ago); a clean configuration fails
# with `Dialect.h.inc: No such file or directory`, and after that one is
# added, with `TritonSDNNAttrInterfaces.h.inc`.  The internal tree declares
# both dependencies in three places; they were lost when SDNN became prebuilt
# objects plus a staged declaration package.
#
# The two SDNN tablegen targets are gone: the declaration surface no longer
# participates in the build.  Its only consumer was the SDNN binding segment of
# gluon_ir.cc, which now lives in the prebuilt gluon_ir_sdnn.cc.o, so nothing
# generates or includes those .h.inc files any more.
#
# Deferred for the same reason as above: the targets do not exist yet here.
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL
  add_dependencies triton TritonXPUTableGen TritonXPUAttrDefsIncGen)

add_subdirectory(lib)

# ==================== FLAGTREE XPU SYNC MARK ====================
# XPU SDNN's Python binding object calls TritonShared pass factories such as
# createTritonToLinalgExperimentalPass().  Previous FlagTree XPU releases
# supplied those symbols through this prebuilt archive instead of vendoring the
# whole third_party/triton_shared source tree.  Keep that model, but use an
# archive regenerated from the reference Triton XPU clean build so it matches the
# current SDNN objects better than the historical bundled archive.
# ==================== FLAGTREE XPU SYNC MARK ====================
add_library(TritonShareForXPU STATIC IMPORTED GLOBAL)
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

      DEPENDS
      MLIRXPUConversionsIncGen        # triton/Dialect/LLVMXPU/IR/*.h.inc
      LLVMXPUTransformsIncGen         # triton/Dialect/LLVMXPU/Transforms/Passes.h.inc
      TritonXPUTableGen               # triton/Dialect/TritonXPU/IR/*.h.inc
      TritonXPUAttrDefsIncGen         # triton/Dialect/TritonXPU/IR/TritonXPUAttrDefs.h.inc
      TritonXPUTransformsIncGen       # triton/Dialect/TritonXPU/Transforms/Passes.h.inc
      TT2TTXConversionPassIncGen      # triton/Conversion/TritonToTritonXPU/Passes.h.inc
      TTX2LLVMConversionPassIncGen    # triton/Conversion/TritonXPUToLLVM/Passes.h.inc

      LINK_LIBS
      TritonXPUTransforms
      TritonSDNNTransforms
      TritonToTritonXPU
      LinalgToTritonSDNN
      TritonXPUToLLVM
      TritonSDNNToLLVM
      TritonShareForXPU
    )
    target_compile_definitions(TritonXPU PRIVATE __XPU__)
    option(TRITON_XPU_CONCEAL_IR "Conceal XPU IR text in distributed builds" ON)
    if(TRITON_XPU_CONCEAL_IR)
      target_compile_definitions(TritonXPU PRIVATE TRITON_CONCEAL_IR=1)
    endif()
    # Both objects are staged into this directory, so both must be listed in the
    # same call: xpu_check_object_file_list() globs the directory and would
    # report the other one as missing.
    add_xpu_sdnn_object(TritonXPUSDNN
      ${CMAKE_CURRENT_SOURCE_DIR}/triton_xpu_sdnn.cc.o
      ${CMAKE_CURRENT_SOURCE_DIR}/gluon_ir_sdnn.cc.o
      ${CMAKE_CURRENT_SOURCE_DIR}/abi_table.cc.o
    )
endif()

# Bind libtriton's own ~1800 weak (template/inline) LLVM symbols to its own
# definitions so that a foreign libLLVM loaded into the global scope first
# (e.g. torch_xmlir's libLLVM-15.so) cannot interpose them; llvm::Module::
# getDataLayout() then reads a wrong offset and link_extern_libs dies with
# std::bad_alloc (XPUTC-7923). XPU-only, mirroring the sync source's fix
# without touching the shared top-level CMakeLists.
if(TRITON_BUILD_PYTHON_MODULE AND UNIX AND NOT APPLE AND TARGET triton)
  target_link_options(triton PRIVATE "-Wl,-Bsymbolic-functions")
endif()
