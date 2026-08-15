# Global Triton build configuration shared by kurama and flagtree.
# kurama includes this from its root CMakeLists.txt; flagtree includes it from
# the top-level enflame CMakeLists.txt. Keep this file byte-identical in both
# trees.

# TRITON_VERSION: upstream Triton version selector (e.g. 3.5/3.6/3.7).
# Overridable via the TRITON_VERSION environment variable.
if(NOT DEFINED TRITON_VERSION)
  if(NOT "$ENV{TRITON_VERSION}" STREQUAL "")
    set(TRITON_VERSION "$ENV{TRITON_VERSION}")
  else()
    set(TRITON_VERSION "3.6")
  endif()
endif()
message(STATUS "TRITON_VERSION = ${TRITON_VERSION}")

# Numeric form (e.g. 36) for C++ #if guards and cmake version dispatch.
# NOTE: the matching add_compile_definitions(TRITON_VERSION=${TRITON_VERSION_NUMERIC})
# is intentionally NOT emitted here. kurama compiles the gcu_compiler subproject
# (which uses its own, older LLVM) as a regular target in the main build and must
# keep it free of this define so its version guards stay on the pre-3.7 path.
# Each consumer therefore adds the compile definition itself, after any such
# version-isolated subproject.
string(REPLACE "." "" TRITON_VERSION_NUMERIC "${TRITON_VERSION}")
set(TRITON_VERDIR "triton${TRITON_VERSION_NUMERIC}")

# TARGET_PROFILE: build profile (e.g. default/gcu300_legacy/gcu300).
# Overridable via the TARGET_PROFILE environment variable.
if(NOT DEFINED TARGET_PROFILE)
  if(NOT "$ENV{TARGET_PROFILE}" STREQUAL "")
    set(TARGET_PROFILE "$ENV{TARGET_PROFILE}")
  else()
    set(TARGET_PROFILE "default")
  endif()
endif()

# Emit the VERSION file (read at runtime by the python backend's
# _triton_version()) into the build dir. Packaging copies it next to the backend
# module. Written under bin/ to match the binaries' output directory.
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  OUTPUT_VARIABLE _triton_global_commit
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(NOT _triton_global_commit)
  set(_triton_global_commit "unknown")
endif()
file(WRITE "${CMAKE_BINARY_DIR}/bin/VERSION"
  "COMMIT=${_triton_global_commit}\nTRITON_VERSION=${TRITON_VERSION}\nTARGET_PROFILE=${TARGET_PROFILE}\n")
