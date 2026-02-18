include(FetchContent)

set(GOOGLETEST_DIR "" CACHE STRING "Location of local GoogleTest repo to build against")

if(NOT GOOGLETEST_DIR)
  set(_default_googletest "$ENV{HOME}/.triton/googletest-release-1.12.1")
  if(EXISTS "${_default_googletest}")
    set(GOOGLETEST_DIR "${_default_googletest}" CACHE STRING "Location of local GoogleTest repo to build against" FORCE)
  else()
    message("Dowloading https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz")
  endif()
  unset(_default_googletest)
endif()

if(GOOGLETEST_DIR)
  set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST ${GOOGLETEST_DIR} CACHE STRING "GoogleTest source directory override")
endif()

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
  )

FetchContent_GetProperties(googletest)

if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  if (MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  endif()
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
