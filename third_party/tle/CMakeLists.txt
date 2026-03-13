# Include and generated-file paths for all subdirs
set(TLE_INCLUDE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(TLE_INCLUDE_BINARY "${CMAKE_CURRENT_BINARY_DIR}/include")

include_directories(${TLE_INCLUDE_SOURCE})
include_directories(${TLE_INCLUDE_BINARY})
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(python)
