cmake_minimum_required(VERSION 3.15)
project(SpineTritonRuntime)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(RUNTIME_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ExecutionEngine/CRunnerUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ExecutionEngine/CpuProtonRuntime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ExecutionEngine/SpinePrintRuntime.cpp
)

add_library(SpineTritonRuntime SHARED ${RUNTIME_SOURCES})


target_compile_features(SpineTritonRuntime PRIVATE cxx_std_17)


install(TARGETS SpineTritonRuntime
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
