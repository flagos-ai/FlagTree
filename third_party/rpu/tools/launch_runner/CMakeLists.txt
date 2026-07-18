# The launch_kernel_runner CLI builds two ways:
#   * standalone:
#       cmake -S . -B build -DCMAKE_PREFIX_PATH=<rhino-launch-kernel install prefix>
#       cmake --build build
#   * as part of the RPU backend build, via -DRPU_BUILD_LAUNCH_RUNNER=ON
#       (declared in ../../CMakeLists.txt).
# When pulled in with add_subdirectory the parent project has already configured
# CMake, so the project() scaffolding below only runs for the standalone build.
# Either way, point RPU_LK_RUNNER at the resulting launch_kernel_runner.
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.10)
  project(rpu_launch_runner CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

find_package(rhino_launch REQUIRED)

# rhino_launch transitively depends on libHxil.so + librpu_api.so. Link the
# system-installed versions (the vendor's real device libraries), not any
# build-tree stubs, otherwise RpuInit fails at runtime.
add_executable(launch_kernel_runner launch_kernel_runner.cc)
target_compile_features(launch_kernel_runner PRIVATE cxx_std_17)
target_link_libraries(launch_kernel_runner
    PRIVATE rhino_launch::rhino_launch Hxil rpu_api pthread)

set_target_properties(launch_kernel_runner PROPERTIES
    BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib"
    INSTALL_RPATH "${CMAKE_PREFIX_PATH}/lib")
