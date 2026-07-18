# RPU currently reuses the root Triton IR stack and adds a narrow backend
# plugin for runtime/AOT support plus the RPU plan-recognition pass.
#
# FlagTree skips the root include/lib subdirectories when FLAGTREE_BACKEND is
# set. A backend that does not yet provide specialized C++ dialects still has
# to add the main include/lib trees so TableGen outputs such as Dialect.h.inc
# are generated for the Python extension build.
#
# The shared TritonGPU transforms in this FlagTree branch still have compile-time
# references to TritonNvidiaGPU ops. Keep those definitions enabled for the
# Python extension build; the public target remains backend="rpu".
add_definitions(-D__NVIDIA__)
set(IS_COMPILE_TritonNvidiaGPU TRUE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/rpu-root/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party)
include_directories(${PROJECT_BINARY_DIR}/rpu-root/third_party)
include_directories(${PROJECT_SOURCE_DIR}/third_party/nvidia/include)
include_directories(${PROJECT_BINARY_DIR}/rpu-root/third_party/nvidia/include)

if(NOT TARGET NVGPUIR)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/nvidia/include/Dialect/NVGPU/IR
                   ${PROJECT_BINARY_DIR}/rpu-root/third_party/nvidia/include/Dialect/NVGPU/IR)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/nvidia/lib/Dialect/NVGPU/IR
                   ${PROJECT_BINARY_DIR}/rpu-root/third_party/nvidia/lib/Dialect/NVGPU/IR)
endif()

if(NOT TARGET TritonIR)
  add_subdirectory(${PROJECT_SOURCE_DIR}/include
                   ${PROJECT_BINARY_DIR}/rpu-root/include)
  add_subdirectory(${PROJECT_SOURCE_DIR}/lib
                   ${PROJECT_BINARY_DIR}/rpu-root/lib)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/include/RPUPlan/IR
                 ${CMAKE_CURRENT_BINARY_DIR}/include/RPUPlan/IR)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUPlan/IR
                 ${CMAKE_CURRENT_BINARY_DIR}/lib/RPUPlan/IR)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/include/RPU/IR
                 ${CMAKE_CURRENT_BINARY_DIR}/include/RPU/IR)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/RPU/IR
                 ${CMAKE_CURRENT_BINARY_DIR}/lib/RPU/IR)

if(TRITON_BUILD_PYTHON_MODULE)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RecognizeRPUPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUTTIRPatternMatcher.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUPlanJSON.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUDSLSourceBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUDSLEmitter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUPlanExecutableBridge.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableEmitter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableLowering.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableElementwise16ValueMapLowering.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableDirectLowering.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableDirectLoweringPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableHighLevelLegalizationPipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableCompactElementwise1DLegalizationPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableDotLegalizationPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableSoftmaxLegalizationPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableValueMapLegalizationPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableRenderableVerificationPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableRPURCEmissionPass.cpp
    PROPERTIES COMPILE_FLAGS "-fno-rtti")
  add_triton_plugin(TritonRPU
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_rpu.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RecognizeRPUPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUTTIRPatternMatcher.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUPlanJSON.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUDSLSourceBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUDSLEmitter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUPlanExecutableBridge.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableEmitter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableLowering.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableElementwise16ValueMapLowering.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableDirectLowering.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableDirectLoweringPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableHighLevelLegalizationPipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableCompactElementwise1DLegalizationPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableDotLegalizationPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableSoftmaxLegalizationPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableValueMapLegalizationPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableRenderableVerificationPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/RPUTransforms/RPUExecutableRPURCEmissionPass.cpp
    LINK_LIBS RPUPlanIR RPUIR)
  message(STATUS "Enable FlagTree RPU backend runtime package")
endif()

# Optional on-board tool: the launch_kernel_runner CLI. OFF by default because
# it links the vendor rhino-launch-kernel runtime, which only exists on the
# board; the compiler build itself does not need it. Enable explicitly with
#   -DRPU_BUILD_LAUNCH_RUNNER=ON -DCMAKE_PREFIX_PATH=<rhino-launch-kernel install>
# This option is declared only here (inside the RPU backend CMake, which is
# processed only when FLAGTREE_BACKEND=rpu), so it has no effect on the main
# build or on any other backend.
option(RPU_BUILD_LAUNCH_RUNNER "Build the on-board RPU launch_kernel_runner CLI" OFF)
if(RPU_BUILD_LAUNCH_RUNNER)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools/launch_runner
                   ${CMAKE_CURRENT_BINARY_DIR}/tools/launch_runner)
  # Emit the runner at a stable in-tree path so RPU_LK_RUNNER is easy to point
  # at, independent of where the package build tree happens to live.
  set_target_properties(launch_kernel_runner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tools/launch_runner)
endif()
