# Compile NVIDIA compat sources for use by TritonInstrumentToLLVM
# (PTXBuilder, createElectPredicateWarp0, etc.)
set(NVIDIA_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../nvidia/lib/TritonNVIDIAGPUToLLVM)

add_library(TritonNVIDIACompat STATIC
  ${NVIDIA_SRC_DIR}/PTXAsmFormat.cpp
  ${NVIDIA_SRC_DIR}/Utility.cpp
)
target_include_directories(TritonNVIDIACompat PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../nvidia/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
)
