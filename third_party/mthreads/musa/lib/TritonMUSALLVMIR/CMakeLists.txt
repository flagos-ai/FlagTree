add_triton_library(TritonMUSALLVMIR
  LLVMIRMTGPUPostSLPRepair.cpp
  LLVMIRMTGPUVectorCanonicalize.cpp
  MTGPUSLPPipeline.cpp
  MTGPUTargetTransformInfo.cpp
)

target_include_directories(TritonMUSALLVMIR PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
