add_subdirectory(AutoBlockify)
add_subdirectory(Dialect)
add_subdirectory(TritonMemoryAccess)
add_subdirectory(Utils)
add_subdirectory(TritonToStructured)
add_subdirectory(TritonToGraph)
add_subdirectory(TritonToLinalg)
add_subdirectory(TritonToAnnotation)
add_subdirectory(TritonToHFusion)
add_subdirectory(TritonToHIVM)
add_subdirectory(TritonControlFlowOpt)
add_subdirectory(DiscreteMaskAccessConversion)
add_subdirectory(TritonToUnstructure)
add_subdirectory(TritonToLLVM)
add_subdirectory(DynamicCVPipeline)
add_subdirectory(LibEntry)
add_subdirectory(CommonIRToHIVM)

if(TRITON_ENABLE_COVERAGE_HITEST)
  set(_instrument_targets
      DiscreteMaskAccessConversion
      TritonToAnnotation
      TritonToHFusion
      TritonToHIVM
      TritonToLinalg
      TritonControlFlowOpt
      TritonToLLVM
      TritonToStructured
      TritonToUnstructure
      MLIRTritonNPUUtils           # from Utils
      TritonMemoryAccessCore
      TritonMemoryAccessAnalysis
      TritonAscendIR               # from Dialect/TritonAscend/IR
      TritonStructuredIR           # from Dialect/TritonStructured/IR
      AutoBlockify
  )

  foreach(_target ${_instrument_targets})
      if(TARGET ${_target})
          set_target_properties(${_target} PROPERTIES
              RULE_LAUNCH_COMPILE "hitestwrapper"
              RULE_LAUNCH_LINK "hitestwrapper"
          )
          message(STATUS "Enabled hitestwrapper for target: ${_target}")
      else()
          message(WARNING "Target ${_target} not found, please check the actual target name")
      endif()
  endforeach()
endif()
