add_subdirectory(Analysis)
add_subdirectory(Dialect)
add_subdirectory(Instrumentation)
# FlagPrism: Proton tests are disabled when the external component owns the dialect.
# add_subdirectory(Proton)
if(NOT TRITON_BUILD_FLAGPRISM)
  add_subdirectory(Proton)
endif()
