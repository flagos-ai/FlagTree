include(CheckIncludeFileCXX)

# LLVM builds without pass-plugin support may not ship PassPlugin.h.
# Skip sanitizer plugin build in that case so the rest of the project can build.
set(CMAKE_REQUIRED_INCLUDES ${LLVM_INCLUDE_DIRS})
check_include_file_cxx("llvm/Passes/PassPlugin.h" HAVE_LLVM_PASSPLUGIN_H)
unset(CMAKE_REQUIRED_INCLUDES)

if(HAVE_LLVM_PASSPLUGIN_H)
	add_subdirectory(SanitizerAttributes)
else()
	message(STATUS "Skipping SanitizerAttributes: llvm/Passes/PassPlugin.h not found")
endif()
