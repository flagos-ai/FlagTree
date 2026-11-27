# The XPU device runtime (launch.cpp, libdevice .xpu, procedure .xpu) must be
# compiled by an XTDK clang that understands --xpu-arch / the XPU target.
# With the public LLVM 22 frontend LLVM_TOOLS_BINARY_DIR points at a tree that
# has no clang at all (the public package ships libraries and headers only),
# so fall back to the LLVM 19 XTDK clang staged by setup.py under
# third_party/xpu/backend/llvm19/bin (resource headers under
# ../lib/clang/19/include).  Same guard as the sync source's device/CMakeLists.
if(NOT EXISTS ${LLVM_TOOLS_BINARY_DIR}/clang++)
  set(LLVM_TOOLS_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../backend/llvm19/bin)
endif()

  set(_xpudev_fpm "-ffile-prefix-map=${CMAKE_SOURCE_DIR}=.")
  if(LLVM_SYSPATH)
    set(_xpudev_fpm "${_xpudev_fpm} -ffile-prefix-map=${LLVM_SYSPATH}=.")
  endif()
  if(DEFINED ENV{HOME})
    set(_xpudev_fpm "${_xpudev_fpm} -ffile-prefix-map=$ENV{HOME}=.")
  endif()
function(add_xpu_libdevice OUTPUT SRC ARCH)
  set(CLANG ${LLVM_TOOLS_BINARY_DIR}/clang)
  file(RELATIVE_PATH _src_rel "${CMAKE_SOURCE_DIR}" "${SRC}")

  add_custom_target(
    libdevice-${ARCH} ALL
    COMMAND ${CLANG} --xpu-arch=${ARCH} ${_src_rel} -c -emit-llvm --xpu-device-only -O3 -o ${OUTPUT} -std=c++11 -Wno-literal-range ${_xpudev_fpm}
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/xpu3/liblaunch.a ${XPU3_LIB_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/liblaunch_shared.so ${XPU3_LIB_DIR}/../so
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/libLLVM-15.so ${XPU3_LIB_DIR}/../so
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/libclang-cpp.so.15 ${XPU3_LIB_DIR}/../so
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/libxpujitc.so ${XPU3_LIB_DIR}/../so
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/libdevice-xpu3s.bc ${XPU3_LIB_DIR}
    DEPENDS ${SRC}
    COMMENT "Building libdevice-${ARCH} ..."
    VERBATIM
  )
endfunction()

set(XPU_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/trigonometric.xpu)
set(XPU3_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../backend/xpu3/lib/)
file(MAKE_DIRECTORY ${XPU3_LIB_DIR})
add_xpu_libdevice(${XPU3_LIB_DIR}/libdevice-xpu3.bc ${XPU_LIB_SRC} xpu3)
