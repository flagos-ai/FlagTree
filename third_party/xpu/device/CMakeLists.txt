function(add_xpu_libdevice OUTPUT SRC ARCH)
  set(CLANG ${LLVM_TOOLS_BINARY_DIR}/clang)

  add_custom_target(
    libdevice-${ARCH} ALL
    COMMAND ${CLANG} --xpu-arch=${ARCH} ${SRC} -c -emit-llvm --xpu-device-only -O3 -o ${OUTPUT} -std=c++11 -Wno-literal-range
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/liblaunch.a ${XPU3_LIB_DIR}
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
