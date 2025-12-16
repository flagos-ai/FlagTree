function(add_xpu_libdevice OUTPUT SRC ARCH)
  set(CLANG ${LLVM_TOOLS_BINARY_DIR}/clang)

  set(LIBLAUNCH_SO_URL "https://klx-sdk-release-public.su.bcebos.com/v1/triton/build_deps/liblaunch_shared.so.1")
  set(LIBLAUNCH_SO_PATH ${CMAKE_CURRENT_SOURCE_DIR}/liblaunch_shared.so)

  if(NOT EXISTS ${LIBLAUNCH_SO_PATH})
    message(STATUS "Downloading liblaunch_shared.so...")
    file(DOWNLOAD ${LIBLAUNCH_SO_URL} ${LIBLAUNCH_SO_PATH}
         SHOW_PROGRESS
         STATUS download_status)
    list(GET download_status 0 status_code)
    if(NOT status_code EQUAL 0)
      message(FATAL_ERROR "Failed to download liblaunch_shared.so")
    endif()
  endif()

  add_custom_target(
    libdevice-${ARCH} ALL
    COMMAND ${CLANG} --xpu-arch=${ARCH} ${SRC} -c -emit-llvm --xpu-device-only -O3 -o ${OUTPUT} -std=c++11 -Wno-literal-range
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/liblaunch.a ${XPU3_LIB_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/liblaunch_shared.so ${XPU3_LIB_DIR}/../so
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
