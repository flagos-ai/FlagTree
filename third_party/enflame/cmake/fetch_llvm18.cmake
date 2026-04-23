include(fetchArtifactory)
message(STATUS "-- downloading llvm18 gcu")

string(REGEX MATCH "^[0-9]+" GCC_MAJOR_VERSION ${CMAKE_CXX_COMPILER_VERSION})
message(STATUS "-- kurama llvm GCC major version: ${GCC_MAJOR_VERSION}")
# 如果编译器大版本不是7/8/9/11，则设置版本号为7
if(NOT GCC_MAJOR_VERSION STREQUAL "7"
AND NOT GCC_MAJOR_VERSION STREQUAL "8"
AND NOT GCC_MAJOR_VERSION STREQUAL "9"
AND NOT GCC_MAJOR_VERSION STREQUAL "11")
  message(STATUS "if not support, we just use gcc7")
  set(GCC_MAJOR_VERSION 7)
endif()

set(SPIRV_LLVM18_HASH 3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff)
set(LLVM_HASH ${SPIRV_LLVM18_HASH})
string(SUBSTRING ${LLVM_HASH} 0 7 LLVM_HASH_SHORT)

set(SPIRV_LLVM_TAR_NAME llvm-${LLVM_HASH_SHORT}-gcc${GCC_MAJOR_VERSION}-x64)
set(LLVM_DOWNLOAD_URL http://artifact.enflame.cn:80/artifactory/module_package/triton_llvm_project/${LLVM_HASH_SHORT}/${SPIRV_LLVM_TAR_NAME}.tar.gz)

if(NOT PROJECT_GIT_URL)
        message(STATUS "-- downloading kurama llvm from ${LLVM_DOWNLOAD_URL}")
        fetchFromArtifactory(kurama_llvm_gcu
                FILE ${LLVM_DOWNLOAD_URL}
                EXTRACT ON
        )
else()
        if(INTERNAL_TX_BUILD)
                get_gongfeng_project_internal("triton_gcu_binary" "${LLVM_HASH_SHORT}/${SPIRV_LLVM_TAR_NAME}.tar.gz"
                        "kurama_llvm_gcu" "${LLVM_DOWNLOAD_URL}"
                        EXTRACT ON)
        else()
                get_gongfeng_tar("triton_gcu_binary" "${LLVM_HASH_SHORT}/${SPIRV_LLVM_TAR_NAME}.tar.gz" extracted_tar_path)
                set(kurama_llvm_gcu_SOURCE_DIR ${extracted_tar_path})
        endif()
endif()

set(KURAMA_LLVM18_GCU_DIR ${kurama_llvm_gcu_SOURCE_DIR})
message(STATUS "-- set kurama llvm18 path: ${KURAMA_LLVM18_GCU_DIR}")
