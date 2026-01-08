#add_subdirectory(triton-adapter triton-adapter)
#add_subdirectory(test)

add_subdirectory(backend/spec/lib)
add_subdirectory(${PROJECT_SOURCE_DIR}/include
                 ${PROJECT_BINARY_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib
                 ${PROJECT_BINARY_DIR}/lib)

add_triton_plugin(TritonAscend ${CMAKE_CURRENT_SOURCE_DIR}/triton_ascend.cpp)
target_include_directories(TritonAscend PRIVATE
	${CMAKE_SOURCE_DIR}/third_party/flir/include
	${CMAKE_BINARY_DIR}/third_party/flir/include)

add_triton_library(Registrar Registrar.cc)
