cmake_minimum_required(VERSION 3.18)

project(Tx81HostTests C)

enable_testing()

set(TX81_TEST_STUB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/stubs)
set(TX81_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include/Tx81)
set(TX81_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../lib/Tx81)

add_executable(tx81_addr_range_test
  tx81_addr_range_test.c
  ${TX81_LIB_DIR}/tx81.c
)

target_include_directories(tx81_addr_range_test PRIVATE
  ${TX81_TEST_STUB_DIR}
  ${TX81_INCLUDE_DIR}
)


add_test(NAME tx81_addr_range_test COMMAND tx81_addr_range_test)
