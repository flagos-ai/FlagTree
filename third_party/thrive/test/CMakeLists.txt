set(THRIVE_TEST_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/triton/backends/thrive/test/")

add_custom_target(thrive_install_test ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${THRIVE_TEST_INSTALL_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/conftest.py"
        "${THRIVE_TEST_INSTALL_DIR}/conftest.py"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
        "${THRIVE_TEST_INSTALL_DIR}/__init__.py"
    COMMENT "[thrive] Installing test base files to ${THRIVE_TEST_INSTALL_DIR}"
)

add_subdirectory(examples)
add_subdirectory(unit)
