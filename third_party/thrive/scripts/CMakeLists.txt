set(THRIVE_TEST_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/triton/backends/thrive/test")

add_custom_target(thrive_install_scripts ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${THRIVE_TEST_INSTALL_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/run-test.sh"
            "${THRIVE_TEST_INSTALL_DIR}/run-test.sh"
    COMMENT "[thrive] Installing run-test.sh to ${THRIVE_TEST_INSTALL_DIR}"
)
