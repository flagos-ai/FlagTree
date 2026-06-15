file(GLOB_RECURSE THRIVE_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

set(THRIVE_TEST_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/triton/backends/thrive/test/unit/runtime")

add_custom_target(thrive_install_test_runtime ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${THRIVE_TEST_INSTALL_DIR}"
    COMMENT "[thrive] Installing test to ${THRIVE_TEST_INSTALL_DIR}"
)

foreach(PY_FILE ${THRIVE_TEST_FILES})
    get_filename_component(FILE_NAME "${PY_FILE}" NAME)
    add_custom_command(
        TARGET thrive_install_test_runtime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PY_FILE}" "${THRIVE_TEST_INSTALL_DIR}/${FILE_NAME}"
        COMMENT "[thrive] Copying ${FILE_NAME}"
    )
endforeach()
