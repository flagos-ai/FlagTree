# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
