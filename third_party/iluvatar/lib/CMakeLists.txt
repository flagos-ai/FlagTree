macro(flagtree_add_iluvatar_library name)
  cmake_parse_arguments(_ILUVATAR_ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  set(_ILUVATAR_SOURCES)
  foreach(_source IN LISTS _ILUVATAR_ARG_UNPARSED_ARGUMENTS)
    if("${_source}" MATCHES "\\$<" OR
       IS_ABSOLUTE "${_source}" OR
       NOT "${_source}" MATCHES "\\.(c|cc|cpp|cxx)$")
      list(APPEND _ILUVATAR_SOURCES "${_source}")
      continue()
    endif()

    get_filename_component(
      _source_path "${CMAKE_CURRENT_SOURCE_DIR}/${_source}" ABSOLUTE)
    file(RELATIVE_PATH _relative_source
      "${TRITON_CORE_SOURCE_DIR}/lib" "${_source_path}")
    set(_main_source "${PROJECT_SOURCE_DIR}/lib/${_relative_source}")
    if(EXISTS "${_main_source}")
      list(APPEND _ILUVATAR_SOURCES "${_main_source}")
    else()
      list(APPEND _ILUVATAR_SOURCES "${_source}")
    endif()
  endforeach()

  add_triton_library("${name}"
    ${_ILUVATAR_SOURCES}
    DEPENDS
    ${_ILUVATAR_ARG_DEPENDS}
    LINK_LIBS
    ${_ILUVATAR_ARG_LINK_LIBS})
endmacro()

add_subdirectory(Analysis)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Target)
add_subdirectory(Tools)
add_subdirectory(Instrumentation)
