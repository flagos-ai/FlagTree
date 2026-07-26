function(_flagtree_collect_build_targets directory output)
  get_property(_local_targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
  set(_targets ${_local_targets})

  get_property(_subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
  foreach(_subdirectory IN LISTS _subdirectories)
    _flagtree_collect_build_targets("${_subdirectory}" _subdirectory_targets)
    list(APPEND _targets ${_subdirectory_targets})
  endforeach()

  list(REMOVE_DUPLICATES _targets)
  set(${output} "${_targets}" PARENT_SCOPE)
endfunction()

function(_flagtree_normalize_target_source target source output)
  if("${source}" MATCHES "\\$<")
    set(${output} "" PARENT_SCOPE)
    return()
  endif()

  get_target_property(_target_source_dir "${target}" SOURCE_DIR)
  if(IS_ABSOLUTE "${source}")
    set(_absolute_source "${source}")
  else()
    get_filename_component(
      _absolute_source "${source}" ABSOLUTE BASE_DIR "${_target_source_dir}")
  endif()

  if(EXISTS "${_absolute_source}")
    get_filename_component(_absolute_source "${_absolute_source}" REALPATH)
  endif()
  set(${output} "${_absolute_source}" PARENT_SCOPE)
endfunction()

function(_flagtree_path_has_suffix path suffix output)
  string(LENGTH "${path}" _path_length)
  string(LENGTH "${suffix}" _suffix_length)
  if(_path_length LESS _suffix_length)
    set(${output} FALSE PARENT_SCOPE)
    return()
  endif()

  math(EXPR _suffix_offset "${_path_length} - ${_suffix_length}")
  string(SUBSTRING "${path}" ${_suffix_offset} ${_suffix_length} _path_suffix)
  if(_path_suffix STREQUAL "${suffix}")
    set(${output} TRUE PARENT_SCOPE)
  else()
    set(${output} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(flagtree_apply_backend_source_overrides backend_root)
  set(_spec_root "${backend_root}/backend/spec")
  set(_spec_lib_root "${_spec_root}/lib")
  if(NOT IS_DIRECTORY "${_spec_lib_root}")
    return()
  endif()

  file(GLOB_RECURSE _spec_sources CONFIGURE_DEPENDS
    "${_spec_lib_root}/*.c"
    "${_spec_lib_root}/*.cc"
    "${_spec_lib_root}/*.cpp"
    "${_spec_lib_root}/*.cxx")
  if(NOT _spec_sources)
    return()
  endif()
  list(SORT _spec_sources)

  _flagtree_collect_build_targets("${PROJECT_SOURCE_DIR}" _build_targets)

  set(_source_index)
  set(_target_index)
  foreach(_target IN LISTS _build_targets)
    get_target_property(_target_type "${_target}" TYPE)
    if(_target_type STREQUAL "UTILITY" OR
       _target_type STREQUAL "INTERFACE_LIBRARY")
      continue()
    endif()

    get_target_property(_target_sources "${_target}" SOURCES)
    if(NOT _target_sources OR _target_sources STREQUAL "_target_sources-NOTFOUND")
      continue()
    endif()

    foreach(_source IN LISTS _target_sources)
      _flagtree_normalize_target_source(
        "${_target}" "${_source}" _absolute_source)
      if(NOT _absolute_source)
        continue()
      endif()
      list(APPEND _source_index "${_absolute_source}")
      list(APPEND _target_index "${_target}")
    endforeach()
  endforeach()

  list(LENGTH _source_index _source_count)
  if(_source_count EQUAL 0)
    message(FATAL_ERROR
      "Backend spec sources exist under ${_spec_lib_root}, but no build target "
      "sources were available for matching")
  endif()
  math(EXPR _last_source_index "${_source_count} - 1")

  foreach(_spec_source IN LISTS _spec_sources)
    get_filename_component(_spec_source "${_spec_source}" REALPATH)
    file(RELATIVE_PATH _logical_source "${_spec_root}" "${_spec_source}")
    set(_logical_suffix "/${_logical_source}")

    set(_preferred_main_sources)
    set(_core_roots "${PROJECT_SOURCE_DIR}")
    if(DEFINED TRITON_CORE_SOURCE_DIR)
      list(PREPEND _core_roots "${TRITON_CORE_SOURCE_DIR}")
    endif()
    list(REMOVE_DUPLICATES _core_roots)
    foreach(_core_root IN LISTS _core_roots)
      get_filename_component(
        _candidate_main "${_core_root}/${_logical_source}" ABSOLUTE)
      if(EXISTS "${_candidate_main}")
        get_filename_component(_candidate_main "${_candidate_main}" REALPATH)
      endif()
      list(FIND _source_index "${_candidate_main}" _candidate_index)
      if(NOT _candidate_index EQUAL -1)
        list(APPEND _preferred_main_sources "${_candidate_main}")
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _preferred_main_sources)

    if(_preferred_main_sources)
      list(LENGTH _preferred_main_sources _preferred_count)
      if(_preferred_count GREATER 1)
        message(FATAL_ERROR
          "Backend spec source ${_spec_source} matches multiple preferred main "
          "sources: ${_preferred_main_sources}")
      endif()
      list(GET _preferred_main_sources 0 _main_source)
    else()
      set(_suffix_main_sources)
      foreach(_index RANGE 0 ${_last_source_index})
        list(GET _source_index ${_index} _indexed_source)
        _flagtree_path_has_suffix(
          "${_indexed_source}" "${_logical_suffix}" _has_logical_suffix)
        if(_has_logical_suffix)
          list(APPEND _suffix_main_sources "${_indexed_source}")
        endif()
      endforeach()
      list(REMOVE_DUPLICATES _suffix_main_sources)
      list(LENGTH _suffix_main_sources _suffix_match_count)
      if(_suffix_match_count EQUAL 0)
        message(FATAL_ERROR
          "Backend spec source ${_spec_source} has no owner target for mirrored "
          "main source ${_logical_source}")
      elseif(_suffix_match_count GREATER 1)
        message(FATAL_ERROR
          "Backend spec source ${_spec_source} ambiguously matches main sources: "
          "${_suffix_main_sources}")
      endif()
      list(GET _suffix_main_sources 0 _main_source)
    endif()

    if(NOT EXISTS "${_main_source}")
      message(FATAL_ERROR
        "Backend spec source ${_spec_source} maps to missing main source "
        "${_main_source}")
    endif()

    set(_owner_targets)
    foreach(_index RANGE 0 ${_last_source_index})
      list(GET _source_index ${_index} _indexed_source)
      if(_indexed_source STREQUAL "${_main_source}")
        list(GET _target_index ${_index} _owner_target)
        list(APPEND _owner_targets "${_owner_target}")
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _owner_targets)

    foreach(_owner_target IN LISTS _owner_targets)
      set(_already_injected FALSE)
      foreach(_index RANGE 0 ${_last_source_index})
        list(GET _source_index ${_index} _indexed_source)
        list(GET _target_index ${_index} _indexed_target)
        if(_indexed_target STREQUAL "${_owner_target}" AND
           _indexed_source STREQUAL "${_spec_source}")
          set(_already_injected TRUE)
          break()
        endif()
      endforeach()
      if(_already_injected)
        message(FATAL_ERROR
          "Backend spec source ${_spec_source} is already present in target "
          "${_owner_target}")
      endif()

      set_source_files_properties(
        "${_main_source}"
        TARGET_DIRECTORY "${_owner_target}"
        PROPERTIES HEADER_FILE_ONLY ON)
      target_sources("${_owner_target}" PRIVATE "${_spec_source}")
      message(STATUS
        "FlagTree backend spec: ${_logical_source} -> ${_owner_target}")
    endforeach()
  endforeach()
endfunction()
