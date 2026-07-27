# Copyright 2025-     FlagOS Contributors
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

# flagtree backend cmake specialization
function(flagtree_apply_backend_source_overrides backend_root)
  # ${backend_root}: third_party/{backend}
  set(_spec_root "${backend_root}/spec_cpp")
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

  # ${_build_targets}: all build targets in the project
  _flagtree_collect_build_targets("${PROJECT_SOURCE_DIR}" _build_targets)

  set(_source_index)
  set(_target_index)
  foreach(_target IN LISTS _build_targets)
    # ${_target}: target
    get_target_property(_target_type "${_target}" TYPE)
    if(_target_type STREQUAL "UTILITY" OR
       _target_type STREQUAL "INTERFACE_LIBRARY")
      continue()
    endif()

    # ${_target_sources}: sources of the target
    get_target_property(_target_sources "${_target}" SOURCES)
    if(NOT _target_sources OR _target_sources STREQUAL "_target_sources-NOTFOUND")
      continue()
    endif()

    # ${_source_index}: all sources in the project
    # ${_target_index}: all targets in the project
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

  # ${_source_count}: number of sources in the project
  list(LENGTH _source_index _source_count)
  if(_source_count EQUAL 0)
    message(FATAL_ERROR
      "Backend spec sources exist under ${_spec_lib_root}, but no build target "
      "sources were available for matching")
  endif()
  math(EXPR _last_source_index "${_source_count} - 1")

  message(STATUS "=====================================")
  message(STATUS "flagtree backend cmake specialization")
  foreach(_spec_source IN LISTS _spec_sources)
    # ${_spec_source}: third_party/{backend}/spec_cpp/lib/*.cpp
    get_filename_component(_spec_source "${_spec_source}" REALPATH)
    # ${_relative_path}: lib/*.cpp in the spec root
    file(RELATIVE_PATH _relative_path "${_spec_root}" "${_spec_source}")

    set(_spec_sources_in_core_root)
    # ${_core_roots}: all core root directories
    set(_core_roots "${PROJECT_SOURCE_DIR}")
    # ${TRITON_CORE_SOURCE_DIR}: third_party/iluvatar
    if(DEFINED TRITON_CORE_SOURCE_DIR)
      list(PREPEND _core_roots "${TRITON_CORE_SOURCE_DIR}")
    endif()
    list(REMOVE_DUPLICATES _core_roots)
    foreach(_core_root IN LISTS _core_roots)
      # ${_spec_source_in_core_root}: lib/*.cpp in the core root
      get_filename_component(
        _spec_source_in_core_root "${_core_root}/${_relative_path}" ABSOLUTE)
      if(EXISTS "${_spec_source_in_core_root}")
        get_filename_component(_spec_source_in_core_root "${_spec_source_in_core_root}" REALPATH)
      endif()
      # Only process ${_spec_source} that appears in the core root
      list(FIND _source_index "${_spec_source_in_core_root}" _candidate_index)
      if(NOT _candidate_index EQUAL -1)
        list(APPEND _spec_sources_in_core_root "${_spec_source_in_core_root}")
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _spec_sources_in_core_root)

    list(LENGTH _spec_sources_in_core_root _candidate_count)
    if(_candidate_count EQUAL 0)
      message(FATAL_ERROR
        "Backend spec source ${_spec_source} has no owner target in the "
        "configured core roots for mirrored main source ${_relative_path}")
    elseif(_candidate_count GREATER 1)
      message(FATAL_ERROR
        "Backend spec source ${_spec_source} matches multiple preferred main "
        "sources: ${_spec_sources_in_core_root}")
    endif()
    list(GET _spec_sources_in_core_root 0 _root_source)

    # Assert ${_spec_source} appears in the core root
    if(NOT EXISTS "${_root_source}")
      message(FATAL_ERROR
        "Backend spec source ${_spec_source} maps to missing main source "
        "${_root_source}")
    endif()

    set(_owner_targets)
    foreach(_index RANGE 0 ${_last_source_index})
      list(GET _source_index ${_index} _indexed_source)
      if(_indexed_source STREQUAL "${_root_source}")
        list(GET _target_index ${_index} _owner_target)
        list(APPEND _owner_targets "${_owner_target}")
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _owner_targets)

    # ${_owner_targets}: all targets that own the ${_root_source}
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

      # Register ${_root_source} as a header file only
      set_source_files_properties(
        "${_root_source}"
        TARGET_DIRECTORY "${_owner_target}"
        PROPERTIES HEADER_FILE_ONLY ON)
      # Add ${_spec_source} to the target
      target_sources("${_owner_target}" PRIVATE "${_spec_source}")
      message(STATUS "SPEC: ${_relative_path} -> ${_owner_target}")
    endforeach()
  endforeach()
  message(STATUS "=====================================")
endfunction()
