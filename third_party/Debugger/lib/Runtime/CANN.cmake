include_guard(GLOBAL)

set(FLAGTREE_DEBUGGER_CANN_ROOT ""
    CACHE PATH "Root directory of the CANN toolkit used by the debugger runtime")

function(_flagtree_debugger_resolve_cann)
  set(_candidates)
  if(FLAGTREE_DEBUGGER_CANN_ROOT)
    list(APPEND _candidates
      "${FLAGTREE_DEBUGGER_CANN_ROOT}"
      "${FLAGTREE_DEBUGGER_CANN_ROOT}/aarch64-linux"
    )
  endif()

  foreach(_env_name ASCEND_TOOLKIT_HOME ASCEND_HOME_PATH)
    if(DEFINED ENV{${_env_name}} AND NOT "$ENV{${_env_name}}" STREQUAL "")
      list(APPEND _candidates
        "$ENV{${_env_name}}"
        "$ENV{${_env_name}}/aarch64-linux"
      )
    endif()
  endforeach()

  list(APPEND _candidates
    "/usr/local/Ascend/ascend-toolkit/latest"
    "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux"
    "/usr/local/Ascend/cann/latest"
    "/usr/local/Ascend/cann/latest/aarch64-linux"
  )

  file(GLOB _glob_candidates LIST_DIRECTORIES true
    "/usr/local/Ascend/cann-*"
    "/usr/local/Ascend/cann-*/*-linux"
  )
  list(APPEND _candidates ${_glob_candidates})
  list(REMOVE_DUPLICATES _candidates)

  set(_resolved_root "")
  set(_resolved_include "")
  set(_resolved_library "")

  foreach(_candidate IN LISTS _candidates)
    if(NOT _candidate)
      continue()
    endif()

    file(TO_CMAKE_PATH "${_candidate}" _candidate)

    set(_root "")
    if(EXISTS "${_candidate}/include/acl/acl_rt.h")
      set(_root "${_candidate}")
    elseif(EXISTS "${_candidate}/aarch64-linux/include/acl/acl_rt.h")
      set(_root "${_candidate}/aarch64-linux")
    endif()

    if(NOT _root)
      continue()
    endif()

    set(_library "")
    if(EXISTS "${_root}/lib64/libascendcl.so")
      set(_library "${_root}/lib64/libascendcl.so")
    elseif(EXISTS "${_root}/devlib/libascendcl.so")
      set(_library "${_root}/devlib/libascendcl.so")
    endif()

    if(_library)
      set(_resolved_root "${_root}")
      set(_resolved_include "${_root}/include")
      set(_resolved_library "${_library}")
      break()
    endif()
  endforeach()

  if(_resolved_root)
    get_filename_component(_resolved_libdir "${_resolved_library}" DIRECTORY)
    set(_resolved_link_libraries "${_resolved_library}")
    set(_has_driver_runtime FALSE)
    if(EXISTS "/usr/local/Ascend/driver/lib64/driver/libascend_hal.so")
      set(_has_driver_runtime TRUE)
    endif()

    set(_runtime_dirs
      "${_resolved_libdir}"
      "${_resolved_root}/lib64"
      "/usr/local/Ascend/driver/lib64"
      "/usr/local/Ascend/driver/lib64/driver"
      "/usr/local/Ascend/driver/lib64/common"
    )
    if(NOT _has_driver_runtime)
      list(APPEND _runtime_dirs
        "${_resolved_root}/devlib"
        "${_resolved_root}/devlib/device"
        "${_resolved_root}/lib64/device/lib64"
      )
      file(GLOB _runtime_glob_dirs LIST_DIRECTORIES true
        "${_resolved_root}/devlib/*"
        "${_resolved_root}/devlib/linux/*"
      )
      list(APPEND _runtime_dirs ${_runtime_glob_dirs})
    endif()
    set(_resolved_runtime_dirs)
    foreach(_dir IN LISTS _runtime_dirs)
      if(IS_DIRECTORY "${_dir}")
        list(APPEND _resolved_runtime_dirs "${_dir}")
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _resolved_runtime_dirs)

    set(FLAGTREE_DEBUGGER_CANN_FOUND TRUE CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_CANN_RESOLVED TRUE CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_CANN_RESOLVED_ROOT "${_resolved_root}" CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_CANN_INCLUDE_DIR "${_resolved_include}" CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_CANN_LIBRARY "${_resolved_library}" CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_CANN_LIBRARY_DIR "${_resolved_libdir}" CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_CANN_LINK_LIBRARIES "${_resolved_link_libraries}" CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_CANN_RUNTIME_DIRS "${_resolved_runtime_dirs}" CACHE INTERNAL "" FORCE)
  else()
    set(FLAGTREE_DEBUGGER_CANN_FOUND FALSE CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_CANN_RESOLVED TRUE CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_CANN_RESOLVED_ROOT "" CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_CANN_INCLUDE_DIR "" CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_CANN_LIBRARY "" CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_CANN_LIBRARY_DIR "" CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_CANN_LINK_LIBRARIES "" CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_CANN_RUNTIME_DIRS "" CACHE INTERNAL "" FORCE)
  endif()
endfunction()

function(flagtree_debugger_enable_cann target)
  _flagtree_debugger_resolve_cann()

  if(FLAGTREE_DEBUGGER_CANN_FOUND)
    message(STATUS
      "FlagTree Debugger: enabling CANN runtime for ${target} "
      "(${FLAGTREE_DEBUGGER_CANN_RESOLVED_ROOT})")
    target_compile_definitions(${target}
      PRIVATE
        FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME=1
    )
    target_include_directories(${target}
      PRIVATE
        ${FLAGTREE_DEBUGGER_CANN_INCLUDE_DIR}
    )
    target_link_libraries(${target}
      PUBLIC
        ${FLAGTREE_DEBUGGER_CANN_LINK_LIBRARIES}
    )
    foreach(_runtime_dir IN LISTS FLAGTREE_DEBUGGER_CANN_RUNTIME_DIRS)
      target_link_directories(${target}
        PUBLIC
          ${_runtime_dir}
      )
      target_link_options(${target}
        PUBLIC
          "-Wl,-rpath-link,${_runtime_dir}"
      )
    endforeach()

    get_target_property(_target_type ${target} TYPE)
    if(NOT _target_type STREQUAL "OBJECT_LIBRARY")
      foreach(_runtime_dir IN LISTS FLAGTREE_DEBUGGER_CANN_RUNTIME_DIRS)
        set_property(TARGET ${target}
          APPEND PROPERTY BUILD_RPATH ${_runtime_dir})
      endforeach()
    endif()
  else()
    message(STATUS
      "FlagTree Debugger: CANN runtime not found for ${target}, "
      "building stub CANN adapter")
    target_compile_definitions(${target}
      PRIVATE
        FLAGTREE_DEBUGGER_HAS_CANN_RUNTIME=0
    )
  endif()
endfunction()
