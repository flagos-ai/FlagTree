#ifndef _TANG_RUNTIME_VERSION_H_
#define _TANG_RUNTIME_VERSION_H_
#define TANG_VERSION_MAJOR 0
#define TANG_VERSION_MINOR 13
#define TANG_VERSION_PATCH 0

#define TANG_VERSION_GIT_SHA ""

/////////////////////////////////////////////////////////

#define TANGRT_VERSION_MAJOR 0
#define TANGRT_VERSION_MINOR 13
#define TANGRT_VERSION_PATCH 0

#define TANGRT_VERSION_GIT_SHA "04137493 Merge branch 'ln/bugfix/taStreamIsCapturing' into 'master'"

/////////////////////////////////////////////////////////
#define TANGRT_TANGCC_VERSION_MAJOR 2
#define TANGRT_TANGCC_VERSION_MINOR 2

#ifdef __TANGC_MAJOR__
#    if (TANGRT_TANGCC_VERSION_MAJOR <= 1) && (__TANGC_MAJOR__ >= 2)
#warning "the ptcc used is not compatible with the tang runtime library\nptcc less than 2.0.0 is required."
//#error "the ptcc used is not compatible with the tang runtime library\nptcc less than 2.0.0 is required."
//#    elif (TANGRT_TANGCC_VERSION_MAJOR >= 2) && (__TANGC_MAJOR__ <= 1)
//#error "the ptcc used is not compatible with the tang runtime library\nptcc 2.0.0 or later is required."
#    endif
#endif  // __TANGC_MAJOR__

#endif  //! _TANG_RUNTIME_VERSION_H_
