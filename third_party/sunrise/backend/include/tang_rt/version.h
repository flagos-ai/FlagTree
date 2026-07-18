// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
