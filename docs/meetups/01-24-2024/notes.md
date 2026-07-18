<!--
 Copyright 2026 FlagOS Contributors

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 -->

#### Agenda:

##### Items:
1. 3rd party refactoring backend update.
2. AMD update about experience with refactored backend and new process.
3. Plan to restore the Intel XPU backend as third-party module.
4. Open discussion.

##### Minutes:
Recording link [here](https://youtu.be/uRlqolhNbRk)

1. 3rd party refactoring backend update.
   - Backends are passes and IRs are shared by the backends to avoid divergence and duplications so that developers do not have to change the Triton source code
   - To discover backend forks in directories, put environment vars in setup.py.
   - Backends can link whatever library they want, they don’t need to copy paste Nvidia code.
   - Nvidia uses the same API as other backends, (refactoring of the C++ code is still remaining). No special casing for Nvidia code.
   - If Triton dependency is on top of the main branch then it will work for forks/branches.
   - Still remaining: LLVM IR conversion – reusuable pattern rewriters update; Reduce complexity in statefulness in Triton GPU - inherit from base pattern
2. AMD update about experience with refactored backend and new process.
   - Skipped due to lack of time. Will be covered in February meetup
3. Plan to restore the Intel XPU backend as third-party module.
   - Prereqs to upstream – Will take into account the system HW and SW, with perf to be ~80% of Nvidia, to allow upstreaming.
   - Consider how useful it is for AI research to allow upstreaming – as it impacts maintenance cost of the backends.
   - Don’t have plans to upstream mobile backends
   - Intel will hold offline discussion with Open AI for being in-tree.
