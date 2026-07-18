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
1. Refactoring plan for 3rd party backends
2. Front end refactoring (AMD)
3. Things like block pointers, ptr_analysis, mask_analysis can be used for GPUs, is there a plan to incrementally include components from Triton shared for GPU development.

##### Minutes:
Recording link [here](https://youtu.be/Lo43DQYkOWM)

1. Refactoring plan for 3rd party backends
   - Refactoring to be completed by end of the year so that all GPU backends can be individual passes on Triton GPU IR instead of being completely out of tree. The goal is for users to get other GPUs besides Cuda when they install Triton. Non-GPU Triton IR expected to stay as is.
3. Front end refactoring (AMD)
   - Will work with Phil for AMD related refactoring. Will share more details in next meetup about where AMD has diverged from Triton GPU IR and in the codeflow.
4. Things like block pointers, ptr_analysis, mask_analysis can be used for GPUs, is there a plan to incrementally include components from Triton shared for GPU development.
   - Can look at it on a case by case basis.
