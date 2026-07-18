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
1. Intel update
2. AMD update
3. Profiler update
4. We are in the process of transitioning to a pro slack plan, so everybody will be able to see history. Expect this to take a few more weeks.
5. We are still working on finalizing a document about our technical governance structure. Expect this to take a few more weeks too.4. Open discussion.

##### Minutes:
Recording link [here](https://youtu.be/JDQCdj18Snc)

1. Intel GPU integration with Triton and Pytorch:
   - No strong requirement from PyTorch for specific backends to be part of Triton official release.
   - Can use a separate branch/fork for CI/CD and testing.
   - Intel team will work with Pytorch offline to close.
2. AMD GPU backend update:
   - AMD team shared the refactored design for AMD backend.
   - The new design is modularized and reduces clutter and duplication in upstream Triton.
   - Further work needed for regression testing and secure runners.
3. Proton profiler update:
   - Keren from the OpenAI team presented a new profiler tool for Triton kernels, which supports multiple vendors, metrics, and formats.
   - Outlined the plan for open-sourcing, integrating, and extending the tool.
