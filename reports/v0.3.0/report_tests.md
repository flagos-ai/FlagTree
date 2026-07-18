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



## FlagTree Test-Report

FlagTree tests are validated on different backends, but currently the tests consist of only unit tests, which we will refine in the future for smaller or larger scale tests.

### 1. Python unit test:

| 　　　　　　　　　　　　 | default                   | iluvatar                                 | klx xpu                                       | mthreads                                       | metax                                       | hcu                                       |
|----------------------|---------------------------|-------------------------------------------|------------------------------------------------|------------------------------------------------|---------------------------------------------|---------------------------------------------|
| Number of unit tests | 9161 items               | 11395 items                               | 4183 items                                    | 4116 items                                    | 6309 items                                 | 309 items                                 |
| Script location      | flagtree/python/test/unit | flagtree/third_party/iluvatar/python/test/unit | flagtree/third_party/xpu/python/test/unit | flagtree/third_party/mthreads/python/test/unit | flagtree/third_party/metax/python/test/unit | flagtree/third_party/hcu/python/test/unit |
| Test command         | python3 -m pytest -s      | python3 -m pytest -s                      | python3 -m pytest -s                           | python3 -m pytest -s                           | python3 -m pytest -s                        | sh flagtree_test.sh                        |
| Passing rate         | 100%                      | 100%                                      | 100%                                           | 100%                                           | 100%                                        | 100%                                        |
