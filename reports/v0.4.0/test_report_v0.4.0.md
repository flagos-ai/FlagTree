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



## FlagTree v0.4.0 Test Report

FlagTree tests are validated on different backends, but currently the tests consist of only python unit tests.

### Python unit-test:

|Backend |Triton|Tests Number|Script location|Test command|Passing rate|
|--------|------|------------|---------------|------------|------------|
|nvidia  |3.1   |9161        |python/test/unit                      |python3 -m pytest -s |100%|
|nvidia  |3.2   |9734        |python/test/unit                      |python3 -m pytest -s |100%|
|nvidia  |3.3   |10579       |python/test/unit                      |python3 -m pytest -s |100%|
|nvidia  |3.4   |17446       |python/test/unit                      |python3 -m pytest -s |100%|
|nvidia  |3.5   |12687       |python/test/unit<br>python/test/tle   |python3 -m pytest -s |100%|
|iluvatar|3.1   |12876       |third_party/iluvatar/python/test/unit |python3 -m pytest -s |100%|
|mthreads|3.1   |4116        |third_party/mthreads/python/test/unit |python3 -m pytest -s |100%|
|metax   |3.1   |6309        |third_party/metax/python/test/unit    |python3 -m pytest -s |100%|
|xpu     |3.0   |4163        |third_party/xpu/python/test/unit      |python3 -m pytest -s |100%|
|hcu     |3.0   |7644        |third_party/hcu/python/test/unit      |sh flagtree_test.sh  |100%|
|ascend  |3.2   |1223        |third_party/ascend/examples/pytest_ut |python3 -m pytest -s |98% |
|aipu    |3.3   |21          |third_party/aipu/python/test/         |python3              |100%|
|enflame |3.3   |145         |third_party/enflame/python/test/unit  |python3 -m pytest -s |100%|
