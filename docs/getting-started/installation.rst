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

============
Installation
============

For supported platform/OS and supported hardware, review the `Compatibility <https://github.com/triton-lang/triton?tab=readme-ov-file#compatibility>`_ section on Github.

--------------------
Binary Distributions
--------------------

You can install the latest stable release of Triton from pip:

.. code-block:: bash

      pip install triton

Binary wheels are available for CPython 3.10-3.14.

-----------
From Source
-----------

++++++++++++++
Python Package
++++++++++++++

You can install the Python package from source by running the following commands:

.. code-block:: bash

      git clone https://github.com/triton-lang/triton.git
      cd triton

      pip install -r python/requirements.txt # build-time dependencies
      pip install -e .

Note that, if llvm is not present on your system, the setup.py script will download the official LLVM static libraries and link against that.

For building with a custom LLVM, review the `Building with a custom LLVM <https://github.com/triton-lang/triton?tab=readme-ov-file#building-with-a-custom-llvm>`_ section on Github.

You can then test your installation by running the tests:

.. code-block:: bash

      # One-time setup
      make dev-install

      # To run all tests (requires a GPU)
      make test

      # Or, to run tests without a GPU
      make test-nogpu
