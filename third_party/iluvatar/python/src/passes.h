// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#define ADD_PASS_WRAPPER_0(name, builder)                                      \
  m.def(name, [](mlir::PassManager &pm) { pm.addPass(builder()); })

#define ADD_PASS_WRAPPER_1(name, builder, ty0)                                 \
  m.def(name,                                                                  \
        [](mlir::PassManager &pm, ty0 val0) { pm.addPass(builder(val0)); })

#define ADD_PASS_WRAPPER_2(name, builder, ty0, ty1)                            \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1) {                  \
    pm.addPass(builder(val0, val1));                                           \
  })

#define ADD_PASS_WRAPPER_3(name, builder, ty0, ty1, ty2)                       \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2) {        \
    pm.addPass(builder(val0, val1, val2));                                     \
  })

#define ADD_PASS_WRAPPER_4(name, builder, ty0, ty1, ty2, ty3)                  \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2,          \
                 ty3 val3) { pm.addPass(builder(val0, val1, val2, val3)); })

#define ADD_PASS_OPTION_WRAPPER_1(name, builder, ty0)                          \
  m.def(name,                                                                  \
        [](mlir::PassManager &pm, ty0 val0) { pm.addPass(builder({val0})); })

#define ADD_PASS_OPTION_WRAPPER_2(name, builder, ty0, ty1)                     \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1) {                  \
    pm.addPass(builder({val0, val1}));                                         \
  })

#define ADD_PASS_OPTION_WRAPPER_3(name, builder, ty0, ty1, ty2)                \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2) {        \
    pm.addPass(builder({val0, val1, val2}));                                   \
  })

#define ADD_PASS_OPTION_WRAPPER_4(name, builder, ty0, ty1, ty2, ty3)           \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2,          \
                 ty3 val3) { pm.addPass(builder({val0, val1, val2, val3})); })

#define ADD_PASS_OPTION_WRAPPER_5(name, builder, ty0, ty1, ty2, ty3, ty4)      \
  m.def(name,                                                                  \
        [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2, ty3 val3,      \
           ty4 val4) { pm.addPass(builder({val0, val1, val2, val3, val4})); })
