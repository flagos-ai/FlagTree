/*
* Copyright 2018-2020 Philippe Tillet
* Copyright 2020-2022 OpenAI
* Copyright 2025-     FlagOS Contributors
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <string>
#include <type_traits>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton {

// Better version of llvm::join.  This one works when T is an integer or any
// other type which defines operator<<(raw_ostream).
template <typename C>
std::string join(C &&container, llvm::StringRef sep = ", ") {
  std::string ret;
  llvm::raw_string_ostream s(ret);
  for (const auto &elem : container) {
    if (!ret.empty())
      s << sep;
    s << elem;
  }
  return ret;
}

// Joins a container of elements into a string, using `sep` as a separator.
//
// fn is called to transform each element of the container before it's added to
// the string.  fn must have one of the following two signatures.
//
//   - void fn(llvm::raw_ostream&, E), where E is the element type of the
//     container, or
//   - T fn(E), where T is a type which can be passed to
//     raw_ostream::operator<<.
//
template <typename C, typename Fn>
std::string join(C &&container, llvm::StringRef sep, Fn &&fn) {
  std::string ret;
  llvm::raw_string_ostream s(ret);
  for (const auto &elem : container) {
    if (!ret.empty())
      s << sep;

    if constexpr (std::is_invocable_v<Fn, llvm::raw_ostream &,
                                      decltype(elem)>) {
      static_assert(
          std::is_void_v<
              std::invoke_result_t<Fn, llvm::raw_ostream &, decltype(elem)>>);
      fn(s, elem);
    } else {
      s << fn(elem);
    }
  }
  return ret;
}

} // namespace mlir::triton
