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

#include "Context/Shadow.h"

#include <stdexcept>
#include <thread>

namespace proton {

void ShadowContextSource::initializeThreadContext() {
  if (!threadContextInitialized[this]) {
    threadContextStack[this] = *mainContextStack;
    threadContextInitialized[this] = true;
  }
}

void ShadowContextSource::enterScope(const Scope &scope) {
  initializeThreadContext();
  threadContextStack[this].push_back(scope);
}

std::vector<Context> ShadowContextSource::getContextsImpl() {
  initializeThreadContext();
  return threadContextStack[this];
}

size_t ShadowContextSource::getDepth() {
  initializeThreadContext();
  return threadContextStack[this].size();
}

void ShadowContextSource::exitScope(const Scope &scope) {
  if (threadContextStack[this].empty()) {
    throw std::runtime_error("Context stack is empty");
  }
  if (threadContextStack[this].back() != scope) {
    throw std::runtime_error("Context stack is not balanced");
  }
  threadContextStack[this].pop_back();
}

void ShadowContextSource::clear() {
  ContextSource::clear();
  threadContextStack[this].clear();
  threadContextInitialized[this] = false;
}

/*static*/ thread_local std::map<ShadowContextSource *, bool>
    ShadowContextSource::threadContextInitialized;

/*static*/ thread_local std::map<ShadowContextSource *, std::vector<Context>>
    ShadowContextSource::threadContextStack;

} // namespace proton
