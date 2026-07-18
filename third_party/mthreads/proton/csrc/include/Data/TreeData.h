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

#ifndef PROTON_DATA_TREE_DATA_H_
#define PROTON_DATA_TREE_DATA_H_

#include "Context/Context.h"
#include "Data.h"
#include "nlohmann/json.hpp"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

namespace proton {

class TreeData : public Data {
public:
  TreeData(const std::string &path, ContextSource *contextSource);
  virtual ~TreeData();

  TreeData(const std::string &path) : TreeData(path, nullptr) {}

  std::string toJsonString(size_t phase) const override;

  std::vector<uint8_t> toMsgPack(size_t phase) const override;

  DataEntry addOp(const std::string &name) override;

  DataEntry addOp(size_t phase, size_t contextId,
                  const std::vector<Context> &contexts) override;

  void
  addMetrics(size_t scopeId,
             const std::map<std::string, MetricValueType> &metrics) override;

  void
  addMetrics(size_t phase, size_t entryId,
             const std::map<std::string, MetricValueType> &metrics) override;

protected:
  // ScopeInterface
  void enterScope(const Scope &scope) override;

  void exitScope(const Scope &scope) override;

private:
  // `tree` and `scopeIdToContextId` can be accessed by both the user thread and
  // the background threads concurrently, so methods that access them should be
  // protected by a (shared) mutex.
  class Tree;
  json buildHatchetJson(TreeData::Tree *tree) const;
  std::vector<uint8_t> buildHatchetMsgPack(TreeData::Tree *tree) const;

  // Data
  void doDump(std::ostream &os, OutputFormat outputFormat,
              size_t phase) const override;

  OutputFormat getDefaultOutputFormat() const override {
    return OutputFormat::Hatchet;
  }

  void dumpHatchet(std::ostream &os, size_t phase) const;
  void dumpHatchetMsgPack(std::ostream &os, size_t phase) const;

  PhaseStore<Tree> treePhases;
  // ScopeId -> ContextId
  std::unordered_map<size_t, size_t> scopeIdToContextId;
};

} // namespace proton

#endif // PROTON_DATA_TREE_DATA_H_
