/*
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

#include "Analysis/CoalescingSlice.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <gtest/gtest.h>

namespace mlir::triton::tle {
namespace {

void expectSameMembership(StringRef source) {
  MLIRContext context;
  context
      .loadDialect<arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  module->walk([&](Operation *root) {
    // getSlice topologically sorts its result and requires a common ancestor;
    // the owning module itself has no parent and is not a valid query root.
    if (!root->getParentOp())
      return;
    auto reference = mlir::getSlice(root);
    auto actual = getCoalescingSlice(root);
    ASSERT_EQ(reference.size(), actual.size())
        << root->getName().getStringRef().str();
    for (Operation *member : reference)
      EXPECT_TRUE(actual.contains(member))
          << member->getName().getStringRef().str();
  });
}

TEST(CoalescingSlice, DiamondAndIndependentFunctions) {
  expectSameMembership(R"mlir(
    module {
      func.func @diamond(%a: i32, %b: i32) -> i32 {
        %c = arith.constant 1 : i32
        %x = arith.addi %a, %c : i32
        %y = arith.addi %b, %c : i32
        %z = arith.muli %x, %y : i32
        return %z : i32
      }
      func.func @independent(%a: i32) -> i32 {
        %c = arith.constant 2 : i32
        %x = arith.addi %a, %c : i32
        return %x : i32
      }
    }
  )mlir");
}

TEST(CoalescingSlice, NestedRegionsAndBlockArguments) {
  expectSameMembership(R"mlir(
    module {
      func.func @regions(%condition: i1, %a: i32, %b: i32) -> i32 {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %four = arith.constant 4 : index
        %loop = scf.for %i = %zero to %four step %one iter_args(%v = %a) -> i32 {
          %next = scf.if %condition -> i32 {
            %sum = arith.addi %v, %b : i32
            scf.yield %sum : i32
          } else {
            %product = arith.muli %v, %b : i32
            scf.yield %product : i32
          }
          scf.yield %next : i32
        }
        return %loop : i32
      }
    }
  )mlir");
}

TEST(CoalescingSlice, AsymmetricRegionReachability) {
  expectSameMembership(R"mlir(
    module {
      func.func @branches(%condition: i1, %a: i32, %b: i32) {
        scf.if %condition {
          %one = arith.constant 1 : i32
          %x = arith.addi %a, %one : i32
        } else {
          %two = arith.constant 2 : i32
          %y = arith.muli %b, %two : i32
        }
        return
      }
    }
  )mlir");
}

TEST(CoalescingSlice, UnusedBlockArgumentsDoNotConnectSiblings) {
  expectSameMembership(R"mlir(
    module {
      func.func @same_argument(%a: i32) {
        %one = arith.constant 1 : i32
        %two = arith.constant 2 : i32
        %x = arith.addi %a, %one : i32
        %y = arith.muli %a, %two : i32
        return
      }
    }
  )mlir");
}

TEST(CoalescingSlice, RepeatedOperandsAndManyOverlappingRoots) {
  std::string source;
  llvm::raw_string_ostream out(source);
  out << "module { func.func @dag() {\n";
  // Deterministic fan-in/fan-out DAG, including disconnected roots and repeated
  // operands. Compare all roots, not just the final arithmetic expression.
  unsigned state = 12345;
  for (unsigned index = 0; index < 64; ++index) {
    if (index < 4 || index % 11 == 0) {
      out << "%v" << index << " = arith.constant " << index << " : i32\n";
      continue;
    }
    state = state * 1664525u + 1013904223u;
    unsigned lhs = state % index;
    state = state * 1664525u + 1013904223u;
    unsigned rhs = index % 3 == 0 ? lhs : state % index;
    out << "%v" << index << " = arith.addi %v" << lhs << ", %v" << rhs
        << " : i32\n";
  }
  out << "return } }";
  expectSameMembership(source);
}

TEST(CoalescingSlice, WhileCarriedArgumentsConnectBothRegions) {
  MLIRContext context;
  context
      .loadDialect<arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @loop(%initial: i32, %limit: i32) -> i32 {
        %one = arith.constant 1 : i32
        %unrelated = arith.constant 99 : i32
        %result = scf.while (%v = %initial) : (i32) -> i32 {
          %condition = arith.cmpi slt, %v, %limit : i32
          scf.condition(%condition) %v : i32
        } do {
        ^bb0(%v: i32):
          %next = arith.addi %v, %one : i32
          scf.yield %next : i32
        }
        return %result : i32
      }
      func.func @independent(%value: i32) -> i32 {
        %one = arith.constant 1 : i32
        %sum = arith.addi %value, %one : i32
        return %sum : i32
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<func::FuncOp>("loop");
  auto loop = *function.getOps<scf::WhileOp>().begin();
  llvm::SetVector<Operation *> expected;
  loop.walk([&](Operation *op) { expected.insert(op); });
  expected.insert(&function.getBody().front().front());
  expected.insert(function.getBody().front().getTerminator());
  // Neither mlir::getSlice's single-region precondition nor a reference call
  // to that API defines membership here. Check the actual loop dependency
  // group explicitly, from both the condition and the carried-value update.
  for (Operation *root :
       {&loop.getBefore().front().front(), &loop.getAfter().front().front()}) {
    auto actual = getCoalescingSlice(root);
    ASSERT_EQ(actual.size(), expected.size());
    for (Operation *op : expected)
      EXPECT_TRUE(actual.contains(op)) << op->getName().getStringRef().str();
  }
}

} // namespace
} // namespace mlir::triton::tle
