// Copyright 2026 FlagOS Contributors

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
  context.loadDialect<arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  module->walk([&](Operation *root) {
    // getSlice topologically sorts its result and requires a common ancestor;
    // the owning module itself has no parent and is not a valid query root.
    if (!root->getParentOp())
      return;
    auto reference = mlir::getSlice(root);
    auto actual = getCoalescingSlice(root);
    ASSERT_EQ(reference.size(), actual.size()) << root->getName().getStringRef().str();
    for (Operation *member : reference)
      EXPECT_TRUE(actual.contains(member)) << member->getName().getStringRef().str();
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

} // namespace
} // namespace mlir::triton::tle
