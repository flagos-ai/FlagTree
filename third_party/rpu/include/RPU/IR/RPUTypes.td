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

#ifndef RPU_EXEC_TYPES
#define RPU_EXEC_TYPES

include "RPU/IR/RPUDialect.td"
include "mlir/IR/AttrTypeBase.td"

class RPU_Type<string name, string _mnemonic, list<Trait> traits = []> :
    TypeDef<RPU_Dialect, name, traits> {
  let mnemonic = _mnemonic;
}

def RPU_TileType : RPU_Type<"Tile", "tile"> {
  let summary = "RPU executable tile type";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    unsigned getRank() const { return getShape().size(); }
    int64_t getExtent() const {
      assert(getShape().size() == 1 && "getExtent requires rank-1 tile");
      return getShape().front();
    }
  }];
}

def RPU_Tile : DialectType<RPU_Dialect,
    CPred<"::mlir::isa<::mlir::rpu::exec::TileType>($_self)">,
    "RPU executable tile", "::mlir::rpu::exec::TileType">;

#endif
