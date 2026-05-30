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
