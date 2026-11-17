add_triton_library(TleToThvTile
  ConvertTleToThvTile.cpp
  SliceOpsToThvTile.cpp
  CumsumOpToThvTile.cpp

  DEPENDS
  TleToThvTilePassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRPass
  MLIRTransforms
  TleIR
  ThvTileIR
)
