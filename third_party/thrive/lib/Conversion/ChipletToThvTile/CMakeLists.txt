add_triton_library(ChipletToThvTile
    ChipletToThvTile.cpp

    DEPENDS
    ThvTileTableGen
    TleTableGen

    LINK_LIBS PUBLIC
    ThvTileIR
    TleIR
    MLIRArithDialect
)
