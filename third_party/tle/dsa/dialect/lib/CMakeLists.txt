# Copyright 2026- Xcoresigma Technology Co., Ltd

add_subdirectory(Analysis)
add_subdirectory(IR)
# Conversion depends on the main TleIR target, so it is added by
# third_party/tle/CMakeLists.txt after dialect/lib is configured.
# add_subdirectory(Conversion)
