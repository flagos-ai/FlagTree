add_library(entryC SHARED libentry.cpp)
if(APPLE)
  target_link_libraries(entryC PRIVATE -undefined dynamic_lookup)
endif()
