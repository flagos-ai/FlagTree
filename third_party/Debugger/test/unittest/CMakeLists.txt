add_triton_ut(
  NAME TestFlagTreeDebuggerInstrumentation
  SRCS InstrumentationTest.cpp
  LIBS
    FlagTreeDebugger
)

add_triton_ut(
  NAME FlagTreeDebuggerFrontendTest
  SRCS FrontendBridgeTest.cpp
  LIBS FlagTreeDebuggerFrontend
)

add_triton_ut(
  NAME FlagTreeDebuggerRuntimeTest
  SRCS RuntimeTransferEngineTest.cpp
  LIBS FlagTreeDebuggerRuntime
)

add_triton_ut(
  NAME DebuggerDecode
  SRCS DecodeTest.cpp
  LIBS
    FlagTreeDebugger
)

add_triton_ut(
  NAME FlagTreeDebuggerMetadataTest
  SRCS MetadataTest.cpp
  LIBS
    FlagTreeDebugger
)
