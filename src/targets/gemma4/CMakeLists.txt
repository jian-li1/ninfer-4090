target_sources(ninfer_engine PRIVATE
  impl/frontend/frontend.cpp
  impl/runtime/continuation.cpp)
target_include_directories(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/export
  ${CMAKE_CURRENT_SOURCE_DIR}/impl)
