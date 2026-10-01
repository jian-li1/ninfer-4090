target_sources(ninfer_engine PRIVATE
  impl/load/bindings.cpp)
target_include_directories(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/export
  ${CMAKE_CURRENT_SOURCE_DIR}/impl)
