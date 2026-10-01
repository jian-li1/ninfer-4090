target_sources(ninfer_engine PRIVATE
  impl/load/bindings.cpp
  impl/runtime/kv_groups.cpp
  impl/runtime/persistent_model.cpp
  impl/runtime/reference_model.cpp
  impl/runtime/reference_kernels.cu)
target_include_directories(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/export
  ${CMAKE_CURRENT_SOURCE_DIR}/impl)
