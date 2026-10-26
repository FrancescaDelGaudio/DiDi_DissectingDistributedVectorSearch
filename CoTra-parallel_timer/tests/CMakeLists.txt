
link_libraries(coromem rdma_anns rdma_index)

add_subdirectory(rdma)
add_subdirectory(anns)
add_subdirectory(index)

add_executable(scala_anns${COTRA_BIN_SUFFIX} scala_anns.cc)
target_link_libraries(scala_anns${COTRA_BIN_SUFFIX} ${TCMALLOC_LIB})
# -rdynamic exports symbols so backtrace_symbols_fd() resolves function names
# in the fatal-signal handler in scala_anns.cc.
target_link_options(scala_anns${COTRA_BIN_SUFFIX} PRIVATE -rdynamic)

add_executable(scala_index${COTRA_BIN_SUFFIX} scala_index.cc)
target_link_libraries(scala_index${COTRA_BIN_SUFFIX} diskann ${TCMALLOC_LIB} ${FAISS_LIB})
