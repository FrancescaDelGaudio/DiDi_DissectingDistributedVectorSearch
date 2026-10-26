cmake_minimum_required(VERSION 3.11)
project(DistributedANN CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---- Required dependencies ----
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# ---- Compiler flags ----
add_compile_options(-O3 -march=native -Wall -Wextra)

# ---- Step 1: k-means + epsilon-replication ----
add_executable(build_index src/build_index.cpp)
target_include_directories(build_index PRIVATE src)
target_link_libraries(build_index PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# ---- Step 2: per-cluster Vamana graphs ----
add_executable(build_graph src/build_graph.cpp)
target_include_directories(build_graph PRIVATE src)
target_link_libraries(build_graph PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# ---- Step 3: global graph stitching ----
add_executable(build_stitch src/build_stitch.cpp)
target_include_directories(build_stitch PRIVATE src)
target_link_libraries(build_stitch PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# ---- Step 4: PQ codebook + vector codes ----
add_executable(build_pq src/build_pq.cpp)
target_include_directories(build_pq PRIVATE src)
target_link_libraries(build_pq PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# ---- Step 5: in-memory head index (HNSW) ----
add_executable(build_head src/build_head.cpp)
target_include_directories(build_head PRIVATE src ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(build_head PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# ---- Step 6: distributed KV store (nodes.bin) ----
add_executable(build_kvstore src/build_kvstore.cpp)
target_include_directories(build_kvstore PRIVATE src)
target_link_libraries(build_kvstore PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# ---- Diagnostic: MPI launcher / wireup check ----
add_executable(mpi_size_test src/tools/mpi_size_test.cpp)
target_link_libraries(mpi_size_test PRIVATE MPI::MPI_CXX)

# ---- Query answering (Algorithms 1 & 2) ----
add_executable(run_query src/run_query.cpp)
target_include_directories(run_query PRIVATE src ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(run_query PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# ---- Batched query answering (replicated orchestrator + head, targeted fan-out) ----
add_executable(run_query_batch src/run_query_batch.cpp)
target_include_directories(run_query_batch PRIVATE src ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(run_query_batch PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "MPI found  : ${MPI_CXX_FOUND}")
message(STATUS "OpenMP     : ${OpenMP_CXX_FOUND}")
