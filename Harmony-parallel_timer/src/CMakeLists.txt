include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SRC_FILES
    Clustering.cpp
    Index.cpp
    IVF.cpp
    node.cpp
    # utils.cpp
)

add_library(Harmony STATIC ${SRC_FILES})

# Eigen must be visible to all Harmony TU (do not rely only on directory-level include_directories)
target_include_directories(Harmony PUBLIC "${HARMONY_EIGEN3_INCLUDE}")

target_link_libraries(Harmony PUBLIC OpenMP::OpenMP_CXX
                                    mkl_intel_lp64
                                    mkl_sequential
                                    mkl_core
                                    MPI::MPI_CXX)

add_executable(main main.cpp)
target_link_libraries(main Harmony faiss)

add_executable(query query.cpp)
target_link_libraries(query Harmony faiss)

# add_executable(divide divide.cpp)
# target_link_libraries(divide Tribase faiss)
# add_subdirectory(tests)