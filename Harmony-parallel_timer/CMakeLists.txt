cmake_minimum_required(VERSION 3.21)
project(Harmony)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Standard)
elseif (NOT CMAKE_BUILD_TYPE MATCHES "Debug|Release|Standard")
    message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(OpenMP REQUIRED)
find_package(GTest REQUIRED)
find_package(MPI REQUIRED)

# Stella: module load intel/20U4/mkl sets MKLROOT; fallback matches that layout.
if(DEFINED ENV{MKLROOT})
    set(MKL_ROOT "$ENV{MKLROOT}")
else()
    set(MKL_ROOT "/gpfs/softs/intel/2020U4/compilers_and_libraries_2020.4.304/linux/mkl")
endif()
message(STATUS "MKL_ROOT: ${MKL_ROOT}")
include_directories(${MKL_ROOT}/include)
link_directories(${MKL_ROOT}/lib/intel64)

# Eigen: standard layout is <Eigen/Dense> with -I.../eigen3 (CMake variable kept Harmony-specific)
set(_eigen_prefix_paths
    "/softs/eigen/3.4.0/include"
    "/usr/include"
    "/usr/local/include"
)
if(DEFINED ENV{EIGEN3_ROOT})
    list(PREPEND _eigen_prefix_paths "$ENV{EIGEN3_ROOT}/include" "$ENV{EIGEN3_ROOT}")
endif()
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND _eigen_prefix_paths "$ENV{CONDA_PREFIX}/include")
endif()

unset(HARMONY_EIGEN3_INCLUDE CACHE)
find_path(HARMONY_EIGEN3_INCLUDE NAMES Eigen/Dense
    PATHS ${_eigen_prefix_paths}
    PATH_SUFFIXES eigen3)
if(NOT HARMONY_EIGEN3_INCLUDE)
    message(FATAL_ERROR
        "Eigen3 not found (expected .../eigen3/Eigen/Dense under a known prefix). "
        "On Stella: module load eigen/3.4.0, then rm -rf build-harmony && re-run cmake. "
        "Or: micromamba install -p ~/local/gcc13 -c conda-forge eigen "
        "Or: export EIGEN3_ROOT=/path/to/prefix containing eigen3/Eigen")
endif()
include_directories(${HARMONY_EIGEN3_INCLUDE})
message(STATUS "HARMONY_EIGEN3_INCLUDE: ${HARMONY_EIGEN3_INCLUDE}")

option(ENABLE_PROFILING "Enable profiling" OFF)
option(HARMONY_ENABLE_AVX512 "Use -mavx512f (needs CPU + matching Faiss flags)" OFF)

if (ENABLE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

# set(W_CXX_FLAGS " -Wall -Wextra")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "Standard")
    if(HARMONY_ENABLE_AVX512)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Ofast -mavx512f -march=native -mtune=native")
    else()
        # AVX2 only: avoids Faiss simdlib AVX512 header mismatch on nodes without AVX-512
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Ofast -mavx2 -mfma -mf16c -mpopcnt")
    endif()
elseif (CMAKE_BUILD_TYPE STREQUAL "MPI")
    if(HARMONY_ENABLE_AVX512)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Ofast -mavx512f -march=native -mtune=native")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Ofast -mavx2 -mfma -mf16c -mpopcnt")
    endif()
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(NSTATS)
endif()

include_directories(${CMAKE_SOURCE_DIR}/third_party)
include_directories(${CMAKE_SOURCE_DIR}/third_party/faiss)
set(FAISS_OPT_LEVEL generic CACHE STRING "Faiss CPU opt level: generic, avx2, avx512")
add_subdirectory(third_party/faiss)
add_subdirectory(src)