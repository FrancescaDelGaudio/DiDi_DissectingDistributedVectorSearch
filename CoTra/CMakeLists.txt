cmake_minimum_required(VERSION 3.15)
project(rdma-anns LANGUAGES CXX)

# Compile-time partition count (must match sbatch -N). Default 4 = build/tests/scala_index.
# For 6 partitions use a separate build tree: bash script_slurm/build_cotra_6p.sh
set(COTRA_MACHINE_NUM "4" CACHE STRING "CoTra MACHINE_NUM (partitions / nodes)")
if(NOT COTRA_MACHINE_NUM MATCHES "^[0-9]+$")
  message(FATAL_ERROR "COTRA_MACHINE_NUM must be a positive integer, got: ${COTRA_MACHINE_NUM}")
endif()
if(NOT COTRA_MACHINE_NUM EQUAL 4)
  add_compile_definitions(MACHINE_NUM=${COTRA_MACHINE_NUM})
  set(COTRA_BIN_SUFFIX "_${COTRA_MACHINE_NUM}p")
  message(STATUS "CoTra MACHINE_NUM=${COTRA_MACHINE_NUM} -> binaries *${COTRA_BIN_SUFFIX}")
else()
  set(COTRA_BIN_SUFFIX "")
  message(STATUS "CoTra MACHINE_NUM=4 (default binaries scala_index / scala_anns)")
endif()

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")
endif()

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCoTraDeps.cmake")

# Optional cluster preset: cmake -C cmake/StellaCluster.cmake ...
if(NOT COTRA_DEPS_PREFIX AND DEFINED ENV{COTRA_DEPS_PREFIX})
  set(COTRA_DEPS_PREFIX "$ENV{COTRA_DEPS_PREFIX}")
endif()
if(NOT COTRA_DEPS_PREFIX)
  set(COTRA_DEPS_PREFIX "/gpfs/scratch/delgaudio/cotra/deps")
endif()

if(NOT MKL_ROOT)
  set(MKL_ROOT "/gpfs/softs/intel/oneapi/mkl/latest")
endif()
if(NOT MKL_INCLUDE_PATH)
  set(MKL_INCLUDE_PATH "${MKL_ROOT}/include")
endif()
if(NOT MKL_LIB_DIR)
  set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64")
endif()
if(NOT INTEL_ONEAPI_ROOT)
  set(INTEL_ONEAPI_ROOT "/gpfs/softs/intel/oneapi")
endif()
if(NOT INTEL_OMP_LIB_DIR)
  set(INTEL_OMP_LIB_DIR "${INTEL_ONEAPI_ROOT}/compiler/2021.2.0/linux/compiler/lib/intel64_lin")
endif()
if(NOT FAISS_ROOT)
  set(FAISS_ROOT "/gpfs/users/delgaudio/exp_eval/Distributed-HNSW/third_party/faiss")
endif()
if(NOT FAISS_BUILD_DIR)
  set(FAISS_BUILD_DIR "${FAISS_ROOT}/build-gcc10")
endif()
if(NOT FAISS_LIB)
  set(FAISS_LIB "${FAISS_BUILD_DIR}/faiss/libfaiss.so")
endif()
if(NOT GPERFTOOLS_ROOT)
  set(GPERFTOOLS_ROOT "${COTRA_DEPS_PREFIX}")
endif()
if(NOT RDMA_INCLUDE_DIR)
  set(RDMA_INCLUDE_DIR "/usr/include")
endif()
if(NOT IBVERBS_LIB)
  set(IBVERBS_LIB "/usr/lib64/libibverbs.so")
endif()
if(NOT RDMACM_LIB)
  set(RDMACM_LIB "/usr/lib64/librdmacm.so")
endif()
if(NOT LIBAIO_LIB)
  if(EXISTS "${COTRA_DEPS_PREFIX}/lib/libaio.so")
    set(LIBAIO_LIB "${COTRA_DEPS_PREFIX}/lib/libaio.so")
  else()
    set(LIBAIO_LIB "/usr/lib64/libaio.so.1")
  endif()
endif()
if(NOT LIBAIO_INCLUDE_DIR)
  if(EXISTS "${COTRA_DEPS_PREFIX}/include/libaio.h")
    set(LIBAIO_INCLUDE_DIR "${COTRA_DEPS_PREFIX}/include")
  else()
    set(LIBAIO_INCLUDE_DIR "/usr/include")
  endif()
endif()

cotra_setup_mkl()
cotra_setup_faiss()
cotra_setup_rdma()
cotra_setup_libaio()
cotra_setup_gperftools()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -lrt -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 -shared-libgcc -fcoroutines")
set(CMAKE_THREAD_LIBS_INIT "-lpthread")

find_package(Threads REQUIRED)
find_package(Boost 1.58.0 REQUIRED COMPONENTS serialization iostreams program_options system filesystem)

find_library(MEMCACHED_LIB memcached
  HINTS "${COTRA_DEPS_PREFIX}/lib" "${COTRA_DEPS_PREFIX}/lib64"
  REQUIRED)

message(STATUS "Build CoroMem library")

include_directories(./third_party)
set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/third_party)
add_subdirectory(${THIRD_PARTY_DIR})

add_custom_target(lib)
add_subdirectory(coromem)

message(STATUS "Build RDMA library")

link_libraries(
  pthread
  ${IBVERBS_LIB}
  ${RDMACM_LIB}
  ${MEMCACHED_LIB}
)
include_directories(
  "${PROJECT_SOURCE_DIR}"
  "${RDMA_INCLUDE_DIR}"
  "${COTRA_DEPS_PREFIX}/include"
)

add_library(rdma_anns)
file(GLOB sources
  ${CMAKE_SOURCE_DIR}/src/*.cc
  ${CMAKE_SOURCE_DIR}/src/rdma/*.cc
  ${CMAKE_SOURCE_DIR}/src/anns/*.cc
)
target_sources(rdma_anns PRIVATE ${sources})
target_include_directories(rdma_anns PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

link_directories(${OMP_PATH})
link_directories(${MKL_PATH})
include_directories(${MKL_INCLUDE_PATH})

link_libraries(mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 pthread m dl ${Boost_LIBRARIES})

add_library(rdma_index)
file(GLOB index_sources
  ${CMAKE_SOURCE_DIR}/src/index/*.cc
)
target_sources(rdma_index PRIVATE ${index_sources})
target_include_directories(rdma_index PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${FAISS_ROOT}>
  $<INSTALL_INTERFACE:include>
)

add_subdirectory(tests)
