# DiskANN + hnswlib vendored for CoTra indexing.

set(PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/diskann")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

include_directories(${PROJECT_SOURCE_DIR}/include)

if(NOT PYBIND)
  set(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS ON)
endif()

if(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS)
  include_directories(${GPERFTOOLS_INCLUDE_DIR})
endif()

include_directories(/usr/local/include)
include_directories(${FAISS_ROOT})
link_libraries(${FAISS_LIB})

if(NOT MSVC)
  find_package(Boost COMPONENTS program_options)
endif()

if(NOT Boost_FOUND)
  message(FATAL_ERROR "Couldn't find Boost dependency")
endif()

include_directories(${Boost_INCLUDE_DIR})
link_directories(${OMP_PATH})
link_directories(${MKL_PATH})
include_directories(${MKL_INCLUDE_PATH})

if(NOT MKL_PATH OR NOT MKL_INCLUDE_PATH)
  message(FATAL_ERROR "MKL not configured (run cmake from project root first)")
endif()

add_compile_options(-m64 -Wl,--no-as-needed)
link_libraries(mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 pthread m dl)
add_definitions(-DMKL_ILP64)

set(DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS ${TCMALLOC_LIB})

if(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS)
  add_definitions(-DRELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS)
endif()

if(NOT MSVC)
  set(DISKANN_ASYNC_LIB ${LIBAIO_LIB})
  include_directories(${LIBAIO_INCLUDE_DIR})
  if(COTRA_DEPS_PREFIX AND EXISTS "${COTRA_DEPS_PREFIX}/include/libaio.h")
    include_directories("${COTRA_DEPS_PREFIX}/include")
  endif()
endif()

set(ENV{TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD} 500000000000)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -fPIC")

file(GLOB CPP_SOURCES diskann/src/*.cpp)
add_library(diskann ${CPP_SOURCES})

include(diskann/clang-format.cmake)
