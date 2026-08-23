# Cluster defaults for Stella (ncpum). Override any path via -DVAR=... or env.
#
# Typical configure:
#   cmake -S . -B build -C cmake/StellaCluster.cmake

if(DEFINED ENV{COTRA_DEPS_PREFIX})
  set(COTRA_DEPS_PREFIX "$ENV{COTRA_DEPS_PREFIX}" CACHE PATH "User-built deps (memcached, gperftools, ...)")
else()
  set(COTRA_DEPS_PREFIX "/gpfs/scratch/delgaudio/cotra/deps" CACHE PATH "User-built deps (memcached, gperftools, ...)")
endif()

set(INTEL_ONEAPI_ROOT "/gpfs/softs/intel/oneapi" CACHE PATH "Intel oneAPI root")

set(MKL_ROOT "${INTEL_ONEAPI_ROOT}/mkl/latest" CACHE PATH "Intel MKL root")
set(MKL_INCLUDE_PATH "${MKL_ROOT}/include" CACHE PATH "Intel MKL headers")
set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64" CACHE PATH "Intel MKL libraries")

set(INTEL_OMP_LIB_DIR "${INTEL_ONEAPI_ROOT}/compiler/2021.2.0/linux/compiler/lib/intel64_lin" CACHE PATH "libiomp5 directory")

set(FAISS_ROOT "/gpfs/users/delgaudio/exp_eval/Distributed-HNSW/third_party/faiss" CACHE PATH "Faiss source tree (contains faiss/Index.h)")
set(FAISS_BUILD_DIR "${FAISS_ROOT}/build-gcc10" CACHE PATH "Faiss build directory")
set(FAISS_LIB "${FAISS_BUILD_DIR}/faiss/libfaiss.so" CACHE FILEPATH "Faiss library to link")

set(GPERFTOOLS_ROOT "${COTRA_DEPS_PREFIX}" CACHE PATH "gperftools install prefix (include/ and lib/)")
set(LIBAIO_INCLUDE_DIR "/usr/include" CACHE PATH "libaio headers")
set(LIBAIO_LIB "/usr/lib64/libaio.so.1" CACHE FILEPATH "libaio shared library")

set(IBVERBS_LIB "/usr/lib64/libibverbs.so" CACHE FILEPATH "libibverbs")
set(RDMACM_LIB "/usr/lib64/librdmacm.so" CACHE FILEPATH "librdmacm")

# RDMA userspace headers ship with rdma-core on RHEL/CentOS.
set(RDMA_INCLUDE_DIR "/usr/include" CACHE PATH "RDMA headers (infiniband/verbs.h)")

list(APPEND CMAKE_PREFIX_PATH "${COTRA_DEPS_PREFIX}")

if(EXISTS "${COTRA_DEPS_PREFIX}/lib/pkgconfig")
  set(ENV{PKG_CONFIG_PATH} "${COTRA_DEPS_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
endif()

message(STATUS "StellaCluster: COTRA_DEPS_PREFIX=${COTRA_DEPS_PREFIX}")
message(STATUS "StellaCluster: MKL_ROOT=${MKL_ROOT}")
message(STATUS "StellaCluster: FAISS_ROOT=${FAISS_ROOT}")
