# Resolve MKL, OpenMP runtime, Faiss, libmemcached, RDMA, libaio for CoTra.

include_guard(GLOBAL)

function(cotra_check_path desc path)
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "CoTra dependency missing: ${desc}\n  Expected: ${path}")
  endif()
endfunction()

function(cotra_setup_mkl)
  if(NOT MKL_LIB_DIR)
    set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64" PARENT_SCOPE)
  endif()
  cotra_check_path("MKL core library" "${MKL_LIB_DIR}/libmkl_core.so")
  cotra_check_path("MKL headers" "${MKL_INCLUDE_PATH}/mkl.h")

  set(_omp_dir "${INTEL_OMP_LIB_DIR}")
  if(NOT _omp_dir OR NOT EXISTS "${_omp_dir}/libiomp5.so")
    file(GLOB _omp_candidates
      "${INTEL_ONEAPI_ROOT}/compiler/*/linux/compiler/lib/intel64_lin/libiomp5.so"
      "${INTEL_ONEAPI_ROOT}/compiler/latest/lib/libiomp5.so")
    if(_omp_candidates)
      list(GET _omp_candidates 0 _omp_lib)
      get_filename_component(_omp_dir "${_omp_lib}" DIRECTORY)
    else()
      message(FATAL_ERROR "Could not find libiomp5.so under ${INTEL_ONEAPI_ROOT}/compiler")
    endif()
  endif()
  cotra_check_path("Intel OpenMP (libiomp5)" "${_omp_dir}/libiomp5.so")

  set(MKL_PATH "${MKL_LIB_DIR}" PARENT_SCOPE)
  set(OMP_PATH "${_omp_dir}" PARENT_SCOPE)
  set(INTEL_OMP_LIB_DIR "${_omp_dir}" PARENT_SCOPE)
endfunction()

function(cotra_setup_faiss)
  if(NOT EXISTS "${FAISS_LIB}")
    if(EXISTS "${FAISS_BUILD_DIR}/libfaiss.so")
      set(FAISS_LIB "${FAISS_BUILD_DIR}/libfaiss.so" PARENT_SCOPE)
    elseif(EXISTS "${FAISS_BUILD_DIR}/faiss/libfaiss.a")
      set(FAISS_LIB "${FAISS_BUILD_DIR}/faiss/libfaiss.a" PARENT_SCOPE)
    else()
      message(FATAL_ERROR
        "Faiss not built. Run:\n"
        "  /gpfs/users/delgaudio/exp_eval/Distributed-HNSW/script/build_faiss_gcc10.sh\n"
        "or set -DFAISS_LIB=...")
    endif()
  endif()
  cotra_check_path("Faiss headers" "${FAISS_ROOT}/faiss/IndexFlat.h")
  cotra_check_path("Faiss library" "${FAISS_LIB}")
endfunction()

function(cotra_setup_rdma)
  cotra_check_path("RDMA header verbs.h" "${RDMA_INCLUDE_DIR}/infiniband/verbs.h")
  cotra_check_path("libibverbs" "${IBVERBS_LIB}")
  cotra_check_path("librdmacm" "${RDMACM_LIB}")
endfunction()

function(cotra_setup_libaio)
  cotra_check_path("libaio" "${LIBAIO_LIB}")
endfunction()

function(cotra_setup_gperftools)
  if(EXISTS "${GPERFTOOLS_ROOT}/include/gperftools/malloc_extension.h")
    set(GPERFTOOLS_INCLUDE_DIR "${GPERFTOOLS_ROOT}/include" PARENT_SCOPE)
    if(EXISTS "${GPERFTOOLS_ROOT}/lib/libtcmalloc.so")
      set(TCMALLOC_LIB "${GPERFTOOLS_ROOT}/lib/libtcmalloc.so" PARENT_SCOPE)
    elseif(EXISTS "${GPERFTOOLS_ROOT}/lib64/libtcmalloc.so")
      set(TCMALLOC_LIB "${GPERFTOOLS_ROOT}/lib64/libtcmalloc.so" PARENT_SCOPE)
    else()
      set(TCMALLOC_LIB "tcmalloc" PARENT_SCOPE)
    endif()
  elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/diskann/gperftools/src/gperftools/malloc_extension.h")
    set(GPERFTOOLS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/diskann/gperftools/src" PARENT_SCOPE)
    set(TCMALLOC_LIB "tcmalloc" PARENT_SCOPE)
  else()
    message(FATAL_ERROR
      "gperftools not found. Run:\n"
      "  bash script_slurm/build_cotra_deps.sh\n"
      "(builds gperftools into COTRA_DEPS_PREFIX)")
  endif()
endfunction()
