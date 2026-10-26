# Find vendored Faiss built by script/build_faiss_gcc10.sh
#
# Sets:
#   FAISS_INCLUDE_DIR
#   FAISS_LIBRARY
#   FAISS_FOUND

set(_FAISS_ROOT "${CMAKE_SOURCE_DIR}/third_party/faiss")
set(_FAISS_BUILD "${_FAISS_ROOT}/build-gcc10")

find_path(FAISS_INCLUDE_DIR
  NAMES faiss/IndexFlat.h
  PATHS "${_FAISS_ROOT}"
  NO_DEFAULT_PATH
)

find_library(FAISS_LIBRARY
  NAMES faiss
  PATHS "${_FAISS_BUILD}/faiss" "${_FAISS_BUILD}"
  NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Faiss DEFAULT_MSG FAISS_INCLUDE_DIR FAISS_LIBRARY)

mark_as_advanced(FAISS_INCLUDE_DIR FAISS_LIBRARY)
