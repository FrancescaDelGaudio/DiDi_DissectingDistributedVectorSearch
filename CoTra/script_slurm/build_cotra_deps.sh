#!/bin/bash
# Build memcached server + libmemcached into user-writable prefix (no root).
# Run once on the LOGIN node (needs network for download).
#
# Usage:
#   cd /gpfs/users/delgaudio/exp_eval/CoTra
#   bash script_slurm/build_cotra_deps.sh
#
# Installs to: /gpfs/scratch/delgaudio/cotra/deps (override with COTRA_DEPS_PREFIX)

set -euo pipefail

PREFIX="${COTRA_DEPS_PREFIX:-/gpfs/scratch/delgaudio/cotra/deps}"
BUILD_DIR="${COTRA_DEPS_BUILD:-/gpfs/scratch/delgaudio/cotra/build-deps}"
JOBS="${JOBS:-8}"

MEMCACHED_VER=1.6.28
LIBEVENT_VER=2.1.12-stable
LIBMEMCACHED_VER=1.0.18

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
module purge 2>/dev/null || true
module load gcc/10.3.0

mkdir -p "$PREFIX" "$BUILD_DIR"
export PATH="$PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CC=gcc
export CXX=g++
export CFLAGS="-O2 -fPIC"
export CXXFLAGS="-O2 -fPIC"
export LDFLAGS="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib"

cd "$BUILD_DIR"

build_libevent() {
  if [[ -f "$PREFIX/lib/libevent.so" || -f "$PREFIX/lib64/libevent.so" ]]; then
    echo "[skip] libevent already in $PREFIX"
    return 0
  fi
  echo "=== build libevent $LIBEVENT_VER ==="
  local t="libevent-${LIBEVENT_VER}.tar.gz"
  [[ -f "$t" ]] || curl -fL -o "$t" "https://github.com/libevent/libevent/releases/download/release-${LIBEVENT_VER}/${t}"
  rm -rf "libevent-${LIBEVENT_VER}"
  tar xzf "$t"
  cd "libevent-${LIBEVENT_VER}"
  ./configure --prefix="$PREFIX" --disable-shared --enable-static
  make -j"$JOBS"
  make install
  cd "$BUILD_DIR"
}

build_memcached_server() {
  if [[ -x "$PREFIX/bin/memcached" ]]; then
    echo "[skip] memcached already in $PREFIX/bin"
    return 0
  fi
  echo "=== build memcached server $MEMCACHED_VER ==="
  local t="memcached-${MEMCACHED_VER}.tar.gz"
  [[ -f "$t" ]] || curl -fL -o "$t" "https://memcached.org/files/${t}"
  rm -rf "memcached-${MEMCACHED_VER}"
  tar xzf "$t"
  cd "memcached-${MEMCACHED_VER}"
  ./configure --prefix="$PREFIX" --with-libevent="$PREFIX"
  make -j"$JOBS"
  make install
  cd "$BUILD_DIR"
}

patch_libmemcached_sources() {
  # libmemcached 1.0.18: opt_servers is char*; GCC 10+ rejects `== false`
  if [[ -f clients/memflush.cc ]]; then
    sed -i 's/if (opt_servers == false)/if (!opt_servers)/g' clients/memflush.cc
  fi
}

build_libmemcached() {
  if [[ -f "$PREFIX/lib/libmemcached.so" || -f "$PREFIX/lib64/libmemcached.so" ]]; then
    echo "[skip] libmemcached already in $PREFIX"
    return 0
  fi
  echo "=== build libmemcached $LIBMEMCACHED_VER ==="
  local t="libmemcached-${LIBMEMCACHED_VER}.tar.gz"
  [[ -f "$t" ]] || curl -fL -o "$t" "https://launchpad.net/libmemcached/1.0/${LIBMEMCACHED_VER}/+download/${t}"
  rm -rf "libmemcached-${LIBMEMCACHED_VER}"
  tar xzf "$t"
  cd "libmemcached-${LIBMEMCACHED_VER}"
  patch_libmemcached_sources
  ./configure --prefix="$PREFIX"
  # Build/install only libraries (top-level Makefile; subdirs have no Makefile)
  make -j"$JOBS" libhashkit/libhashkit.la libmemcached/libmemcached.la
  make install-libLTLIBRARIES install-pkgconfigDATA install-nobase_includeHEADERS
  cd "$BUILD_DIR"
}

build_libaio() {
  if [[ -f "$PREFIX/include/libaio.h" ]]; then
    echo "[skip] libaio headers already in $PREFIX"
    return 0
  fi
  echo "=== build libaio 0.3.113 (headers + lib) ==="
  local ver="0.3.113"
  local t="libaio-${ver}.tar.gz"
  [[ -f "$t" ]] || curl -fL -o "$t" "https://pagure.io/libaio/archive/libaio-${ver}/libaio-${ver}.tar.gz"
  rm -rf "libaio-${ver}"
  tar xzf "$t"
  cd "libaio-${ver}"
  make prefix="$PREFIX" -j"$JOBS"
  make prefix="$PREFIX" install
  cd "$BUILD_DIR"
}

build_fmt() {
  if [[ -f "$PREFIX/lib/cmake/fmt/fmt-config.cmake" || -f "$PREFIX/lib64/cmake/fmt/fmt-config.cmake" ]]; then
    echo "[skip] fmt already in $PREFIX"
    return 0
  fi
  echo "=== build fmt 10.2.1 ==="
  local ver="10.2.1"
  local t="fmt-${ver}.zip"
  [[ -f "$t" ]] || curl -fL -o "$t" "https://github.com/fmtlib/fmt/archive/refs/tags/${ver}.zip"
  rm -rf "fmt-${ver}"
  unzip -q "$t"
  cmake -S "fmt-${ver}" -B "fmt-${ver}-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DFMT_DOC=OFF -DFMT_TEST=OFF
  cmake --build "fmt-${ver}-build" -j"$JOBS"
  cmake --install "fmt-${ver}-build"
}

build_gperftools() {
  if [[ -f "$PREFIX/include/gperftools/malloc_extension.h" ]]; then
    echo "[skip] gperftools already in $PREFIX"
    return 0
  fi
  echo "=== build gperftools 2.15 (libtcmalloc for DiskANN) ==="
  local ver="2.15"
  local t="gperftools-${ver}.tar.gz"
  [[ -f "$t" ]] || curl -fL -o "$t" "https://github.com/gperftools/gperftools/releases/download/gperftools-${ver}/${t}"
  rm -rf "gperftools-${ver}"
  tar xzf "$t"
  cd "gperftools-${ver}"
  ./configure --prefix="$PREFIX"
  make -j"$JOBS"
  make install
  cd "$BUILD_DIR"
}

build_libevent
build_memcached_server
build_libmemcached
build_libaio
build_fmt
build_gperftools

echo
echo "==== Done ===="
echo "PREFIX=$PREFIX"
echo "memcached: $PREFIX/bin/memcached"
echo "libmemcached: $(ls "$PREFIX"/lib/libmemcached.so* 2>/dev/null | head -1 || echo 'check lib64')"
echo
echo "Add to your Slurm jobs (env_slurm.sh does this if COTRA_DEPS_PREFIX is set):"
echo "  export COTRA_DEPS_PREFIX=$PREFIX"
echo "  export PATH=\$COTRA_DEPS_PREFIX/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=\$COTRA_DEPS_PREFIX/lib:\$LD_LIBRARY_PATH"
echo
echo "When compiling CoTra, point CMake to libmemcached:"
echo "  export CMAKE_PREFIX_PATH=\$COTRA_DEPS_PREFIX"
