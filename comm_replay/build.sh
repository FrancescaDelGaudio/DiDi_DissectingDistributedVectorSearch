#!/usr/bin/bash
# Build transport_replay with libibverbs and the MPI of each system:
#   transport_replay          OpenMPI 4.1.1 + gcc 11   (GP-ANN, DistributedANN, Harmony)
#   transport_replay_ompi418  OpenMPI 4.1.8 + gcc 15   (Pyramid)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source /etc/profile.d/modules.sh
module purge
module load gcc/11.3.0
M=/softs/openmpi/gcc-10.3.0/4.1.1
OMPI_CXX=g++ "$M/bin/mpicxx" -O2 -std=c++17 -Wall -Wextra -o "$DIR/transport_replay" "$DIR/transport_replay.cpp" -libverbs
module purge
module load gcc/15.3.0 openmpi/gcc-15.3.0/4.1.8
mpicxx -O2 -std=c++17 -Wall -Wextra -o "$DIR/transport_replay_ompi418" "$DIR/transport_replay.cpp" -libverbs
ls -la "$DIR/transport_replay" "$DIR/transport_replay_ompi418"
