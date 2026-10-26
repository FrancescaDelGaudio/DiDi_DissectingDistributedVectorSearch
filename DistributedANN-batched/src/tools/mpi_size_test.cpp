// Minimal MPI launcher diagnostic: each rank prints its rank/size and host.
// A correct multi-rank launch prints size == N on every line.
#include <mpi.h>
#include <cstdio>
#include <unistd.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    char host[256]; gethostname(host, sizeof(host));
    printf("rank=%d size=%d host=%s\n", rank, size, host);
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
