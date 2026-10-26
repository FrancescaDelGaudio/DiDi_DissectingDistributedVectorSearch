#include <cstdio>
//
// Distributed Pyramid MIPS query engine (Algorithm 4 + step 17 via mode_code).
//

#include "distributed/partition.hpp"
#include "mpi_executor.hpp"
#include "parameters.hpp"
#include "executor.hpp"

int main(int argc, char** argv){
	// parallel_timer: OpenMP threads search/route, only the main thread calls MPI.
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "#[error ] MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
	mt::Partition partition(1);
	ss::parameter para;
	LoadOptions(argc, argv, para);
	mt::mpiBody(para, partition);
}
