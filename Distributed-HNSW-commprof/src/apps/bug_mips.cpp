//
// Distributed Pyramid MIPS query engine (Algorithm 4 + step 17 via mode_code).
//

#include "distributed/partition.hpp"
#include "mpi_executor.hpp"
#include "parameters.hpp"
#include "executor.hpp"

int main(int argc, char** argv){
	MPI_Init(&argc, &argv);
	mt::Partition partition(1);
	ss::parameter para;
	LoadOptions(argc, argv, para);
	mt::mpiBody(para, partition);
}
