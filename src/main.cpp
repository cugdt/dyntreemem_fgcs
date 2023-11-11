#include "IEvolutionaryAlg.h"
#include "CudaWorkerRepTest.cuh"

int main() {

	#if !TEST
	IEvolutionaryAlg alg;
	alg.Run();
	#else
	CCudaWorkerRepTest cudaWorker;
	cudaWorker.Run();	
	#endif

	return 0;
}