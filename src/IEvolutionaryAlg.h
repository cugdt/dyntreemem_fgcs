#pragma once
#include "CudaWorker.cuh"

#if !TEST
class IEvolutionaryAlg
{
public:
	void Run();
	void CrossoverOperator(Population population);
	void MutationOperator(Population population);
	void ApplyTestChangeChooser(Individual individual);		//choose CUDA or seq version
	void ApplyTestChangeByCUDA(Individual individual);		//call CUDA

	Population CreateInitialPopulation();
	void EvaluateFitness(Population population);			//update fitness
	void Selection(Population population);					//selection operator
	void PostEAOperations(Population population);			//prune etc.
	Pairs CreatePairs(Population population);				//create pairs of DTs for the crossover

	#if CUDA_EA_ON
	CCudaWorker worker;	
	#endif
};
#endif