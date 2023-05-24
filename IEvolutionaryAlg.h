#pragma once

#include "CudaWorker.cuh"

class IEvolutionaryAlg
{
public:
	void Run();
	void CrossoverOperator(Population population);
	void MutationOperator(Population population);
	void ApplyTestChangeChooser(Individual individual);
	void ApplyTestChangeByCUDA(Individual individual);

	Population CreateInitialPopulation();
	void EvaluateFitness(Population population);
	void Selection(Population population);
	void PostEAOperations(Population population);
	Pairs CreatePairs(Population population);

	#if CUDA_EA_ON
	CCudaWorker worker;	
	#endif
};