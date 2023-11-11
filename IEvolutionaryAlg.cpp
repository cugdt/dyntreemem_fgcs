#include "IEvolutionaryAlg.h"

#if !TEST
#include <utility>

//run the basic evolution
void IEvolutionaryAlg::Run()
{
	Dataset dataset = ReadDataset(path);
	#if CUDA_EA_ON				
		worker.InitSimulation(dataset, 1, 0);		
	#endif
		
	Population population = CreateInitialPopulation();	
	EvaluateFitness(population);		//ApplyTestChangeChooser for all individuals
	Selection(population);

	//evolutionary loop
	do{		
		CrossoverOperator(population);
		MutationOperator(population);
		Selection(population);
		PostEAOperations(population);
	}while(!stopCondition);	
}

void IEvolutionaryAlg::CrossoverOperator(Population population)
{
	vector<pair<Individual, Individual>> pairsVec = CreatePairs(population);

	//for all pairs of DTs try to apply crossover operator
	for (auto iter : pairsVec) {
		if (Crossover(iter) {
			ApplyTestChangeChooser(iter.first)
			ApplyTestChangeChooser(iter.second)
		}
}

void IEvolutionaryAlg::MutationOperator(Population population) {
	//for all DTs try to apply mutation operator
	for (auto iter : population.GetIndiv()) {
		if (Mutation(iter))
			ApplyTestChangeChooser(iter)
}

void IEvolutionaryAlg::ApplyTestChangeChooser(Individual individual) {
	//run CUDA or C++ to update a DT
	#if CUDA_EA_ON
		ApplyTestChangeByCUDA(individual);
	#else	
		ApplyTestChangeByCPU(individual);
	#endif
}

//prepare and go to function where CUDA is used
void IEvolutionaryAlg::ApplyTestChangeByCUDA(Individual individual) {
	unsigned int* indivDetailedClassDistTab = NULL;
	unsigned int* indivDipolTab = NULL;
	unsigned int* indivDetailedErrTab = NULL;

	indivDetailedErrTab = worker.CalcIndivDetailedErrAndClassDistAndDipol_V2b(individual.GetRoot(), &indivDetailedClassDistTab, &indivDipolTab );
	worker.FillDTreeByExternalResultsChooser(individual.GetRoot(), indivDetailedErrTab, indivDetailedClassDistTab, indivDipolTab, 0, dataset);

	if (indivDetailedClassDistTab != NULL) delete[]indivDetailedClassDistTab;
	if (indivDipolTab != NULL) delete[]indivDipolTab;
	if (indivDetailedErrTab != NULL) delete[]indivDetailedErrTab;
}
#endif