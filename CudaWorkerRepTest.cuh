#pragma once

#include "DTreeNodeSim.h"
#include "CudaTimeStats.h"
#define N_THREADS 256
#define N_BLOCKS 256
#define DATASET_TRANING "sdd.data"

#define DS_REAL float						//double/float for dataset

#define N_DIPOL_OBJECTS				2		//number of objects of the same class for the dipol mechanism/algorithm (CT)
#define CUDA_MALLOC_OPTIM_1			1
#define FULL_BINARY_TREE_REP		0		//if memory is allocated for full binart tree - not finished yet
#if	!FULL_BINARY_TREE_REP
 #define ADDAPTIVE_TREE_REP			0		//full/complete or compact based on a metric 
 #if ADDAPTIVE_TREE_REP
  #define ADAPTIVE_TREE_REP_SWITCHING_POINT 0.7	//switching point between the in-memory representation complete(full)/compact
 #endif
#endif
#define N_INFO_CORR_ERR				2		//number of information about corrected and false classified objects in each tree node

#define CUDA_TREE_PART_MECH			0			//if send a part of tree or the whole tree
#define CUDA_MALLOC_OPTIM_1			1			//optimizalization - allocate once before simulation most of memory at GPU
#define CUDA_SHARED_MEM_POST_CALC	0			//using shared memory when reducting results, when 1 indiv is considered at time, then it is possible to use only globcal memory
#define CUDA_SHARED_MEM_PRE_CALC	0			//if use also shared memory in pre calc kernel function when in all leafs class dist is found (finally, no save in time)

#if FULL_BINARY_TREE_REP
 #define AT_LEAST_WISH_TREE_DEPTH_LIMIT	  10				//how many levels in trees we wish (at least) to be able to have, 0 means as much as memory it is accessible
 #define INIT_TREE_DEPTH_LIMIT			   5				//used when there is not limit to the tree size
 #define MAX_MEMORY_TREE_DEPTH			  18				//if more than 'out of memory' is ... (roboczo)
 //#define MAX_N_TREE_NODES			1024 				//at least at the beginning
 #define MAX_N_INFO_TREE_NODES		5120				//CT, since 49K at GPU; 5120 * 4 for each element * 2 tables = 41K, place for all (with all info) nodes, so actually less nodes can be stored			

 #if USE_DOUBLE_FOR_RT_RESULTS == 1
  #define RT_MAX_N_INFO_TREE_NODES	2048//2560				
 #else		
  #define RT_MAX_N_INFO_TREE_NODES	5120			//RT and MT, since 49K at GPU; 5120 * 4 for each element * 2 tables = 41K, place for all (with all info) nodes, so actually less nodes can be stored	
 #endif
#else
 //#define MAX_SHARED_MEMORY_SIZE				49			//[KB]
 #define INIT_N_TREE_NODES_LIMIT				128
#define MAX_MEMORY_TREE_DEPTH					  18		//CURRENTLY AS FOR FULL (duplicated)
#if ADDAPTIVE_TREE_REP  
  #define MAX_N_INFO_TREE_NODES					5120		//but here it can be done more efficiently
 #endif
#endif

class CCudaWorkerRepTest {
protected:
	int nObjects;
	int nAttrs;
	int nClasses;
	int nThreads;
	int nBlocks;
		
#if !FULL_BINARY_TREE_REP	
	bool bCompactOrFullTreeRep;
public:
	float adaptiveTreeRepSwitch;	
#endif

private:
	DS_REAL *dev_datasetTab;
	unsigned int *dev_classTab;
	int *dev_populationAttrNumTab;	
	float *dev_populationValueTab;	
	int *dev_individualPosInTab;

	#if !FULL_BINARY_TREE_REP //tree structure - position of left/right children or parent node
	//MEMORY OPTIMIZATION
	int *dev_populationNodePosTab;										//for the efficient data transfer (one transfer for all tables)	
	int *dev_populationLeftNodePosTab;
	int *dev_populationRightNodePosTab;	
	int *dev_populationParentNodePosTab;	
	#endif

	//memory at device for results
	unsigned int *dev_populationClassDistTab_ScatOverBlocks;	
	unsigned int *dev_populationDetailedErrTab;	
	unsigned int *dev_populationDetailedClassDistTab;	
	unsigned int *dev_populationDipolTab_ScatOverBlocks;	
	unsigned int *dev_populationDipolTab;	

	int maxPopulationTabSize;
	#if FULL_BINARY_TREE_REP	
	int currTreeDepthLimit;
	#else
	int currNTreeNodesLimit;
	int currTreeDepthLimit_FULL_BINARY_TREE_REP;
	 #if ADDAPTIVE_TREE_REP	 
	 #endif
	#endif

	CCudaTimeStats timeStats;

public:	
	int GetCurrMaxTreeTabSize();
	void GenerateDT(CDTreeNodeSim* node, int depth);
	void DeleteDT(CDTreeNodeSim* node);
	#if CUDA_MALLOC_OPTIM_1	
	void AllocateMemoryPopAndResultsAtGPU(int nIndividuals, int maxTreeTabSize);
	void DeleteMemoryPopAndResultsAtGPU();
	#endif
	void SendDatasetToGPU();
	void DeleteDatasetAtGPU();
	void InitSimulation();
	void EndSimulation();
	void PrepareIndivBeforeCUDA(CDTreeNodeSim* root);
	void PruneNodeBeforeCUDA(CDTreeNodeSim* node);
	#if FULL_BINARY_TREE_REP
	void SetInitTreeDepthLimit();
	void PruneIndivBeforeCUDA(CDTreeNodeSim* node, int treeLevel );
	#else
	bool DetermineCompactOrFullTreeRep(CDTreeNodeSim* node);
	void PruneIndivBeforeCUDA_FULL_BINARY_TREE_REP(CDTreeNodeSim* node, int treeLevel);
	#endif
	void Run();
	#if FULL_BINARY_TREE_REP
	void CopyBinaryTreeToTable(const CDTreeNodeSim* node, int* attrTab, float* valueTab, unsigned int index);
	#else
	int CopyBinaryTreeToTable(const CDTreeNodeSim* node, int* attrTab, float* valueTab, unsigned int index, int* leftChildPosTab, int* rightChildPosTab, int* parentPosTab);
	void CopyBinaryTreeToTable_FULL_BINARY_TREE_REP(const CDTreeNodeSim* node, int* attrTab, float* valueTab, unsigned int index);
	#endif
	int GetDTArrayRepTabSize( CDTreeNodeSim* root );
	unsigned int* CalcIndivDetailedErrAndClassDistAndDipol_V2b(CDTreeNodeSim* root, unsigned int** populationDetailedClassDistTab, unsigned int** populationDipolTab);
};