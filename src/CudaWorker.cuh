#pragma once

#include "Worker.h"

#if CUDA_EA_ON
#include <omp.h>

#include <cusolverDn.h>						//cuSOLVER	- MLR
#include <cublas_v2.h>						//cuBLAS	- MLR
#define MIN(X, Y) ( (X) < (Y) ? (X):(Y) )

//#include "../../mlpdt/MLP_CUDA/CudaTimeStats.h"

#define CUDA_TREE_PART_MECH			0			// optimization - calc only in the change part of the tree
#define CUDA_MALLOC_OPTIM_1			1			// alloc once before the evolution
#define CUDA_SHARED_MEM_POST_CALC	0			// used shared memory
#define CUDA_SHARED_MEM_PRE_CALC	0			// used shared memory

#if FULL_BINARY_TREE_REP						// use complete in-memory representation 
 #define AT_LEAST_WISH_TREE_DEPTH_LIMIT	  10
 #define INIT_TREE_DEPTH_LIMIT			   5
 #define MAX_MEMORY_TREE_DEPTH			  18
 //#define MAX_N_TREE_NODES			1024
 #define MAX_N_INFO_TREE_NODES		5120

 #if USE_DOUBLE_FOR_RT_RESULTS == 1				//use double for the results
  #define RT_MAX_N_INFO_TREE_NODES	2048//2560			
 #else		
  #define RT_MAX_N_INFO_TREE_NODES	5120
 #endif
#else
 //#define MAX_SHARED_MEMORY_SIZE				49
 #define INIT_N_TREE_NODES_LIMIT				128		//starting tree nodes limit
#define MAX_MEMORY_TREE_DEPTH					  18	//max depth when complete representation
#if ADDAPTIVE_TREE_REP									
  #define MAX_N_INFO_TREE_NODES					5120
 #endif
#endif

class CCudaWorker : public CWorker{
private:
	int device;

	const IDataSet *dataset;					//dataset
	eModelType modelType;						//type of DTs
	DS_REAL *dev_datasetTab;					//dataset for the device
	DS_REAL *dev_datasetTabMGPU[CUDA_EA_ON];	//used for multi-gpu
	int nObjects;								//number of objects
	int nObjectsMGPU[CUDA_EA_ON];				//used for multi-gpu
	unsigned int *dev_classTab;					//DTs in classification
	unsigned int *dev_classTabMGPU[CUDA_EA_ON];	//DTs in classification
	DS_REAL *dev_predictionTab;					//DTs in classification
	DS_REAL *dev_predictionTabMGPU[CUDA_EA_ON];	//DTs in classification
	int error;

	//memory at device for population	
	int *dev_populationAttrNumTab;						//population od device
	int *dev_populationAttrNumTabMGPU[CUDA_EA_ON];		//population od device used for multi-gpu
	float *dev_populationValueTab;						//population od device
	float *dev_populationValueTabMGPU[CUDA_EA_ON];		//population od device used for multi-gpu
	RT_TREE_TRESHOLD_REAL *dev_RT_populationValueTab;	
	char* dev_MT_populationBUpdateTab;					
	MT_ATTRIBUTES_ONOFF_TYPE* dev_MT_populationAttrsONOFFTab;	//
	int *dev_individualPosInTab;								//DTs sizes
	int *dev_individualPosInTabMGPU[CUDA_EA_ON];				//DTs sizes used for multi-gpu
	int maxTreeTabSize;											
	int maxPopulationTabSize;									
	#if !FULL_BINARY_TREE_REP //tree structure - position of left/right children or parent node	
	int *dev_populationNodePosTab;								
	int *dev_populationNodePosTabMGPU[CUDA_EA_ON];
	int *dev_populationLeftNodePosTab;
	int *dev_populationLeftNodePosTabMGPU[CUDA_EA_ON];
	int *dev_populationRightNodePosTab;
	int *dev_populationRightNodePosTabMGPU[CUDA_EA_ON];
	int *dev_populationParentNodePosTab;
	int *dev_populationParentNodePosTabMGPU[CUDA_EA_ON];
	#endif

	//memory at device for results - CT and RT
	unsigned int *dev_populationClassDistTab_ScatOverBlocks;
	unsigned int *dev_populationClassDistTabMGPU_ScatOverBlocks[CUDA_EA_ON];
	unsigned int *dev_populationDetailedErrTab;
	unsigned int *dev_populationDetailedErrTabMGPU[CUDA_EA_ON];	
	unsigned int *dev_populationDetailedClassDistTab;
	unsigned int *dev_populationDetailedClassDistTabMGPU[CUDA_EA_ON];	
	unsigned int *dev_populationDipolTab_ScatOverBlocks;
	unsigned int *dev_populationDipolTabMGPU_ScatOverBlocks[CUDA_EA_ON];
	unsigned int *dev_populationDipolTab;
	unsigned int *dev_populationDipolTabMGPU[CUDA_EA_ON];	

	//memory at device for results - RT
	RT_REAL *dev_RT_populationErrTab_ScatOverBlocks;
	RT_REAL *dev_RT_populationDetailedErrTab;
	RT_REAL *dev_RT_populationModelTab_ScatOverBlocks;
	RT_REAL *dev_RT_populationModelTab;

	//memory at device for results - MT
	unsigned int *dev_MT_objectToLeafAssignTab;	
	unsigned int *dev_MT_objectToLeafAssignTab_out;
	unsigned int *dev_MT_objectToLeafAssignIndexTab;
	unsigned int *dev_MT_objectToLeafAssignIndexTab_out;
	void *dev_CUDACUB_Sort_temp_storage;
	size_t CUDACUB_Sort_temp_storage_bytes;
	unsigned int *dev_MT_populationNObjectsInLeafsTab;
	unsigned int *dev_MT_populationStartLeafDataMatrixTab;
	unsigned int *dev_MT_populationShiftLeafDataMatrixTab;
	RT_REAL* dev_MT_objectsMLRMatrixA;
	RT_REAL* dev_MT_objectsMLRMatrixb;

	CCudaTimeStats* timeStats;
	
	//cuBLAS, cuSOLVER
	cusolverDnHandle_t mt_cusolverDnH;
	cublasHandle_t mt_cublasH;
	float *mt_tau = 0, *mt_work;
	int *mt_devInfo = 0, mt_Lwork, mt_Lwork_max;
	const float mt_alpha = 1;
	int mt_min_m_n;							//MIN(m, n)

	#if FULL_BINARY_TREE_REP	
	int currTreeDepthLimit;
	#else
	int currNTreeNodesLimit;
	int currTreeDepthLimit_FULL_BINARY_TREE_REP;
	 #if ADDAPTIVE_TREE_REP	 
	 #endif
	#endif
	
public:
	CCudaWorker();
	~CCudaWorker();
	void CheckCUDAStatus(cudaError_t cudaStatus, char* errorInfo);					
	int GetDTArrayRepTabSize( CDTreeNode* root );
	int GetCurrMaxTreeTabSize();
	void InitSimulation(const DataSet *dataset, int nIndividuals, eModelType modelType);		//memory alloc, send dataset
	void EndSimulation();
	void ShowSimulationInfo();
	void SendDatasetToGPU(const IDataSet *dataset);
	void SendDatasetToGPUs(const IDataSet *dataset);
	void DeleteDatasetAtGPU();
	void DeleteDatasetAtGPUs();
	int GetBeginDatasetObjectIndexMGPU(int whichGPU);
	int GetEndDatasetObjectIndexMGPU(int whichGPU);
	#if CUDA_MALLOC_OPTIM_1	
	void AllocateMemoryPopAndResultsAtGPU(int nIndividuals, int maxTreeTabSize);
	void AllocateMemoryPopAndResultsAtGPUs(int nIndividuals, int maxTreeTabSize);
	void DeleteMemoryPopAndResultsAtGPU();	
	void DeleteMemoryPopAndResultsAtGPUs();
	#endif
	
	//use GPU to update the modified DT
	unsigned int* CalcIndivDetailedErrAndClassDistAndDipol_V2b( CDTreeNode* root, unsigned int** populationDetailedClassDistTab, unsigned int** populationDipolTab );
	//use GPU to update the modified DT - tree part version
	unsigned int* CalcIndivPartDetailedErrAndClassDistAndDipol_V2b( CDTreeNode* startNode, CDTreeNode* root, unsigned int** populationDetailedClassDistTab, unsigned int** populationDipolTab, int& startNodeTabIndex );

	//fill DT by the GPU calculated data
	int FillDTreeByExternalResults(CDTreeNode *node, unsigned int *indivDetailedErrTab, unsigned int *indivDetailedClassDistTab, unsigned int *indivDipolTab, unsigned int index, const IDataSet *pDS);
	int FillDTreeByExternalResults_FULL_BINARY_TREE_REP(CDTreeNode *node, unsigned int *indivDetailedErrTab, unsigned int *indivDetailedClassDistTab, unsigned int *indivDipolTab, unsigned int index, const IDataSet *pDS);
	int FillDTreeByExternalResultsChooser(CDTreeNode *root, unsigned int *indivDetailedErrTab, unsigned int *indivDetailedClassDistTab, unsigned int* indivDipolTab, unsigned int index, const IDataSet *pDS);
		
	void PruneNodeBeforeCUDA(CDTreeNode* node);
	#if FULL_BINARY_TREE_REP
	void SetInitTreeDepthLimit(const IDataSet *datasetCheckTreeDepthLimit, eModelType modelType);
	void PruneIndivBeforeCUDA( CDTreeNode* node, int treeLevel );
	#else
	bool DetermineCompactOrFullTreeRep(CDTreeNode* node);
	void PruneIndivBeforeCUDA_FULL_BINARY_TREE_REP(CDTreeNode* node, int treeLevel);
	#if ADDAPTIVE_TREE_REP
	#endif
    #endif	
	
	//time stats
	//void ClearAllStats(){ timeStats -> ClearAllStats(); }
	//void ClearCurrStats(){ timeStats -> ClearCurrStats(); }
	//void ShowTimeStatsLegend( eModelType modelType ){ timeStats -> ShowTimeStatsLegend( modelType ); }
	//void ShowTimeStats_V1a() { timeStats -> ShowTimeStats_V1a();  }
	//void ShowTimeStats_V1b() { timeStats -> ShowTimeStats_V1b();  }
	//void ShowTimeStats_V2a() { timeStats -> ShowTimeStats_V2a();  }
	//void ShowTimeStats_V2b() { timeStats -> ShowTimeStats_V2b();  }
	//void ShowTimeStats_DetailedV1a(){ timeStats -> ShowTimeStats_DetailedV1a(); }
	//void ShowTimeStats_DetailedV1b(){ timeStats -> ShowTimeStats_DetailedV1b(); }
	//void ShowTimeStats_DetailedV2a(){ timeStats -> ShowTimeStats_DetailedV2a(); }
	//void ShowTimeStats_DetailedV2b(){ timeStats -> ShowTimeStats_DetailedV2b(); }
	//void ShowTimeStats_DetailedIndivV2b( eModelType modelType ){ timeStats -> ShowTimeStats_DetailedIndivV2b( modelType ); }
	//void ShowTimeStats_Seq(){ timeStats -> ShowTimeStats_Seq(); }

	//void MoveTimeStats_DetailedIndivV2b(){ timeStats->MoveTimeStats_DetailedIndivV2b(); }

	//void SeqTimeBegin(){ timeStats -> SeqTimeBegin(); }
	//void SeqTimeEnd()  { timeStats -> SeqTimeEnd();   }

	//void ShowNodeStats_MLRCalc(){ timeStats -> ShowNodeStats_MLRCalc(); }

	int GetMaxTreeTabSize(){ return maxTreeTabSize; }
};
#endif