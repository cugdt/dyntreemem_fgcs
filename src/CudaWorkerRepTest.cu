#include "CudaWorkerRepTest.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "device_functions.h"
#include "cuda_runtime_api.h"

#include "RealTest.h"
//#include "assert.h"

#if FULL_BINARY_TREE_REP
//kernel pre complete
__global__ void dev_CalcPopClassDistAndDipolAtLeafs_Pre_V2b( DS_REAL *datasetTab, unsigned int *classTab, int nObjects, int nAttrs, int nIndividuals,
															 int *populationAttrNumTab, float *populationValueTab,  int *individualPosInTab, int populationTabSize, int nClasses,
															 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks );
#else
//kernel pre compact
__global__ void dev_CalcPopClassDistAndDipolAtLeafs_Pre_V2b( DS_REAL *datasetTab, unsigned int *classTab, int nObjects, int nAttrs, int nIndividuals,
															 int *populationAttrNumTab, float *populationValueTab,  int *individualPosInTab, int populationTabSize, int nClasses,
															 int *populationLeftNodePosTab, int *populationRightNodePosTab, int *populationParentNodePosTab,
															 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks );
#if ADDAPTIVE_TREE_REP
//kernel pre adaptive
__global__ void dev_CalcPopClassDistAndDipolAtLeafs_Pre_FULL_BINARY_TREE_REP_V2b( DS_REAL *datasetTab, unsigned int *classTab, int nObjects, int nAttrs, int nIndividuals,
															 int *populationAttrNumTab, float *populationValueTab,  int *individualPosInTab, int populationTabSize, int nClasses,
															 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks );
#endif
#endif

#if FULL_BINARY_TREE_REP
//kernel post for complete - 2 class version
__global__ void dev_CalcPopDetailedErrAndClassDistAndDipol_Post_V2b_2Classes( int *individualPosInTab, int populationTabSize, int nClasses,
															 		    unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks,
																	    unsigned int *populationDetailedErrTab, unsigned int *populationDetailedClassDistTab, unsigned int *populationDipolTab );
//kernel post for complete
__global__ void dev_CalcPopDetailedErrAndClassDistAndDipol_Post_V2b( int *individualPosInTab, int populationTabSize, int nClasses,
															 		 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks,
																	 unsigned int *populationDetailedErrTab, unsigned int *populationDetailedClassDistTab, unsigned int *populationDipolTab );
#else
//kernel post for compact
__global__ void dev_CalcPopDetailedErrAndClassDistAndDipol_Post_V2b( int *individualPosInTab, int populationTabSize, int nClasses, int *populationParentNodePosTab,
															 		 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks,
																	 unsigned int *populationDetailedErrTab, unsigned int *populationDetailedClassDistTab, unsigned int *populationDipolTab );
#if ADDAPTIVE_TREE_REP
//kernel post for adaptive
__global__ void dev_CalcPopDetailedErrAndClassDistAndDipol_Post_FULL_BINARY_TREE_REP_V2b( int *individualPosInTab, int populationTabSize, int nClasses,
															 		 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks,
																	 unsigned int *populationDetailedErrTab, unsigned int *populationDetailedClassDistTab, unsigned int *populationDipolTab );
#endif
#endif

inline void CheckCUDAStatus(cudaError_t cudaStatus, char* errorInfo){
	if( cudaStatus != cudaSuccess ){
		printf( "%s", errorInfo );
		printf( "%s\n", cudaGetErrorString(cudaStatus) );
		exit(1);
	}
}

int CCudaWorkerRepTest::GetCurrMaxTreeTabSize(){
	#if FULL_BINARY_TREE_REP
		return pow(2.0, currTreeDepthLimit) - 1;
	#else
		return currNTreeNodesLimit;
	#endif
}

//generate the random tree
void CCudaWorkerRepTest::GenerateDT(CDTreeNodeSim* node, int depth) {
	if (depth <= 0) {
		node->m_bLeaf = true;
		if(node->m_pITest) delete node->m_pITest;
		node->m_pITest = NULL;
		return;
	}

	if (!(rand() % depth/2)) {
		node->m_bLeaf = true;
		if (node->m_pITest) delete node->m_pITest;
		node->m_pITest = NULL;
		return;
	}

	node -> m_vBranch.push_back(new CDTreeNodeSim((rand() / (float)RAND_MAX) * 10, rand() % nAttrs));
	node -> m_vBranch.push_back(new CDTreeNodeSim((rand() / (float)RAND_MAX) * 10, rand() % nAttrs));

	size_t nOutcomes = node -> m_vBranch.size();
	for (size_t i = 0; i < nOutcomes; i++) 
		GenerateDT(node->m_vBranch[i], depth - 1);
}

//delete the random generated DT
void CCudaWorkerRepTest::DeleteDT(CDTreeNodeSim* node) {
	size_t nOutcomes = node -> m_vBranch.size();
	for (size_t i = 0; i < nOutcomes; i++)
		DeleteDT(node->m_vBranch[i]);
		
	delete node;
}

#if CUDA_MALLOC_OPTIM_1
//alloc one before the evolution - for DT structure and for results
void CCudaWorkerRepTest::AllocateMemoryPopAndResultsAtGPU(int nIndividuals, int maxTreeTabSize){

	//timeStats->WholeTimeBegin();						//time stats
	//timeStats->MemoryAllocDeallocTimeBegin();			//time stats	
	//allocate memory at device for population
	cudaError_t cudaStatus;

	maxPopulationTabSize = maxTreeTabSize * nIndividuals;
	
	cudaStatus = cudaMalloc((void**)&dev_populationAttrNumTab, maxPopulationTabSize * sizeof(int));
	CheckCUDAStatus( cudaStatus, "InitSimulation - cudaMalloc failed - 1 !!!\n" );

	cudaStatus = cudaMalloc((void**)&dev_individualPosInTab, nIndividuals * sizeof(int));
	CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - 2 !!!\n");

	#if !FULL_BINARY_TREE_REP
	cudaStatus = cudaMalloc((void**)&dev_populationNodePosTab, 3 * maxPopulationTabSize * sizeof(int));
	CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - 4, 5, 6 !!!\n");

	dev_populationLeftNodePosTab = dev_populationNodePosTab;
	
	dev_populationRightNodePosTab = dev_populationNodePosTab + maxPopulationTabSize;
	
	dev_populationParentNodePosTab = dev_populationNodePosTab + 2*maxPopulationTabSize;
	#endif	

	//allocate memory at device for results

	//CT
	cudaStatus = cudaMalloc((void**)&dev_populationValueTab, maxPopulationTabSize * sizeof(float));
	CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - CT - 7 !!!\n");

	cudaStatus = cudaMalloc((void**)&dev_populationClassDistTab_ScatOverBlocks, nBlocks * maxPopulationTabSize * nClasses * sizeof(unsigned int));
	CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - CT - 8 !!!\n");

	cudaStatus = cudaMalloc((void**)&dev_populationDetailedClassDistTab, maxPopulationTabSize * nClasses * sizeof(unsigned int));
	CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - CT - 9 !!!\n");

	cudaStatus = cudaMalloc((void**)&dev_populationDetailedErrTab, maxPopulationTabSize * N_INFO_CORR_ERR * sizeof(unsigned int));
	CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - CT - 10 !!!\n");

	cudaStatus = cudaMalloc((void**)&dev_populationDipolTab_ScatOverBlocks, nBlocks * maxPopulationTabSize * nClasses * (N_DIPOL_OBJECTS + 1) * sizeof(unsigned int));
	CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - CT - 11 !!!\n");
		
	cudaStatus = cudaMalloc((void**)&dev_populationDipolTab, maxPopulationTabSize * nClasses * N_DIPOL_OBJECTS * sizeof(unsigned int));
	CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - CT - 12 !!!\n");		
	
	//timeStats->MemoryAllocDeallocTimeEnd();			//time stats
	//timeStats->WholeTimeEnd();						//time stats
}

void CCudaWorkerRepTest::DeleteMemoryPopAndResultsAtGPU(){
	//timeStats->WholeTimeBegin();				//time stats
	//timeStats->MemoryAllocDeallocTimeBegin();	//time stats

	#if !FULL_BINARY_TREE_REP //tree structure
	if (dev_populationNodePosTab		 != NULL) cudaFree(dev_populationNodePosTab);
	dev_populationNodePosTab = NULL;
	dev_populationLeftNodePosTab	= NULL;
	dev_populationRightNodePosTab	= NULL;
	dev_populationParentNodePosTab	= NULL;
	#endif

	if (dev_populationAttrNumTab != NULL) cudaFree(dev_populationAttrNumTab);	
	if (dev_individualPosInTab   != NULL) cudaFree(dev_individualPosInTab);
	dev_populationAttrNumTab	= NULL;
	dev_individualPosInTab		= NULL;	

	//CT
	if (dev_populationValueTab != NULL) cudaFree(dev_populationValueTab);
	if (dev_populationClassDistTab_ScatOverBlocks != NULL) cudaFree(dev_populationClassDistTab_ScatOverBlocks);	
	if (dev_populationDetailedClassDistTab != NULL) cudaFree(dev_populationDetailedClassDistTab);	
	if (dev_populationDetailedErrTab != NULL) cudaFree(dev_populationDetailedErrTab);	
	if (dev_populationDipolTab_ScatOverBlocks != NULL) cudaFree(dev_populationDipolTab_ScatOverBlocks);
	if (dev_populationDipolTab != NULL) cudaFree(dev_populationDipolTab);	
	dev_populationValueTab = NULL;
	dev_populationClassDistTab_ScatOverBlocks = NULL;
	dev_populationDetailedClassDistTab = NULL;
	dev_populationDetailedErrTab = NULL;
	dev_populationDipolTab_ScatOverBlocks = NULL;
	dev_populationDipolTab = NULL;

	//timeStats->MemoryAllocDeallocTimeEnd();	//time stats
	//timeStats->WholeTimeEnd();				//time stats
}
#endif

//sending dataset to device
void CCudaWorkerRepTest::SendDatasetToGPU(){
	//timeStats -> WholeTimeBegin();						//time stats
	//timeStats -> WholeDatasetTransportTimeBegin();

	FILE* file = fopen(DATASET_TRANING, "rt");
	if( !file ){
		printf( "Error during reading the dataset\n" );
		exit( EXIT_FAILURE );
	}

	fscanf( file, "%d %d %d", &nObjects, &nAttrs, &nClasses );
		
	if( nObjects <= 0 || nAttrs <= 0 || nClasses <= 0 ){
		printf( "SendDatasetToGPU - 1  !!!\n" );		
		printf( "nObjects == 0 || nAttrs == 0 || nClasses == 0\n" );		
		exit( EXIT_FAILURE );
	}

	DS_REAL *datasetTab = NULL;	
	datasetTab = new DS_REAL[ nObjects * nAttrs ];
	unsigned int *classTab = NULL;
	classTab = new unsigned int[ nObjects ];
	char dot;

	for( int i = 0; i < nObjects; i++){
		for( int j = 0; j < nAttrs; j++)
			fscanf( file, "%f,", &datasetTab[ i * nAttrs + j ] );
		fscanf( file, "%u", &classTab[ i ] );		
	}

	fclose(file);

	cudaError_t cudaStatus;
	
	// allocate memory at device for dataset
	//timeStats -> MemoryAllocForDatasetTimeBegin();
	cudaStatus = cudaMalloc( (void**)&dev_datasetTab, nObjects * nAttrs * sizeof( DS_REAL ) );
	if( cudaStatus != cudaSuccess ){
		printf( "SendDatasetToGPU - cudaMalloc failed - 2 !!!" );		
		exit( 1 );
	}

	// allocate memory at device for class id (CT)
	cudaStatus = cudaMalloc( (void**)&dev_classTab, nObjects * sizeof( unsigned int ) );	//CT
	if( cudaStatus != cudaSuccess ){
		printf( "SendDatasetToGPU - cudaMalloc failed - 3 !!!" );		
		exit( 1 );
	}
	//timeStats -> MemoryAllocForDatasetTimeEnd();


	// send dataset from host to device
	//timeStats -> SendDatasetToGPUTimeBegin();
	cudaStatus = cudaMemcpy( dev_datasetTab, datasetTab, nObjects * nAttrs * sizeof( DS_REAL ), cudaMemcpyHostToDevice );
	if( cudaStatus != cudaSuccess ){
        printf( "SendDatasetToGPU - cudaMemcpy failed - 4 !!!" );		
		exit( 1 );
    }

	// send class id (CT) from host to device
	cudaStatus = cudaMemcpy( dev_classTab, classTab, nObjects * sizeof( unsigned int ), cudaMemcpyHostToDevice );	//CT
	if( cudaStatus != cudaSuccess ) {
        printf( "SendDatasetToGPU - cudaMemcpy failed - 5 !!!" );		
		exit( 1 );
    }
	//timeStats -> SendDatasetToGPUTimeEnd();

	if( datasetTab != NULL )delete[]datasetTab;
	if( classTab != NULL ) delete []classTab;
	
	//timeStats -> WholeDatasetTransportTimeEnd();		//time stats
	//timeStats -> WholeTimeEnd();						//time stats	
}

//deallocate dataset memory at GPU
void CCudaWorkerRepTest::DeleteDatasetAtGPU(){
	if (dev_datasetTab != NULL) cudaFree(dev_datasetTab);
	if (dev_classTab != NULL) cudaFree(dev_classTab);
}

//default settings
void CCudaWorkerRepTest::InitSimulation() {
	char datasetPath[256] = "sdd.data";

	SendDatasetToGPU();

	#if FULL_BINARY_TREE_REP
	SetInitTreeDepthLimit();
	#else
	currNTreeNodesLimit = INIT_N_TREE_NODES_LIMIT;
	#endif

	#if CUDA_MALLOC_OPTIM_1
	AllocateMemoryPopAndResultsAtGPU( 1, GetCurrMaxTreeTabSize() );
	#endif
}

void CCudaWorkerRepTest::EndSimulation() {
	DeleteDatasetAtGPU();	

	#if CUDA_MALLOC_OPTIM_1
		DeleteMemoryPopAndResultsAtGPU();
	#endif
}

//if needed prune tree
void CCudaWorkerRepTest::PrepareIndivBeforeCUDA(CDTreeNodeSim* root ){
	//printf("PrepareIndivBeforeCUDA - begin\n");
	#if FULL_BINARY_TREE_REP		
		if( root -> GetDepth() > currTreeDepthLimit ){
			#if CUDA_SHARED_MEM_POST_CALC
				//prune the tree to the max size
				PruneIndivBeforeCUDA( root, 0 );
			#else		
				int mod_limit = (MAX_MEMORY_TREE_DEPTH - (nClasses / 4));
				int previous_limit = currTreeDepthLimit;
				if( root -> GetDepth() > mod_limit ){
					currTreeDepthLimit = mod_limit;
					PruneIndivBeforeCUDA( root, 0 );
					if (previous_limit >= currTreeDepthLimit) return;
					//return;
				}

				currTreeDepthLimit = root -> GetDepth();
				#if CUDA_MALLOC_OPTIM_1
					//printf( "The tree depth size of the tables for trees and results at GPU(s) is increased to %d - begin\n", currTreeDepthLimit );										
					DeleteMemoryPopAndResultsAtGPU();
					AllocateMemoryPopAndResultsAtGPU( 1, GetCurrMaxTreeTabSize() );					
					//printf( "The tree depth size of the tables for trees and results at GPU(s) is increased to %d - end\n", currTreeDepthLimit );
				#endif				
			#endif
		}
	#else
		if( root -> GetDepth() > MAX_MEMORY_TREE_DEPTH ){
			currTreeDepthLimit_FULL_BINARY_TREE_REP = MAX_MEMORY_TREE_DEPTH;
			PruneIndivBeforeCUDA_FULL_BINARY_TREE_REP( root, 0 );
			//return;
		}

		bCompactOrFullTreeRep = DetermineCompactOrFullTreeRep(root);
		#if ADDAPTIVE_TREE_REP		
		if (!bCompactOrFullTreeRep){		//full (complete)
			if ((pow(2.0, root->GetDepth()) - 1) > currNTreeNodesLimit) {
				currNTreeNodesLimit = pow(2.0, root->GetDepth()) - 1;

				#if CUDA_MALLOC_OPTIM_1
				DeleteMemoryPopAndResultsAtGPU();
				AllocateMemoryPopAndResultsAtGPU( 1, currNTreeNodesLimit );
				//printf("ADDAPTIVE: The number of tree nodes of the arrays at GPU(s) is increased to %d - end\n", currNTreeNodesLimit);
				#endif
			}
			return;
		}
        #endif
		if( root ->GetNodeCount() > currNTreeNodesLimit ){
			//currNTreeNodesLimit *= 2;
			currNTreeNodesLimit = root->GetNodeCount();

			#if CUDA_MALLOC_OPTIM_1
			DeleteMemoryPopAndResultsAtGPU();
			AllocateMemoryPopAndResultsAtGPU( 1, GetCurrMaxTreeTabSize() );
			//printf( "The number of tree nodes in the arrays at GPU(s) is increased to %d - end\n", currNTreeNodesLimit );
			#endif
		}
	#endif
}

void CCudaWorkerRepTest::PruneNodeBeforeCUDA(CDTreeNodeSim* node) {
	if (node->m_bLeaf) return;
	size_t nBranches = node->m_vBranch.size();
	for (size_t i = 0; i < nBranches; i++) {
		//if( i == 0 ) PruneNodeBeforeCUDA( node -> m_vBranch[ 0 ] );
		//if( i == 1 ) PruneNodeBeforeCUDA( node -> m_vBranch[ 1 ] );
		PruneNodeBeforeCUDA(node->m_vBranch[i]);
		delete node->m_vBranch[i];
	}
	node->m_vBranch.clear();
	delete node->m_pITest;
	node->m_pITest = NULL;
	node->m_bLeaf = true;
}

//check how big tree can be stored
//and check if it is enough
#if FULL_BINARY_TREE_REP
void CCudaWorkerRepTest::SetInitTreeDepthLimit(){
	
	#if CUDA_SHARED_MEM_POST_CALC
	this -> nClasses = dataset -> GetClassCount();
	//nClasses = nClasses * 5;				//TO VERIFY , DEPTH, 2 CLASSES, TO DO
	int depthLimit = 0;
	
	int maxFactor = 1;
	int nTreeNodes = 0;
	if (dynamic_cast<CDataSetRT*> (const_cast<IDataSet*>(dataset)) == NULL)	//CT
		nTreeNodes = MAX_N_INFO_TREE_NODES / nClasses / N_DIPOL_OBJECTS;				
	else{	
		if (modelType == NONE_MODEL){										//RT

			if (RT_N_DIPOLS * 2 >= RT_N_MODEL_VALUES)
				maxFactor = RT_N_DIPOLS * 2;
			else
				maxFactor = RT_N_MODEL_VALUES;
			nTreeNodes = RT_MAX_N_INFO_TREE_NODES / maxFactor;
		}
		else{																//MT
			maxFactor = RT_N_DIPOLS * 2;
		}
		nTreeNodes = RT_MAX_N_INFO_TREE_NODES / maxFactor;
	}

	//first, find max tree depth
	while( nTreeNodes / 2.0 >= 1.0 ){
		depthLimit++;
		nTreeNodes /= 2;
	}
	
	//second, verification
	if( depthLimit < AT_LEAST_WISH_TREE_DEPTH_LIMIT ){
		printf( "SetInitTreeDepthLimit - possible depthLimit(%d) is lower then AT_LEAST_WISH_TREE_DEPTH_LIMIT(%d)\n", depthLimit, AT_LEAST_WISH_TREE_DEPTH_LIMIT );
		exit( 1 );
	}
	printf( "CUDA: Possible tree depth limit: %d\n", depthLimit );

	currTreeDepthLimit = depthLimit;
	#else
	currTreeDepthLimit = INIT_TREE_DEPTH_LIMIT;
	#endif	
}

void CCudaWorkerRepTest::PruneIndivBeforeCUDA(CDTreeNodeSim* node, int treeLevel ){
	treeLevel++;
	if( treeLevel == currTreeDepthLimit ){
		if( !node -> m_bLeaf ){
			PruneNodeBeforeCUDA( node );
			//printf( "PruneIndivBeforeCUDA\n" );
		}
	}
	else{
		size_t nBranches = node -> m_vBranch.size();
		for( size_t i = 0; i < nBranches; i++ ){
			//if( i == 0 ) PruneIndivBeforeCUDA( node -> m_vBranch[ 0 ], treeLevel );		//old, code for two classes only
			//if( i == 1 ) PruneIndivBeforeCUDA( node -> m_vBranch[ 1 ], treeLevel );		//old, code for two classes only
			PruneIndivBeforeCUDA( node -> m_vBranch[i], treeLevel );
		}
	}			
}
#else
//which population to choose
bool CCudaWorkerRepTest::DetermineCompactOrFullTreeRep(CDTreeNodeSim* root) {
	#if !ADDAPTIVE_TREE_REP
	return 1;													//compact
	#else
	
	if (root->GetDepth() <= 2) return 0;						//full (complete)

	//if (root->GetDepth() > MAX_MEMORY_TREE_DEPTH) return 1;		//compact

	if (root->GetNodeCount() / ((double)pow(2.0, root->GetDepth())-1) > adaptiveTreeRepSwitch)
		return 0;												//full (complete) 
	else
		return 1;												//compact
	#endif
}
void CCudaWorkerRepTest::PruneIndivBeforeCUDA_FULL_BINARY_TREE_REP(CDTreeNodeSim* node, int treeLevel) {
	treeLevel++;
	if (treeLevel == currTreeDepthLimit_FULL_BINARY_TREE_REP) {
		if (!node->m_bLeaf) {
			PruneNodeBeforeCUDA(node);
			//printf( "PruneIndivBeforeCUDA\n" );
		}
	}
	else {
		size_t nBranches = node->m_vBranch.size();
		for (size_t i = 0; i < nBranches; i++) {
			//if( i == 0 ) PruneIndivBeforeCUDA( node -> m_vBranch[ 0 ], treeLevel );		//old, code for two classes only
			//if( i == 1 ) PruneIndivBeforeCUDA( node -> m_vBranch[ 1 ], treeLevel );		//old, code for two classes only
			PruneIndivBeforeCUDA_FULL_BINARY_TREE_REP(node->m_vBranch[i], treeLevel);
		}
	}
}
#endif

void CCudaWorkerRepTest::Run() {
	//set params
	#if !FULL_BINARY_TREE_REP
		#if ADDAPTIVE_TREE_REP
			adaptiveTreeRepSwitch = ADAPTIVE_TREE_REP_SWITCHING_POINT;		//switching point
		#endif
	#endif
	nThreads = N_THREADS;
	nBlocks = N_BLOCKS;

	InitSimulation();					//init dataset and model type	

	printf("blocks: %d, number of threads: %d\n", nBlocks, nThreads);
	printf("dataset:%s, samples: %d, attrs: %d, classes: %d\n", DATASET_TRANING, nObjects, nAttrs, nClasses);
	#if FULL_BINARY_TREE_REP
		printf("Complete in-memory representation\n");
	#else
		#if ADDAPTIVE_TREE_REP
		printf("Adaptive in-memory representation %.2f\n", adaptiveTreeRepSwitch);
		#else	
		printf("Compact in-memory representation\n");
		#endif	
	#endif
	for (int i = 0; i < 20; i++ ) {
		int maxTreeDepth = rand() % 15+2;

		CDTreeNodeSim* root = new CDTreeNodeSim((rand()/(float)RAND_MAX)*10, rand()%nAttrs);
		GenerateDT(root, maxTreeDepth);
		PrepareIndivBeforeCUDA(root);
		//printf("%d %d %d\n", root->GetDepth(), root->GetLeavesCount(), root->GetNodeCount());
	
		unsigned int* indivDetailedClassDistTab = NULL;
		unsigned int* indivDipolTab = NULL;
		unsigned int* indivDetailedErrTab = NULL;
		timeStats.WholeTimeBegin();					//time stats
		printf("Tree depth:%2d .... ", maxTreeDepth);
		indivDetailedErrTab = CalcIndivDetailedErrAndClassDistAndDipol_V2b(root, &indivDetailedClassDistTab, &indivDipolTab);
		timeStats.WholeTimeEnd();						//time stats
		printf( "time:%f\n", timeStats.GetWholeTimeLastDiff() );
		if (indivDetailedClassDistTab != NULL) delete[]indivDetailedClassDistTab;
		if (indivDipolTab != NULL) delete[]indivDipolTab;
		if (indivDetailedErrTab != NULL) delete[]indivDetailedErrTab;
		//DeleteDT(root);
		delete root;
	}	
	EndSimulation();
}

//copy binary tree to the table (one-dimensional table)
//-1 means leaf
//>= 0 means the attribute number
#if FULL_BINARY_TREE_REP
void CCudaWorkerRepTest::CopyBinaryTreeToTable( const CDTreeNodeSim* node, int* attrTab, float* valueTab, unsigned int index ){
	if( node -> m_bLeaf ){
		attrTab[ index ] = -1;
		return;
	}

	attrTab[ index ] = ((CRealTest*)(node->m_pITest)) -> GetAttrNum();		//mozna sprawdzac typ testu, GetType, zakladam ze jest CRealTest
	valueTab[ index ] = ((CRealTest*)(node->m_pITest)) -> m_dThreshold;		//moze dynamic_cast	

	if( node->m_vBranch.size() > 0 ) CopyBinaryTreeToTable( node -> m_vBranch[0], attrTab, valueTab, 2 * index + 1 );
	if( node->m_vBranch.size() > 1 ) CopyBinaryTreeToTable( node -> m_vBranch[1], attrTab, valueTab, 2 * index + 2 );
}
#else
int CCudaWorkerRepTest::CopyBinaryTreeToTable( const CDTreeNodeSim* node, int* attrTab, float* valueTab, unsigned int index, int* leftChildPosTab, int* rightChildPosTab, int* parentPosTab ){
	if( node -> m_bLeaf ){
		attrTab[ index ] = -1;
		leftChildPosTab[ index ] = -1;
		rightChildPosTab[ index ] = -1;
		return index;
	}

	attrTab[ index ] = ((CRealTest*)(node->m_pITest)) -> GetAttrNum();		//mozna sprawdzac typ testu, GetType, zakladam ze jest CRealTest
	valueTab[ index ] = ((CRealTest*)(node->m_pITest)) -> m_dThreshold;		//moze dynamic_cast	
	
	int oldIndex;
	if( node->m_vBranch.size() > 0 ){
		leftChildPosTab[ index ] = index + 1;
		parentPosTab[ index + 1 ] = index;
		oldIndex = index;
		index = CopyBinaryTreeToTable( node -> m_vBranch[0], attrTab, valueTab, index + 1, leftChildPosTab, rightChildPosTab, parentPosTab );
	}
	if( node->m_vBranch.size() > 1 ){
		rightChildPosTab[ oldIndex ] = index + 1;
		parentPosTab[ index + 1 ] = oldIndex;
		index = CopyBinaryTreeToTable( node -> m_vBranch[1], attrTab, valueTab, index + 1, leftChildPosTab, rightChildPosTab, parentPosTab );
	}

	return index;
}
void CCudaWorkerRepTest::CopyBinaryTreeToTable_FULL_BINARY_TREE_REP( const CDTreeNodeSim* node, int* attrTab, float* valueTab, unsigned int index ){
	if( node -> m_bLeaf ){
		attrTab[ index ] = -1;
		return;
	}

	attrTab[ index ] = ((CRealTest*)(node->m_pITest)) -> GetAttrNum();		//mozna sprawdzac typ testu, GetType, zakladam ze jest CRealTest
	valueTab[ index ] = ((CRealTest*)(node->m_pITest)) -> m_dThreshold;		//moze dynamic_cast	

	if( node->m_vBranch.size() > 0 ) CopyBinaryTreeToTable_FULL_BINARY_TREE_REP( node -> m_vBranch[0], attrTab, valueTab, 2 * index + 1 );
	if( node->m_vBranch.size() > 1 ) CopyBinaryTreeToTable_FULL_BINARY_TREE_REP( node -> m_vBranch[1], attrTab, valueTab, 2 * index + 2 );
}
#endif


int CCudaWorkerRepTest::GetDTArrayRepTabSize( CDTreeNodeSim* root ){
	int size;
	#if FULL_BINARY_TREE_REP
		size = pow(2.0, (int)(root->GetDepth())) - 1;
		//size = pow( 2.0, (int)(population[ i ] -> GetRoot() -> GetDepth()) ) - 1;	
		//printf( "%d ", size );
	#else
		#if ADDAPTIVE_TREE_REP
		if(!bCompactOrFullTreeRep)
			size = pow(2.0, (int)(root->GetDepth())) - 1;
		else
		#endif
		size = root -> GetNodeCount();
	#endif
	
	return size;
}

//classification trees
unsigned int* CCudaWorkerRepTest::CalcIndivDetailedErrAndClassDistAndDipol_V2b(CDTreeNodeSim* root, unsigned int** populationDetailedClassDistTab, unsigned int** populationDipolTab) {
	cudaError_t cudaStatus;

	//timeStats -> WholeTimeBegin();						//time stats

	//timeStats -> DataReorganizationTimeBegin();			//time stats
	//////////////////////////////////////////////////////////////////////////
	//create tables with necessary population data to send to device (GPU)
	//int nIndividuals = population.size();	
	int nIndividuals = 1;
	int *individualPosInTab = new int[ nIndividuals ];
	int populationTabSize = 0;	
	
	//individualPosInTab[ 0 ] = 0;
	//individualSizeSum = (population[ 0 ]) -> GetRoot() -> GetDepth();
	for( int i = 0; i < nIndividuals; i++ ){
		individualPosInTab[ i ] = populationTabSize;
		populationTabSize += GetDTArrayRepTabSize( root );		
	}
	//printf( "\n" );
	int *populationAttrNumTab = new int[ populationTabSize ];
	float *populationValueTab = new float[ populationTabSize ];	
	for( int i = 0; i < populationTabSize; i++ ){
		populationAttrNumTab[ i ] = -2;
		populationValueTab	[ i ] = 0.0;		
	}
	#if !FULL_BINARY_TREE_REP
	//MEMORY OPTIMIZATION
	int *populationNodePosTab = NULL;
	int *populationLeftNodePosTab = NULL;
	int *populationRightNodePosTab = NULL;
	int *populationParentNodePosTab = NULL;
	if (bCompactOrFullTreeRep) {
		populationNodePosTab = new int[3*populationTabSize];
		populationLeftNodePosTab = populationNodePosTab;
		populationRightNodePosTab = populationNodePosTab + populationTabSize;
		populationParentNodePosTab = populationNodePosTab + 2*populationTabSize;
		//populationLeftNodePosTab = new int[populationTabSize];
		//populationRightNodePosTab = new int[populationTabSize];
		//populationParentNodePosTab = new int[populationTabSize];

		for (int i = 0; i < populationTabSize; i++) {
			populationLeftNodePosTab[i] = -1;
			populationRightNodePosTab[i] = -1;
			populationParentNodePosTab[i] = -1;
		}		
	}
	#endif
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	//copy population data to tables
	for( int i = 0; i < nIndividuals; i++ ){
		int shift = individualPosInTab[i];
		#if FULL_BINARY_TREE_REP
		CopyBinaryTreeToTable( root, populationAttrNumTab + shift, populationValueTab + shift, 0 );
		#else
		if(bCompactOrFullTreeRep){
			CopyBinaryTreeToTable( root, populationAttrNumTab + shift, populationValueTab + shift, 0,
				                   populationLeftNodePosTab + 3*shift, populationRightNodePosTab + 3*shift, populationParentNodePosTab + 3*shift );
			//CheckDTLinkedArrayRep(root, 0, populationLeftNodePosTab + 3*shift, populationRightNodePosTab + 3*shift, populationParentNodePosTab + 3*shift);
		}
		else
			CopyBinaryTreeToTable_FULL_BINARY_TREE_REP(root, populationAttrNumTab + shift, populationValueTab + shift, 0);		

		#if DEBUG_ADAPTIVE_TREE_REP
		if (bCompactOrFullTreeRep)	{ nCompactRepDTsRun++;	nCompactRepDTsSum++;}
		else						{ nFullRepDTsRun++;		nFullRepDTsSum++;}
		//switching frequency
		if (bLastCompactOrFullTreeRep != bCompactOrFullTreeRep) { nSwitchingCompactOrFullTreeRepRun++; nSwitchingCompactOrFullTreeRepSum++; }
		bLastCompactOrFullTreeRep = bCompactOrFullTreeRep;
		#endif
		#endif
	}
	//timeStats -> DataReorganizationTimeEnd();			//time stats
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	#if !CUDA_MALLOC_OPTIM_1
	//allocate memory at device for population		
	//timeStats -> MemoryAllocDeallocTimeBegin();			//time stats

	cudaStatus = cudaMalloc( (void**)&dev_populationAttrNumTab, populationTabSize * sizeof( int ) );
	if( cudaStatus != cudaSuccess ){
		printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMalloc failed - 1 !!!\n" );
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
	}
	cudaStatus = cudaMalloc( (void**)&dev_populationValueTab, populationTabSize * sizeof( float ) );
	if( cudaStatus != cudaSuccess ){
		printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMalloc failed - 2 !!!\n" );
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
	}
	cudaStatus = cudaMalloc( (void**)&dev_individualPosInTab, nIndividuals * sizeof( int ) );
	if( cudaStatus != cudaSuccess ){
		printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMalloc failed - 3 !!!\n" );
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
	}

	#if !FULL_BINARY_TREE_REP
	if (bCompactOrFullTreeRep) {
		cudaStatus = cudaMalloc((void**)&dev_populationNodePosTab, 3 * populationTabSize * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			printf("CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMalloc failed - 4, 5, 6 !!!\n");
			printf("%s\n", cudaGetErrorString(cudaStatus));
			exit(EXIT_FAILURE);
		}		
	}
	#endif

	//allocate memory at device for results	
	//cudaStatus = cudaMalloc( (void**)&dev_populationClassDistTab, nBlocks * nIndividuals * MAX_N_INFO_TREE_NODES * nClasses * sizeof( unsigned int ) );
	cudaStatus = cudaMalloc( (void**)&dev_populationClassDistTab_ScatOverBlocks, nBlocks * populationTabSize * nClasses * sizeof( unsigned int ) );
	if( cudaStatus != cudaSuccess ){
		printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMalloc failed - 7 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
	}	

	cudaStatus = cudaMalloc( (void**)&dev_populationDetailedErrTab,  populationTabSize * N_INFO_CORR_ERR * sizeof( unsigned int ) );
	if( cudaStatus != cudaSuccess ){
		printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMalloc failed - 8 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
	}

	cudaStatus = cudaMalloc( (void**)&dev_populationDetailedClassDistTab, populationTabSize * nClasses * sizeof( unsigned int ) );
	if( cudaStatus != cudaSuccess ){
		printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMalloc failed - 9 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
	}

	cudaStatus = cudaMalloc( (void**)&dev_populationDipolTab_ScatOverBlocks, nBlocks * populationTabSize * nClasses * (N_DIPOL_OBJECTS + 1) * sizeof( unsigned int ) );
	if( cudaStatus != cudaSuccess ){
		printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMalloc failed - 10 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
	}

	cudaStatus = cudaMalloc( (void**)&dev_populationDipolTab, populationTabSize * nClasses * N_DIPOL_OBJECTS * sizeof( unsigned int ) );
	if( cudaStatus != cudaSuccess ){
		printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMalloc failed - 11 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
	}
	//timeStats -> MemoryAllocDeallocTimeEnd();			//time stats
	#endif
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	//transfer population data to device	
	//timeStats -> DataTransferToGPUTimeBegin();			//time stats
	cudaStatus = cudaMemcpy( dev_populationAttrNumTab, populationAttrNumTab, populationTabSize * sizeof( int ), cudaMemcpyHostToDevice );
	if( cudaStatus != cudaSuccess ){
        printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 12 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
    }

	cudaStatus = cudaMemcpy( dev_populationValueTab, populationValueTab, populationTabSize * sizeof( float ), cudaMemcpyHostToDevice );
	if( cudaStatus != cudaSuccess ){
        printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 13 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
    }
	cudaStatus = cudaMemcpy( dev_individualPosInTab, individualPosInTab, nIndividuals * sizeof( int ), cudaMemcpyHostToDevice );
	if( cudaStatus != cudaSuccess ){
        printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 14 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
    }
	#if DEBUG_ADAPTIVE_TREE_REP
	popHostDeviceSendBytesRun += populationTabSize * ( sizeof(int) + sizeof(float) );
	popHostDeviceSendBytesSum += populationTabSize * ( sizeof(int) + sizeof(float) );
	#endif
	#if !FULL_BINARY_TREE_REP
	if (bCompactOrFullTreeRep) {
		//MEMORY OPTIMIZATION
		cudaStatus = cudaMemcpy(dev_populationNodePosTab, populationNodePosTab, 3 * populationTabSize * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf("CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 15, 16, 17 !!!\n");
			printf("%s\n", cudaGetErrorString(cudaStatus));
			exit(EXIT_FAILURE);
		}
		dev_populationLeftNodePosTab = dev_populationNodePosTab;
		dev_populationRightNodePosTab = dev_populationNodePosTab + populationTabSize;
		dev_populationParentNodePosTab = dev_populationNodePosTab + 2 * populationTabSize;
		#if DEBUG_ADAPTIVE_TREE_REP
		popHostDeviceSendBytesRun += populationTabSize * (3 * sizeof(int));
		popHostDeviceSendBytesSum += populationTabSize * (3 * sizeof(int));
		#endif
	}	
	#endif
	//////////////////////////////////////////////////////////////////////////
	//timeStats -> DataTransferToGPUTimeEnd();			//time stats

	//////////////////////////////////////////////////////////////////////////
	//timeStats -> DataReorganizationTimeBegin();			//time stats
	delete []individualPosInTab;
	delete []populationAttrNumTab;
	delete []populationValueTab;
	#if !FULL_BINARY_TREE_REP
	if (bCompactOrFullTreeRep) {
		//MEMORY OPTIMIZATION
		delete[]populationNodePosTab;
		//delete[]populationLeftNodePosTab;
		//delete[]populationRightNodePosTab;
		//delete[]populationParentNodePosTab;
	}
	#endif
	//timeStats -> DataReorganizationTimeEnd();			//time stats
	//////////////////////////////////////////////////////////////////////////

	
	//////////////////////////////////////////////////////////////////////////
	//two CUDA kernels - first, to pass all samples in the training dataset through the tree starting from the root node
	//to an appropriate leaf, second, to reduce (gather and merge) the results as well as to propagate them from the leaves towards the root node
	//timeStats -> CalcTimeBegin();						//time stats
	//the number of blocks cannot be bigger than the number of objects
	//the number of objects cannot be bigger than nBlocks * nThreads
	if( nObjects < nBlocks ) printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - warning - nObjects < nBlocks - not good results are guaranteed !!!\n" );
	#if FULL_BINARY_TREE_REP
	dev_CalcPopClassDistAndDipolAtLeafs_Pre_V2b<<< nBlocks, nThreads >>>( dev_datasetTab, dev_classTab, nObjects, nAttrs, nIndividuals, 											//datset
																		  dev_populationAttrNumTab, dev_populationValueTab, dev_individualPosInTab, populationTabSize, nClasses,	//population
																		  dev_populationClassDistTab_ScatOverBlocks, dev_populationDipolTab_ScatOverBlocks );						//results
	#else
	if (bCompactOrFullTreeRep)
		dev_CalcPopClassDistAndDipolAtLeafs_Pre_V2b<<< nBlocks, nThreads >>>( dev_datasetTab, dev_classTab, nObjects, nAttrs, nIndividuals, 													//datset
																			  dev_populationAttrNumTab, dev_populationValueTab, dev_individualPosInTab, populationTabSize, nClasses,	//population
																			  dev_populationLeftNodePosTab, dev_populationRightNodePosTab, dev_populationParentNodePosTab,				//DT links
																			  dev_populationClassDistTab_ScatOverBlocks, dev_populationDipolTab_ScatOverBlocks );						//results
	#if ADDAPTIVE_TREE_REP	
	else
		dev_CalcPopClassDistAndDipolAtLeafs_Pre_FULL_BINARY_TREE_REP_V2b<<< nBlocks, nThreads >>>( dev_datasetTab, dev_classTab, nObjects, nAttrs, nIndividuals, 								//datset
																					 			   dev_populationAttrNumTab, dev_populationValueTab, dev_individualPosInTab, populationTabSize, nClasses,	//population
																								   dev_populationClassDistTab_ScatOverBlocks, dev_populationDipolTab_ScatOverBlocks );						//results
	#endif
	#endif
	cudaStatus = cudaGetLastError();
    if( cudaStatus != cudaSuccess ){		
        printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - dev_CalcPopClassDistAndDipolAtLeafs_Pre_V2b failed - 18 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
        exit( EXIT_FAILURE );
    }

	cudaStatus = cudaDeviceSynchronize();
	if( cudaStatus != cudaSuccess ){
		printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaDeviceSynchronize failed - 19 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
        exit( EXIT_FAILURE );        
    }
	
	//aggregate/sum results in table (nBlocks equals at least 2)
	#if FULL_BINARY_TREE_REP
	dev_CalcPopDetailedErrAndClassDistAndDipol_Post_V2b<<< nIndividuals, nBlocks >>>( dev_individualPosInTab, populationTabSize, nClasses,
								      												  dev_populationClassDistTab_ScatOverBlocks, dev_populationDipolTab_ScatOverBlocks,
																					  dev_populationDetailedErrTab, dev_populationDetailedClassDistTab, dev_populationDipolTab );	
	#else
	if (bCompactOrFullTreeRep)
		dev_CalcPopDetailedErrAndClassDistAndDipol_Post_V2b<<< nIndividuals, nBlocks >>>( dev_individualPosInTab, populationTabSize, nClasses, dev_populationParentNodePosTab,
									      												  dev_populationClassDistTab_ScatOverBlocks, dev_populationDipolTab_ScatOverBlocks,
																						  dev_populationDetailedErrTab, dev_populationDetailedClassDistTab, dev_populationDipolTab );	
	#if ADDAPTIVE_TREE_REP
	else
		dev_CalcPopDetailedErrAndClassDistAndDipol_Post_FULL_BINARY_TREE_REP_V2b<<< nIndividuals, nBlocks >>>( dev_individualPosInTab, populationTabSize, nClasses,
								      																		   dev_populationClassDistTab_ScatOverBlocks, dev_populationDipolTab_ScatOverBlocks,
																											   dev_populationDetailedErrTab, dev_populationDetailedClassDistTab, dev_populationDipolTab );	
	#endif
	#endif	
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
    if( cudaStatus != cudaSuccess ){		
        printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - dev_CalcPopDetailedErrAndClassDistAndDipol_Post_V2b failed - 20 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		printf( "Probablity individual/tree size was too big, it equals %d", populationTabSize );
        exit( EXIT_FAILURE );
    }
	//timeStats -> CalcTimeEnd();							//time stats
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	//transfer results from device	
	//timeStats -> DataTransferFromGPUTimeBegin();		//time stats
	unsigned int* populationDetailedErrTab = new unsigned int[ populationTabSize * N_INFO_CORR_ERR ];
	cudaStatus = cudaMemcpy( populationDetailedErrTab, dev_populationDetailedErrTab, populationTabSize * N_INFO_CORR_ERR * sizeof( unsigned int ), cudaMemcpyDeviceToHost );
	if( cudaStatus != cudaSuccess ){
        printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 21 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		printf( "Probablity individual/tree size was too big, it equals %d", populationTabSize );
		exit( EXIT_FAILURE );
    }

	(*populationDetailedClassDistTab) = new unsigned int[ populationTabSize * nClasses ];
	cudaStatus = cudaMemcpy( (*populationDetailedClassDistTab), dev_populationDetailedClassDistTab, populationTabSize * nClasses * sizeof( unsigned int ), cudaMemcpyDeviceToHost );
	if( cudaStatus != cudaSuccess ){
        printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 22 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		printf( "Probablity individual/tree size was too big, it equals %d", populationTabSize );
		exit( EXIT_FAILURE );
	}

	(*populationDipolTab) = new unsigned int[ populationTabSize * nClasses * N_DIPOL_OBJECTS ];
	cudaStatus = cudaMemcpy( (*populationDipolTab), dev_populationDipolTab, populationTabSize * nClasses * N_DIPOL_OBJECTS * sizeof( unsigned int ), cudaMemcpyDeviceToHost );
	if( cudaStatus != cudaSuccess ){
        printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 23 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		printf( "Probablity individual/tree size was too big, it equals %d", populationTabSize );
		exit( EXIT_FAILURE );
	}
	//printf( "popSize=%d rootTrain=%d rootError=%d\n", populationTabSize, populationDetailedErrTab[ 0 ], populationDetailedErrTab[ 1 ] );		
	//timeStats -> DataTransferFromGPUTimeEnd();			//time stats
	#if DEBUG_ADAPTIVE_TREE_REP	
	resultDeviceHostSendBytesRun += populationTabSize * ( (N_INFO_CORR_ERR * sizeof(unsigned int) + nClasses * sizeof(unsigned int) + nClasses * N_DIPOL_OBJECTS * sizeof(unsigned int) ) );
	resultDeviceHostSendBytesSum += populationTabSize * ( (N_INFO_CORR_ERR * sizeof(unsigned int) + nClasses * sizeof(unsigned int) + nClasses * N_DIPOL_OBJECTS * sizeof(unsigned int) ) );
	#endif
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	
	//////////////////////////////////////////////////////////////////////////
	//clear memory
	//delete []populationErr;
	#if !CUDA_MALLOC_OPTIM_1
	timeStats -> MemoryAllocDeallocTimeBegin();			//time stats	
	cudaFree( dev_populationAttrNumTab );	
	cudaFree( dev_populationValueTab );	
	cudaFree( dev_individualPosInTab );
	cudaFree( dev_populationClassDistTab_ScatOverBlocks );
	cudaFree( dev_populationDipolTab_ScatOverBlocks );
	cudaFree( dev_populationDetailedClassDistTab );
	cudaFree( dev_populationDetailedErrTab );
	cudaFree( dev_populationDipolTab );
	#if !FULL_BINARY_TREE_REP
	if (bCompactOrFullTreeRep){
		//MEMORY OPTIMIZATION
		cudaFree( dev_populationNodePosTab );
		//cudaFree( dev_populationLeftNodePosTab );
		//cudaFree( dev_populationRightNodePosTab );
		//cudaFree( dev_populationParentNodePosTab );
	}
	#endif
	//timeStats -> MemoryAllocDeallocTimeEnd();			//time stats
	#endif
	//////////////////////////////////////////////////////////////////////////
	
	//timeStats -> WholeTimeEnd();						//time stats

	#if DEBUG_TIME_PER_TREE
	treeDepthAndTime.back().back().push_back(make_pair((int)root->GetDepth(), timeStats->GetWholeTimeLastDiff()));
	//treeNLeaves.back().back().push_back((int)root->GetLeavesCount());
	//treeNAllNodes.back().back().push_back((int)root->GetNAllNodes());
	#endif

	//final time stats actions
	//timeStats -> MoveTimeStats_DetailedIndivV2b();		//time stats

	return populationDetailedErrTab;
}

#if FULL_BINARY_TREE_REP
__global__ void dev_CalcPopClassDistAndDipolAtLeafs_Pre_V2b( DS_REAL *datasetTab, unsigned int *classTab, int nObjects, int nAttrs, int nIndividuals,
															 int *populationAttrNumTab, float *populationValueTab,  int *individualPosInTab, int populationTabSize, int nClasses,
															 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks ){
	//individual parameters are set later in for loop
	//since each thread goes through all population individuals																
																	
	//initial object to check		
	int startObjectIndex = 0;
	//FIRST - BASED ON BLOCK ID
	if( blockIdx.x < nObjects % gridDim.x )
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + blockIdx.x;		
	else
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + nObjects % gridDim.x;
	int nObjectsToCheck = nObjects / gridDim.x + (int)( ( nObjects % gridDim.x ) > blockIdx.x );	
	//SECOND - BASED ON THREAD ID (to spread objects further)
	if( threadIdx.x < nObjectsToCheck % blockDim.x )
		startObjectIndex += threadIdx.x * ( nObjectsToCheck / blockDim.x ) + threadIdx.x;		
	else
		startObjectIndex += threadIdx.x * ( nObjectsToCheck / blockDim.x ) + nObjectsToCheck % blockDim.x;

	nObjectsToCheck = nObjectsToCheck / blockDim.x + (int)( ( nObjectsToCheck % blockDim.x ) > threadIdx.x );

	//MOZE
	if( nObjectsToCheck <= 0 ) return;

	//clear results table
	//for class distribution
	unsigned int* populationClassDistTab_Local = populationClassDistTab_ScatOverBlocks + blockIdx.x * populationTabSize * nClasses;
	if( threadIdx.x == 0 ){		
		for( int i = 0; i < populationTabSize * nClasses; i++ )
			populationClassDistTab_Local[ i ] = 0;
	}
	//also for dipol
	unsigned int* populationDipolTab_Local = populationDipolTab_ScatOverBlocks + blockIdx.x * populationTabSize * nClasses * (N_DIPOL_OBJECTS + 1);
	if( threadIdx.x == 0 ){		
		for( int i = 0; i < populationTabSize * nClasses * (N_DIPOL_OBJECTS + 1); i++ )
			populationDipolTab_Local[ i ] = 0;
	}
	__syncthreads();

	/////////////////////////////////////////////// !!!!!!!!!!!
	/////////////////////////////////////////////// !!!!!!!!!!!
	/////////////////////////////////////////////// !!!!!!!!!!!
	//atomicAdd( &(populationClassDistTabLocal[ treeNodeIndex * nClasses + classTab[ startObjectIndex + index ] ]), 1 );
	//TU CHYBA MOZNA TEZ PRZYSPIESZYC, pamietajac wskaznik dla lokalnej czesci tablicy, classTab[ startObjectIndex + index ]
	/////////////////////////////////////////////// !!!!!!!!!!!
	/////////////////////////////////////////////// !!!!!!!!!!!
	/////////////////////////////////////////////// !!!!!!!!!!!

	unsigned int* individualClassDistTab_Local;
	unsigned int* individualDipolTab_Local;
	int dipolObjectStartIndex, dipolObjectSubIndex;
	for( int i = 0; i < nIndividuals; i++ ){
	
		//which invidual in population
		int individualIndex = i;
		int individualTabSize = 0;
	
		//the individual size in 1D table
		if( individualIndex < nIndividuals - 1 )
			individualTabSize = individualPosInTab[ individualIndex + 1 ] - individualPosInTab[ individualIndex ];
		else
			individualTabSize = populationTabSize - individualPosInTab[ individualIndex ];

		//to choose the desired individual in 1D population table	
		int *individualAttrNumTab = populationAttrNumTab + individualPosInTab[ individualIndex ];
		float *individualValueTab = populationValueTab + individualPosInTab[ individualIndex ];

		//table to remember the number of objects in tree nodes (such a table is the result of dev_CalcClassDist kernel)
		//here based on this table, the number of badly classified objects are finally obtained (that is, individual error)
		//in 2b case, several tables are needed, so local tables are used, and in the next kernel function they are aggregated/summed
		//populationClassDistTab_Local = populationClassDistTab_ScatOverBlocks + blockIdx.x * populationTabSize * nClasses + individualPosInTab[ individualIndex ] * nClasses;
		individualClassDistTab_Local = populationClassDistTab_Local + individualPosInTab[ individualIndex ] * nClasses;
		individualDipolTab_Local = populationDipolTab_Local + individualPosInTab[ individualIndex ] * nClasses * (N_DIPOL_OBJECTS + 1);

		//to choose the start object in 1D table
		DS_REAL* objectTab = datasetTab + startObjectIndex * nAttrs;

		int treeNodeIndex;		
		unsigned int index = 0;
		while( index < nObjectsToCheck ){			
			//start from root node
			treeNodeIndex = 0;

			while( treeNodeIndex < individualTabSize ){

				//if we are in a leaf
				if( individualAttrNumTab[ treeNodeIndex ] == -1 ){
					//remember the object class in this leaf
					atomicAdd( &(individualClassDistTab_Local[ treeNodeIndex * nClasses + classTab[ startObjectIndex + index ] ]), 1 );
					
					//remember the object number for the dipol mechanism and increment the index for the next object of the same class
					//wydaje mi sie ze mimo ze dwie nastepne operacje nie wykonaja sie laczenie razem to i tak nie spowoduje to bledu
					dipolObjectStartIndex = treeNodeIndex * nClasses * (N_DIPOL_OBJECTS + 1) + classTab[ startObjectIndex + index ] * (N_DIPOL_OBJECTS + 1);																	//where is the place for the first object for the dipol mechanism
					dipolObjectSubIndex = atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ]), (individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ] + 1) % N_DIPOL_OBJECTS );	//+1 to know where any object was not set					
					atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + dipolObjectSubIndex ]), startObjectIndex + index + 1 );											
					//old version with the race of threads					
					//dipolObjectStartIndex = treeNodeIndex * nClasses * (N_DIPOL_OBJECTS + 1) + classTab[ startObjectIndex + index ] * (N_DIPOL_OBJECTS + 1);		//where is the place for the first object for the dipol mechanism
					//dipolObjectSubIndex = individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ];								//which (first, second, etc) object for the dipol mechanism
					//atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + dipolObjectSubIndex ]), startObjectIndex + index + 1 );
					//atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ]), (dipolObjectSubIndex + 1) % N_DIPOL_OBJECTS );		//+1 to know where any object was not set
					break;
				}
				else				
					//go left in the tree
					if( objectTab[ individualAttrNumTab[ treeNodeIndex ] ] <= individualValueTab[ treeNodeIndex ] )
						treeNodeIndex = 2 * treeNodeIndex + 1;
					//go right in the tree
					else
						treeNodeIndex = 2 * treeNodeIndex + 2;
			}

			//move to the next object in dataset
			index++;		
			objectTab += nAttrs;
		}		
	}
}
#else
__global__ void dev_CalcPopClassDistAndDipolAtLeafs_Pre_V2b( DS_REAL *datasetTab, unsigned int *classTab, int nObjects, int nAttrs, int nIndividuals,
															 int *populationAttrNumTab, float *populationValueTab,  int *individualPosInTab, int populationTabSize, int nClasses,
															 int *populationLeftNodePosTab, int *populationRightNodePosTab, int *populationParentNodePosTab,
															 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks ){
	//individual parameters are set later in for loop
	//since each thread goes through all population individuals																
																	
	//initial object to check		
	int startObjectIndex = 0;
	//FIRST - BASED ON BLOCK ID
	if( blockIdx.x < nObjects % gridDim.x )
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + blockIdx.x;		
	else
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + nObjects % gridDim.x;
	int nObjectsToCheck = nObjects / gridDim.x + (int)( ( nObjects % gridDim.x ) > blockIdx.x );	
	//SECOND - BASED ON THREAD ID (to spread objects further)
	if( threadIdx.x < nObjectsToCheck % blockDim.x )
		startObjectIndex += threadIdx.x * ( nObjectsToCheck / blockDim.x ) + threadIdx.x;		
	else
		startObjectIndex += threadIdx.x * ( nObjectsToCheck / blockDim.x ) + nObjectsToCheck % blockDim.x;

	nObjectsToCheck = nObjectsToCheck / blockDim.x + (int)( ( nObjectsToCheck % blockDim.x ) > threadIdx.x );

	//MOZE
	if( nObjectsToCheck <= 0 ) return;

	//clear results table
	//for class distribution
	unsigned int* populationClassDistTab_Local = populationClassDistTab_ScatOverBlocks + blockIdx.x * populationTabSize * nClasses;
	if( threadIdx.x == 0 ){		
		for( int i = 0; i < populationTabSize * nClasses; i++ )
			populationClassDistTab_Local[ i ] = 0;
	}
	//also for dipol
	unsigned int* populationDipolTab_Local = populationDipolTab_ScatOverBlocks + blockIdx.x * populationTabSize * nClasses * (N_DIPOL_OBJECTS + 1);
	if( threadIdx.x == 0 ){		
		for( int i = 0; i < populationTabSize * nClasses * (N_DIPOL_OBJECTS + 1); i++ )
			populationDipolTab_Local[ i ] = 0;
	}
	__syncthreads();

	/////////////////////////////////////////////// !!!!!!!!!!!
	/////////////////////////////////////////////// !!!!!!!!!!!
	/////////////////////////////////////////////// !!!!!!!!!!!
	//atomicAdd( &(populationClassDistTabLocal[ treeNodeIndex * nClasses + classTab[ startObjectIndex + index ] ]), 1 );
	//TU CHYBA MOZNA TEZ PRZYSPIESZYC, pamietajac wskaznik dla lokalnej czesci tablicy, classTab[ startObjectIndex + index ]
	/////////////////////////////////////////////// !!!!!!!!!!!
	/////////////////////////////////////////////// !!!!!!!!!!!
	/////////////////////////////////////////////// !!!!!!!!!!!

	unsigned int* individualClassDistTab_Local;
	unsigned int* individualDipolTab_Local;
	int dipolObjectStartIndex, dipolObjectSubIndex;
	for( int i = 0; i < nIndividuals; i++ ){
	
		//which invidual in population
		int individualIndex = i;
		int individualTabSize = 0;
	
		//the individual size in 1D table
		if( individualIndex < nIndividuals - 1 )
			individualTabSize = individualPosInTab[ individualIndex + 1 ] - individualPosInTab[ individualIndex ];
		else
			individualTabSize = populationTabSize - individualPosInTab[ individualIndex ];

		//to choose the desired individual in 1D population table	
		int *individualAttrNumTab = populationAttrNumTab + individualPosInTab[ individualIndex ];
		float *individualValueTab = populationValueTab	 + individualPosInTab[ individualIndex ];

		//NEW IN LINKED ARRAY DT REPRESENATION - !FULL_BINARY_TREE_REP
		int *individualLeftNodePosTab	= populationLeftNodePosTab	 + individualPosInTab[ individualIndex ];
		int *individualRightNodePosTab	= populationRightNodePosTab	 + individualPosInTab[ individualIndex ];
		int *individualParentNodePosTab = populationParentNodePosTab + individualPosInTab[ individualIndex ];

		//table to remember the number of objects in tree nodes (such a table is the result of dev_CalcClassDist kernel)
		//here based on this table, the number of badly classified objects are finally obtained (that is, individual error)
		//in 2b case, several tables are needed, so local tables are used, and in the next kernel function they are aggregated/summed
		//populationClassDistTab_Local = populationClassDistTab_ScatOverBlocks + blockIdx.x * populationTabSize * nClasses + individualPosInTab[ individualIndex ] * nClasses;
		individualClassDistTab_Local = populationClassDistTab_Local + individualPosInTab[ individualIndex ] * nClasses;
		individualDipolTab_Local = populationDipolTab_Local + individualPosInTab[ individualIndex ] * nClasses * (N_DIPOL_OBJECTS + 1);

		//to choose the start object in 1D table
		DS_REAL* objectTab = datasetTab + startObjectIndex * nAttrs;

		int treeNodeIndex;		
		unsigned int index = 0;
		while( index < nObjectsToCheck ){			
			//start from root node
			treeNodeIndex = 0;

			while( treeNodeIndex < individualTabSize ){

				//if we are in a leaf
				if (individualAttrNumTab[treeNodeIndex] == -1) {
					//remember the object class in this leaf
					atomicAdd(&(individualClassDistTab_Local[treeNodeIndex * nClasses + classTab[startObjectIndex + index]]), 1);

					//remember the object number for the dipol mechanism and increment the index for the next object of the same class
					//wydaje mi sie ze mimo ze dwie nastepne operacje nie wykonaja sie laczenie razem to i tak nie spowoduje to bledu
					dipolObjectStartIndex = treeNodeIndex * nClasses * (N_DIPOL_OBJECTS + 1) + classTab[startObjectIndex + index] * (N_DIPOL_OBJECTS + 1);																	//where is the place for the first object for the dipol mechanism
					dipolObjectSubIndex = atomicExch(&(individualDipolTab_Local[dipolObjectStartIndex + N_DIPOL_OBJECTS]), (individualDipolTab_Local[dipolObjectStartIndex + N_DIPOL_OBJECTS] + 1) % N_DIPOL_OBJECTS);	//+1 to know where any object was not set					
					atomicExch(&(individualDipolTab_Local[dipolObjectStartIndex + dipolObjectSubIndex]), startObjectIndex + index + 1);
					//old version with the race of threads					
					//dipolObjectStartIndex = treeNodeIndex * nClasses * (N_DIPOL_OBJECTS + 1) + classTab[ startObjectIndex + index ] * (N_DIPOL_OBJECTS + 1);		//where is the place for the first object for the dipol mechanism
					//dipolObjectSubIndex = individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ];								//which (first, second, etc) object for the dipol mechanism
					//atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + dipolObjectSubIndex ]), startObjectIndex + index + 1 );
					//atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ]), (dipolObjectSubIndex + 1) % N_DIPOL_OBJECTS );		//+1 to know where any object was not set
					break;
				}
				else
					//go left in the tree
					if (objectTab[individualAttrNumTab[treeNodeIndex]] <= individualValueTab[treeNodeIndex])
						//NEW IN LINKED ARRAY DT REPRESENATION - !FULL_BINARY_TREE_REP
						//treeNodeIndex = 2 * treeNodeIndex + 1;
						treeNodeIndex = individualLeftNodePosTab[ treeNodeIndex ];
					//go right in the tree
					else
						//NEW IN LINKED ARRAY DT REPRESENATION - !FULL_BINARY_TREE_REP
						//treeNodeIndex = 2 * treeNodeIndex + 2;
						treeNodeIndex = individualRightNodePosTab[ treeNodeIndex ];
			}

			//move to the next object in dataset
			index++;		
			objectTab += nAttrs;
		}		
	}
}
__global__ void dev_CalcPopClassDistAndDipolAtLeafs_Pre_FULL_BINARY_TREE_REP_V2b( DS_REAL *datasetTab, unsigned int *classTab, int nObjects, int nAttrs, int nIndividuals,
															 int *populationAttrNumTab, float *populationValueTab,  int *individualPosInTab, int populationTabSize, int nClasses,
															 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks ){
	//individual parameters are set later in for loop
	//since each thread goes through all population individuals																
																	
	//initial object to check		
	int startObjectIndex = 0;
	//FIRST - BASED ON BLOCK ID
	if( blockIdx.x < nObjects % gridDim.x )
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + blockIdx.x;		
	else
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + nObjects % gridDim.x;
	int nObjectsToCheck = nObjects / gridDim.x + (int)( ( nObjects % gridDim.x ) > blockIdx.x );	
	//SECOND - BASED ON THREAD ID (to spread objects further)
	if( threadIdx.x < nObjectsToCheck % blockDim.x )
		startObjectIndex += threadIdx.x * ( nObjectsToCheck / blockDim.x ) + threadIdx.x;		
	else
		startObjectIndex += threadIdx.x * ( nObjectsToCheck / blockDim.x ) + nObjectsToCheck % blockDim.x;

	nObjectsToCheck = nObjectsToCheck / blockDim.x + (int)( ( nObjectsToCheck % blockDim.x ) > threadIdx.x );

	//MOZE
	if( nObjectsToCheck <= 0 ) return;

	//clear results table
	//for class distribution
	unsigned int* populationClassDistTab_Local = populationClassDistTab_ScatOverBlocks + blockIdx.x * populationTabSize * nClasses;
	if( threadIdx.x == 0 ){		
		for( int i = 0; i < populationTabSize * nClasses; i++ )
			populationClassDistTab_Local[ i ] = 0;
	}
	//also for dipol
	unsigned int* populationDipolTab_Local = populationDipolTab_ScatOverBlocks + blockIdx.x * populationTabSize * nClasses * (N_DIPOL_OBJECTS + 1);
	if( threadIdx.x == 0 ){		
		for( int i = 0; i < populationTabSize * nClasses * (N_DIPOL_OBJECTS + 1); i++ )
			populationDipolTab_Local[ i ] = 0;
	}
	__syncthreads();

	/////////////////////////////////////////////// !!!!!!!!!!!
	/////////////////////////////////////////////// !!!!!!!!!!!
	/////////////////////////////////////////////// !!!!!!!!!!!
	//atomicAdd( &(populationClassDistTabLocal[ treeNodeIndex * nClasses + classTab[ startObjectIndex + index ] ]), 1 );
	//TU CHYBA MOZNA TEZ PRZYSPIESZYC, pamietajac wskaznik dla lokalnej czesci tablicy, classTab[ startObjectIndex + index ]
	/////////////////////////////////////////////// !!!!!!!!!!!
	/////////////////////////////////////////////// !!!!!!!!!!!
	/////////////////////////////////////////////// !!!!!!!!!!!

	unsigned int* individualClassDistTab_Local;
	unsigned int* individualDipolTab_Local;
	int dipolObjectStartIndex, dipolObjectSubIndex;
	for( int i = 0; i < nIndividuals; i++ ){
	
		//which invidual in population
		int individualIndex = i;
		int individualTabSize = 0;
	
		//the individual size in 1D table
		if( individualIndex < nIndividuals - 1 )
			individualTabSize = individualPosInTab[ individualIndex + 1 ] - individualPosInTab[ individualIndex ];
		else
			individualTabSize = populationTabSize - individualPosInTab[ individualIndex ];

		//to choose the desired individual in 1D population table	
		int *individualAttrNumTab = populationAttrNumTab + individualPosInTab[ individualIndex ];
		float *individualValueTab = populationValueTab + individualPosInTab[ individualIndex ];

		//table to remember the number of objects in tree nodes (such a table is the result of dev_CalcClassDist kernel)
		//here based on this table, the number of badly classified objects are finally obtained (that is, individual error)
		//in 2b case, several tables are needed, so local tables are used, and in the next kernel function they are aggregated/summed
		//populationClassDistTab_Local = populationClassDistTab_ScatOverBlocks + blockIdx.x * populationTabSize * nClasses + individualPosInTab[ individualIndex ] * nClasses;
		individualClassDistTab_Local = populationClassDistTab_Local + individualPosInTab[ individualIndex ] * nClasses;
		individualDipolTab_Local = populationDipolTab_Local + individualPosInTab[ individualIndex ] * nClasses * (N_DIPOL_OBJECTS + 1);

		//to choose the start object in 1D table
		DS_REAL* objectTab = datasetTab + startObjectIndex * nAttrs;

		int treeNodeIndex;		
		unsigned int index = 0;
		while( index < nObjectsToCheck ){			
			//start from root node
			treeNodeIndex = 0;

			while( treeNodeIndex < individualTabSize ){

				//if we are in a leaf
				if( individualAttrNumTab[ treeNodeIndex ] == -1 ){
					//remember the object class in this leaf
					atomicAdd( &(individualClassDistTab_Local[ treeNodeIndex * nClasses + classTab[ startObjectIndex + index ] ]), 1 );
					
					//remember the object number for the dipol mechanism and increment the index for the next object of the same class
					//wydaje mi sie ze mimo ze dwie nastepne operacje nie wykonaja sie laczenie razem to i tak nie spowoduje to bledu
					dipolObjectStartIndex = treeNodeIndex * nClasses * (N_DIPOL_OBJECTS + 1) + classTab[ startObjectIndex + index ] * (N_DIPOL_OBJECTS + 1);																	//where is the place for the first object for the dipol mechanism
					dipolObjectSubIndex = atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ]), (individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ] + 1) % N_DIPOL_OBJECTS );	//+1 to know where any object was not set					
					atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + dipolObjectSubIndex ]), startObjectIndex + index + 1 );											
					//old version with the race of threads					
					//dipolObjectStartIndex = treeNodeIndex * nClasses * (N_DIPOL_OBJECTS + 1) + classTab[ startObjectIndex + index ] * (N_DIPOL_OBJECTS + 1);		//where is the place for the first object for the dipol mechanism
					//dipolObjectSubIndex = individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ];								//which (first, second, etc) object for the dipol mechanism
					//atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + dipolObjectSubIndex ]), startObjectIndex + index + 1 );
					//atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ]), (dipolObjectSubIndex + 1) % N_DIPOL_OBJECTS );		//+1 to know where any object was not set
					break;
				}
				else				
					//go left in the tree
					if( objectTab[ individualAttrNumTab[ treeNodeIndex ] ] <= individualValueTab[ treeNodeIndex ] )
						treeNodeIndex = 2 * treeNodeIndex + 1;
					//go right in the tree
					else
						treeNodeIndex = 2 * treeNodeIndex + 2;
			}

			//move to the next object in dataset
			index++;		
			objectTab += nAttrs;
		}		
	}
}
#endif

#if FULL_BINARY_TREE_REP
__global__ void dev_CalcPopDetailedErrAndClassDistAndDipol_Post_V2b_2Classes( int *individualPosInTab, int populationTabSize, int nClasses,
																			  unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks,
																			  unsigned int *populationDetailedErrTab, unsigned int *populationDetailedClassDistTab, unsigned int *populationDipolTab ){
	//which invidual in population based on the block id
	int individualIndex = blockIdx.x;
	int individualTabSize = 0;
	//unsigned int *individualDetailedErrTab = populationDetailedErrTab + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int *individualDetailedErrTab = populationDetailedErrTab + individualPosInTab[ individualIndex ] * N_INFO_CORR_ERR;
	unsigned int *individualDetailedClassDistTab = populationDetailedClassDistTab + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int *individualDipolTab = populationDipolTab + individualPosInTab[ individualIndex ] * nClasses * N_DIPOL_OBJECTS;

	//the individual size in 1D table
	if( individualIndex < gridDim.x - 1 )
		individualTabSize = individualPosInTab[ individualIndex + 1 ] - individualPosInTab[ individualIndex ];
	else
		individualTabSize = populationTabSize - individualPosInTab[ individualIndex ];

	//set a pointer in individualClassTabResults to read the data for an appropriate individual		
	//unsigned int* individualClassTabResultsLocal = individualClassTabResults + threadIdx.x * gridDim.x * MAX_N_INFO_TREE_NODES + blockIdx.x * MAX_N_INFO_TREE_NODES;
	unsigned int* individualClassDistTab_Local = populationClassDistTab_ScatOverBlocks + threadIdx.x * populationTabSize * nClasses + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int* individualDipolTab_Local = populationDipolTab_ScatOverBlocks + threadIdx.x * populationTabSize * nClasses * (N_DIPOL_OBJECTS + 1)  + individualPosInTab[ individualIndex ] * nClasses * (N_DIPOL_OBJECTS + 1);
	
	//table to remember the number of objects in tree nodes (such a table is the result of dev_CalcClassDist kernel)
	//here based on this table, the number of badly classified objects are finally obtained (that is, individual error)
	__shared__ unsigned int individualClassDistTab_Sum[ MAX_N_INFO_TREE_NODES ];
	__shared__ unsigned int individualDipolTab_Random[ MAX_N_INFO_TREE_NODES ];

	//clear results table
	if( threadIdx.x == 0 ){		
		for( int i = 0; i < MAX_N_INFO_TREE_NODES; i++ ){
			individualClassDistTab_Sum[ i ] = 0;
			individualDipolTab_Random[ i ] = 0.0;
		}
		for( int i = 0; i < individualTabSize * N_INFO_CORR_ERR; i++ )
			individualDetailedErrTab[ i ] = 0.0;
	}
	__syncthreads();
	

	//for( int i = 0; i < MAX_N_INFO_TREE_NODES; i++ )
	for( int i = 0; i < individualTabSize * nClasses; i++ )
		atomicAdd( &( individualClassDistTab_Sum[ i ] ), individualClassDistTab_Local[ i ] );

	//to juz wprowadzi troche losowosci, bo watki beda sie sciagac ktory pierwszy ustawi obiekt
	/*for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
		atomicCAS( &( individualDipolTab_Random[ i ] ), false, individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] );		
		*/

	for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
		if( individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] != 0 )
			atomicCAS( &( individualDipolTab_Random[ i ] ), false, individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] );		
		
	
	__syncthreads();
	//in this moment a block has a table with number objects in each tree leaf

	//calculate individual errors and class distribution - ERROR and CLASS DISTRIBUTION IN ALL TREE NODES
	if( threadIdx.x == 0 ){
		int helpVar;

		//setting error/correct values in an appriopriate order
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex >= 0; treeNodeIndex-- ){
			
			//before setting at first place in table 'individualClassTabResults_Sum' errors, class distribution is copied
			for( int classIndex = 0; classIndex < nClasses; classIndex++ )
				individualDetailedClassDistTab[ treeNodeIndex * nClasses + classIndex ] = individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ];

			//setting error/correct values in an appriopriate order
			if( individualClassDistTab_Sum[ treeNodeIndex * nClasses ] > individualClassDistTab_Sum[ treeNodeIndex * nClasses + 1 ] ){
				helpVar = individualClassDistTab_Sum[ treeNodeIndex * nClasses + 1 ];
				individualClassDistTab_Sum[ treeNodeIndex * nClasses + 1 ] = individualClassDistTab_Sum[ treeNodeIndex * nClasses ];	
				individualClassDistTab_Sum[ treeNodeIndex * nClasses ] = helpVar;
			}
		}
		
		//propagate error/correct values from leafs to the root node
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){			
				if( treeNodeIndex % 2 == 0 ){
					individualClassDistTab_Sum[ treeNodeIndex - 2 ] += individualClassDistTab_Sum[ treeNodeIndex * N_INFO_CORR_ERR];
					individualClassDistTab_Sum[ treeNodeIndex - 1 ] += individualClassDistTab_Sum[ treeNodeIndex * N_INFO_CORR_ERR + 1 ];
				}
				else{
					individualClassDistTab_Sum[ treeNodeIndex - 1 ] += individualClassDistTab_Sum[ treeNodeIndex * N_INFO_CORR_ERR];
					individualClassDistTab_Sum[ treeNodeIndex     ] += individualClassDistTab_Sum[ treeNodeIndex * N_INFO_CORR_ERR + 1 ];
				}			
		}
		for( int i = 0; i < individualTabSize * N_INFO_CORR_ERR; i++ )
			individualDetailedErrTab[ i ] = individualClassDistTab_Sum[ i ];

		//TEN FOR DZIALA TYLKO DLA DWOCH KLAS - DODAC W SRODKU IF PO WIELKU KLASACH !!!!!!!!!!!!!!!!!!!! tak samo w funkcjach wyzej
		//propagate class distribution from leafs to the root node
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){			
				if( treeNodeIndex % 2 == 0 ){
					individualDetailedClassDistTab[ treeNodeIndex - 2 ] += individualDetailedClassDistTab[ treeNodeIndex * nClasses     ];
					individualDetailedClassDistTab[ treeNodeIndex - 1 ] += individualDetailedClassDistTab[ treeNodeIndex * nClasses + 1 ];
				}
				else{
					individualDetailedClassDistTab[ treeNodeIndex - 1 ] += individualDetailedClassDistTab[ treeNodeIndex * nClasses     ];
					individualDetailedClassDistTab[ treeNodeIndex     ] += individualDetailedClassDistTab[ treeNodeIndex * nClasses + 1 ];
				}			
		}		


		//NA SZYBKO GDY N_DIPOL_OBJECTS = 2
		//propagate dipol values from leafs to the root node
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){			
				if( treeNodeIndex % 2 == 0 ){
					if( individualDipolTab_Random[ treeNodeIndex * 2 - 4 ] == 0 ) individualDipolTab_Random[ treeNodeIndex * 2 - 4 ] = individualDipolTab_Random[ treeNodeIndex * nClasses * N_DIPOL_OBJECTS     ];
					if( individualDipolTab_Random[ treeNodeIndex * 2 - 3 ] == 0 ) individualDipolTab_Random[ treeNodeIndex * 2 - 3 ] = individualDipolTab_Random[ treeNodeIndex * nClasses * N_DIPOL_OBJECTS + 1 ];
					if( individualDipolTab_Random[ treeNodeIndex * 2 - 2 ] == 0 ) individualDipolTab_Random[ treeNodeIndex * 2 - 2 ] = individualDipolTab_Random[ treeNodeIndex * nClasses * N_DIPOL_OBJECTS + 2 ];
					if( individualDipolTab_Random[ treeNodeIndex * 2 - 1 ] == 0 ) individualDipolTab_Random[ treeNodeIndex * 2 - 1 ] = individualDipolTab_Random[ treeNodeIndex * nClasses * N_DIPOL_OBJECTS + 3 ];
				}
				else{
					if( individualDipolTab_Random[ treeNodeIndex * 2 - 2 ] == 0 ) individualDipolTab_Random[ treeNodeIndex * 2 - 2 ] = individualDipolTab_Random[ treeNodeIndex * nClasses * N_DIPOL_OBJECTS     ];
					if( individualDipolTab_Random[ treeNodeIndex * 2 - 1 ] == 0 ) individualDipolTab_Random[ treeNodeIndex * 2 - 1 ] = individualDipolTab_Random[ treeNodeIndex * nClasses * N_DIPOL_OBJECTS + 1 ];
					if( individualDipolTab_Random[ treeNodeIndex * 2     ] == 0 ) individualDipolTab_Random[ treeNodeIndex * 2     ] = individualDipolTab_Random[ treeNodeIndex * nClasses * N_DIPOL_OBJECTS + 2 ];
					if( individualDipolTab_Random[ treeNodeIndex * 2 + 1 ] == 0 ) individualDipolTab_Random[ treeNodeIndex * 2 + 1 ] = individualDipolTab_Random[ treeNodeIndex * nClasses * N_DIPOL_OBJECTS + 3 ];
				}			
		}
		//in order set dipol objects currently without propagation from leafs to the root please place ten above lines in a comment
		
		for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
			individualDipolTab[ i ] = individualDipolTab_Random[ i ];
	}
}

__global__ void dev_CalcPopDetailedErrAndClassDistAndDipol_Post_V2b( int *individualPosInTab, int populationTabSize, int nClasses,
																	 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks,
																	 unsigned int *populationDetailedErrTab, unsigned int *populationDetailedClassDistTab, unsigned int *populationDipolTab ){
	//which invidual in population based on the block id
	int individualIndex = blockIdx.x;
	int individualTabSize = 0;
	//unsigned int *individualDetailedErrTab = populationDetailedErrTab + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int *individualDetailedErrTab = populationDetailedErrTab + individualPosInTab[ individualIndex ] * N_INFO_CORR_ERR;
	unsigned int *individualDetailedClassDistTab = populationDetailedClassDistTab + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int *individualDipolTab = populationDipolTab + individualPosInTab[ individualIndex ] * nClasses * N_DIPOL_OBJECTS;

	//the individual size in 1D table
	if( individualIndex < gridDim.x - 1 )
		individualTabSize = individualPosInTab[ individualIndex + 1 ] - individualPosInTab[ individualIndex ];
	else
		individualTabSize = populationTabSize - individualPosInTab[ individualIndex ];

	//set a pointer in individualClassTabResults to read the data for an appropriate individual		
	//unsigned int* individualClassTabResultsLocal = individualClassTabResults + threadIdx.x * gridDim.x * MAX_N_INFO_TREE_NODES + blockIdx.x * MAX_N_INFO_TREE_NODES;
	unsigned int* individualClassDistTab_Local = populationClassDistTab_ScatOverBlocks + threadIdx.x * populationTabSize * nClasses + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int* individualDipolTab_Local = populationDipolTab_ScatOverBlocks + threadIdx.x * populationTabSize * nClasses * (N_DIPOL_OBJECTS + 1)  + individualPosInTab[ individualIndex ] * nClasses * (N_DIPOL_OBJECTS + 1);
	
	//table to remember the number of objects in tree nodes (such a table is the result of dev_CalcClassDist kernel)
	//here based on this table, the number of badly classified objects are finally obtained (that is, individual error)
	#if CUDA_SHARED_MEM_POST_CALC
	__shared__ unsigned int individualClassDistTab_Sum[ MAX_N_INFO_TREE_NODES ];
	__shared__ unsigned int individualDipolTab_Random[ MAX_N_INFO_TREE_NODES ];
	#else
	unsigned int *individualClassDistTab_Sum = individualDetailedClassDistTab;
	unsigned int *individualDipolTab_Random = individualDipolTab;
	#endif


	//clear results table
	if( threadIdx.x == 0 ){		
		for (int i = 0; i < individualTabSize * nClasses; i++)
			individualClassDistTab_Sum[ i ] = 0;
		for (int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++)
			individualDipolTab_Random[ i ] = 0.0;
		for( int i = 0; i < individualTabSize * N_INFO_CORR_ERR; i++ )
			individualDetailedErrTab[ i ] = 0.0;
	}
	__syncthreads();
	

	//for( int i = 0; i < MAX_N_INFO_TREE_NODES; i++ )
	for( int i = 0; i < individualTabSize * nClasses; i++ )
		atomicAdd( &( individualClassDistTab_Sum[ i ] ), individualClassDistTab_Local[ i ] );

	//to juz wprowadzi troche losowosci, bo watki beda sie sciagac ktory pierwszy ustawi obiekt
	/*for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
		atomicCAS( &( individualDipolTab_Random[ i ] ), false, individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] );		
		*/

	for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
		if( individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] != 0 )
			atomicCAS( &( individualDipolTab_Random[ i ] ), false, individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] );		
		
	
	__syncthreads();
	//in this moment a block has a table with number objects in each tree leaf

	//calculate individual errors and class distribution - ERROR and CLASS DISTRIBUTION IN ALL TREE NODES
	if( threadIdx.x == 0 ){
		int helpVar, max, sum;

		//setting error/correct values in an appriopriate order
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex >= 0; treeNodeIndex-- ){
			
			//find number of errors and correct classified objects ('max' and 'sum' - 'max' )
			sum = 0;
			max = individualClassDistTab_Sum[ treeNodeIndex * nClasses ];			
			for( int classIndex = 0; classIndex < nClasses; classIndex++ ){
				if( individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ] > max ) max = individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ];
				sum += individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ]; 
			}
			//push errors/correct from min/max to individualDetailedErrTab
			individualDetailedErrTab[treeNodeIndex * N_INFO_CORR_ERR] = sum - max;
			individualDetailedErrTab[treeNodeIndex * N_INFO_CORR_ERR + 1] = max;
		}
		
		//propagate error/correct values from leafs to the root node
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){			
				if( treeNodeIndex % 2 == 0 ){	//for full binary tree structure
					individualDetailedErrTab[ treeNodeIndex - 2 ] += individualDetailedErrTab[ treeNodeIndex * N_INFO_CORR_ERR     ];
					individualDetailedErrTab[ treeNodeIndex - 1 ] += individualDetailedErrTab[ treeNodeIndex * N_INFO_CORR_ERR + 1 ];
				}
				else{
					individualDetailedErrTab[ treeNodeIndex - 1 ] += individualDetailedErrTab[ treeNodeIndex * N_INFO_CORR_ERR     ];
					individualDetailedErrTab[ treeNodeIndex     ] += individualDetailedErrTab[ treeNodeIndex * N_INFO_CORR_ERR + 1 ];
				}			
		}
		
		//propagate class distribution from leafs to the root node
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){			
				if( treeNodeIndex % 2 == 0 )
					for( int classIndex = 0; classIndex < nClasses; classIndex++ )
						individualClassDistTab_Sum[ ( treeNodeIndex - 2 ) / 2 * nClasses + classIndex ] += individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ];
				else
					for( int classIndex = 0; classIndex < nClasses; classIndex++ )
						individualClassDistTab_Sum[ ( treeNodeIndex - 1 ) / 2 * nClasses + classIndex ] += individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ];
		}
		#if CUDA_SHARED_MEM_POST_CALC
		for( int i = 0; i < individualTabSize * N_INFO_CORR_ERR; i++ )
			individualDetailedClassDistTab[ i ] = individualClassDistTab_Sum[ i ];
		#endif

		//NA SZYBKO GDY N_DIPOL_OBJECTS = 2
		//propagate dipol values from leafs to the root node
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){			
				if( treeNodeIndex % 2 == 0 )
					for( int classIndex = 0; classIndex < nClasses; classIndex++ ){
						for( int dipolIndex = 0; dipolIndex < N_DIPOL_OBJECTS; dipolIndex++ )
							if( individualDipolTab_Random[ ( treeNodeIndex - 2 ) / 2 * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ] == 0 )
								individualDipolTab_Random[ ( treeNodeIndex - 2 ) / 2 * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ] = 
								individualDipolTab_Random[ treeNodeIndex * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ];
					}
				else
					for( int classIndex = 0; classIndex < nClasses; classIndex++ ){
						for( int dipolIndex = 0; dipolIndex < N_DIPOL_OBJECTS; dipolIndex++ )
							if( individualDipolTab_Random[ ( treeNodeIndex - 1 ) / 2 * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ] == 0 )
								individualDipolTab_Random[ ( treeNodeIndex - 1 ) / 2 * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ] = 
								individualDipolTab_Random[ treeNodeIndex * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ];
					}							
		}
		//in order set dipol objects currently without propagation from leafs to the root please place ten above lines in a comment		
		#if CUDA_SHARED_MEM_POST_CALC
		for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
			individualDipolTab[ i ] = individualDipolTab_Random[ i ];
		#endif
	}
}
#else
__global__ void dev_CalcPopDetailedErrAndClassDistAndDipol_Post_V2b( int *individualPosInTab, int populationTabSize, int nClasses, int *populationParentNodePosTab,
																	 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks,
																	 unsigned int *populationDetailedErrTab, unsigned int *populationDetailedClassDistTab, unsigned int *populationDipolTab ){
	//which invidual in population based on the block id
	int individualIndex = blockIdx.x;
	int individualTabSize = 0;
	//unsigned int *individualDetailedErrTab = populationDetailedErrTab + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int *individualDetailedErrTab = populationDetailedErrTab + individualPosInTab[ individualIndex ] * N_INFO_CORR_ERR;
	unsigned int *individualDetailedClassDistTab = populationDetailedClassDistTab + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int *individualDipolTab = populationDipolTab + individualPosInTab[ individualIndex ] * nClasses * N_DIPOL_OBJECTS;

	//NEW IN LINKED ARRAY DT REPRESENATION - !FULL_BINARY_TREE_REP	
	int *individualParentNodePosTab = populationParentNodePosTab + individualPosInTab[ individualIndex ];

	//the individual size in 1D table
	if( individualIndex < gridDim.x - 1 )
		individualTabSize = individualPosInTab[ individualIndex + 1 ] - individualPosInTab[ individualIndex ];
	else
		individualTabSize = populationTabSize - individualPosInTab[ individualIndex ];

	//set a pointer in individualClassTabResults to read the data for an appropriate individual		
	//unsigned int* individualClassTabResultsLocal = individualClassTabResults + threadIdx.x * gridDim.x * MAX_N_INFO_TREE_NODES + blockIdx.x * MAX_N_INFO_TREE_NODES;
	unsigned int* individualClassDistTab_Local = populationClassDistTab_ScatOverBlocks + threadIdx.x * populationTabSize * nClasses + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int* individualDipolTab_Local = populationDipolTab_ScatOverBlocks + threadIdx.x * populationTabSize * nClasses * (N_DIPOL_OBJECTS + 1)  + individualPosInTab[ individualIndex ] * nClasses * (N_DIPOL_OBJECTS + 1);
	
	//table to remember the number of objects in tree nodes (such a table is the result of dev_CalcClassDist kernel)
	//here based on this table, the number of badly classified objects are finally obtained (that is, individual error)
	#if CUDA_SHARED_MEM_POST_CALC
	__shared__ unsigned int individualClassDistTab_Sum[ MAX_N_INFO_TREE_NODES ];
	__shared__ unsigned int individualDipolTab_Random[ MAX_N_INFO_TREE_NODES ];
	#else
	unsigned int *individualClassDistTab_Sum = individualDetailedClassDistTab;
	unsigned int *individualDipolTab_Random = individualDipolTab;
	#endif


	//clear results table
	if( threadIdx.x == 0 ){		
		for (int i = 0; i < individualTabSize * nClasses; i++)
			individualClassDistTab_Sum[ i ] = 0;
		for (int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++)
			individualDipolTab_Random[ i ] = 0.0;
		for( int i = 0; i < individualTabSize * N_INFO_CORR_ERR; i++ )
			individualDetailedErrTab[ i ] = 0.0;
	}
	__syncthreads();
	

	//for( int i = 0; i < MAX_N_INFO_TREE_NODES; i++ )
	for( int i = 0; i < individualTabSize * nClasses; i++ )
		atomicAdd( &( individualClassDistTab_Sum[ i ] ), individualClassDistTab_Local[ i ] );

	//to juz wprowadzi troche losowosci, bo watki beda sie sciagac ktory pierwszy ustawi obiekt
	/*for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
		atomicCAS( &( individualDipolTab_Random[ i ] ), false, individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] );		
		*/

	for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
		if( individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] != 0 )
			atomicCAS( &( individualDipolTab_Random[ i ] ), false, individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] );		
		
	
	__syncthreads();
	//in this moment a block has a table with number objects in each tree leaf

	//calculate individual errors and class distribution - ERROR and CLASS DISTRIBUTION IN ALL TREE NODES
	if( threadIdx.x == 0 ){
		int helpVar, max, sum;

		//setting error/correct values in an appriopriate order
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex >= 0; treeNodeIndex-- ){
			
			//find number of errors and correct classified objects ('max' and 'sum' - 'max' )
			sum = 0;
			max = individualClassDistTab_Sum[ treeNodeIndex * nClasses ];			
			for( int classIndex = 0; classIndex < nClasses; classIndex++ ){
				if( individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ] > max ) max = individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ];
				sum += individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ]; 
			}
			//push errors/correct from min/max to individualDetailedErrTab
			individualDetailedErrTab[treeNodeIndex * N_INFO_CORR_ERR] = sum - max;
			individualDetailedErrTab[treeNodeIndex * N_INFO_CORR_ERR + 1] = max;
		}
		

		//NEW IN LINKED ARRAY DT REPRESENATION - !FULL_BINARY_TREE_REP
		//propagate error/correct values from leafs to the root node
		/*for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){			
				if( treeNodeIndex % 2 == 0 ){	//for full binary tree structure
					individualDetailedErrTab[ treeNodeIndex - 2 ] += individualDetailedErrTab[ treeNodeIndex * N_INFO_CORR_ERR     ];
					individualDetailedErrTab[ treeNodeIndex - 1 ] += individualDetailedErrTab[ treeNodeIndex * N_INFO_CORR_ERR + 1 ];
				}
				else{
					individualDetailedErrTab[ treeNodeIndex - 1 ] += individualDetailedErrTab[ treeNodeIndex * N_INFO_CORR_ERR     ];
					individualDetailedErrTab[ treeNodeIndex     ] += individualDetailedErrTab[ treeNodeIndex * N_INFO_CORR_ERR + 1 ];
				}			
		}*/
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){
			individualDetailedErrTab[ individualParentNodePosTab[ treeNodeIndex ] * N_INFO_CORR_ERR     ] += individualDetailedErrTab[ treeNodeIndex * N_INFO_CORR_ERR     ];
			individualDetailedErrTab[ individualParentNodePosTab[ treeNodeIndex ] * N_INFO_CORR_ERR + 1 ] += individualDetailedErrTab[ treeNodeIndex * N_INFO_CORR_ERR + 1 ];
		}
		
		//NEW IN LINKED ARRAY DT REPRESENATION - !FULL_BINARY_TREE_REP
		//propagate class distribution from leafs to the root node
		/*for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){			
				if( treeNodeIndex % 2 == 0 )
					for( int classIndex = 0; classIndex < nClasses; classIndex++ )
						individualClassDistTab_Sum[ ( treeNodeIndex - 2 ) / 2 * nClasses + classIndex ] += individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ];
				else
					for( int classIndex = 0; classIndex < nClasses; classIndex++ )
						individualClassDistTab_Sum[ ( treeNodeIndex - 1 ) / 2 * nClasses + classIndex ] += individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ];
		}*/
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){
			for( int classIndex = 0; classIndex < nClasses; classIndex++ )
				individualClassDistTab_Sum[ individualParentNodePosTab[treeNodeIndex] * nClasses + classIndex ] += individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ];
		}
		#if CUDA_SHARED_MEM_POST_CALC
		for( int i = 0; i < individualTabSize * N_INFO_CORR_ERR; i++ )
			individualDetailedClassDistTab[ i ] = individualClassDistTab_Sum[ i ];
		#endif

		//NA SZYBKO GDY N_DIPOL_OBJECTS = 2
		//propagate dipol values from leafs to the root node
		//NEW IN LINKED ARRAY DT REPRESENATION - !FULL_BINARY_TREE_REP
		/*for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){			
				if( treeNodeIndex % 2 == 0 )
					for( int classIndex = 0; classIndex < nClasses; classIndex++ ){
						for( int dipolIndex = 0; dipolIndex < N_DIPOL_OBJECTS; dipolIndex++ )
							if( individualDipolTab_Random[ ( treeNodeIndex - 2 ) / 2 * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ] == 0 )
								individualDipolTab_Random[ ( treeNodeIndex - 2 ) / 2 * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ] = 
								individualDipolTab_Random[ treeNodeIndex * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ];
					}
				else
					for( int classIndex = 0; classIndex < nClasses; classIndex++ ){
						for( int dipolIndex = 0; dipolIndex < N_DIPOL_OBJECTS; dipolIndex++ )
							if( individualDipolTab_Random[ ( treeNodeIndex - 1 ) / 2 * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ] == 0 )
								individualDipolTab_Random[ ( treeNodeIndex - 1 ) / 2 * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ] = 
								individualDipolTab_Random[ treeNodeIndex * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ];
					}							
		}*/
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){			
			for( int classIndex = 0; classIndex < nClasses; classIndex++ )
				for( int dipolIndex = 0; dipolIndex < N_DIPOL_OBJECTS; dipolIndex++ )
					if( individualDipolTab_Random[ individualParentNodePosTab[treeNodeIndex] * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ] == 0 )
						individualDipolTab_Random[ individualParentNodePosTab[treeNodeIndex] * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ] = 
						individualDipolTab_Random[ treeNodeIndex * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ];
		}

		//in order set dipol objects currently without propagation from leafs to the root please place ten above lines in a comment		
		#if CUDA_SHARED_MEM_POST_CALC
		for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
			individualDipolTab[ i ] = individualDipolTab_Random[ i ];
		#endif
	}
}
__global__ void dev_CalcPopDetailedErrAndClassDistAndDipol_Post_FULL_BINARY_TREE_REP_V2b( int *individualPosInTab, int populationTabSize, int nClasses,
																	 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks,
																	 unsigned int *populationDetailedErrTab, unsigned int *populationDetailedClassDistTab, unsigned int *populationDipolTab ){
	//which invidual in population based on the block id
	int individualIndex = blockIdx.x;
	int individualTabSize = 0;
	//unsigned int *individualDetailedErrTab = populationDetailedErrTab + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int *individualDetailedErrTab = populationDetailedErrTab + individualPosInTab[ individualIndex ] * N_INFO_CORR_ERR;
	unsigned int *individualDetailedClassDistTab = populationDetailedClassDistTab + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int *individualDipolTab = populationDipolTab + individualPosInTab[ individualIndex ] * nClasses * N_DIPOL_OBJECTS;

	//the individual size in 1D table
	if( individualIndex < gridDim.x - 1 )
		individualTabSize = individualPosInTab[ individualIndex + 1 ] - individualPosInTab[ individualIndex ];
	else
		individualTabSize = populationTabSize - individualPosInTab[ individualIndex ];

	//set a pointer in individualClassTabResults to read the data for an appropriate individual		
	//unsigned int* individualClassTabResultsLocal = individualClassTabResults + threadIdx.x * gridDim.x * MAX_N_INFO_TREE_NODES + blockIdx.x * MAX_N_INFO_TREE_NODES;
	unsigned int* individualClassDistTab_Local = populationClassDistTab_ScatOverBlocks + threadIdx.x * populationTabSize * nClasses + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int* individualDipolTab_Local = populationDipolTab_ScatOverBlocks + threadIdx.x * populationTabSize * nClasses * (N_DIPOL_OBJECTS + 1)  + individualPosInTab[ individualIndex ] * nClasses * (N_DIPOL_OBJECTS + 1);
	
	//table to remember the number of objects in tree nodes (such a table is the result of dev_CalcClassDist kernel)
	//here based on this table, the number of badly classified objects are finally obtained (that is, individual error)
	#if CUDA_SHARED_MEM_POST_CALC
	__shared__ unsigned int individualClassDistTab_Sum[ MAX_N_INFO_TREE_NODES ];
	__shared__ unsigned int individualDipolTab_Random[ MAX_N_INFO_TREE_NODES ];
	#else
	unsigned int *individualClassDistTab_Sum = individualDetailedClassDistTab;
	unsigned int *individualDipolTab_Random = individualDipolTab;
	#endif


	//clear results table
	if( threadIdx.x == 0 ){		
		for (int i = 0; i < individualTabSize * nClasses; i++)
			individualClassDistTab_Sum[ i ] = 0;
		for (int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++)
			individualDipolTab_Random[ i ] = 0.0;
		for( int i = 0; i < individualTabSize * N_INFO_CORR_ERR; i++ )
			individualDetailedErrTab[ i ] = 0.0;
	}
	__syncthreads();
	

	//for( int i = 0; i < MAX_N_INFO_TREE_NODES; i++ )
	for( int i = 0; i < individualTabSize * nClasses; i++ )
		atomicAdd( &( individualClassDistTab_Sum[ i ] ), individualClassDistTab_Local[ i ] );

	//to juz wprowadzi troche losowosci, bo watki beda sie sciagac ktory pierwszy ustawi obiekt
	/*for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
		atomicCAS( &( individualDipolTab_Random[ i ] ), false, individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] );		
		*/

	for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
		if( individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] != 0 )
			atomicCAS( &( individualDipolTab_Random[ i ] ), false, individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] );		
		
	
	__syncthreads();
	//in this moment a block has a table with number objects in each tree leaf

	//calculate individual errors and class distribution - ERROR and CLASS DISTRIBUTION IN ALL TREE NODES
	if( threadIdx.x == 0 ){
		int helpVar, max, sum;

		//setting error/correct values in an appriopriate order
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex >= 0; treeNodeIndex-- ){
			
			//find number of errors and correct classified objects ('max' and 'sum' - 'max' )
			sum = 0;
			max = individualClassDistTab_Sum[ treeNodeIndex * nClasses ];			
			for( int classIndex = 0; classIndex < nClasses; classIndex++ ){
				if( individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ] > max ) max = individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ];
				sum += individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ]; 
			}
			//push errors/correct from min/max to individualDetailedErrTab
			individualDetailedErrTab[treeNodeIndex * N_INFO_CORR_ERR] = sum - max;
			individualDetailedErrTab[treeNodeIndex * N_INFO_CORR_ERR + 1] = max;
		}
		
		//propagate error/correct values from leafs to the root node
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){			
				if( treeNodeIndex % 2 == 0 ){	//for full binary tree structure
					individualDetailedErrTab[ treeNodeIndex - 2 ] += individualDetailedErrTab[ treeNodeIndex * N_INFO_CORR_ERR     ];
					individualDetailedErrTab[ treeNodeIndex - 1 ] += individualDetailedErrTab[ treeNodeIndex * N_INFO_CORR_ERR + 1 ];
				}
				else{
					individualDetailedErrTab[ treeNodeIndex - 1 ] += individualDetailedErrTab[ treeNodeIndex * N_INFO_CORR_ERR     ];
					individualDetailedErrTab[ treeNodeIndex     ] += individualDetailedErrTab[ treeNodeIndex * N_INFO_CORR_ERR + 1 ];
				}			
		}
		
		//propagate class distribution from leafs to the root node
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){			
				if( treeNodeIndex % 2 == 0 )
					for( int classIndex = 0; classIndex < nClasses; classIndex++ )
						individualClassDistTab_Sum[ ( treeNodeIndex - 2 ) / 2 * nClasses + classIndex ] += individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ];
				else
					for( int classIndex = 0; classIndex < nClasses; classIndex++ )
						individualClassDistTab_Sum[ ( treeNodeIndex - 1 ) / 2 * nClasses + classIndex ] += individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ];
		}
		#if CUDA_SHARED_MEM_POST_CALC
		for( int i = 0; i < individualTabSize * N_INFO_CORR_ERR; i++ )
			individualDetailedClassDistTab[ i ] = individualClassDistTab_Sum[ i ];
		#endif

		//NA SZYBKO GDY N_DIPOL_OBJECTS = 2
		//propagate dipol values from leafs to the root node
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){			
				if( treeNodeIndex % 2 == 0 )
					for( int classIndex = 0; classIndex < nClasses; classIndex++ ){
						for( int dipolIndex = 0; dipolIndex < N_DIPOL_OBJECTS; dipolIndex++ )
							if( individualDipolTab_Random[ ( treeNodeIndex - 2 ) / 2 * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ] == 0 )
								individualDipolTab_Random[ ( treeNodeIndex - 2 ) / 2 * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ] = 
								individualDipolTab_Random[ treeNodeIndex * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ];
					}
				else
					for( int classIndex = 0; classIndex < nClasses; classIndex++ ){
						for( int dipolIndex = 0; dipolIndex < N_DIPOL_OBJECTS; dipolIndex++ )
							if( individualDipolTab_Random[ ( treeNodeIndex - 1 ) / 2 * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ] == 0 )
								individualDipolTab_Random[ ( treeNodeIndex - 1 ) / 2 * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ] = 
								individualDipolTab_Random[ treeNodeIndex * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ];
					}							
		}
		//in order set dipol objects currently without propagation from leafs to the root please place ten above lines in a comment		
		#if CUDA_SHARED_MEM_POST_CALC
		for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
			individualDipolTab[ i ] = individualDipolTab_Random[ i ];
		#endif
	}
}
#endif