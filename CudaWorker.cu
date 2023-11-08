#include "CudaWorker.cuh"

#if CUDA_EA_ON

#include <cub/cub.cuh> 
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "device_functions.h"
#include "cuda_runtime_api.h"

#include <stdio.h>

CCudaWorker::CCudaWorker(){
	//dataset related
	dev_datasetTab = NULL;	
	nObjects = 0;
	nAttrs = 0;
	nClasses = 0;
	dev_classTab = NULL;
	dev_predictionTab = NULL;
	timeStats = new CCudaTimeStats();	

	//population related
	#if !FULL_BINARY_TREE_REP	
	dev_populationNodePosTab		= NULL;			
	dev_populationLeftNodePosTab	= NULL;
	dev_populationRightNodePosTab	= NULL;
	dev_populationParentNodePosTab	= NULL;
	#endif
	dev_populationAttrNumTab		= NULL;	
	dev_populationValueTab			= NULL;	
	dev_RT_populationValueTab		= NULL;	
	dev_MT_populationBUpdateTab		= NULL;
	dev_MT_populationAttrsONOFFTab	= NULL;
	dev_individualPosInTab			= NULL;
	maxPopulationTabSize			= 0;

	for (int i = 0; i < CUDA_EA_ON; i++){
		dev_populationAttrNumTabMGPU[i] = NULL;
		dev_populationValueTabMGPU[i]	= NULL;
		dev_individualPosInTabMGPU[i]	= NULL;
	}	

	dev_populationClassDistTab_ScatOverBlocks	= NULL;
	dev_populationDetailedErrTab				= NULL;	
	dev_populationDetailedClassDistTab			= NULL;	
	dev_populationDipolTab_ScatOverBlocks		= NULL;
	dev_populationDipolTab						= NULL;	

	for (int i = 0; i < CUDA_EA_ON; i++){
		dev_populationClassDistTabMGPU_ScatOverBlocks[i] = NULL;
		dev_populationDetailedErrTabMGPU[i]				 = NULL;
		dev_populationDetailedClassDistTabMGPU[i]		 = NULL;
		dev_populationDipolTabMGPU_ScatOverBlocks[i]	 = NULL;
		dev_populationDipolTabMGPU[i]					 = NULL;
	}

	dev_RT_populationErrTab_ScatOverBlocks	 = NULL;
	dev_RT_populationDetailedErrTab			 = NULL;
	dev_RT_populationModelTab_ScatOverBlocks = NULL;
	dev_RT_populationModelTab				 = NULL;

	dev_MT_objectToLeafAssignTab			= NULL;
	dev_MT_objectToLeafAssignTab_out		= NULL;
	dev_MT_objectToLeafAssignIndexTab		= NULL;	
	dev_MT_objectToLeafAssignIndexTab_out	= NULL;
	dev_CUDACUB_Sort_temp_storage			= NULL;	
	dev_MT_populationNObjectsInLeafsTab		= NULL;
	dev_MT_populationStartLeafDataMatrixTab = NULL;
	dev_MT_populationShiftLeafDataMatrixTab = NULL;
	dev_MT_objectsMLRMatrixA				= NULL;
	dev_MT_objectsMLRMatrixb				= NULL;

	mt_cusolverDnH = NULL;
	mt_cublasH = NULL;
	mt_tau = NULL;
	mt_work = NULL;
	mt_devInfo = NULL;
		
	#if CUDA_EA_ON == 1
	device = 0;	
	cudaSetDevice(device);	
	#endif	
}

CCudaWorker::~CCudaWorker(){
	EndSimulation();
	if( timeStats != NULL ) delete timeStats;
}


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
/////////////     DEFINITION OF KERNEL (DEVICE) FUNCTIONS - BEGIN ////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

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
//kernel pre for adaptive
__global__ void dev_CalcPopClassDistAndDipolAtLeafs_Pre_FULL_BINARY_TREE_REP_V2b( DS_REAL *datasetTab, unsigned int *classTab, int nObjects, int nAttrs, int nIndividuals,
															 int *populationAttrNumTab, float *populationValueTab,  int *individualPosInTab, int populationTabSize, int nClasses,
															 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks );
#endif
#endif

#if FULL_BINARY_TREE_REP
//kernel pre for complete - part version
__global__ void dev_CalcPopPartClassDistAndDipolAtLeafs_Pre_V2b( DS_REAL *datasetTab, unsigned int *classTab, int nObjects, int nAttrs, int nIndividuals,
																 int *populationAttrNumTab, float *populationValueTab,  int *individualPosInTab, int populationTabSize, int nClasses,
																 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks );
#else
//kernel pre for compact - part version
__global__ void dev_CalcPopPartClassDistAndDipolAtLeafs_Pre_V2b( DS_REAL *datasetTab, unsigned int *classTab, int nObjects, int nAttrs, int nIndividuals,
																 int *populationAttrNumTab, float *populationValueTab,  int *individualPosInTab, int populationTabSize, int nClasses,
																 int *populationLeftNodePosTab, int *populationRightNodePosTab, int *populationParentNodePosTab,
																 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks );
#if ADDAPTIVE_TREE_REP
//kernel pre for adaptive - part version
__global__ void dev_CalcPopPartClassDistAndDipolAtLeafs_Pre_FULL_BINARY_TREE_REP_V2b( DS_REAL *datasetTab, unsigned int *classTab, int nObjects, int nAttrs, int nIndividuals,
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

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
/////////////     DEFINITION OF KERNEL (DEVICE) FUNCTIONS - END///////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
inline void CCudaWorker::CheckCUDAStatus(cudaError_t cudaStatus, char* errorInfo) {
	if (cudaStatus != cudaSuccess) {
		printf("%s", errorInfo);
		printf("%s\n", cudaGetErrorString(cudaStatus));
		exit(1);
	}
}

inline void CudaSetDevice(int whichGPU){
	cudaSetDevice( whichGPU );
}

int CCudaWorker::GetCurrMaxTreeTabSize(){
	#if FULL_BINARY_TREE_REP
		return pow(2.0, currTreeDepthLimit) - 1;
	#else
		return currNTreeNodesLimit;
	#endif
}

int CCudaWorker::GetDTArrayRepTabSize( CDTreeNode* root ){
	int size;
	#if FULL_BINARY_TREE_REP
		size = pow(2.0, (int)(root->GetDepth())) - 1;
	#else
		#if ADDAPTIVE_TREE_REP
		if(!bCompactOrFullTreeRep)
			size = pow(2.0, (int)(root->GetDepth())) - 1;
		else
		#endif
		size = root -> GetNAllNodes();
	#endif
	
	return size;
}

//init GPU simulation
void CCudaWorker::InitSimulation(const DataSet *dataset, int nIndividuals, eModelType modelType){
	this -> dataset = dataset;
	this -> modelType = modelType;

	ShowSimulationInfo();

	#if FULL_BINARY_TREE_REP
	SetInitTreeDepthLimit( dataset, modelType );
	#else
	currNTreeNodesLimit = INIT_N_TREE_NODES_LIMIT;
	#endif

	//allocate memory at GPU
	#if CUDA_EA_ON > 1
		int nGPUs = 0;
		cudaGetDeviceCount( &nGPUs );
		if( nGPUs < CUDA_EA_ON ){
			printf( "InitSimulation failed - 1  !!!\n" );		
			printf( "Number of physical GPUs is lower than you request (%d)\n", CUDA_EA_ON );
			exit(1);
		}
		omp_set_num_threads( nGPUs );
		SendDatasetToGPUs( dataset );
		#if CUDA_MALLOC_OPTIM_1
		AllocateMemoryPopAndResultsAtGPUs( nIndividuals, GetCurrMaxTreeTabSize() );		
		#endif		
	#else
		SendDatasetToGPU(dataset);
		#if CUDA_MALLOC_OPTIM_1
		AllocateMemoryPopAndResultsAtGPU( nIndividuals, GetCurrMaxTreeTabSize() );
		#endif		
	#endif
}

//end/finish GPU simulation
void CCudaWorker::EndSimulation(){
	#if CUDA_EA_ON > 1
	DeleteDatasetAtGPUs();	
	#else
	DeleteDatasetAtGPU();
	#endif
	
	#if CUDA_MALLOC_OPTIM_1
		#if CUDA_EA_ON > 1
		DeleteMemoryPopAndResultsAtGPUs();
		#else
		DeleteMemoryPopAndResultsAtGPU();
		#endif
	#endif
}

void CCudaWorker::ShowSimulationInfo() {
	#if CUDA_EA_ON > 1
	printf("CUDA: multi-gpu: %d, blocks: %d, threads: %d\n", CUDA_EA_ON, nBlocks, nThreads);
	#else
	printf("CUDA: device: %d, blocks: %d, threads: %d\n", device, nBlocks, nThreads);
	#endif
	
	#if FULL_BINARY_TREE_REP
	printf( "Full binary tree (array) representation\n" );
	#else
		#if ADDAPTIVE_TREE_REP
		printf("Adaptive tree (array) representation - compact or complete/full - switch factor %.4f\n", adaptiveTreeRepSwitch);
		#else
		printf( "Compact (linked) tree (array) representation\n" );
		#endif
	#endif
}

void CCudaWorker::SendDatasetToGPU( const IDataSet *dataset ){
	//timeStats -> WholeTimeBegin();						//time stats
	//timeStats -> WholeDatasetTransportTimeBegin();

	//create tables with necessary dataset (objects) data to send to device (GPU)
	nObjects = dataset -> GetSize();
	nAttrs = dataset -> GetCurrentAttrCount();	
	nClasses = dataset-> GetClassCount();		
	if (dynamic_cast<CDataSetRT*> (const_cast<IDataSet*>(dataset)) != NULL) nClasses = 1;					//for RT, set only locally
	if( nObjects == 0 || nAttrs == 0 || nClasses == 0 ){		
		printf( "SendDatasetToGPU - 1  !!!\n" );		
		printf( "nObjects == 0 || nAttrs == 0 || nClasses == 0\n" );		
		exit( 1 );
	}

	DS_REAL *datasetTab = NULL;
	datasetTab = new DS_REAL[ nObjects * nAttrs ];
	for( int i = 0; i < nObjects; i++ ){
		for( int j = 0; j < nAttrs; j++ )
			datasetTab[ i * nAttrs + j ] = (*dataset)[i] -> GetReal( j, dataset );
	}

	unsigned int *classTab = NULL;
	DS_REAL *predictionTab = NULL;
	int predictionSum = 0;
	if (dynamic_cast<CDataSetRT*> (const_cast<IDataSet*>(dataset)) == NULL){
		//CT
		classTab = new unsigned int[nObjects];
		for (int i = 0; i < nObjects; i++)
			classTab[i] = (*dataset)[i]->GetClassId();
	}
	else{
		//RT
		predictionTab = new DS_REAL[nObjects];
		for (int i = 0; i < nObjects; i++)
			predictionTab[i] = (*dataset)[i]->GetPrediction();
	}

	cudaError_t cudaStatus;
	
	// allocate memory at device for dataset
	//timeStats -> MemoryAllocForDatasetTimeBegin();
	cudaStatus = cudaMalloc( (void**)&dev_datasetTab, nObjects * nAttrs * sizeof( DS_REAL ) );
	if( cudaStatus != cudaSuccess ){
		printf( "SendDatasetToGPU - cudaMalloc failed - 2 !!!" );		
		exit( 1 );
	}

	// allocate memory at device for class id (CT) or prediction value (RT)
	if (dynamic_cast<CDataSetRT*> (const_cast<IDataSet*>(dataset)) == NULL)
		cudaStatus = cudaMalloc( (void**)&dev_classTab, nObjects * sizeof( unsigned int ) );	//CT
	else
		cudaStatus = cudaMalloc((void**)&dev_predictionTab, nObjects * sizeof(DS_REAL));			//RT
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

	// send class id (CT) or prediction value (RT) from host to device
	if (dynamic_cast<CDataSetRT*> (const_cast<IDataSet*>(dataset)) == NULL)
		cudaStatus = cudaMemcpy( dev_classTab, classTab, nObjects * sizeof( unsigned int ), cudaMemcpyHostToDevice );	//CT
	else
		cudaStatus = cudaMemcpy( dev_predictionTab, predictionTab, nObjects * sizeof(DS_REAL), cudaMemcpyHostToDevice );	//RT
	if( cudaStatus != cudaSuccess ) {
        printf( "SendDatasetToGPU - cudaMemcpy failed - 5 !!!" );		
		exit( 1 );
    }
	//timeStats -> SendDatasetToGPUTimeEnd();

	if( datasetTab != NULL )delete[]datasetTab;
	if( classTab != NULL ) delete []classTab;
	if( predictionTab != NULL ) delete []predictionTab;

	//timeStats -> WholeDatasetTransportTimeEnd();		//time stats
	//timeStats -> WholeTimeEnd();						//time stats	
}

void CCudaWorker::SendDatasetToGPUs(const IDataSet *dataset){
	//timeStats->WholeTimeBegin();						//time stats
	//timeStats->WholeDatasetTransportTimeBegin();

	nObjects = dataset->GetSize();
	nAttrs = dataset->GetCurrentAttrCount();
	nClasses = dataset->GetClassCount();
	
	if (dynamic_cast<CDataSetRT*> (const_cast<IDataSet*>(dataset)) != NULL) nClasses = 1;					//for RT, set only locally
	if (nObjects == 0 || nAttrs == 0 || nClasses == 0){
		printf("SendDatasetToGPUs - 1  !!!\n");
		printf("nObjects == 0 || nAttrs == 0 || nClasses == 0\n");
		exit(1);
	}

	//divide dataset into parts
	int nGPUs = CUDA_EA_ON;	
	int maxNObjectsPerGPU = nObjects / nGPUs;
	int nObjectsCurrGPU;
	if (nObjects % nGPUs > 0) maxNObjectsPerGPU++;
	
	//allocate memory for attributes
	DS_REAL *datasetTabMGPU = NULL;
	datasetTabMGPU = new DS_REAL[maxNObjectsPerGPU * nAttrs];

	//allocate memory for a decision
	unsigned int *classTabMGPU = NULL;
	DS_REAL *predictionTabMGPU = NULL;
	int predictionSum = 0;
	if (dynamic_cast<CDataSetRT*> (const_cast<IDataSet*>(dataset)) == NULL)		//CT		
		classTabMGPU = new unsigned int[maxNObjectsPerGPU];
	else																		//	RT/MT
		predictionTabMGPU = new DS_REAL[maxNObjectsPerGPU];

	//spread/send dataset over GPUs
	cudaError_t cudaStatus;
	int beginObjectIdx, endObjectIdx;

	for (int k = 0; k < nGPUs; k++){
		cudaSetDevice( k );

		beginObjectIdx = GetBeginDatasetObjectIndexMGPU(k);
		endObjectIdx = GetEndDatasetObjectIndexMGPU(k);

		nObjectsCurrGPU = endObjectIdx - beginObjectIdx + 1;
		nObjectsMGPU[k] = nObjectsCurrGPU;
				
		for (int i = beginObjectIdx, t=0; i < endObjectIdx; i++, t++)
			for (int j = 0; j < nAttrs; j++)
				datasetTabMGPU[t * nAttrs + j] = (*dataset)[i]->GetReal(j, dataset);

		if (dynamic_cast<CDataSetRT*> (const_cast<IDataSet*>(dataset)) == NULL){	//CT			
			for (int i = beginObjectIdx, t = 0; i < endObjectIdx; i++, t++)
				classTabMGPU[t] = (*dataset)[i]->GetClassId();
		}
		else{																		//RT/MT			
			for (int i = beginObjectIdx, t = 0; i < endObjectIdx; i++, t++)
				predictionTabMGPU[t] = (*dataset)[i]->GetPrediction();
		}

		// allocate memory at device for dataset
		//timeStats -> MemoryAllocForDatasetTimeBegin();
		cudaStatus = cudaMalloc((void**)&(dev_datasetTabMGPU[k]), nObjectsCurrGPU * nAttrs * sizeof(DS_REAL));
		CheckCUDAStatus(cudaStatus, "SendDatasetToGPUs - cudaMalloc failed - 2 !!!");

		// allocate memory at device for class id (CT) or prediction value (RT)
		if (dynamic_cast<CDataSetRT*> (const_cast<IDataSet*>(dataset)) == NULL)
			cudaStatus = cudaMalloc((void**)&dev_classTabMGPU[k], nObjectsCurrGPU * sizeof(unsigned int));			//CT
		else
			cudaStatus = cudaMalloc((void**)&dev_predictionTabMGPU[k], nObjectsCurrGPU * sizeof(DS_REAL));			//RT
		CheckCUDAStatus(cudaStatus, "SendDatasetToGPUs - cudaMalloc failed - 3 !!!");
		//timeStats->MemoryAllocForDatasetTimeEnd();


		// send dataset from host to device
		//timeStats->SendDatasetToGPUTimeBegin();
		cudaStatus = cudaMemcpy(dev_datasetTabMGPU[k], datasetTabMGPU, nObjectsCurrGPU * nAttrs * sizeof(DS_REAL), cudaMemcpyHostToDevice);
		CheckCUDAStatus(cudaStatus, "SendDatasetToGPUs - cudaMemcpy failed - 4 !!!");
		

		// send class id (CT) or prediction value (RT) from host to device
		if (dynamic_cast<CDataSetRT*> (const_cast<IDataSet*>(dataset)) == NULL)
			cudaStatus = cudaMemcpy(dev_classTabMGPU[k], classTabMGPU, nObjectsCurrGPU * sizeof(unsigned int), cudaMemcpyHostToDevice);	//CT
		else
			cudaStatus = cudaMemcpy(dev_predictionTabMGPU[k], predictionTabMGPU, nObjectsCurrGPU * sizeof(DS_REAL), cudaMemcpyHostToDevice);	//RT
		CheckCUDAStatus(cudaStatus, "SendDatasetToGPUs - cudaMemcpy failed - 5 !!!");
		//timeStats->SendDatasetToGPUTimeEnd();

		printf( "GPU %d received object from %d to %d (in total %d objects)\n", k + 1, beginObjectIdx + 1, endObjectIdx + 1, nObjectsCurrGPU );
	}
	
	if (datasetTabMGPU != NULL) delete[]datasetTabMGPU;
	if (classTabMGPU != NULL) delete[]classTabMGPU;
	if (predictionTabMGPU != NULL) delete[]predictionTabMGPU;

	//timeStats->WholeDatasetTransportTimeEnd();		//time stats
	//timeStats->WholeTimeEnd();						//time stats
}

//deallocate dataset memory at GPU
void CCudaWorker::DeleteDatasetAtGPU(){
	if (dev_datasetTab != NULL) cudaFree(dev_datasetTab);
	if (dev_classTab != NULL) cudaFree(dev_classTab);
	if (dev_predictionTab != NULL) cudaFree(dev_predictionTab);
}

//deallocate dataset memory at GPUs
void CCudaWorker::DeleteDatasetAtGPUs(){
	int nGPUs = CUDA_EA_ON;
	for (int k = 0; k < nGPUs; k++){
		cudaSetDevice(k);

		if (dev_datasetTabMGPU[k]	 != NULL) cudaFree(dev_datasetTabMGPU[k]);
		if (dev_classTabMGPU[k]		 != NULL) cudaFree(dev_classTabMGPU[k]);
		if (dev_predictionTabMGPU[k] != NULL) cudaFree(dev_predictionTabMGPU[k]);
	}	
}

inline int CCudaWorker::GetBeginDatasetObjectIndexMGPU(int whichGPU){
	if (nObjects == 0){
		printf("GetBeginDatasetObject - 1  !!!\n");
		printf("nObjects == 0");
		exit(1);
	}

	if (whichGPU < 0) return 0;

	int nGPUs = CUDA_EA_ON;
	int index;
	int nObjectsPerGPU = nObjects / nGPUs;

	index = whichGPU * nObjectsPerGPU;
	
	if (whichGPU < nObjects % nGPUs)
		index += whichGPU;
	else
		index += nObjects % nGPUs;

	return index;
}

int CCudaWorker::GetEndDatasetObjectIndexMGPU(int whichGPU){
	if (nObjects == 0){
		printf("GetBeginDatasetObject - 1  !!!\n");
		printf("nObjects == 0");
		exit(1);
	}

	if (whichGPU < 0) return 0;

	int nGPUs = CUDA_EA_ON;
	if (whichGPU == (nGPUs - 1))
		return nObjects - 1;
	else
		whichGPU++;

	int index;
	int nObjectsPerGPU = nObjects / nGPUs;

	index = whichGPU * nObjectsPerGPU;

	if (whichGPU < nObjects % nGPUs)
		index += whichGPU;
	else
		index += nObjects % nGPUs;

	return index - 1;
}

#if CUDA_MALLOC_OPTIM_1
void CCudaWorker::AllocateMemoryPopAndResultsAtGPU(int nIndividuals, int maxTreeTabSize){

	//timeStats->WholeTimeBegin();						//time stats
	//timeStats->MemoryAllocDeallocTimeBegin();			//time stats	
	//allocate memory at device for population
	cudaError_t cudaStatus;

	maxPopulationTabSize = maxTreeTabSize * nIndividuals;
	
	//allocate memory for the population
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
	if (dynamic_cast<CDataSetRT*> (const_cast<IDataSet*>(dataset)) == NULL){
		cudaStatus = cudaMalloc((void**)&dev_populationValueTab, maxPopulationTabSize * sizeof(float));
		CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - CT - 7 !!!\n");

		//allocate memory for the results
		cudaStatus = cudaMalloc((void**)&dev_populationClassDistTab_ScatOverBlocks, nBlocks * maxPopulationTabSize * nClasses * sizeof(unsigned int));
		CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - CT - 8 !!!\n");

		//allocate memory for the results
		cudaStatus = cudaMalloc((void**)&dev_populationDetailedClassDistTab, maxPopulationTabSize * nClasses * sizeof(unsigned int));
		CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - CT - 9 !!!\n");

		//allocate memory for the results
		cudaStatus = cudaMalloc((void**)&dev_populationDetailedErrTab, maxPopulationTabSize * N_INFO_CORR_ERR * sizeof(unsigned int));
		CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - CT - 10 !!!\n");

		//allocate memory for the results
		cudaStatus = cudaMalloc((void**)&dev_populationDipolTab_ScatOverBlocks, nBlocks * maxPopulationTabSize * nClasses * (N_DIPOL_OBJECTS + 1) * sizeof(unsigned int));
		CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - CT - 11 !!!\n");
		
		//allocate memory for the results
		cudaStatus = cudaMalloc((void**)&dev_populationDipolTab, maxPopulationTabSize * nClasses * N_DIPOL_OBJECTS * sizeof(unsigned int));
		CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - CT - 12 !!!\n");		
	}	
	else{//RT, MT
		cudaStatus = cudaMalloc((void**)&dev_RT_populationValueTab, maxPopulationTabSize * sizeof(RT_TREE_TRESHOLD_REAL));
		CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - RT/MT - 13 !!!\n");

		//allocate memory for the results
		cudaStatus = cudaMalloc((void**)&dev_RT_populationErrTab_ScatOverBlocks, nBlocks * maxPopulationTabSize * sizeof(RT_REAL));
		CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - RT/MT - 14 !!!\n");

		//allocate memory for the results
		cudaStatus = cudaMalloc((void**)&dev_RT_populationDetailedErrTab, maxPopulationTabSize * sizeof(RT_REAL));
		CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - RT/MT - 15 !!!\n");

		//allocate memory for the results
		cudaStatus = cudaMalloc((void**)&dev_populationDipolTab_ScatOverBlocks, nBlocks * maxPopulationTabSize * MT_N_DIPOL_TMP * sizeof(unsigned int));
		CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - RT/MT - 16 !!!\n");

		//allocate memory for the results
		cudaStatus = cudaMalloc((void**)&dev_populationDipolTab, maxPopulationTabSize * (RT_N_DIPOLS * 2) * sizeof(unsigned int));
		CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - RT/MT - 17 !!!\n");

		if (modelType == NONE_MODEL){	//RT
			cudaStatus = cudaMalloc((void**)&dev_RT_populationModelTab_ScatOverBlocks, nBlocks * maxPopulationTabSize * RT_N_MODEL_VALUES * sizeof(RT_REAL));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - RT - 18 !!!\n");

			cudaStatus = cudaMalloc((void**)&dev_RT_populationModelTab, maxPopulationTabSize * RT_N_MODEL_VALUES * sizeof(RT_REAL));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - RT - 19 !!!\n");
		}
		else{//MT
			assert(nObjects > 0);
		
			cudaStatus = cudaMalloc((void**)&dev_MT_populationBUpdateTab, maxPopulationTabSize * sizeof(char));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - MT - 20 !!!\n");

			#if !ALL_ATTRIBUTES_IN_MODELS
			cudaStatus = cudaMalloc((void**)&dev_MT_populationAttrsONOFFTab, maxPopulationTabSize * nAttrs * sizeof(MT_ATTRIBUTES_ONOFF_TYPE));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - MT - 21 !!!\n");			
			#endif

			cudaStatus = cudaMalloc((void**)&dev_MT_objectToLeafAssignTab, nObjects * sizeof(unsigned int));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - MT - 22 !!!\n");

			cudaStatus = cudaMalloc((void**)&dev_MT_objectToLeafAssignTab_out, nObjects * sizeof(unsigned int));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - MT - 23 !!!\n");

			cudaStatus = cudaMalloc((void**)&dev_MT_objectToLeafAssignIndexTab, nObjects * sizeof(unsigned int));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - MT - 24 !!!\n");

			cudaStatus = cudaMalloc((void**)&dev_MT_objectToLeafAssignIndexTab_out, nObjects * sizeof(unsigned int));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - MT - 25 !!!\n");

			cudaStatus = cudaMalloc((void**)&dev_MT_populationNObjectsInLeafsTab, maxPopulationTabSize * sizeof(unsigned int));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - MT - 26 !!!\n");

			cudaStatus = cudaMalloc((void**)&dev_MT_populationStartLeafDataMatrixTab, maxPopulationTabSize * sizeof(unsigned int));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - MT - 27 !!!\n");

			cudaStatus = cudaMalloc((void**)&dev_MT_populationShiftLeafDataMatrixTab, maxPopulationTabSize * sizeof(unsigned int));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - MT - 28 !!!\n");			

			cudaStatus = cudaMalloc((void**)&dev_MT_objectsMLRMatrixA, nObjects * (nAttrs + 1) * sizeof(RT_REAL));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - MT - 29 !!!\n");

			cudaStatus = cudaMalloc((void**)&dev_MT_objectsMLRMatrixb, nObjects * sizeof(RT_REAL));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - MT - 30 !!!\n");

			mt_cusolverDnH = 0;
			mt_cublasH = 0;
			cusolverDnCreate(&mt_cusolverDnH);
			cublasCreate(&mt_cublasH);

			mt_tau = 0;
			mt_work = 0;
			mt_Lwork_max = 0;
			mt_devInfo = 0;
			mt_Lwork = 0;
			mt_min_m_n = nAttrs+1;
			cudaStatus = cudaMalloc(&mt_tau, mt_min_m_n*sizeof(float));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - MT - 31 !!!\n");
			cudaStatus = cudaMalloc(&mt_devInfo, sizeof(int));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - MT - 32 !!!\n");
		}
	}

	//timeStats->MemoryAllocDeallocTimeEnd();			//time stats
	//timeStats->WholeTimeEnd();						//time stats
}

void CCudaWorker::AllocateMemoryPopAndResultsAtGPUs(int nIndividuals, int maxTreeTabSize){
	//timeStats->WholeTimeBegin();						//time stats
	//timeStats->MemoryAllocDeallocTimeBegin();			//time stats	
	//allocate memory at device for population
	cudaError_t cudaStatus;

	maxPopulationTabSize = maxTreeTabSize * nIndividuals;

	int nGPUs = CUDA_EA_ON;
	for (int i = 0; i < nGPUs; i++){
		cudaSetDevice(i);

		cudaStatus = cudaMalloc((void**)&dev_populationAttrNumTabMGPU[i], maxPopulationTabSize * sizeof(int));
		CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - 1 !!!\n");

		cudaStatus = cudaMalloc((void**)&dev_individualPosInTabMGPU[i], nIndividuals * sizeof(int));
		CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - 2 !!!\n");

		//allocate memory at device for results

		//CT
		if (dynamic_cast<CDataSetRT*> (const_cast<IDataSet*>(dataset)) == NULL){
			cudaStatus = cudaMalloc((void**)&dev_populationValueTabMGPU[i], maxPopulationTabSize * sizeof(float));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - 3 !!!\n");

			cudaStatus = cudaMalloc((void**)&dev_populationClassDistTabMGPU_ScatOverBlocks[i], nBlocks * maxPopulationTabSize * nClasses * sizeof(unsigned int));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - CT - 4 !!!\n");

			cudaStatus = cudaMalloc((void**)&dev_populationDetailedClassDistTabMGPU[i], maxPopulationTabSize * nClasses * sizeof(unsigned int));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - CT - 5 !!!\n");

			cudaStatus = cudaMalloc((void**)&dev_populationDetailedErrTabMGPU[i], maxPopulationTabSize * N_INFO_CORR_ERR * sizeof(unsigned int));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - CT - 6 !!!\n");

			cudaStatus = cudaMalloc((void**)&dev_populationDipolTabMGPU_ScatOverBlocks[i], nBlocks * maxPopulationTabSize * nClasses * (N_DIPOL_OBJECTS + 1) * sizeof(unsigned int));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - CT - 7 !!!\n");

			cudaStatus = cudaMalloc((void**)&dev_populationDipolTabMGPU[i], maxPopulationTabSize * nClasses * N_DIPOL_OBJECTS * sizeof(unsigned int));
			CheckCUDAStatus(cudaStatus, "InitSimulation - cudaMalloc failed - CT - 8 !!!\n");
		}		
	}

	//timeStats->MemoryAllocDeallocTimeEnd();			//time stats
	//timeStats->WholeTimeEnd();						//time stats
}

void CCudaWorker::DeleteMemoryPopAndResultsAtGPU(){
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

	//RT, MT
	if (dev_RT_populationValueTab != NULL) cudaFree(dev_RT_populationValueTab);	
	if (dev_RT_populationErrTab_ScatOverBlocks != NULL) cudaFree(dev_RT_populationErrTab_ScatOverBlocks);
	if (dev_RT_populationDetailedErrTab != NULL) cudaFree(dev_RT_populationDetailedErrTab);	
	dev_RT_populationValueTab = NULL;
	dev_RT_populationDetailedErrTab = NULL;
	dev_RT_populationErrTab_ScatOverBlocks = NULL;
	
	//RT
	if (dev_RT_populationModelTab_ScatOverBlocks != NULL) cudaFree(dev_RT_populationModelTab_ScatOverBlocks);
	if (dev_RT_populationModelTab != NULL) cudaFree(dev_RT_populationModelTab);	
	dev_RT_populationModelTab_ScatOverBlocks = NULL;
	dev_RT_populationModelTab = NULL;

	//MT
	if (dev_MT_populationBUpdateTab != NULL) cudaFree(dev_MT_populationBUpdateTab);
	if (dev_MT_populationAttrsONOFFTab != NULL) cudaFree(dev_MT_populationAttrsONOFFTab);
	if (dev_MT_objectToLeafAssignTab != NULL) cudaFree(dev_MT_objectToLeafAssignTab);
	if (dev_MT_objectToLeafAssignTab_out != NULL) cudaFree(dev_MT_objectToLeafAssignTab_out);
	if (dev_MT_objectToLeafAssignIndexTab != NULL) cudaFree(dev_MT_objectToLeafAssignIndexTab);	
	if (dev_MT_objectToLeafAssignIndexTab_out != NULL) cudaFree(dev_MT_objectToLeafAssignIndexTab_out);
	dev_MT_populationBUpdateTab = NULL;
	dev_MT_populationAttrsONOFFTab = NULL;
	dev_MT_objectToLeafAssignTab = NULL;
	dev_MT_objectToLeafAssignTab_out = NULL;
	dev_MT_objectToLeafAssignIndexTab = NULL;
	dev_MT_objectToLeafAssignIndexTab_out = NULL;

	if (dev_MT_populationNObjectsInLeafsTab != NULL) cudaFree(dev_MT_populationNObjectsInLeafsTab);
	if (dev_MT_populationStartLeafDataMatrixTab != NULL) cudaFree(dev_MT_populationStartLeafDataMatrixTab);
	if (dev_MT_populationShiftLeafDataMatrixTab != NULL) cudaFree(dev_MT_populationShiftLeafDataMatrixTab);
	if (dev_MT_objectsMLRMatrixA != NULL) cudaFree(dev_MT_objectsMLRMatrixA);
	if (dev_MT_objectsMLRMatrixb != NULL) cudaFree(dev_MT_objectsMLRMatrixb);
	dev_MT_populationNObjectsInLeafsTab = NULL;
	dev_MT_populationStartLeafDataMatrixTab = NULL;
	dev_MT_populationShiftLeafDataMatrixTab = NULL;
	dev_MT_objectsMLRMatrixA = NULL;
	dev_MT_objectsMLRMatrixb = NULL;

	if (mt_cusolverDnH != NULL)  cusolverDnDestroy(mt_cusolverDnH);
	if (mt_cublasH != NULL) cublasDestroy(mt_cublasH);
	mt_cusolverDnH = NULL;
	mt_cublasH = NULL;

	if (mt_tau != NULL) cudaFree(mt_tau);
	if (mt_devInfo != NULL) cudaFree(mt_devInfo);	
	if (mt_work != NULL) cudaFree(mt_work);
	mt_tau = NULL;
	mt_devInfo = NULL;	
	mt_work = NULL;	

	//timeStats->MemoryAllocDeallocTimeEnd();	//time stats
	//timeStats->WholeTimeEnd();				//time stats
}

//ready only for CT
void CCudaWorker::DeleteMemoryPopAndResultsAtGPUs(){
	//timeStats->WholeTimeBegin();				//time stats
	//timeStats->MemoryAllocDeallocTimeBegin();	//time stats

	int nGPUs = CUDA_EA_ON;
	for (int i = 0; i < nGPUs; i++){
		cudaSetDevice(i);

		if (dev_populationAttrNumTabMGPU[i] != NULL) cudaFree(dev_populationAttrNumTabMGPU[i]);
		if (dev_individualPosInTabMGPU[i] != NULL) cudaFree(dev_individualPosInTabMGPU[i]);
		dev_populationAttrNumTabMGPU[i] = NULL;
		dev_individualPosInTabMGPU[i] = NULL;

		//CT	
		if (dev_populationValueTabMGPU[i] != NULL) cudaFree(dev_populationValueTabMGPU[i]);
		if (dev_populationClassDistTabMGPU_ScatOverBlocks[i] != NULL) cudaFree(dev_populationClassDistTabMGPU_ScatOverBlocks[i]);
		if (dev_populationDetailedClassDistTabMGPU[i] != NULL) cudaFree(dev_populationDetailedClassDistTabMGPU[i]);
		if (dev_populationDetailedErrTabMGPU[i] != NULL) cudaFree(dev_populationDetailedErrTabMGPU[i]);
		if (dev_populationDipolTabMGPU_ScatOverBlocks[i] != NULL) cudaFree(dev_populationDipolTabMGPU_ScatOverBlocks[i]);		
		if (dev_populationDipolTabMGPU[i] != NULL) cudaFree(dev_populationDipolTabMGPU[i]);
		dev_populationValueTabMGPU[i] = NULL;		
		dev_populationClassDistTabMGPU_ScatOverBlocks[i] = NULL;
		dev_populationDetailedClassDistTabMGPU[i] = NULL;
		dev_populationDetailedErrTabMGPU[i] = NULL;		
		dev_populationDipolTabMGPU_ScatOverBlocks[i] = NULL;
		dev_populationDipolTabMGPU[i] = NULL;
	}

	//timeStats->MemoryAllocDeallocTimeEnd();	//time stats	
	//timeStats->WholeTimeEnd();				//time stats
}
#endif

//classification trees
unsigned int* CCudaWorker::CalcIndivDetailedErrAndClassDistAndDipol_V2b( CDTreeNode* root, unsigned int** populationDetailedClassDistTab, unsigned int** populationDipolTab ){
	cudaError_t cudaStatus;

	//timeStats -> WholeTimeBegin();						//time stats

	//timeStats -> DataReorganizationTimeBegin();			//time stats
	//////////////////////////////////////////////////////////////////////////	
	//alloc population
	int nIndividuals = 1;
	int *individualPosInTab = new int[ nIndividuals ];
	int populationTabSize = 0;	
	
	for( int i = 0; i < nIndividuals; i++ ){
		individualPosInTab[ i ] = populationTabSize;
		populationTabSize += GetDTArrayRepTabSize( root );		
	}
	int *populationAttrNumTab = new int[ populationTabSize ];
	float *populationValueTab = new float[ populationTabSize ];	
	for( int i = 0; i < populationTabSize; i++ ){
		populationAttrNumTab[ i ] = -2;
		populationValueTab	[ i ] = 0.0;		
	}
	#if !FULL_BINARY_TREE_REP	
	int *populationNodePosTab = NULL;
	int *populationLeftNodePosTab = NULL;
	int *populationRightNodePosTab = NULL;
	int *populationParentNodePosTab = NULL;
	if (bCompactOrFullTreeRep) {
		populationNodePosTab = new int[3*populationTabSize];
		populationLeftNodePosTab = populationNodePosTab;
		populationRightNodePosTab = populationNodePosTab + populationTabSize;
		populationParentNodePosTab = populationNodePosTab + 2*populationTabSize;

		for (int i = 0; i < populationTabSize; i++) {
			populationLeftNodePosTab[i] = -1;
			populationRightNodePosTab[i] = -1;
			populationParentNodePosTab[i] = -1;
		}		
	}
	#endif
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////	
	//pack population
	for( int i = 0; i < nIndividuals; i++ ){
		int shift = individualPosInTab[i];
		#if FULL_BINARY_TREE_REP
		CopyBinaryTreeToTable( root, populationAttrNumTab + shift, populationValueTab + shift, 0 );
		#else
		if(bCompactOrFullTreeRep){
			CopyBinaryTreeToTable( root, populationAttrNumTab + shift, populationValueTab + shift, 0,
				                   populationLeftNodePosTab + 3*shift, populationRightNodePosTab + 3*shift, populationParentNodePosTab + 3*shift );
		}
		else
			CopyBinaryTreeToTable_FULL_BINARY_TREE_REP(root, populationAttrNumTab + shift, populationValueTab + shift, 0);		

		#if DEBUG_COMPACT_TREE
		if (bCompactOrFullTreeRep)	{ nCompactRepDTsRun++;	nCompactRepDTsSum++;}
		else						{ nFullRepDTsRun++;		nFullRepDTsSum++;}
		#endif
		#endif
	}
	//timeStats -> DataReorganizationTimeEnd();			//time stats
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	#if !CUDA_MALLOC_OPTIM_1
	//allocate memory at device for population
	//timeStats -> MemoryAllocDeallocTimeBegin();			//time stats

	//send population
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
	#if DEBUG_COMPACT_TREE
	popHostDeviceSendBytesRun += populationTabSize * ( sizeof(int) + sizeof(float) );
	popHostDeviceSendBytesSum += populationTabSize * ( sizeof(int) + sizeof(float) );
	#endif
	#if !FULL_BINARY_TREE_REP
	if (bCompactOrFullTreeRep) {
		cudaStatus = cudaMemcpy(dev_populationNodePosTab, populationNodePosTab, 3 * populationTabSize * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf("CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 15, 16, 17 !!!\n");
			printf("%s\n", cudaGetErrorString(cudaStatus));
			exit(EXIT_FAILURE);
		}
		dev_populationLeftNodePosTab = dev_populationNodePosTab;
		dev_populationRightNodePosTab = dev_populationNodePosTab + populationTabSize;
		dev_populationParentNodePosTab = dev_populationNodePosTab + 2 * populationTabSize;		
		#if DEBUG_COMPACT_TREE
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
		delete[]populationNodePosTab;
	}
	#endif
	//timeStats -> DataReorganizationTimeEnd();			//time stats
	//////////////////////////////////////////////////////////////////////////
	

	//////////////////////////////////////////////////////////////////////////	
	//kernel fitness pre
	//timeStats -> CalcTimeBegin();						//time stats	
	#if FULL_BINARY_TREE_REP
	dev_CalcPopClassDistAndDipolAtLeafs_Pre_V2b<<< nBlocks, nThreads >>>( dev_datasetTab, dev_classTab, nObjects, nAttrs, nIndividuals, 													//datset
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
	
	//kernel fitness post
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
	//transfer results from the GPU to the CPU
	//timeStats -> DataTransferFromGPUTimeBegin();		//time stats
	unsigned int* populationDetailedErrTab = new unsigned int[ populationTabSize * N_INFO_CORR_ERR ];
	cudaStatus = cudaMemcpy( populationDetailedErrTab, dev_populationDetailedErrTab, populationTabSize * N_INFO_CORR_ERR * sizeof( unsigned int ), cudaMemcpyDeviceToHost );
	if( cudaStatus != cudaSuccess ){
        printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 21 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );	
		exit( EXIT_FAILURE );
    }

	(*populationDetailedClassDistTab) = new unsigned int[ populationTabSize * nClasses ];
	cudaStatus = cudaMemcpy( (*populationDetailedClassDistTab), dev_populationDetailedClassDistTab, populationTabSize * nClasses * sizeof( unsigned int ), cudaMemcpyDeviceToHost );
	if( cudaStatus != cudaSuccess ){
        printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 22 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );		
		exit( EXIT_FAILURE );
	}

	(*populationDipolTab) = new unsigned int[ populationTabSize * nClasses * N_DIPOL_OBJECTS ];
	cudaStatus = cudaMemcpy( (*populationDipolTab), dev_populationDipolTab, populationTabSize * nClasses * N_DIPOL_OBJECTS * sizeof( unsigned int ), cudaMemcpyDeviceToHost );
	if( cudaStatus != cudaSuccess ){
        printf( "CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 23 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );		
		exit( EXIT_FAILURE );
	}	
	//timeStats -> DataTransferFromGPUTimeEnd();			//time stats
	#if DEBUG_COMPACT_TREE	
	resultDeviceHostSendBytesRun += populationTabSize * ( (N_INFO_CORR_ERR * sizeof(unsigned int) + nClasses * sizeof(unsigned int) + nClasses * N_DIPOL_OBJECTS * sizeof(unsigned int) ) );
	resultDeviceHostSendBytesSum += populationTabSize * ( (N_INFO_CORR_ERR * sizeof(unsigned int) + nClasses * sizeof(unsigned int) + nClasses * N_DIPOL_OBJECTS * sizeof(unsigned int) ) );
	#endif
	//////////////////////////////////////////////////////////////////////////

	//cleaning
	//////////////////////////////////////////////////////////////////////////	
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
		cudaFree( dev_populationNodePosTab );		
	}
	#endif
	timeStats -> MemoryAllocDeallocTimeEnd();			//time stats
	#endif
	//////////////////////////////////////////////////////////////////////////
	
	//timeStats -> WholeTimeEnd();						//time stats

	return populationDetailedErrTab;
}

//classification trees - for a tree part
unsigned int* CCudaWorker::CalcIndivPartDetailedErrAndClassDistAndDipol_V2b( CDTreeNode* startNode, CDTreeNode* root, unsigned int** populationDetailedClassDistTab, unsigned int** populationDipolTab, int& startNodeTabIndex ){
	
	cudaError_t cudaStatus;

	//timeStats -> WholeTimeBegin();						//time stats

	//timeStats -> DataReorganizationTimeBegin();			//time stats
	//////////////////////////////////////////////////////////////////////////
	//alloc population
	int nIndividuals = 1;
	int *individualPosInTab = new int[ nIndividuals ];
	int populationTabSize = 0;	
	
	for( int i = 0; i < nIndividuals; i++ ){
		individualPosInTab[ i ] = populationTabSize;
		populationTabSize += GetDTArrayRepTabSize( root );
	}
	int *populationAttrNumTab = new int[ populationTabSize ];
	float *populationValueTab = new float[ populationTabSize ];	
	for( int i = 0; i < populationTabSize; i++ ){
		populationAttrNumTab[ i ] = -2;
		populationValueTab[ i ] = 0.0;		
	}
	#if !FULL_BINARY_TREE_REP
	int *populationNodePosTab = NULL;
	int *populationLeftNodePosTab = NULL;
	int *populationRightNodePosTab = NULL;
	int *populationParentNodePosTab = NULL;
	if (bCompactOrFullTreeRep) {
		populationNodePosTab = new int[3 * populationTabSize];
		populationLeftNodePosTab = populationNodePosTab;
		populationRightNodePosTab = populationNodePosTab + populationTabSize;
		populationParentNodePosTab = populationNodePosTab + 2 * populationTabSize;
	
		for (int i = 0; i < populationTabSize; i++) {
			populationLeftNodePosTab[i] = -1;
			populationRightNodePosTab[i] = -1;
			populationParentNodePosTab[i] = -1;
		}
	}
	#endif
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////	
	//pack population
	for( int i = 0; i < nIndividuals; i++ )
		#if FULL_BINARY_TREE_REP
		startNodeTabIndex = CopyBinaryTreePartToTable( startNode, root, populationAttrNumTab + individualPosInTab[ i ], populationValueTab + individualPosInTab[ i ], 0 );				
		#endif
	//timeStats -> DataReorganizationTimeEnd();			//time stats
	//////////////////////////////////////////////////////////////////////////
	
	//////////////////////////////////////////////////////////////////////////
	#if !CUDA_MALLOC_OPTIM_1
	//allocate memory at device for population
	//timeStats->MemoryAllocDeallocTimeBegin();			//time stats

	//send population
	cudaStatus = cudaMalloc( (void**)&dev_populationAttrNumTab, populationTabSize * sizeof( int ) );
	if( cudaStatus != cudaSuccess ){
		printf( "CalcIndivPartDetailedErrAndClassDistAndDipol_V2b - cudaMalloc failed - 1 !!!\n" );
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
	}
	cudaStatus = cudaMalloc( (void**)&dev_populationValueTab, populationTabSize * sizeof( float ) );
	if( cudaStatus != cudaSuccess ){
		printf( "CalcIndivPartDetailedErrAndClassDistAndDipol_V2b - cudaMalloc failed - 2 !!!\n" );
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
	}
	cudaStatus = cudaMalloc( (void**)&dev_individualPosInTab, nIndividuals * sizeof( int ) );
	if( cudaStatus != cudaSuccess ){
		printf( "CalcIndivPartDetailedErrAndClassDistAndDipol_V2b - cudaMalloc failed - 3 !!!\n" );
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

	cudaStatus = cudaMalloc( (void**)&dev_populationClassDistTab_ScatOverBlocks, nBlocks * populationTabSize * nClasses * sizeof( unsigned int ) );
	if( cudaStatus != cudaSuccess ){
		printf( "CalcIndivPartDetailedErrAndClassDistAndDipol_V2b - cudaMalloc failed - 4 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
	}	

	cudaStatus = cudaMalloc( (void**)&dev_populationDetailedErrTab,  populationTabSize * N_INFO_CORR_ERR * sizeof( unsigned int ) );
	if( cudaStatus != cudaSuccess ){
		printf( "CalcIndivPartDetailedErrAndClassDistAndDipol_V2b - cudaMalloc failed - 5 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
	}

	cudaStatus = cudaMalloc( (void**)&dev_populationDetailedClassDistTab, populationTabSize * nClasses * sizeof( unsigned int ) );
	if( cudaStatus != cudaSuccess ){
		printf( "CalcIndivPartDetailedErrAndClassDistAndDipol_V2b - cudaMalloc failed - 6 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
	}

	cudaStatus = cudaMalloc( (void**)&dev_populationDipolTab_ScatOverBlocks, nBlocks * populationTabSize * nClasses * (N_DIPOL_OBJECTS + 1) * sizeof( unsigned int ) );
	if( cudaStatus != cudaSuccess ){
		printf( "CalcIndivPartDetailedErrAndClassDistAndDipol_V2b - cudaMalloc failed - 7 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
	}

	cudaStatus = cudaMalloc( (void**)&dev_populationDipolTab, populationTabSize * nClasses * N_DIPOL_OBJECTS * sizeof( unsigned int ) );
	if( cudaStatus != cudaSuccess ){
		printf( "CalcIndivPartDetailedErrAndClassDistAndDipol_V2b - cudaMalloc failed - 8 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
	}
	timeStats->MemoryAllocDeallocTimeEnd();			//time stats
	#endif
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////	
	//transfer population data to device
	//timeStats -> DataTransferToGPUTimeBegin();			//time stats
	cudaStatus = cudaMemcpy( dev_populationAttrNumTab, populationAttrNumTab, populationTabSize * sizeof( int ), cudaMemcpyHostToDevice );
	if( cudaStatus != cudaSuccess ){
        printf( "CalcIndivPartDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 9 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( 1 );
    }
	cudaStatus = cudaMemcpy( dev_populationValueTab, populationValueTab, populationTabSize * sizeof( float ), cudaMemcpyHostToDevice );
	if( cudaStatus != cudaSuccess ){
        printf( "CalcIndivPartDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 10 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( 1 );
    }
	cudaStatus = cudaMemcpy( dev_individualPosInTab, individualPosInTab, nIndividuals * sizeof( int ), cudaMemcpyHostToDevice );
	if( cudaStatus != cudaSuccess ){
        printf( "CalcIndivPartDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 11 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( 1 );
    }
	#if DEBUG_COMPACT_TREE
	popHostDeviceSendBytesRun += populationTabSize * ( sizeof(int) + sizeof(float) );
	popHostDeviceSendBytesSum += populationTabSize * ( sizeof(int) + sizeof(float) );
	#endif
	#if !FULL_BINARY_TREE_REP
	if (bCompactOrFullTreeRep) {
		cudaStatus = cudaMemcpy(dev_populationNodePosTab, populationNodePosTab, 3 * populationTabSize * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf("CalcIndivDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 15, 16, 17 !!!\n");
			printf("%s\n", cudaGetErrorString(cudaStatus));
			exit(EXIT_FAILURE);
		}
		#if DEBUG_COMPACT_TREE
		popHostDeviceSendBytesRun += populationTabSize * ( 3 * sizeof(int) );
		popHostDeviceSendBytesSum += populationTabSize * ( 3 * sizeof(int) );
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
		delete[]populationNodePosTab;
	}
	#endif
	//timeStats -> DataReorganizationTimeEnd();			//time stats
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	//kernel fitness pre
	//timeStats -> CalcTimeBegin();						//time stats	
	#if FULL_BINARY_TREE_REP
	dev_CalcPopPartClassDistAndDipolAtLeafs_Pre_V2b<<< nBlocks, nThreads >>>( dev_datasetTab, dev_classTab, nObjects, nAttrs, nIndividuals, 											//datset
																					  dev_populationAttrNumTab, dev_populationValueTab, dev_individualPosInTab, populationTabSize, nClasses,	//population
																					  dev_populationClassDistTab_ScatOverBlocks, dev_populationDipolTab_ScatOverBlocks );						//results
	#else
	if (bCompactOrFullTreeRep)
		dev_CalcPopPartClassDistAndDipolAtLeafs_Pre_V2b<<< nBlocks, nThreads >>>( dev_datasetTab, dev_classTab, nObjects, nAttrs, nIndividuals, 											//datset
																						  dev_populationAttrNumTab, dev_populationValueTab, dev_individualPosInTab, populationTabSize, nClasses,	//population
																						  dev_populationLeftNodePosTab, dev_populationRightNodePosTab, dev_populationParentNodePosTab,				//DT links
																						  dev_populationClassDistTab_ScatOverBlocks, dev_populationDipolTab_ScatOverBlocks );						//results
	#if ADDAPTIVE_TREE_REP
	else
		dev_CalcPopPartClassDistAndDipolAtLeafs_Pre_FULL_BINARY_TREE_REP_V2b<<< nBlocks, nThreads >>>( dev_datasetTab, dev_classTab, nObjects, nAttrs, nIndividuals, 											//datset
																						dev_populationAttrNumTab, dev_populationValueTab, dev_individualPosInTab, populationTabSize, nClasses,		//population
																						dev_populationClassDistTab_ScatOverBlocks, dev_populationDipolTab_ScatOverBlocks );							//results
	#endif
	#endif
	cudaDeviceSynchronize();

	cudaStatus = cudaGetLastError();
    if( cudaStatus != cudaSuccess ){		
        printf( "CalcIndivPartDetailedErrAndClassDistAndDipol_V2b - dev_CalcPopClassDistAndDipolAtLeafs_Pre_V2b failed - 12 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
        exit( EXIT_FAILURE );
    }
	
	//kernel fitness post
	#if FULL_BINARY_TREE_REP
	dev_CalcPopDetailedErrAndClassDistAndDipol_Post_V2b<<< nIndividuals, nBlocks >>>( dev_individualPosInTab, populationTabSize, nClasses,
								      												   dev_populationClassDistTab_ScatOverBlocks, dev_populationDipolTab_ScatOverBlocks,
																					   dev_populationDetailedErrTab, dev_populationDetailedClassDistTab, dev_populationDipolTab );	
	#else
	if (bCompactOrFullTreeRep)
		dev_CalcPopDetailedErrAndClassDistAndDipol_Post_V2b<<< nIndividuals, nBlocks >>>( dev_individualPosInTab, populationTabSize, nClasses, dev_populationParentNodePosTab,		//DT links
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
        printf( "CalcIndivPartDetailedErrAndClassDistAndDipol_V2b - dev_CalcPopDetailedErrAndClassDistAndDipol_Post_V2b failed - 13 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );	
        exit( EXIT_FAILURE );
    }
	//timeStats -> CalcTimeEnd();							//time stats
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////	
	//transfer results from the GPU to the CPU
	//timeStats -> DataTransferFromGPUTimeBegin();		//time stats
	unsigned int* populationDetailedErrTab = new unsigned int[ populationTabSize * N_INFO_CORR_ERR ];	
	cudaStatus = cudaMemcpy( populationDetailedErrTab, dev_populationDetailedErrTab, populationTabSize * N_INFO_CORR_ERR * sizeof( unsigned int ), cudaMemcpyDeviceToHost );
	if( cudaStatus != cudaSuccess ){
        printf( "CalcIndivPartDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 14 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
    }

	(*populationDetailedClassDistTab) = new unsigned int[ populationTabSize * nClasses ];
	cudaStatus = cudaMemcpy( (*populationDetailedClassDistTab), dev_populationDetailedClassDistTab, populationTabSize * nClasses * sizeof( unsigned int ), cudaMemcpyDeviceToHost );
	if( cudaStatus != cudaSuccess ){
        printf( "CalcIndivPartDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 15 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
	}

	(*populationDipolTab) = new unsigned int[ populationTabSize * nClasses * N_DIPOL_OBJECTS ];
	cudaStatus = cudaMemcpy( (*populationDipolTab), dev_populationDipolTab, populationTabSize * nClasses * N_DIPOL_OBJECTS * sizeof( unsigned int ), cudaMemcpyDeviceToHost );
	if( cudaStatus != cudaSuccess ){
        printf( "CalcIndivPartDetailedErrAndClassDistAndDipol_V2b - cudaMemcpy failed - 16 !!!\n" );		
		printf( "%s\n", cudaGetErrorString( cudaStatus ) );
		exit( EXIT_FAILURE );
	}
	//timeStats -> DataTransferFromGPUTimeEnd();			//time stats
	#if DEBUG_COMPACT_TREE	
	resultDeviceHostSendBytesRun += populationTabSize * ( (N_INFO_CORR_ERR * sizeof(unsigned int) + nClasses * sizeof(unsigned int) + nClasses * N_DIPOL_OBJECTS * sizeof(unsigned int) ) );
	resultDeviceHostSendBytesSum += populationTabSize * ( (N_INFO_CORR_ERR * sizeof(unsigned int) + nClasses * sizeof(unsigned int) + nClasses * N_DIPOL_OBJECTS * sizeof(unsigned int) ) );
	#endif
		
	//cleaning
	#if !CUDA_MALLOC_OPTIM_1
	//timeStats->MemoryAllocDeallocTimeBegin();			//time stats
	cudaFree( dev_populationAttrNumTab );	
	cudaFree( dev_populationValueTab );	
	cudaFree( dev_individualPosInTab );
	cudaFree( dev_populationClassDistTab_ScatOverBlocks );
	cudaFree( dev_populationDipolTab_ScatOverBlocks );
	cudaFree( dev_populationDetailedClassDistTab );
	cudaFree( dev_populationDetailedErrTab );
	cudaFree( dev_populationDipolTab );
	#if !FULL_BINARY_TREE_REP	
	if (bCompactOrFullTreeRep) {		
		cudaFree(dev_populationNodePosTab);
	}
	#endif
	//timeStats->MemoryAllocDeallocTimeEnd();			//time stats
	#endif
	//////////////////////////////////////////////////////////////////////////

	//timeStats -> WholeTimeEnd();						//time stats

	return populationDetailedErrTab;
}

//fill decision tree by the results obtained by an external device, e.g. GPU: errors, class distribution, dipol objects
int CCudaWorker::FillDTreeByExternalResults( CDTreeNode *node, unsigned int *indivDetailedErrTab, unsigned int *indivDetailedClassDistTab, unsigned int* indivDipolTab, unsigned int index, const IDataSet *pDS ){
	unsigned int errTabIndex = index * N_INFO_CORR_ERR;
	unsigned int classTabIndex = index * nClasses;
	unsigned int dipolTabIndex = index * nClasses * N_DIPOL_OBJECTS;
	
	//set number of samples and error
	node -> SetNMisClassifiedObjects( indivDetailedErrTab[ errTabIndex ] );
	node -> SetNTrainObjects( indivDetailedErrTab[ errTabIndex ] + indivDetailedErrTab[ errTabIndex + 1 ] );
	node -> m_TrainObjs.m_uSize = node -> GetNTrainObjects();

	//set class distribution
	int best = 0;
	for( int i = 0; i < nClasses; i++ ){
		node -> SetNClassObjects( i, indivDetailedClassDistTab[ classTabIndex + i ] );
		if( indivDetailedClassDistTab[ classTabIndex + i ] > indivDetailedClassDistTab[ classTabIndex + best ] )
			best = i;
	}
	node -> m_pIAClass -> m_AttrValue.u_iNom = best;
	
	node -> m_TrainObjs.ClearObjects4Dipoles();

	//set dipole objects
	for( int i = 0; i < nClasses; i++ )
		for( int j = 0; j < N_DIPOL_OBJECTS; j++ ){
			if( indivDipolTab[ dipolTabIndex + N_DIPOL_OBJECTS * i + j ] > 0 )
				node -> m_TrainObjs.SetObject4Dipoles( i, j, (*pDS)[ indivDipolTab[ dipolTabIndex + N_DIPOL_OBJECTS * i + j ] - 1 ] );
			else
				node -> m_TrainObjs.SetObject4Dipoles( i, j, NULL );

			#if TREE_REPO
			node -> m_TrainObjs.SetObject4Dipoles_TreeRepo( i, j, indivDipolTab[ dipolTabIndex + N_DIPOL_OBJECTS * i + j ] );
			#endif
		}

	#if FULL_BINARY_TREE_REP
	if( node -> m_vBranch.size() > 0 )	FillDTreeByExternalResults( node -> m_vBranch[ 0 ], indivDetailedErrTab, indivDetailedClassDistTab, indivDipolTab, index * 2 + 1, pDS );
	if( node -> m_vBranch.size() > 1 )	FillDTreeByExternalResults( node -> m_vBranch[ 1 ], indivDetailedErrTab, indivDetailedClassDistTab, indivDipolTab, index * 2 + 2, pDS );
	#else
	if( node -> m_vBranch.size() > 0 )	index = FillDTreeByExternalResults( node -> m_vBranch[ 0 ], indivDetailedErrTab, indivDetailedClassDistTab, indivDipolTab, index + 1, pDS );
	if( node -> m_vBranch.size() > 1 )	index = FillDTreeByExternalResults( node -> m_vBranch[ 1 ], indivDetailedErrTab, indivDetailedClassDistTab, indivDipolTab, index + 1, pDS );
	#endif


	//search objects for cutted pure and notcutted mixed dipols
	if( node -> m_vBranch.size() > 0 ){						//not in leaf
		//mixed notcutted dipol
		pair<CDataObject*, CDataObject*> p;
		SearchObjects4NotCuttedMixedDipole( node -> m_vBranch[ 0 ], p );
		node -> m_TrainObjs.SetNotCuttedMixedDipoleObject( p.first, 0 );
		node -> m_TrainObjs.SetNotCuttedMixedDipoleObject( p.second, 1 );
		if( !node -> m_TrainObjs.AreObjects4NotCuttedMixedDipoles() ){
			if( node -> m_vBranch.size() > 1 ){
				SearchObjects4NotCuttedMixedDipole( node -> m_vBranch[ 1 ], p );
				node -> m_TrainObjs.SetNotCuttedMixedDipoleObject( p.first, 0 );
				node -> m_TrainObjs.SetNotCuttedMixedDipoleObject( p.second, 1 );
			}
		}
		p.first = NULL;
		p.second = NULL;
		//pure cutted dipol
		if( node -> m_vBranch.size() > 1 ){
			SearchObjects4CuttedPureDipole( node -> m_vBranch[ 0 ], node -> m_vBranch[ 1 ], p );
			node -> m_TrainObjs.SetCuttedPureDipoleObject( p.first, 0 );
			node -> m_TrainObjs.SetCuttedPureDipoleObject( p.second, 1 );
		}
		p.first = NULL;
		p.second = NULL;
		//mixed cutted dipol
		if( node -> m_vBranch.size() > 1 ){
			bool odp = SearchObjects4CuttedMixedDipole( node -> m_vBranch[ 0 ], node -> m_vBranch[ 1 ], p );			
			node -> m_TrainObjs.SetCuttedMixedDipoleObject( p.first, 0 );
			node -> m_TrainObjs.SetCuttedMixedDipoleObject( p.second, 1 );
		}
	}

	return index;
}

//fill decision tree by the results obtained by an external device, e.g. GPU: errors, class distribution, dipol objects (FULL_BINARY_TREE_REP version)
int CCudaWorker::FillDTreeByExternalResults_FULL_BINARY_TREE_REP( CDTreeNode *node, unsigned int *indivDetailedErrTab, unsigned int *indivDetailedClassDistTab, unsigned int* indivDipolTab, unsigned int index, const IDataSet *pDS ){
	unsigned int errTabIndex = index * N_INFO_CORR_ERR;
	unsigned int classTabIndex = index * nClasses;
	unsigned int dipolTabIndex = index * nClasses * N_DIPOL_OBJECTS;

	//set number of samples and error
	node -> SetNMisClassifiedObjects( indivDetailedErrTab[ errTabIndex ] );
	node -> SetNTrainObjects( indivDetailedErrTab[ errTabIndex ] + indivDetailedErrTab[ errTabIndex + 1 ] );
	node -> m_TrainObjs.m_uSize = node -> GetNTrainObjects();

	//set class distribution
	int best = 0;
	for( int i = 0; i < nClasses; i++ ){
		node -> SetNClassObjects( i, indivDetailedClassDistTab[ classTabIndex + i ] );
		if( indivDetailedClassDistTab[ classTabIndex + i ] > indivDetailedClassDistTab[ classTabIndex + best ] )
			best = i;
	}
	node -> m_pIAClass -> m_AttrValue.u_iNom = best;

	node -> m_TrainObjs.ClearObjects4Dipoles();

	//set dipole objects
	for( int i = 0; i < nClasses; i++ )
		for( int j = 0; j < N_DIPOL_OBJECTS; j++ ){
			if( indivDipolTab[ dipolTabIndex + N_DIPOL_OBJECTS * i + j ] > 0 )
				node -> m_TrainObjs.SetObject4Dipoles( i, j, (*pDS)[ indivDipolTab[ dipolTabIndex + N_DIPOL_OBJECTS * i + j ] - 1 ] );
			else
				node -> m_TrainObjs.SetObject4Dipoles( i, j, NULL );

			#if TREE_REPO
			node -> m_TrainObjs.SetObject4Dipoles_TreeRepo( i, j, indivDipolTab[ dipolTabIndex + N_DIPOL_OBJECTS * i + j ] );
			#endif
		}
	
	if( node -> m_vBranch.size() > 0 )	FillDTreeByExternalResults_FULL_BINARY_TREE_REP( node -> m_vBranch[ 0 ], indivDetailedErrTab, indivDetailedClassDistTab, indivDipolTab, index * 2 + 1, pDS );
	if( node -> m_vBranch.size() > 1 )	FillDTreeByExternalResults_FULL_BINARY_TREE_REP( node -> m_vBranch[ 1 ], indivDetailedErrTab, indivDetailedClassDistTab, indivDipolTab, index * 2 + 2, pDS );
	

	//search objects for cutted pure and notcutted mixed dipols
	if( node -> m_vBranch.size() > 0 ){						//not in leaf		
		pair<CDataObject*, CDataObject*> p;
		SearchObjects4NotCuttedMixedDipole( node -> m_vBranch[ 0 ], p );
		node -> m_TrainObjs.SetNotCuttedMixedDipoleObject( p.first, 0 );
		node -> m_TrainObjs.SetNotCuttedMixedDipoleObject( p.second, 1 );
		if( !node -> m_TrainObjs.AreObjects4NotCuttedMixedDipoles() ){
			if( node -> m_vBranch.size() > 1 ){
				SearchObjects4NotCuttedMixedDipole( node -> m_vBranch[ 1 ], p );
				node -> m_TrainObjs.SetNotCuttedMixedDipoleObject( p.first, 0 );
				node -> m_TrainObjs.SetNotCuttedMixedDipoleObject( p.second, 1 );
			}
		}
		p.first = NULL;
		p.second = NULL;
		//pure cutted dipol
		if( node -> m_vBranch.size() > 1 ){
			SearchObjects4CuttedPureDipole( node -> m_vBranch[ 0 ], node -> m_vBranch[ 1 ], p );
			node -> m_TrainObjs.SetCuttedPureDipoleObject( p.first, 0 );
			node -> m_TrainObjs.SetCuttedPureDipoleObject( p.second, 1 );
		}
		p.first = NULL;
		p.second = NULL;
		//mixed cutted dipol
		if( node -> m_vBranch.size() > 1 ){
			bool odp = SearchObjects4CuttedMixedDipole( node -> m_vBranch[ 0 ], node -> m_vBranch[ 1 ], p );			
			node -> m_TrainObjs.SetCuttedMixedDipoleObject( p.first, 0 );
			node -> m_TrainObjs.SetCuttedMixedDipoleObject( p.second, 1 );
		}
	}

	return index;
}

int CCudaWorker::FillDTreeByExternalResultsChooser(CDTreeNode *root, unsigned int *indivDetailedErrTab, unsigned int *indivDetailedClassDistTab, unsigned int* indivDipolTab, unsigned int index, const IDataSet *pDS) {
#if FULL_BINARY_TREE_REP
	return FillDTreeByExternalResults(root, indivDetailedErrTab, indivDetailedClassDistTab, indivDipolTab, 0, pDS)
#else
	if (bCompactOrFullTreeRep)
		return FillDTreeByExternalResults(root, indivDetailedErrTab, indivDetailedClassDistTab, indivDipolTab, 0, pDS);
	else
		return FillDTreeByExternalResults_FULL_BINARY_TREE_REP(root, indivDetailedErrTab, indivDetailedClassDistTab, indivDipolTab, 0, pDS);
#endif
}

void CCudaWorker::PruneNodeBeforeCUDA(CDTreeNode* node) {
	if (node->m_bLeaf) return;
	size_t nBranches = node->m_vBranch.size();
	//propagate delete operation
	for (size_t i = 0; i < nBranches; i++) {
		PruneNodeBeforeCUDA(node->m_vBranch[i]);
		delete node->m_vBranch[i];
	}
	//clean data branches
	node->m_vBranch.clear();
	delete node->m_pITest;
	node->m_pITest = NULL;
	node->m_bLeaf = true;
}

#if FULL_BINARY_TREE_REP
void CCudaWorker::SetInitTreeDepthLimit( const IDataSet *dataset, eModelType modelType ){
	
	//share memory is limited and thus we 
	#if CUDA_SHARED_MEM_POST_CALC
	this -> nClasses = dataset -> GetClassCount();	
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

	while( nTreeNodes / 2.0 >= 1.0 ){
		depthLimit++;
		nTreeNodes /= 2;
	}
	
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

void CCudaWorker::PruneIndivBeforeCUDA( CDTreeNode* node, int treeLevel ){
	treeLevel++;
	if( treeLevel == currTreeDepthLimit ){
		if( !node -> m_bLeaf ){
			PruneNodeBeforeCUDA( node );
		}
	}
	else{
		size_t nBranches = node -> m_vBranch.size();
		for( size_t i = 0; i < nBranches; i++ ){
			PruneIndivBeforeCUDA( node -> m_vBranch[i], treeLevel );
		}
	}			
}
#else
bool CCudaWorker::DetermineCompactOrFullTreeRep(CDTreeNode* root) {
	#if !ADDAPTIVE_TREE_REP
	return 1;													//compact
	#else
	
	//for small trees
	//if (root->GetDepth() <= 2) return 0;						//full (complete)

	//for large trees or over the chosen limit
	//if (root->GetDepth() > MAX_MEMORY_TREE_DEPTH) return 1;		//compact

	//printf("%f ? %f (%d %d)\n", root->GetNAllNodes() / ((double)pow(2.0, root->GetDepth()) - 1), adaptiveTreeRepSwitch,
	//	                        root->GetNAllNodes(), root->GetDepth() );
	//smart switching using a factor (adaptiveTreeRepSwitch)
	if (root->GetNAllNodes() / ((double)pow(2.0, root->GetDepth())-1) > adaptiveTreeRepSwitch)
		return 0;												//full (complete) 
	else
		return 1;												//compact
	#endif
}
void CCudaWorker::PruneIndivBeforeCUDA_FULL_BINARY_TREE_REP(CDTreeNode* node, int treeLevel) {
	treeLevel++;
	if (treeLevel == currTreeDepthLimit_FULL_BINARY_TREE_REP) {
		if (!node->m_bLeaf) {
			PruneNodeBeforeCUDA(node);
		}
	}
	else {
		size_t nBranches = node->m_vBranch.size();
		for (size_t i = 0; i < nBranches; i++) {
			PruneIndivBeforeCUDA_FULL_BINARY_TREE_REP(node->m_vBranch[i], treeLevel);
		}
	}
}
#endif

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////     IMPLEMENTATION OF KERNEL (DEVICE) FUNCTIONS    ///////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
__device__ int dev_random( int max ) {	
	curandState_t state;
	
	curand_init(0, /* the seed controls the sequence of random values that are produced */
				0, /* the sequence number is only important with multiple cores */
				0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
				&state);
	printf( "%d\n", max );
	return curand(&state) % max;
}

__global__ void dev_CudaTest( float* x, float *y, float *sum, int N ){
	unsigned int index = threadIdx.x;

	x[ index ] += y[ index ];

	__syncthreads();
	if( index == 0 )
		for( int i = 0; i < N; i++ )  
			(*sum) += x[ i ];
	__syncthreads();	
}

#if FULL_BINARY_TREE_REP
__global__ void dev_CalcPopClassDistAndDipolAtLeafs_Pre_V2b( DS_REAL *datasetTab, unsigned int *classTab, int nObjects, int nAttrs, int nIndividuals,
															 int *populationAttrNumTab, float *populationValueTab,  int *individualPosInTab, int populationTabSize, int nClasses,
															 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks ){
	int startObjectIndex = 0;
	
	//range of samples to process
	if( blockIdx.x < nObjects % gridDim.x )
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + blockIdx.x;		
	else
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + nObjects % gridDim.x;
	int nObjectsToCheck = nObjects / gridDim.x + (int)( ( nObjects % gridDim.x ) > blockIdx.x );	
	
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

		individualClassDistTab_Local = populationClassDistTab_Local + individualPosInTab[ individualIndex ] * nClasses;
		individualDipolTab_Local = populationDipolTab_Local + individualPosInTab[ individualIndex ] * nClasses * (N_DIPOL_OBJECTS + 1);

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
					
					dipolObjectStartIndex = treeNodeIndex * nClasses * (N_DIPOL_OBJECTS + 1) + classTab[ startObjectIndex + index ] * (N_DIPOL_OBJECTS + 1);																	//where is the place for the first object for the dipol mechanism
					dipolObjectSubIndex = atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ]), (individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ] + 1) % N_DIPOL_OBJECTS );	//+1 to know where any object was not set					
					atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + dipolObjectSubIndex ]), startObjectIndex + index + 1 );											
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
	int startObjectIndex = 0;
	
	//range of samples to process
	if( blockIdx.x < nObjects % gridDim.x )
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + blockIdx.x;		
	else
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + nObjects % gridDim.x;
	int nObjectsToCheck = nObjects / gridDim.x + (int)( ( nObjects % gridDim.x ) > blockIdx.x );	
	
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

		int *individualLeftNodePosTab	= populationLeftNodePosTab	 + individualPosInTab[ individualIndex ];
		int *individualRightNodePosTab	= populationRightNodePosTab	 + individualPosInTab[ individualIndex ];
		int *individualParentNodePosTab = populationParentNodePosTab + individualPosInTab[ individualIndex ];

		individualClassDistTab_Local = populationClassDistTab_Local + individualPosInTab[ individualIndex ] * nClasses;
		individualDipolTab_Local = populationDipolTab_Local + individualPosInTab[ individualIndex ] * nClasses * (N_DIPOL_OBJECTS + 1);

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

					dipolObjectStartIndex = treeNodeIndex * nClasses * (N_DIPOL_OBJECTS + 1) + classTab[startObjectIndex + index] * (N_DIPOL_OBJECTS + 1);
					dipolObjectSubIndex = atomicExch(&(individualDipolTab_Local[dipolObjectStartIndex + N_DIPOL_OBJECTS]), (individualDipolTab_Local[dipolObjectStartIndex + N_DIPOL_OBJECTS] + 1) % N_DIPOL_OBJECTS);
					atomicExch(&(individualDipolTab_Local[dipolObjectStartIndex + dipolObjectSubIndex]), startObjectIndex + index + 1);
					break;
				}
				else
					//go left in the tree
					if (objectTab[individualAttrNumTab[treeNodeIndex]] <= individualValueTab[treeNodeIndex])
						treeNodeIndex = individualLeftNodePosTab[ treeNodeIndex ];
					//go right in the tree
					else
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
	int startObjectIndex = 0;
	
	//range of samples to process
	if( blockIdx.x < nObjects % gridDim.x )
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + blockIdx.x;		
	else
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + nObjects % gridDim.x;
	int nObjectsToCheck = nObjects / gridDim.x + (int)( ( nObjects % gridDim.x ) > blockIdx.x );	
	
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

		individualClassDistTab_Local = populationClassDistTab_Local + individualPosInTab[ individualIndex ] * nClasses;
		individualDipolTab_Local = populationDipolTab_Local + individualPosInTab[ individualIndex ] * nClasses * (N_DIPOL_OBJECTS + 1);

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
					
					dipolObjectStartIndex = treeNodeIndex * nClasses * (N_DIPOL_OBJECTS + 1) + classTab[ startObjectIndex + index ] * (N_DIPOL_OBJECTS + 1);
					dipolObjectSubIndex = atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ]), (individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ] + 1) % N_DIPOL_OBJECTS );
					atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + dipolObjectSubIndex ]), startObjectIndex + index + 1 );											
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

			index++;		
			objectTab += nAttrs;
		}		
	}
}
#endif

#if FULL_BINARY_TREE_REP
__global__ void dev_CalcPopPartClassDistAndDipolAtLeafs_Pre_V2b( DS_REAL *datasetTab, unsigned int *classTab, int nObjects, int nAttrs, int nIndividuals,
																 int *populationAttrNumTab, float *populationValueTab,  int *individualPosInTab, int populationTabSize, int nClasses,
																 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks ){
	int startObjectIndex = 0;
	
	//range of samples to process
	if( blockIdx.x < nObjects % gridDim.x )
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + blockIdx.x;		
	else
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + nObjects % gridDim.x;
	int nObjectsToCheck = nObjects / gridDim.x + (int)( ( nObjects % gridDim.x ) > blockIdx.x );	
	
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

		individualClassDistTab_Local = populationClassDistTab_Local + individualPosInTab[ individualIndex ] * nClasses;
		individualDipolTab_Local = populationDipolTab_Local + individualPosInTab[ individualIndex ] * nClasses * (N_DIPOL_OBJECTS + 1);

		DS_REAL* objectTab = datasetTab + startObjectIndex * nAttrs;

		int treeNodeIndex;		
		unsigned int index = 0;
		while( index < nObjectsToCheck ){			
			//start from root node
			treeNodeIndex = 0;

			while( treeNodeIndex < individualTabSize ){

				//if we are in the tree part that is not to search
				if( individualAttrNumTab[ treeNodeIndex ] == -2 )
					break;
				
				//if we are in a leaf
				if( individualAttrNumTab[ treeNodeIndex ] == -1 ){
					//remember the object class in this leaf
					atomicAdd( &(individualClassDistTab_Local[ treeNodeIndex * nClasses + classTab[ startObjectIndex + index ] ]), 1 );
					
					dipolObjectStartIndex = treeNodeIndex * nClasses * (N_DIPOL_OBJECTS + 1) + classTab[ startObjectIndex + index ] * (N_DIPOL_OBJECTS + 1);																	//where is the place for the first object for the dipol mechanism
					dipolObjectSubIndex = atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ]), (individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ] + 1) % N_DIPOL_OBJECTS );	//+1 to know where any object was not set					
					atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + dipolObjectSubIndex ]), startObjectIndex + index + 1 );											
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
__global__ void dev_CalcPopPartClassDistAndDipolAtLeafs_Pre_V2b( DS_REAL *datasetTab, unsigned int *classTab, int nObjects, int nAttrs, int nIndividuals,
																 int *populationAttrNumTab, float *populationValueTab,  int *individualPosInTab, int populationTabSize, int nClasses,
																 int *populationLeftNodePosTab, int *populationRightNodePosTab, int *populationParentNodePosTab,
																 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks ){
	int startObjectIndex = 0;
	
	//range of samples to process
	if( blockIdx.x < nObjects % gridDim.x )
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + blockIdx.x;		
	else
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + nObjects % gridDim.x;
	int nObjectsToCheck = nObjects / gridDim.x + (int)( ( nObjects % gridDim.x ) > blockIdx.x );	
	
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

				//if we are in the tree part that is not to search
				if( individualAttrNumTab[ treeNodeIndex ] == -2 )
					break;
				
				//if we are in a leaf
				if( individualAttrNumTab[ treeNodeIndex ] == -1 ){
					//remember the object class in this leaf
					atomicAdd( &(individualClassDistTab_Local[ treeNodeIndex * nClasses + classTab[ startObjectIndex + index ] ]), 1 );
					
					dipolObjectStartIndex = treeNodeIndex * nClasses * (N_DIPOL_OBJECTS + 1) + classTab[ startObjectIndex + index ] * (N_DIPOL_OBJECTS + 1);
					dipolObjectSubIndex = atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ]), (individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ] + 1) % N_DIPOL_OBJECTS );
					atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + dipolObjectSubIndex ]), startObjectIndex + index + 1 );											
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

			index++;		
			objectTab += nAttrs;
		}		
	}
}
__global__ void dev_CalcPopPartClassDistAndDipolAtLeafs_Pre_FULL_BINARY_TREE_REP_V2b( DS_REAL *datasetTab, unsigned int *classTab, int nObjects, int nAttrs, int nIndividuals,
																 int *populationAttrNumTab, float *populationValueTab,  int *individualPosInTab, int populationTabSize, int nClasses,
																 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks ){
	int startObjectIndex = 0;
	
	//range of samples to process
	if( blockIdx.x < nObjects % gridDim.x )
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + blockIdx.x;		
	else
		startObjectIndex = blockIdx.x * ( nObjects / gridDim.x ) + nObjects % gridDim.x;
	int nObjectsToCheck = nObjects / gridDim.x + (int)( ( nObjects % gridDim.x ) > blockIdx.x );	
	
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

		individualClassDistTab_Local = populationClassDistTab_Local + individualPosInTab[ individualIndex ] * nClasses;
		individualDipolTab_Local = populationDipolTab_Local + individualPosInTab[ individualIndex ] * nClasses * (N_DIPOL_OBJECTS + 1);

		DS_REAL* objectTab = datasetTab + startObjectIndex * nAttrs;

		int treeNodeIndex;		
		unsigned int index = 0;
		while( index < nObjectsToCheck ){			
			//start from root node
			treeNodeIndex = 0;

			while( treeNodeIndex < individualTabSize ){

				//if we are in the tree part that is not to search
				if( individualAttrNumTab[ treeNodeIndex ] == -2 )
					break;
				
				//if we are in a leaf
				if( individualAttrNumTab[ treeNodeIndex ] == -1 ){
					//remember the object class in this leaf
					atomicAdd( &(individualClassDistTab_Local[ treeNodeIndex * nClasses + classTab[ startObjectIndex + index ] ]), 1 );
					
					dipolObjectStartIndex = treeNodeIndex * nClasses * (N_DIPOL_OBJECTS + 1) + classTab[ startObjectIndex + index ] * (N_DIPOL_OBJECTS + 1);
					dipolObjectSubIndex = atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ]), (individualDipolTab_Local[ dipolObjectStartIndex + N_DIPOL_OBJECTS ] + 1) % N_DIPOL_OBJECTS );
					atomicExch( &(individualDipolTab_Local[ dipolObjectStartIndex + dipolObjectSubIndex ]), startObjectIndex + index + 1 );											
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
	int individualIndex = blockIdx.x;
	int individualTabSize = 0;
	unsigned int *individualDetailedErrTab = populationDetailedErrTab + individualPosInTab[ individualIndex ] * N_INFO_CORR_ERR;
	unsigned int *individualDetailedClassDistTab = populationDetailedClassDistTab + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int *individualDipolTab = populationDipolTab + individualPosInTab[ individualIndex ] * nClasses * N_DIPOL_OBJECTS;

	//the individual size in 1D table
	if( individualIndex < gridDim.x - 1 )
		individualTabSize = individualPosInTab[ individualIndex + 1 ] - individualPosInTab[ individualIndex ];
	else
		individualTabSize = populationTabSize - individualPosInTab[ individualIndex ];

	unsigned int* individualClassDistTab_Local = populationClassDistTab_ScatOverBlocks + threadIdx.x * populationTabSize * nClasses + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int* individualDipolTab_Local = populationDipolTab_ScatOverBlocks + threadIdx.x * populationTabSize * nClasses * (N_DIPOL_OBJECTS + 1)  + individualPosInTab[ individualIndex ] * nClasses * (N_DIPOL_OBJECTS + 1);
	
	//shared memory
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
	
	//reduction class distribution
	for( int i = 0; i < individualTabSize * nClasses; i++ )
		atomicAdd( &( individualClassDistTab_Sum[ i ] ), individualClassDistTab_Local[ i ] );

	//reduction dipoles
	for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
		if( individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] != 0 )
			atomicCAS( &( individualDipolTab_Random[ i ] ), false, individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] );		
		
	
	__syncthreads();
	
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

		//propagate dipoles
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
		
		for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
			individualDipolTab[ i ] = individualDipolTab_Random[ i ];
	}
}

__global__ void dev_CalcPopDetailedErrAndClassDistAndDipol_Post_V2b( int *individualPosInTab, int populationTabSize, int nClasses,
																	 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks,
																	 unsigned int *populationDetailedErrTab, unsigned int *populationDetailedClassDistTab, unsigned int *populationDipolTab ){
	
	int individualIndex = blockIdx.x;
	int individualTabSize = 0;
	
	unsigned int *individualDetailedErrTab = populationDetailedErrTab + individualPosInTab[ individualIndex ] * N_INFO_CORR_ERR;
	unsigned int *individualDetailedClassDistTab = populationDetailedClassDistTab + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int *individualDipolTab = populationDipolTab + individualPosInTab[ individualIndex ] * nClasses * N_DIPOL_OBJECTS;

	//the individual size in 1D table
	if( individualIndex < gridDim.x - 1 )
		individualTabSize = individualPosInTab[ individualIndex + 1 ] - individualPosInTab[ individualIndex ];
	else
		individualTabSize = populationTabSize - individualPosInTab[ individualIndex ];

	unsigned int* individualClassDistTab_Local = populationClassDistTab_ScatOverBlocks + threadIdx.x * populationTabSize * nClasses + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int* individualDipolTab_Local = populationDipolTab_ScatOverBlocks + threadIdx.x * populationTabSize * nClasses * (N_DIPOL_OBJECTS + 1)  + individualPosInTab[ individualIndex ] * nClasses * (N_DIPOL_OBJECTS + 1);
	
	//if set use shared memory
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
	

	for( int i = 0; i < individualTabSize * nClasses; i++ )
		atomicAdd( &( individualClassDistTab_Sum[ i ] ), individualClassDistTab_Local[ i ] );

	for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
		if( individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] != 0 )
			atomicCAS( &( individualDipolTab_Random[ i ] ), false, individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] );		
		
	
	__syncthreads();
	
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

		//propagate dipoles
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
	int individualIndex = blockIdx.x;
	int individualTabSize = 0;
	
	unsigned int *individualDetailedErrTab = populationDetailedErrTab + individualPosInTab[ individualIndex ] * N_INFO_CORR_ERR;
	unsigned int *individualDetailedClassDistTab = populationDetailedClassDistTab + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int *individualDipolTab = populationDipolTab + individualPosInTab[ individualIndex ] * nClasses * N_DIPOL_OBJECTS;

	int *individualParentNodePosTab = populationParentNodePosTab + individualPosInTab[ individualIndex ];

	//the individual size in 1D table
	if( individualIndex < gridDim.x - 1 )
		individualTabSize = individualPosInTab[ individualIndex + 1 ] - individualPosInTab[ individualIndex ];
	else
		individualTabSize = populationTabSize - individualPosInTab[ individualIndex ];

	unsigned int* individualClassDistTab_Local = populationClassDistTab_ScatOverBlocks + threadIdx.x * populationTabSize * nClasses + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int* individualDipolTab_Local = populationDipolTab_ScatOverBlocks + threadIdx.x * populationTabSize * nClasses * (N_DIPOL_OBJECTS + 1)  + individualPosInTab[ individualIndex ] * nClasses * (N_DIPOL_OBJECTS + 1);
	
	//if set use shared memory
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
	

	for( int i = 0; i < individualTabSize * nClasses; i++ )
		atomicAdd( &( individualClassDistTab_Sum[ i ] ), individualClassDistTab_Local[ i ] );

	for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
		if( individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] != 0 )
			atomicCAS( &( individualDipolTab_Random[ i ] ), false, individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] );		
		
	
	__syncthreads();
	
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
	
			
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){
			individualDetailedErrTab[ individualParentNodePosTab[ treeNodeIndex ] * N_INFO_CORR_ERR     ] += individualDetailedErrTab[ treeNodeIndex * N_INFO_CORR_ERR     ];
			individualDetailedErrTab[ individualParentNodePosTab[ treeNodeIndex ] * N_INFO_CORR_ERR + 1 ] += individualDetailedErrTab[ treeNodeIndex * N_INFO_CORR_ERR + 1 ];
		}
		
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){
			for( int classIndex = 0; classIndex < nClasses; classIndex++ )
				individualClassDistTab_Sum[ individualParentNodePosTab[treeNodeIndex] * nClasses + classIndex ] += individualClassDistTab_Sum[ treeNodeIndex * nClasses + classIndex ];
		}
		#if CUDA_SHARED_MEM_POST_CALC
		for( int i = 0; i < individualTabSize * N_INFO_CORR_ERR; i++ )
			individualDetailedClassDistTab[ i ] = individualClassDistTab_Sum[ i ];
		#endif
				
		//propagate dipoles
		for( int treeNodeIndex = individualTabSize - 1; treeNodeIndex > 0; treeNodeIndex-- ){			
			for( int classIndex = 0; classIndex < nClasses; classIndex++ )
				for( int dipolIndex = 0; dipolIndex < N_DIPOL_OBJECTS; dipolIndex++ )
					if( individualDipolTab_Random[ individualParentNodePosTab[treeNodeIndex] * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ] == 0 )
						individualDipolTab_Random[ individualParentNodePosTab[treeNodeIndex] * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ] = 
						individualDipolTab_Random[ treeNodeIndex * nClasses * N_DIPOL_OBJECTS + classIndex * dipolIndex + dipolIndex ];
		}

		#if CUDA_SHARED_MEM_POST_CALC
		for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
			individualDipolTab[ i ] = individualDipolTab_Random[ i ];
		#endif
	}
}
__global__ void dev_CalcPopDetailedErrAndClassDistAndDipol_Post_FULL_BINARY_TREE_REP_V2b( int *individualPosInTab, int populationTabSize, int nClasses,
																	 unsigned int *populationClassDistTab_ScatOverBlocks, unsigned int *populationDipolTab_ScatOverBlocks,
																	 unsigned int *populationDetailedErrTab, unsigned int *populationDetailedClassDistTab, unsigned int *populationDipolTab ){

	int individualIndex = blockIdx.x;
	int individualTabSize = 0;

	unsigned int *individualDetailedErrTab = populationDetailedErrTab + individualPosInTab[ individualIndex ] * N_INFO_CORR_ERR;
	unsigned int *individualDetailedClassDistTab = populationDetailedClassDistTab + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int *individualDipolTab = populationDipolTab + individualPosInTab[ individualIndex ] * nClasses * N_DIPOL_OBJECTS;

	//the individual size in 1D table
	if( individualIndex < gridDim.x - 1 )
		individualTabSize = individualPosInTab[ individualIndex + 1 ] - individualPosInTab[ individualIndex ];
	else
		individualTabSize = populationTabSize - individualPosInTab[ individualIndex ];

	unsigned int* individualClassDistTab_Local = populationClassDistTab_ScatOverBlocks + threadIdx.x * populationTabSize * nClasses + individualPosInTab[ individualIndex ] * nClasses;
	unsigned int* individualDipolTab_Local = populationDipolTab_ScatOverBlocks + threadIdx.x * populationTabSize * nClasses * (N_DIPOL_OBJECTS + 1)  + individualPosInTab[ individualIndex ] * nClasses * (N_DIPOL_OBJECTS + 1);
	
	//if set use shared memory
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
	

	for( int i = 0; i < individualTabSize * nClasses; i++ )
		atomicAdd( &( individualClassDistTab_Sum[ i ] ), individualClassDistTab_Local[ i ] );

	for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
		if( individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] != 0 )
			atomicCAS( &( individualDipolTab_Random[ i ] ), false, individualDipolTab_Local[ i + i / N_DIPOL_OBJECTS ] );		
		
	
	__syncthreads();

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

		//propagate dipoles
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
		#if CUDA_SHARED_MEM_POST_CALC
		for( int i = 0; i < individualTabSize * nClasses * N_DIPOL_OBJECTS; i++ )
			individualDipolTab[ i ] = individualDipolTab_Random[ i ];
		#endif
	}
}
#endif

#endif