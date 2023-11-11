#ifdef __linux__
#include <sys/time.h>
//#define CLOCK_GETTIME 1		//0-gettimeofday, 1-clock_gettime
#else
#include <time.h>
#endif

//time and memory results/peformance
class CCudaTimeStats{
private:
	//two kinds of time measurements under linux
	#ifdef __linux__
		#if CLOCK_GETTIME
		struct timespec wholeTimeBegin, wholeTimeEnd;
		struct timespec calcTimeBegin, calcTimeEnd;
		struct timespec wholeDatasetTransportTimeBegin, wholeDatasetTransportTimeEnd;
		struct timespec memoryAllocForDatasetTimeBegin, memoryAllocForDatasetTimeEnd;
		struct timespec sendDatasetToGPUTimeBegin, sendDatasetToGPUTimeEnd;
		struct timespec dataTransferToGPUTimeBegin, dataTransferToGPUTimeEnd;
		struct timespec dataTransferFromGPUTimeBegin, dataTransferFromGPUTimeEnd;
		struct timespec dataReorganizationTimeBegin, dataReorganizationTimeEnd;
		struct timespec memoryAllocDeallocTimeBegin, memoryAllocDeallocTimeEnd;
		struct timespec treeRepoSearchTimeBegin, treeRepoSearchTimeEnd;
		struct timespec treeRepoApplyTimeBegin, treeRepoApplyTimeEnd;
		struct timespec treeRepoInsertTimeBegin, treeRepoInsertTimeEnd;

		struct timespec MT_kernel1TimeBegin, MT_kernel1TimeEnd;
		struct timespec MT_sort1TimeBegin, MT_sort1TimeEnd;
		struct timespec MT_kernel2TimeBegin, MT_kernel2TimeEnd;
		struct timespec MT_kernel3TimeBegin, MT_kernel3TimeEnd;

		struct timespec MT_modelCalcTimeBegin, MT_modelCalcTimeEnd;
		struct timespec MT_fullMLRCalcTimeBegin, MT_fullMLRCalcTimeEnd;
		struct timespec MT_meanRTCalcTimeBegin, MT_meanRTCalcTimeEnd;

		struct timespec MT_kernel4TimeBegin, MT_kernel4TimeEnd;
		struct timespec MT_kernel5TimeBegin, MT_kernel5TimeEnd;			
		#else
		timeval wholeTimeBegin, wholeTimeEnd;
		timeval calcTimeBegin, calcTimeEnd;
		timeval wholeDatasetTransportTimeBegin, wholeDatasetTransportTimeEnd;
		timeval memoryAllocForDatasetTimeBegin, memoryAllocForDatasetTimeEnd;
		timeval sendDatasetToGPUTimeBegin, sendDatasetToGPUTimeEnd;
		timeval dataTransferToGPUTimeBegin, dataTransferToGPUTimeEnd;
		timeval dataTransferFromGPUTimeBegin, dataTransferFromGPUTimeEnd;
		timeval dataReorganizationTimeBegin, dataReorganizationTimeEnd;
		timeval memoryAllocDeallocTimeBegin, memoryAllocDeallocTimeEnd;
		timeval treeRepoSearchTimeBegin, treeRepoSearchTimeEnd;
		timeval treeRepoApplyTimeBegin, treeRepoApplyTimeEnd;
		timeval treeRepoInsertTimeBegin, treeRepoInsertTimeEnd;

		timeval MT_kernel1TimeBegin, MT_kernel1TimeEnd;
		timeval MT_sort1TimeBegin, MT_sort1TimeEnd;
		timeval MT_kernel2TimeBegin, MT_kernel2TimeEnd;
		timeval MT_kernel3TimeBegin, MT_kernel3TimeEnd;

		timeval MT_modelCalcTimeBegin, MT_modelCalcTimeEnd;
		timeval MT_fullMLRCalcTimeBegin, MT_fullMLRCalcTimeEnd;
		timeval MT_meanRTCalcTimeBegin, MT_meanRTCalcTimeEnd;

		timeval MT_kernel4TimeBegin, MT_kernel4TimeEnd;
		timeval MT_kernel5TimeBegin, MT_kernel5TimeEnd;
		#endif
	#else
	time_t wholeTimeBegin, wholeTimeEnd;
	time_t calcTimeBegin, calcTimeEnd;
	time_t wholeDatasetTransportTimeBegin, wholeDatasetTransportTimeEnd;
	time_t memoryAllocForDatasetTimeBegin, memoryAllocForDatasetTimeEnd;
	time_t sendDatasetToGPUTimeBegin, sendDatasetToGPUTimeEnd;
	time_t dataTransferToGPUTimeBegin, dataTransferToGPUTimeEnd;
	time_t dataTransferFromGPUTimeBegin, dataTransferFromGPUTimeEnd;
	time_t dataReorganizationTimeBegin, dataReorganizationTimeEnd;
	time_t memoryAllocDeallocTimeBegin, memoryAllocDeallocTimeEnd;
	time_t treeRepoSearchTimeBegin, treeRepoSearchTimeEnd;
	time_t treeRepoApplyTimeBegin, treeRepoApplyTimeEnd;
	time_t treeRepoInsertTimeBegin, treeRepoInsertTimeEnd;

	time_t MT_kernel1TimeBegin, MT_kernel1TimeEnd;
	time_t MT_sort1TimeBegin, MT_sort1TimeEnd;
	time_t MT_kernel2TimeBegin, MT_kernel2TimeEnd;
	time_t MT_kernel3TimeBegin, MT_kernel3TimeEnd;

	time_t MT_modelCalcTimeBegin, MT_modelCalcTimeEnd;
	time_t MT_fullMLRCalcTimeBegin, MT_fullMLRCalcTimeEnd;
	time_t MT_meanRTCalcTimeBegin, MT_meanRTCalcTimeEnd;

	time_t MT_kernel4TimeBegin, MT_kernel4TimeEnd;
	time_t MT_kernel5TimeBegin, MT_kernel5TimeEnd;
	#endif

	double wholeTime;												//[seconds]
	double calcTime;												//[seconds]
	double wholeDatasetTransportTime;								//[seconds]
	double memoryAllocForDatasetTime;								//[seconds]
	double sendDatasetToGPUTime;									//[seconds]
	double dataTransferToGPUTime;									//[seconds]
	double dataTransferFromGPUTime;									//[seconds]
	double dataReorganizationTime;									//[seconds]
	double memoryAllocDeallocTime;									//[seconds]
	double MT_kernel1Time;											//[seconds] - dev_MT_CalcPopObjectToLeafAssign_V2b
	double MT_sort1Time;											//[seconds] - Sort_DeviceRadixSort_SortPairs
	double MT_kernel2Time;											//[seconds] - dev_MT_FillCalcPopStartLeafDataTab_V2b
	double MT_kernel3Time;											//[seconds] - dev_MT_FillPopMatrixForModelsCalc_V2b_ver2
	double MT_modelCalcTime;										//[seconds] - MT_CalcModelsAtLeafs	
	double MT_fullMLRCalcTime;										//[seconds] - MT_CalcModelMLR	
	double MT_meanRTCalcTime;										//[seconds] - MT_CalcModelRT
	double MT_kernel4Time;											//[seconds] - dev_MT_CalcPopErrAndDipolAtLeafs_Pre_V2b
	double MT_kernel5Time;											//[seconds] - dev_MT_CalcPopDetailedErrAndDipol_Post_V2b
	double treeRepoSearchTime;										//[seconds] - dev_SearchTheSameTreeInTreeRepo
	double treeRepoApplyTime;										//[seconds]
	double treeRepoInsertTime;										//[seconds] 
	
	//double sumWholeTime;											//[seconds]
	//double sumCalcTime;											//[seconds]	
	//double sumDataTransferToGPUTime;								//[seconds]
	//double sumDataTransferFromGPUTime;							//[seconds]
	//double sumDataReorganizationTime;								//[seconds]
	//double sumMemoryAllocDeallocTime;								//[seconds]

	double sumWholeTime_V1a;										//[seconds]
	double sumCalcTime_V1a;											//[seconds]
	double sumDataTransferToGPUTime_V1a;							//[seconds]
	double sumDataTransferFromGPUTime_V1a;							//[seconds]
	double sumDataReorganizationTime_V1a;							//[seconds]
	double sumMemoryAllocDeallocTime_V1a;							//[seconds]

	double sumWholeTime_DetailedV1a;								//[seconds]
	double sumCalcTime_DetailedV1a;									//[seconds]
	double sumDataTransferToGPUTime_DetailedV1a;					//[seconds]
	double sumDataTransferFromGPUTime_DetailedV1a;					//[seconds]
	double sumDataReorganizationTime_DetailedV1a;					//[seconds]
	double sumMemoryAllocDeallocTime_DetailedV1a;					//[seconds]

	double sumWholeTime_V1b;										//[seconds]
	double sumCalcTime_V1b;											//[seconds]
	double sumDataTransferToGPUTime_V1b;							//[seconds]
	double sumDataTransferFromGPUTime_V1b;							//[seconds]
	double sumDataReorganizationTime_V1b;							//[seconds]
	double sumMemoryAllocDeallocTime_V1b;							//[seconds]

	double sumWholeTime_DetailedV1b;								//[seconds]
	double sumCalcTime_DetailedV1b;									//[seconds]
	double sumDataTransferToGPUTime_DetailedV1b;					//[seconds]
	double sumDataTransferFromGPUTime_DetailedV1b;					//[seconds]
	double sumDataReorganizationTime_DetailedV1b;					//[seconds]
	double sumMemoryAllocDeallocTime_DetailedV1b;					//[seconds]

	double sumWholeTime_V2a;										//[seconds]
	double sumCalcTime_V2a;											//[seconds]
	double sumDataTransferToGPUTime_V2a;							//[seconds]
	double sumDataTransferFromGPUTime_V2a;							//[seconds]
	double sumDataReorganizationTime_V2a;							//[seconds]
	double sumMemoryAllocDeallocTime_V2a;							//[seconds]

	double sumWholeTime_DetailedV2a;								//[seconds]
	double sumCalcTime_DetailedV2a;									//[seconds]
	double sumDataTransferToGPUTime_DetailedV2a;					//[seconds]
	double sumDataTransferFromGPUTime_DetailedV2a;					//[seconds]
	double sumDataReorganizationTime_DetailedV2a;					//[seconds]
	double sumMemoryAllocDeallocTime_DetailedV2a;					//[seconds]

	double sumWholeTime_V2b;										//[seconds]
	double sumCalcTime_V2b;											//[seconds]
	double sumDataTransferToGPUTime_V2b;							//[seconds]
	double sumDataTransferFromGPUTime_V2b;							//[seconds]
	double sumDataReorganizationTime_V2b;							//[seconds]
	double sumMemoryAllocDeallocTime_V2b;							//[seconds]

	double sumWholeTime_DetailedV2b;								//[seconds]
	double sumCalcTime_DetailedV2b;									//[seconds]
	double sumDataTransferToGPUTime_DetailedV2b;					//[seconds]
	double sumDataTransferFromGPUTime_DetailedV2b;					//[seconds]
	double sumDataReorganizationTime_DetailedV2b;					//[seconds]
	double sumMemoryAllocDeallocTime_DetailedV2b;					//[seconds]

	//currently used for time stats
	double sumWholeTime_DetailedIndivV2b;							//[seconds]
	double sumCalcTime_DetailedIndivV2b;							//[seconds]
	double sumDataTransferToGPUTime_DetailedIndivV2b;				//[seconds]
	double sumDataTransferFromGPUTime_DetailedIndivV2b;				//[seconds]
	double sumDataReorganizationTime_DetailedIndivV2b;				//[seconds]
	double sumMemoryAllocDeallocTime_DetailedIndivV2b;				//[seconds]
	double sumTreeRepoSearchTime_DetailedIndivV2b;					//[seconds] - dev_SearchTheSameTreeInTreeRepo
	double sumTreeRepoApplyTime_DetailedIndivV2b;					//[seconds]
	double sumTreeRepoInsertTime_DetailedIndivV2b;					//[seconds]
	double MT_sumKernel1Time_DetailedIndivV2b;						//[seconds] - dev_MT_CalcPopObjectToLeafAssign_V2b
	double MT_sumSort1Time_DetailedIndivV2b;						//[seconds] - Sort_DeviceRadixSort_SortPairs
	double MT_sumKernel2Time_DetailedIndivV2b;						//[seconds] - dev_MT_FillCalcPopStartLeafDataTab_V2b
	double MT_sumKernel3Time_DetailedIndivV2b;						//[seconds] - dev_MT_FillPopMatrixForModelsCalc_V2b_ver2	
	double MT_sumModelCalcTime_DetailedIndivV2b;					//[seconds] - MT_CalcModelsAtLeafs
	double MT_sumFullMLRCalcTime_DetailedIndivV2b;					//[seconds] - MT_CalcModelMLR
	double MT_sumMeanRTCalcTime_DetailedIndivV2b;					//[seconds] - MT_CalcModelRT	
	double MT_sumKernel4Time_DetailedIndivV2b;						//[seconds] - dev_MT_CalcPopErrAndDipolAtLeafs_Pre_V2b
	double MT_sumKernel5Time_DetailedIndivV2b;						//[seconds] - dev_MT_CalcPopDetailedErrAndDipol_Post_V2b
	

	//additional time variables for sequential and OpenMP algorithms
	#ifdef __linux__
	#if CLOCK_GETTIME
	struct timespec seqTimeBegin, seqTimeEnd;
	#else
	timeval seqTimeBegin, seqTimeEnd;
	#endif
	#else
	time_t seqTimeBegin, seqTimeEnd;	
	#endif

	

	double seqTime;														//[seconds]	
	double sumSeqTime;													//[seconds]	

	//used when MT are applied
	int nMLRNodes;														//number of all nodes (leafs) with a model
	int nReCalcRTNodes;													//number of nodes (leafs) where a model (mean - RT) has to be recalculated (e.g. when not enough objects is located in the node)
	int nReCalcMLRNodes;												//number of nodes (leafs) where a model has to be recalculated

public:
	CCudaTimeStats(void);
	~CCudaTimeStats(void);
	void ClearAllStats();		//clean
	void ClearCurrStats();		//clean
	void MoveTimeStats_V1a();	//move before new run
	void MoveTimeStats_V1b();	//move before new run
	void MoveTimeStats_V2a();	//move before new run
	void MoveTimeStats_V2b();	//move before new run
	void MoveTimeStats_DetailedV1a(); //move before new run
	void MoveTimeStats_DetailedV1b(); //move before new run
	void MoveTimeStats_DetailedV2a(); //move before new run
	void MoveTimeStats_DetailedV2b(); //move before new run
	void MoveTimeStats_DetailedIndivV2b(); //move before new run
	void ShowTimeStatsLegend(); //show legend
	void ShowTimeStats_V1a();	//show statistics
	void ShowTimeStats_V1b();	//show statistics
	void ShowTimeStats_V2a();	//show statistics
	void ShowTimeStats_V2b();	//show statistics
	void ShowTimeStats_DetailedV1a();		//show details
	void ShowTimeStats_DetailedV1b();		//show details
	void ShowTimeStats_DetailedV2a();		//show details
	void ShowTimeStats_DetailedV2b();		//show details
	void ShowTimeStats_DetailedIndivV2b();	//show details
	void ShowTimeStats_Seq();				//show details
	void ShowNodeStats_MLRCalc();			//show details

	double getWholeTime(){ return wholeTime; }
	double getCalcTime(){ return calcTime; }
	double getMemoryAllocForDatasetTime(){ return memoryAllocForDatasetTime; }
	double getSendDatasetToGPUTime(){ return sendDatasetToGPUTime; }
	double getDataTransferToGPUTime(){ return dataTransferToGPUTime; }
	double getDataTransferFromGPUTime(){ return dataTransferFromGPUTime; }
	double getDataReorganizationTime(){ return dataReorganizationTime; }
	double getMemoryAllocDeallocTime(){ return memoryAllocDeallocTime; }
	double getTreeRepoSearchTime(){ return treeRepoSearchTime; }
	double getTreeRepoApplyTime(){ return treeRepoApplyTime; }
	double getTreeRepoInsertTime(){ return treeRepoInsertTime; }
	double getMT_Kernel1Time(){ return MT_kernel1Time; }
	double getMT_Sort1Time(){ return MT_sort1Time; }
	double getMT_Kernel2Time(){ return MT_kernel2Time; }
	double getMT_Kernel3Time(){ return MT_kernel3Time; }
	double getMT_ModelCalcTime(){ return MT_modelCalcTime; }	
	double getMT_FullMLRCalcTime(){ return MT_fullMLRCalcTime; }	
	double getMT_MeanRTCalcTime(){ return MT_meanRTCalcTime; }	
	double getMT_Kernel4Time(){ return MT_kernel4Time; }
	double getMT_Kernel5Time(){ return MT_kernel5Time; }	

	double getSumWholeTime_V1a(){ return sumWholeTime_V1a; }
	double getSumCalcTime_V1a(){ return sumCalcTime_V1a; }	
	double getSumDataTransferToGPUTime_V1a(){ return sumDataTransferToGPUTime_V1a; }
	double getSumDataTransferFromGPUTime_V1a(){ return sumDataTransferFromGPUTime_V1a; }
	double getSumDataReorganizationTime_V1a(){ return sumDataReorganizationTime_V1a; }
	double getSumMemoryAllocDeallocTime_V1a(){ return sumMemoryAllocDeallocTime_V1a; }

	double getSumWholeTime_DetailedV1a(){ return sumWholeTime_DetailedV1a; }
	double getSumCalcTime_DetailedV1a(){ return sumCalcTime_DetailedV1a; }	
	double getSumDataTransferToGPUTime_DetailedV1a(){ return sumDataTransferToGPUTime_DetailedV1a; }
	double getSumDataTransferFromGPUTime_DetailedV1a(){ return sumDataTransferFromGPUTime_DetailedV1a; }
	double getSumDataReorganizationTime_DetailedV1a(){ return sumDataReorganizationTime_DetailedV1a; }
	double getSumMemoryAllocDeallocTime_DetailedV1a(){ return sumMemoryAllocDeallocTime_DetailedV1a; }

	double getSumWholeTime_V1b(){ return sumWholeTime_V1b; }
	double getSumCalcTime_V1b(){ return sumCalcTime_V1b; }
	double getSumDataTransferToGPUTime_V1b(){ return sumDataTransferToGPUTime_V1b; }
	double getSumDataTransferFromGPUTime_V1b(){ return sumDataTransferFromGPUTime_V1b; }
	double getSumDataReorganizationTime_V1b(){ return sumDataReorganizationTime_V1b; }
	double getSumMemoryAllocDeallocTime_V1b(){ return sumMemoryAllocDeallocTime_V1b; }

	double getSumWholeTime_DetailedV1b(){ return sumWholeTime_DetailedV1b; }
	double getSumCalcTime_DetailedV1b(){ return sumCalcTime_DetailedV1b; }
	double getSumDataTransferToGPUTime_DetailedV1b(){ return sumDataTransferToGPUTime_DetailedV1b; }
	double getSumDataTransferFromGPUTime_DetailedV1b(){ return sumDataTransferFromGPUTime_DetailedV1b; }
	double getSumDataReorganizationTime_DetailedV1b(){ return sumDataReorganizationTime_DetailedV1b; }
	double getSumMemoryAllocDeallocTime_DetailedV1b(){ return sumMemoryAllocDeallocTime_DetailedV1b; }

	double getSumWholeTime_V2a(){ return sumWholeTime_V2a; }
	double getSumCalcTime_V2a(){ return sumCalcTime_V2a; }	
	double getSumDataTransferToGPUTime_V2a(){ return sumDataTransferToGPUTime_V2a; }
	double getSumDataTransferFromGPUTime_V2a(){ return sumDataTransferFromGPUTime_V2a; }
	double getSumDataReorganizationTime_V2a(){ return sumDataReorganizationTime_V2a; }
	double getSumMemoryAllocDeallocTime_V2a(){ return sumMemoryAllocDeallocTime_V2a; }

	double getSumWholeTime_DetailedV2a(){ return sumWholeTime_DetailedV2a; }
	double getSumCalcTime_DetailedV2a(){ return sumCalcTime_DetailedV2a; }	
	double getSumDataTransferToGPUTime_DetailedV2a(){ return sumDataTransferToGPUTime_DetailedV2a; }
	double getSumDataTransferFromGPUTime_DetailedV2a(){ return sumDataTransferFromGPUTime_DetailedV2a; }
	double getSumDataReorganizationTime_DetailedV2a(){ return sumDataReorganizationTime_DetailedV2a; }
	double getSumMemoryAllocDeallocTime_DetailedV2a(){ return sumMemoryAllocDeallocTime_DetailedV2a; }

	double getSumWholeTime_V2b(){ return sumWholeTime_V2b; }
	double getSumCalcTime_V2b(){ return sumCalcTime_V2b; }
	double getSumDataTransferToGPUTime_V2b(){ return sumDataTransferToGPUTime_V2b; }
	double getSumDataTransferFromGPUTime_V2b(){ return sumDataTransferFromGPUTime_V2b; }
	double getSumDataReorganizationTime_V2b(){ return sumDataReorganizationTime_V2b; }
	double getSumMemoryAllocDeallocTime_V2b(){ return sumMemoryAllocDeallocTime_V2b; }

	double getSumWholeTime_DetailedV2b(){ return sumWholeTime_DetailedV2b; }
	double getSumCalcTime_DetailedV2b(){ return sumCalcTime_DetailedV2b; }
	double getSumDataTransferToGPUTime_DetailedV2b(){ return sumDataTransferToGPUTime_DetailedV2b; }
	double getSumDataTransferFromGPUTime_DetailedV2b(){ return sumDataTransferFromGPUTime_DetailedV2b; }
	double getSumDataReorganizationTime_DetailedV2b(){ return sumDataReorganizationTime_DetailedV2b; }
	double getSumMemoryAllocDeallocTime_DetailedV2b(){ return sumMemoryAllocDeallocTime_DetailedV2b; }

	double getSumWholeTime_DetailedIndivV2b(){ return sumWholeTime_DetailedIndivV2b; }
	double getSumCalcTime_DetailedIndivV2b(){ return sumCalcTime_DetailedIndivV2b; }
	double getSumDataTransferToGPUTime_DetailedIndivV2b(){ return sumDataTransferToGPUTime_DetailedIndivV2b; }
	double getSumDataTransferFromGPUTime_DetailedIndivV2b(){ return sumDataTransferFromGPUTime_DetailedIndivV2b; }
	double getSumDataReorganizationTime_DetailedIndivV2b(){ return sumDataReorganizationTime_DetailedIndivV2b; }
	double getSumMemoryAllocDeallocTime_DetailedIndivV2b(){ return sumMemoryAllocDeallocTime_DetailedIndivV2b; }	
	double getSumTreeRepoSearchTime_DetailedIndivV2b(){ return sumTreeRepoSearchTime_DetailedIndivV2b; }
	double getSumTreeRepoApplyTime_DetailedIndivV2b(){ return sumTreeRepoApplyTime_DetailedIndivV2b; }
	double getSumTreeRepoInsertTime_DetailedIndivV2b(){ return sumTreeRepoInsertTime_DetailedIndivV2b; }

	double getMT_SumKernel1Time_DetailedIndivV2b(){ return MT_sumKernel1Time_DetailedIndivV2b; }
	double getMT_SumSort1Time_DetailedIndivV2b(){ return MT_sumSort1Time_DetailedIndivV2b; }
	double getMT_SumKernel2Time_DetailedIndivV2b(){ return MT_sumKernel2Time_DetailedIndivV2b; }
	double getMT_SumKernel3Time_DetailedIndivV2b(){ return MT_sumKernel3Time_DetailedIndivV2b; }
	double getMT_SumModelCalcTime_DetailedIndivV2b(){ return MT_sumModelCalcTime_DetailedIndivV2b; }
	double getMT_SumFullMLRCalcTime_DetailedIndivV2b(){ return MT_sumFullMLRCalcTime_DetailedIndivV2b; }
	double getMT_SumMeanRTCalcTime_DetailedIndivV2b(){ return MT_sumMeanRTCalcTime_DetailedIndivV2b; }	
	double getMT_SumKernel4Time_DetailedIndivV2b(){ return MT_sumKernel4Time_DetailedIndivV2b; }
	double getMT_SumKernel5Time_DetailedIndivV2b(){ return MT_sumKernel5Time_DetailedIndivV2b; }


	#ifdef __linux__
	#if CLOCK_GETTIME
	void SaveTimeBegin( struct timespec* timeBegin );
	#else
	void SaveTimeBegin( timeval* timeBegin );
	#endif
	#else
	void SaveTimeBegin( time_t* timeBegin );
	#endif

	#ifdef __linux__
	#if CLOCK_GETTIME
	void SaveTimeEnd( struct timespec* timeBegin, struct timespec* timeEnd, double* timeSum );
	#else
	void SaveTimeEnd( timeval* timeBegin, timeval* timeEnd, double* timeSum );
	#endif
	#else
	void SaveTimeEnd( time_t* timeBegin, time_t* timeEnd, double* timeSum );
	#endif	

	//used when #if DEBUG_TIME_PER_TREE in Worker.h		
	double GetWholeTimeLastDiff();


	void WholeTimeBegin();
	void WholeTimeEnd();

	void CalcTimeBegin();
	void CalcTimeEnd();

	void WholeDatasetTransportTimeBegin();
	void WholeDatasetTransportTimeEnd();

	void MemoryAllocForDatasetTimeBegin();
	void MemoryAllocForDatasetTimeEnd();

	void SendDatasetToGPUTimeBegin();
	void SendDatasetToGPUTimeEnd();

	void DataTransferToGPUTimeBegin();
	void DataTransferToGPUTimeEnd();
	void DataTransferFromGPUTimeBegin();
	void DataTransferFromGPUTimeEnd();

	void DataReorganizationTimeBegin();
	void DataReorganizationTimeEnd();

	void MemoryAllocDeallocTimeBegin();
	void MemoryAllocDeallocTimeEnd();

	void TreeRepoSearchTimeBegin();
	void TreeRepoSearchTimeEnd();

	void TreeRepoApplyTimeBegin();
	void TreeRepoApplyTimeEnd();

	void TreeRepoInsertTimeBegin();
	void TreeRepoInsertTimeEnd();

	void MT_Kernel1TimeBegin();
	void MT_Kernel1TimeEnd();

	void MT_Sort1TimeBegin();
	void MT_Sort1TimeEnd();

	void MT_Kernel2TimeBegin();
	void MT_Kernel2TimeEnd();

	void MT_Kernel3TimeBegin();
	void MT_Kernel3TimeEnd();	

	void MT_ModelCalcTimeBegin();
	void MT_ModelCalcTimeEnd();

	void MT_FullMLRCalcTimeBegin();
	void MT_FullMLRCalcTimeEnd();

	void MT_MeanRTCalcTimeBegin();
	void MT_MeanRTCalcTimeEnd();

	void MT_Kernel4TimeBegin();
	void MT_Kernel4TimeEnd();

	void MT_Kernel5TimeBegin();
	void MT_Kernel5TimeEnd();

	//for additional time variables for sequential and OpenMP algorithms
	double getSeqTime(){ return seqTime; }
	double getSumSeqTime(){ return sumSeqTime; }
	
	void SeqTimeBegin();
	void SeqTimeEnd();	

	void incNMLRNodes(){ nMLRNodes++; }
	void incNReCalcRTNodes(){ nReCalcRTNodes++; nMLRNodes++; }
	void incNReCalcMLRNodes(){ nReCalcMLRNodes++; nMLRNodes++; }

	int getNMLRNodes(){ return nMLRNodes++; }
	int getNRTNodes(){ return nReCalcRTNodes++; }
	int getNReCalcMLRNodes(){ return nReCalcMLRNodes; }
};