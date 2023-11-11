#include "CudaTimeStats.h"

#include "stdio.h"

#define MILLION 1000000.0
#define BILLION 1000000000.0

CCudaTimeStats::CCudaTimeStats(void){
	wholeTime = 0;
	calcTime = 0;
	wholeDatasetTransportTime = 0;
	memoryAllocForDatasetTime = 0;
	sendDatasetToGPUTime = 0;
	dataTransferToGPUTime = 0;
	dataTransferFromGPUTime = 0;
	dataReorganizationTime = 0;
	memoryAllocDeallocTime = 0;
	treeRepoSearchTime = 0;
	treeRepoApplyTime = 0;
	treeRepoInsertTime = 0;
	MT_kernel1Time = 0;
	MT_sort1Time = 0;
	MT_kernel2Time = 0;
	MT_kernel3Time = 0;
	MT_modelCalcTime = 0;
	MT_fullMLRCalcTime = 0;
	MT_meanRTCalcTime = 0;
	MT_kernel4Time = 0;
	MT_kernel5Time = 0;

	//sumWholeTime = 0;
	//sumCalcTime = 0;
	//sumDataTransferToGPUTime = 0;
	//sumDataTransferFromGPUTime = 0;
	//sumDataReorganizationTime = 0;
	//sumMemoryAllocDeallocTime = 0;

	sumWholeTime_V1a = 0;
	sumCalcTime_V1a = 0;
	sumDataTransferToGPUTime_V1a = 0;
	sumDataTransferFromGPUTime_V1a = 0;
	sumDataReorganizationTime_V1a = 0;
	sumMemoryAllocDeallocTime_V1a = 0;

	sumWholeTime_DetailedV1a = 0;
	sumCalcTime_DetailedV1a = 0;
	sumDataTransferToGPUTime_DetailedV1a = 0;
	sumDataTransferFromGPUTime_DetailedV1a = 0;
	sumDataReorganizationTime_DetailedV1a = 0;
	sumMemoryAllocDeallocTime_DetailedV1a = 0;

	sumWholeTime_V1b = 0;
	sumCalcTime_V1b = 0;
	sumDataTransferToGPUTime_V1b = 0;
	sumDataTransferFromGPUTime_V1b = 0;
	sumDataReorganizationTime_V1b = 0;
	sumMemoryAllocDeallocTime_V1b = 0;

	sumWholeTime_DetailedV1b = 0;
	sumCalcTime_DetailedV1b = 0;
	sumDataTransferToGPUTime_DetailedV1b = 0;
	sumDataTransferFromGPUTime_DetailedV1b = 0;
	sumDataReorganizationTime_DetailedV1b = 0;
	sumMemoryAllocDeallocTime_DetailedV1b = 0;

	sumWholeTime_V2a = 0;
	sumCalcTime_V2a = 0;
	sumDataTransferToGPUTime_V2a = 0;
	sumDataTransferFromGPUTime_V2a = 0;
	sumDataReorganizationTime_V2a = 0;
	sumMemoryAllocDeallocTime_V2a = 0;

	sumWholeTime_DetailedV2a = 0;
	sumCalcTime_DetailedV2a = 0;
	sumDataTransferToGPUTime_DetailedV2a = 0;
	sumDataTransferFromGPUTime_DetailedV2a = 0;
	sumDataReorganizationTime_DetailedV2a = 0;
	sumMemoryAllocDeallocTime_DetailedV2a = 0;

	sumWholeTime_V2b = 0;
	sumCalcTime_V2b = 0;
	sumDataTransferToGPUTime_V2b = 0;
	sumDataTransferFromGPUTime_V2b = 0;
	sumDataReorganizationTime_V2b = 0;
	sumMemoryAllocDeallocTime_V2b = 0;

	sumWholeTime_DetailedV2b = 0;
	sumCalcTime_DetailedV2b = 0;
	sumDataTransferToGPUTime_DetailedV2b = 0;
	sumDataTransferFromGPUTime_DetailedV2b = 0;
	sumDataReorganizationTime_DetailedV2b = 0;
	sumMemoryAllocDeallocTime_DetailedV2b = 0;

	sumWholeTime_DetailedIndivV2b = 0;
	sumCalcTime_DetailedIndivV2b = 0;
	sumDataTransferToGPUTime_DetailedIndivV2b = 0;
	sumDataTransferFromGPUTime_DetailedIndivV2b = 0;
	sumDataReorganizationTime_DetailedIndivV2b = 0;
	sumMemoryAllocDeallocTime_DetailedIndivV2b = 0;
	sumTreeRepoSearchTime_DetailedIndivV2b = 0;
	sumTreeRepoApplyTime_DetailedIndivV2b = 0;
	sumTreeRepoInsertTime_DetailedIndivV2b = 0;
	MT_sumKernel1Time_DetailedIndivV2b = 0;
	MT_sumSort1Time_DetailedIndivV2b = 0;
	MT_sumKernel2Time_DetailedIndivV2b = 0;
	MT_sumKernel3Time_DetailedIndivV2b = 0;
	MT_sumModelCalcTime_DetailedIndivV2b = 0;
	MT_sumFullMLRCalcTime_DetailedIndivV2b = 0;
	MT_sumMeanRTCalcTime_DetailedIndivV2b = 0;
	MT_sumKernel4Time_DetailedIndivV2b = 0;
	MT_sumKernel5Time_DetailedIndivV2b = 0;

	seqTime = 0;
	sumSeqTime = 0;

	nMLRNodes = 0;
	nReCalcRTNodes = 0;
	nReCalcMLRNodes = 0;
}


CCudaTimeStats::~CCudaTimeStats(void){
}

void CCudaTimeStats::ClearAllStats(){
	wholeTime = 0;
	calcTime = 0;
	wholeDatasetTransportTime = 0;
	memoryAllocForDatasetTime = 0;
	sendDatasetToGPUTime = 0;
	dataTransferToGPUTime = 0;
	dataTransferFromGPUTime = 0;
	dataReorganizationTime = 0;
	memoryAllocDeallocTime = 0;
	treeRepoSearchTime = 0;
	treeRepoApplyTime = 0;
	treeRepoInsertTime = 0;
	MT_kernel1Time = 0;
	MT_sort1Time = 0;
	MT_kernel2Time = 0;
	MT_kernel3Time = 0;	
	MT_modelCalcTime = 0;
	MT_fullMLRCalcTime = 0;
	MT_meanRTCalcTime = 0;		
	MT_kernel4Time = 0;
	MT_kernel5Time = 0;

	sumWholeTime_V1a = 0;
	sumCalcTime_V1a = 0;
	sumDataTransferToGPUTime_V1a = 0;
	sumDataTransferFromGPUTime_V1a = 0;
	sumDataReorganizationTime_V1a = 0;
	sumMemoryAllocDeallocTime_V1a = 0;

	sumWholeTime_DetailedV1a = 0;
	sumCalcTime_DetailedV1a = 0;
	sumDataTransferToGPUTime_DetailedV1a = 0;
	sumDataTransferFromGPUTime_DetailedV1a = 0;
	sumDataReorganizationTime_DetailedV1a = 0;
	sumMemoryAllocDeallocTime_DetailedV1a = 0;

	sumWholeTime_V1b = 0;
	sumCalcTime_V1b = 0;
	sumDataTransferToGPUTime_V1b = 0;
	sumDataTransferFromGPUTime_V1b = 0;
	sumDataReorganizationTime_V1b = 0;
	sumMemoryAllocDeallocTime_V1b = 0;

	sumWholeTime_DetailedV1b = 0;
	sumCalcTime_DetailedV1b = 0;
	sumDataTransferToGPUTime_DetailedV1b = 0;
	sumDataTransferFromGPUTime_DetailedV1b = 0;
	sumDataReorganizationTime_DetailedV1b = 0;
	sumMemoryAllocDeallocTime_DetailedV1b = 0;

	sumWholeTime_V2a = 0;
	sumCalcTime_V2a = 0;
	sumDataTransferToGPUTime_V2a = 0;
	sumDataTransferFromGPUTime_V2a = 0;
	sumDataReorganizationTime_V2a = 0;
	sumMemoryAllocDeallocTime_V2a = 0;

	sumWholeTime_DetailedV2a = 0;
	sumCalcTime_DetailedV2a = 0;
	sumDataTransferToGPUTime_DetailedV2a = 0;
	sumDataTransferFromGPUTime_DetailedV2a = 0;
	sumDataReorganizationTime_DetailedV2a = 0;
	sumMemoryAllocDeallocTime_DetailedV2a = 0;

	sumWholeTime_V2b = 0;
	sumCalcTime_V2b = 0;
	sumDataTransferToGPUTime_V2b = 0;
	sumDataTransferFromGPUTime_V2b = 0;
	sumDataReorganizationTime_V2b = 0;
	sumMemoryAllocDeallocTime_V2b = 0;

	sumWholeTime_DetailedV2b = 0;
	sumCalcTime_DetailedV2b = 0;
	sumDataTransferToGPUTime_DetailedV2b = 0;
	sumDataTransferFromGPUTime_DetailedV2b = 0;
	sumDataReorganizationTime_DetailedV2b = 0;
	sumMemoryAllocDeallocTime_DetailedV2b = 0;

	sumWholeTime_DetailedIndivV2b = 0;
	sumCalcTime_DetailedIndivV2b = 0;
	sumDataTransferToGPUTime_DetailedIndivV2b = 0;
	sumDataTransferFromGPUTime_DetailedIndivV2b = 0;
	sumDataReorganizationTime_DetailedIndivV2b = 0;
	sumMemoryAllocDeallocTime_DetailedIndivV2b = 0;
	sumTreeRepoSearchTime_DetailedIndivV2b = 0;
	sumTreeRepoApplyTime_DetailedIndivV2b = 0;
	sumTreeRepoInsertTime_DetailedIndivV2b = 0;
	MT_sumKernel1Time_DetailedIndivV2b = 0;
	MT_sumSort1Time_DetailedIndivV2b = 0;
	MT_sumKernel2Time_DetailedIndivV2b = 0;
	MT_sumKernel3Time_DetailedIndivV2b = 0;
	MT_sumModelCalcTime_DetailedIndivV2b = 0;	
	MT_sumFullMLRCalcTime_DetailedIndivV2b = 0;	
	MT_sumMeanRTCalcTime_DetailedIndivV2b = 0;
	MT_sumKernel4Time_DetailedIndivV2b = 0;
	MT_sumKernel5Time_DetailedIndivV2b = 0;

	seqTime = 0;
	sumSeqTime = 0;
}

void CCudaTimeStats::ClearCurrStats(){
	wholeTime = 0.0;
	calcTime = 0.0;
	dataTransferFromGPUTime = 0.0;
	dataTransferToGPUTime = 0.0;
	dataReorganizationTime = 0.0;
	memoryAllocDeallocTime = 0;
	treeRepoSearchTime = 0.0;
	treeRepoApplyTime = 0;
	treeRepoInsertTime = 0.0;
	seqTime = 0.0;

	MT_kernel1Time = 0;
	MT_sort1Time = 0;
	MT_kernel2Time = 0;
	MT_kernel3Time = 0;
	MT_modelCalcTime = 0;
	MT_fullMLRCalcTime = 0;
	MT_meanRTCalcTime = 0;
	MT_kernel4Time = 0;
	MT_kernel5Time = 0;
}

void CCudaTimeStats::MoveTimeStats_V1a(){	
	sumWholeTime_V1a += wholeTime;
	sumCalcTime_V1a += calcTime;
	sumDataTransferFromGPUTime_V1a += dataTransferFromGPUTime;
	sumDataTransferToGPUTime_V1a += dataTransferToGPUTime;
	sumDataReorganizationTime_V1a += dataReorganizationTime;
	sumMemoryAllocDeallocTime_V1a += memoryAllocDeallocTime;
}

void CCudaTimeStats::MoveTimeStats_DetailedV1a(){	
	sumWholeTime_DetailedV1a += wholeTime;
	sumCalcTime_DetailedV1a += calcTime;
	sumDataTransferFromGPUTime_DetailedV1a += dataTransferFromGPUTime;
	sumDataTransferToGPUTime_DetailedV1a += dataTransferToGPUTime;
	sumDataReorganizationTime_DetailedV1a += dataReorganizationTime;
	sumMemoryAllocDeallocTime_DetailedV1a += memoryAllocDeallocTime;
}

void CCudaTimeStats::MoveTimeStats_V1b(){	
	sumWholeTime_V1b += wholeTime;
	sumCalcTime_V1b += calcTime;
	sumDataTransferFromGPUTime_V1b += dataTransferFromGPUTime;
	sumDataTransferToGPUTime_V1b += dataTransferToGPUTime;
	sumDataReorganizationTime_V1b += dataReorganizationTime;
	sumMemoryAllocDeallocTime_V1b += memoryAllocDeallocTime;
}

void CCudaTimeStats::MoveTimeStats_DetailedV1b(){	
	sumWholeTime_DetailedV1b += wholeTime;
	sumCalcTime_DetailedV1b += calcTime;
	sumDataTransferFromGPUTime_DetailedV1b += dataTransferFromGPUTime;
	sumDataTransferToGPUTime_DetailedV1b += dataTransferToGPUTime;
	sumDataReorganizationTime_DetailedV1b += dataReorganizationTime;
	sumMemoryAllocDeallocTime_DetailedV1b += memoryAllocDeallocTime;
}

void CCudaTimeStats::MoveTimeStats_V2a(){	
	sumWholeTime_V2a += wholeTime;
	sumCalcTime_V2a += calcTime;
	sumDataTransferFromGPUTime_V2a += dataTransferFromGPUTime;
	sumDataTransferToGPUTime_V2a += dataTransferToGPUTime;
	sumDataReorganizationTime_V2a += dataReorganizationTime;
	sumMemoryAllocDeallocTime_V2a += memoryAllocDeallocTime;
}

void CCudaTimeStats::MoveTimeStats_DetailedV2a(){	
	sumWholeTime_DetailedV2a += wholeTime;
	sumCalcTime_DetailedV2a += calcTime;
	sumDataTransferFromGPUTime_DetailedV2a += dataTransferFromGPUTime;
	sumDataTransferToGPUTime_DetailedV2a += dataTransferToGPUTime;
	sumDataReorganizationTime_DetailedV2a += dataReorganizationTime;
	sumMemoryAllocDeallocTime_DetailedV2a += memoryAllocDeallocTime;
}

void CCudaTimeStats::MoveTimeStats_V2b(){	
	sumWholeTime_V2b += wholeTime;
	sumCalcTime_V2b += calcTime;
	sumDataTransferFromGPUTime_V2b += dataTransferFromGPUTime;
	sumDataTransferToGPUTime_V2b += dataTransferToGPUTime;
	sumDataReorganizationTime_V2b += dataReorganizationTime;
	sumMemoryAllocDeallocTime_V2b += memoryAllocDeallocTime;
}

void CCudaTimeStats::MoveTimeStats_DetailedV2b(){	
	sumWholeTime_DetailedV2b += wholeTime;
	sumCalcTime_DetailedV2b += calcTime;
	sumDataTransferFromGPUTime_DetailedV2b += dataTransferFromGPUTime;
	sumDataTransferToGPUTime_DetailedV2b += dataTransferToGPUTime;
	sumDataReorganizationTime_DetailedV2b += dataReorganizationTime;
	sumMemoryAllocDeallocTime_DetailedV2b += memoryAllocDeallocTime;
}

void CCudaTimeStats::MoveTimeStats_DetailedIndivV2b(){	
	sumWholeTime_DetailedIndivV2b += wholeTime;
	sumCalcTime_DetailedIndivV2b += calcTime;
	sumDataTransferFromGPUTime_DetailedIndivV2b += dataTransferFromGPUTime;
	sumDataTransferToGPUTime_DetailedIndivV2b += dataTransferToGPUTime;
	sumDataReorganizationTime_DetailedIndivV2b += dataReorganizationTime;
	sumMemoryAllocDeallocTime_DetailedIndivV2b += memoryAllocDeallocTime;
	sumTreeRepoSearchTime_DetailedIndivV2b += treeRepoSearchTime;
	sumTreeRepoApplyTime_DetailedIndivV2b += treeRepoApplyTime;
	sumTreeRepoInsertTime_DetailedIndivV2b += treeRepoInsertTime;

	MT_sumKernel1Time_DetailedIndivV2b += MT_kernel1Time;
	MT_sumSort1Time_DetailedIndivV2b += MT_sort1Time;
	MT_sumKernel2Time_DetailedIndivV2b += MT_kernel2Time;
	MT_sumKernel3Time_DetailedIndivV2b += MT_kernel3Time;
	MT_sumModelCalcTime_DetailedIndivV2b += MT_modelCalcTime;
	MT_sumFullMLRCalcTime_DetailedIndivV2b += MT_fullMLRCalcTime;
	MT_sumMeanRTCalcTime_DetailedIndivV2b += MT_meanRTCalcTime;	
	MT_sumKernel4Time_DetailedIndivV2b += MT_kernel4Time;
	MT_sumKernel5Time_DetailedIndivV2b += MT_kernel5Time;
	
}

void CCudaTimeStats::ShowTimeStatsLegend(){
	printf( "\t" );
	printf( "whole time (p)\t" );
	printf( "calc time\t" );	
	printf( "send to GPU\t" );
	printf( "recv from GPU\t" );	
	printf( "data reorgan\t" );
	printf( "de/alloc\t" );	
	printf( "whole DS transp\t" );	
	printf( "DS alloc\t" );
	printf( "DS send\t" );
	printf( "REPO search\t" );
	printf( "REPO apply\t" );
	printf( "REPO insert\t\n" );
}

void CCudaTimeStats::ShowTimeStats_V1a(){
	printf( "V1a-\t" );
	printf( "%.2f/%.3f(%d)(%d)\t", wholeTime, sumWholeTime_V1a, ( int ) ( sumSeqTime / sumWholeTime_V1a ), ( int ) ( sumSeqTime / ( sumWholeTime_V1a - wholeDatasetTransportTime ) ) );
	printf( "%.2f/%.3f (%d%%)\t", calcTime, sumCalcTime_V1a, ( int )( sumCalcTime_V1a / sumWholeTime_V1a * 100 ) );	
	printf( "%.2f/%.3f (%d%%)\t", dataTransferToGPUTime, sumDataTransferToGPUTime_V1a, ( int )(sumDataTransferToGPUTime_V1a / sumWholeTime_V1a * 100 ) );
	printf( "%.2f/%.3f (%d%%)\t", dataTransferFromGPUTime, sumDataTransferFromGPUTime_V1a, ( int )(sumDataTransferFromGPUTime_V1a / sumWholeTime_V1a * 100 ) );	
	printf( "%.2f/%.3f (%d%%)\t", dataReorganizationTime, sumDataReorganizationTime_V1a, ( int )(sumDataReorganizationTime_V1a / sumWholeTime_V1a * 100 ) );
	printf( "%.2f/%.3f (%d%%)\t", memoryAllocDeallocTime, sumMemoryAllocDeallocTime_V1a, ( int )(sumMemoryAllocDeallocTime_V1a / sumWholeTime_V1a * 100 ) );	
	printf( "%.2f (%d%%)\t", wholeDatasetTransportTime, ( int )(wholeDatasetTransportTime / sumWholeTime_V1a * 100 ) );
	printf( "%.2f (%d%%)\t", memoryAllocForDatasetTime, ( int )(memoryAllocForDatasetTime / sumWholeTime_V1a * 100 ) );
	printf( "%.2f (%d%%)\n", sendDatasetToGPUTime, ( int )(sendDatasetToGPUTime / sumWholeTime_V1a * 100 ) );
}

void CCudaTimeStats::ShowTimeStats_DetailedV1a(){
	printf( "DV1a-\t" );
	printf( "%.2f/%.3f(%d)(%d)\t", wholeTime, sumWholeTime_DetailedV1a, ( int ) ( sumSeqTime / sumWholeTime_DetailedV1a ), ( int ) ( sumSeqTime / ( sumWholeTime_DetailedV1a - wholeDatasetTransportTime ) ) );
	printf( "%.2f/%.3f (%d%%)\t", calcTime, sumCalcTime_DetailedV1a, ( int )( sumCalcTime_DetailedV1a / sumWholeTime_DetailedV1a * 100 ) );	
	printf( "%.2f/%.3f (%d%%)\t", dataTransferToGPUTime, sumDataTransferToGPUTime_DetailedV1a, ( int )(sumDataTransferToGPUTime_DetailedV1a / sumWholeTime_DetailedV1a * 100 ) );
	printf( "%.2f/%.3f (%d%%)\t", dataTransferFromGPUTime, sumDataTransferFromGPUTime_DetailedV1a, ( int )(sumDataTransferFromGPUTime_DetailedV1a / sumWholeTime_DetailedV1a * 100 ) );	
	printf( "%.2f/%.3f (%d%%)\t", dataReorganizationTime, sumDataReorganizationTime_DetailedV1a, ( int )(sumDataReorganizationTime_DetailedV1a / sumWholeTime_DetailedV1a * 100 ) );
	printf( "%.2f/%.3f (%d%%)\t", memoryAllocDeallocTime, sumMemoryAllocDeallocTime_DetailedV1a, ( int )(sumMemoryAllocDeallocTime_DetailedV1a / sumWholeTime_DetailedV1a * 100 ) );	
	printf( "%.2f (%d%%)\t", wholeDatasetTransportTime, ( int )(wholeDatasetTransportTime / sumWholeTime_DetailedV1a * 100 ) );
	printf( "%.2f (%d%%)\t", memoryAllocForDatasetTime, ( int )(memoryAllocForDatasetTime / sumWholeTime_DetailedV1a * 100 ) );
	printf( "%.2f (%d%%)\n", sendDatasetToGPUTime, ( int )(sendDatasetToGPUTime / sumWholeTime_DetailedV1a * 100 ) );
}

void CCudaTimeStats::ShowTimeStats_V1b(){
	printf( "V1b-\t" );
	printf( "%.2f/%.3f(%d)(%d)\t", wholeTime, sumWholeTime_V1b, ( int ) ( sumSeqTime / sumWholeTime_V1b ), ( int ) ( sumSeqTime / ( sumWholeTime_V1b - wholeDatasetTransportTime ) ) );
	printf( "%.2f/%.3f (%d%%)\t", calcTime, sumCalcTime_V1b, ( int )( sumCalcTime_V1b / sumWholeTime_V1b * 100 ) );	
	printf( "%.2f/%.3f (%d%%)\t", dataTransferToGPUTime, sumDataTransferToGPUTime_V1b, ( int )(sumDataTransferToGPUTime_V1b / sumWholeTime_V1b * 100 ) );
	printf( "%.2f/%.3f (%d%%)\t", dataTransferFromGPUTime, sumDataTransferFromGPUTime_V1b, ( int )(sumDataTransferFromGPUTime_V1b / sumWholeTime_V1b * 100 ) );	
	printf( "%.2f/%.3f (%d%%)\t", dataReorganizationTime, sumDataReorganizationTime_V1b, ( int )(sumDataReorganizationTime_V1b / sumWholeTime_V1b * 100 ) );
	printf( "%.2f/%.3f (%d%%)\t", memoryAllocDeallocTime, sumMemoryAllocDeallocTime_V1b, ( int )(sumMemoryAllocDeallocTime_V1b / sumWholeTime_V1b * 100 ) );	
	printf( "%.2f (%d%%)\t", wholeDatasetTransportTime, ( int )(wholeDatasetTransportTime / sumWholeTime_V1b * 100 ) );
	printf( "%.2f (%d%%)\t", memoryAllocForDatasetTime, ( int )(memoryAllocForDatasetTime / sumWholeTime_V1b * 100 ) );
	printf( "%.2f (%d%%)\n", sendDatasetToGPUTime, ( int )(sendDatasetToGPUTime / sumWholeTime_V1b * 100 ) );
}

void CCudaTimeStats::ShowTimeStats_DetailedV1b(){
	printf( "DV1b-\t" );
	printf( "%.2f/%.3f(%d)(%d)\t", wholeTime, sumWholeTime_DetailedV1b, ( int ) ( sumSeqTime / sumWholeTime_DetailedV1b ), ( int ) ( sumSeqTime / ( sumWholeTime_DetailedV1b - wholeDatasetTransportTime ) ) );
	printf( "%.2f/%.3f (%d%%)\t", calcTime, sumCalcTime_DetailedV1b, ( int )( sumCalcTime_DetailedV1b / sumWholeTime_DetailedV1b * 100 ) );	
	printf( "%.2f/%.3f (%d%%)\t", dataTransferToGPUTime, sumDataTransferToGPUTime_DetailedV1b, ( int )(sumDataTransferToGPUTime_DetailedV1b / sumWholeTime_DetailedV1b * 100 ) );
	printf( "%.2f/%.3f (%d%%)\t", dataTransferFromGPUTime, sumDataTransferFromGPUTime_DetailedV1b, ( int )(sumDataTransferFromGPUTime_DetailedV1b / sumWholeTime_DetailedV1b * 100 ) );	
	printf( "%.2f/%.3f (%d%%)\t", dataReorganizationTime, sumDataReorganizationTime_DetailedV1b, ( int )(sumDataReorganizationTime_DetailedV1b / sumWholeTime_DetailedV1b * 100 ) );
	printf( "%.2f/%.3f (%d%%)\t", memoryAllocDeallocTime, sumMemoryAllocDeallocTime_DetailedV1b, ( int )(sumMemoryAllocDeallocTime_DetailedV1b / sumWholeTime_DetailedV1b * 100 ) );	
	printf( "%.2f (%d%%)\t", wholeDatasetTransportTime, ( int )(wholeDatasetTransportTime / sumWholeTime_DetailedV1b * 100 ) );
	printf( "%.2f (%d%%)\t", memoryAllocForDatasetTime, ( int )(memoryAllocForDatasetTime / sumWholeTime_DetailedV1b * 100 ) );
	printf( "%.2f (%d%%)\n", sendDatasetToGPUTime, ( int )(sendDatasetToGPUTime / sumWholeTime_DetailedV1b * 100 ) );
}

void CCudaTimeStats::ShowTimeStats_V2a(){
	printf( "V2a-\t" );
	printf( "%.2f/%.3f(%d)(%d)\t", wholeTime, sumWholeTime_V2a, ( int ) ( sumSeqTime / sumWholeTime_V2a ), ( int ) ( sumSeqTime / ( sumWholeTime_V2a - wholeDatasetTransportTime ) ) );
	printf( "%.2f/%.3f (%d%%)\t", calcTime, sumCalcTime_V2a, ( int )( sumCalcTime_V2a / sumWholeTime_V2a * 100 ) );	
	printf( "%.2f/%.3f (%d%%)\t", dataTransferToGPUTime, sumDataTransferToGPUTime_V2a, ( int )(sumDataTransferToGPUTime_V2a / sumWholeTime_V2a * 100 ) );
	printf( "%.2f/%.3f (%d%%)\t", dataTransferFromGPUTime, sumDataTransferFromGPUTime_V2a, ( int )(sumDataTransferFromGPUTime_V2a / sumWholeTime_V2a * 100 ) );	
	printf( "%.2f/%.3f (%d%%)\t", dataReorganizationTime, sumDataReorganizationTime_V2a, ( int )(sumDataReorganizationTime_V2a / sumWholeTime_V2a * 100 ) );
	printf( "%.2f/%.3f (%d%%)\t", memoryAllocDeallocTime, sumMemoryAllocDeallocTime_V2a, ( int )(sumMemoryAllocDeallocTime_V2a / sumWholeTime_V2a * 100 ) );	
	printf( "%.2f (%d%%)\t", wholeDatasetTransportTime, ( int )(wholeDatasetTransportTime / sumWholeTime_V2a * 100 ) );
	printf( "%.2f (%d%%)\t", memoryAllocForDatasetTime, ( int )(memoryAllocForDatasetTime / sumWholeTime_V2a * 100 ) );
	printf( "%.2f (%d%%)\n", sendDatasetToGPUTime, ( int )(sendDatasetToGPUTime / sumWholeTime_V2a * 100 ) );
}

void CCudaTimeStats::ShowTimeStats_DetailedV2a(){
	printf( "DV2a-\t" );
	printf( "%.2f/%.3f(%d)(%d)\t", wholeTime, sumWholeTime_DetailedV2a, ( int ) ( sumSeqTime / sumWholeTime_DetailedV2a ), ( int ) ( sumSeqTime / ( sumWholeTime_DetailedV2a - wholeDatasetTransportTime ) ) );
	printf( "%.2f/%.3f (%d%%)\t", calcTime, sumCalcTime_DetailedV2a, ( int )( sumCalcTime_DetailedV2a / sumWholeTime_DetailedV2a * 100 ) );	
	printf( "%.2f/%.3f (%d%%)\t", dataTransferToGPUTime, sumDataTransferToGPUTime_DetailedV2a, ( int )(sumDataTransferToGPUTime_DetailedV2a / sumWholeTime_DetailedV2a * 100 ) );
	printf( "%.2f/%.3f (%d%%)\t", dataTransferFromGPUTime, sumDataTransferFromGPUTime_DetailedV2a, ( int )(sumDataTransferFromGPUTime_DetailedV2a / sumWholeTime_DetailedV2a * 100 ) );	
	printf( "%.2f/%.3f (%d%%)\t", dataReorganizationTime, sumDataReorganizationTime_DetailedV2a, ( int )(sumDataReorganizationTime_DetailedV2a / sumWholeTime_DetailedV2a * 100 ) );
	printf( "%.2f/%.3f (%d%%)\t", memoryAllocDeallocTime, sumMemoryAllocDeallocTime_DetailedV2a, ( int )(sumMemoryAllocDeallocTime_DetailedV2a / sumWholeTime_DetailedV2a * 100 ) );	
	printf( "%.2f (%d%%)\t", wholeDatasetTransportTime, ( int )(wholeDatasetTransportTime / sumWholeTime_DetailedV2a * 100 ) );
	printf( "%.2f (%d%%)\t", memoryAllocForDatasetTime, ( int )(memoryAllocForDatasetTime / sumWholeTime_DetailedV2a * 100 ) );
	printf( "%.2f (%d%%)\n", sendDatasetToGPUTime, ( int )(sendDatasetToGPUTime / sumWholeTime_DetailedV2a * 100 ) );
}

void CCudaTimeStats::ShowTimeStats_V2b(){
	printf( "V2b-\t" );
	printf( "%.2f/%.3f(%d)(%d)\t", wholeTime, sumWholeTime_V2b, ( int ) ( sumSeqTime / sumWholeTime_V2b ), ( int ) ( sumSeqTime / ( sumWholeTime_V2b - wholeDatasetTransportTime ) ) );
	printf( "%.2f/%.3f (%d%%)\t", calcTime, sumCalcTime_V2b, ( int )( sumCalcTime_V2b / sumWholeTime_V2b * 100 ) );	
	printf( "%.2f/%.3f (%d%%)\t", dataTransferToGPUTime, sumDataTransferToGPUTime_V2b, ( int )(sumDataTransferToGPUTime_V2b / sumWholeTime_V2b * 100 ) );
	printf( "%.2f/%.3f (%d%%)\t", dataTransferFromGPUTime, sumDataTransferFromGPUTime_V2b, ( int )(sumDataTransferFromGPUTime_V2b / sumWholeTime_V2b * 100 ) );	
	printf( "%.2f/%.3f (%d%%)\t", dataReorganizationTime, sumDataReorganizationTime_V2b, ( int )(sumDataReorganizationTime_V2b / sumWholeTime_V2b * 100 ) );
	printf( "%.2f/%.3f (%d%%)\t", memoryAllocDeallocTime, sumMemoryAllocDeallocTime_V2b, ( int )(sumMemoryAllocDeallocTime_V2b / sumWholeTime_V2b * 100 ) );	
	printf( "%.2f (%d%%)\t", wholeDatasetTransportTime, ( int )(wholeDatasetTransportTime / sumWholeTime_V2b * 100 ) );
	printf( "%.2f (%d%%)\t", memoryAllocForDatasetTime, ( int )(memoryAllocForDatasetTime / sumWholeTime_V2b * 100 ) );
	printf( "%.2f (%d%%)\n", sendDatasetToGPUTime, ( int )(sendDatasetToGPUTime / sumWholeTime_V2b * 100 ) );
}

void CCudaTimeStats::ShowTimeStats_DetailedV2b(){
	printf( "DV2b-\t" );
	printf( "%.2f/%.3f(%d)(%d)\t", wholeTime, sumWholeTime_DetailedV2b, ( int ) ( sumSeqTime / sumWholeTime_DetailedV2b ), ( int ) ( sumSeqTime / ( sumWholeTime_DetailedV2b - wholeDatasetTransportTime ) ) );
	printf( "%.2f/%.3f (%d%%)\t", calcTime, sumCalcTime_DetailedV2b, ( int )( sumCalcTime_DetailedV2b / sumWholeTime_DetailedV2b * 100 ) );	
	printf( "%.2f/%.3f (%d%%)\t", dataTransferToGPUTime, sumDataTransferToGPUTime_DetailedV2b, ( int )(sumDataTransferToGPUTime_DetailedV2b / sumWholeTime_DetailedV2b * 100 ) );
	printf( "%.2f/%.3f (%d%%)\t", dataTransferFromGPUTime, sumDataTransferFromGPUTime_DetailedV2b, ( int )(sumDataTransferFromGPUTime_DetailedV2b / sumWholeTime_DetailedV2b * 100 ) );	
	printf( "%.2f/%.3f (%d%%)\t", dataReorganizationTime, sumDataReorganizationTime_DetailedV2b, ( int )(sumDataReorganizationTime_DetailedV2b / sumWholeTime_DetailedV2b * 100 ) );
	printf( "%.2f/%.3f (%d%%)\t", memoryAllocDeallocTime, sumMemoryAllocDeallocTime_DetailedV2b, ( int )(sumMemoryAllocDeallocTime_DetailedV2b / sumWholeTime_DetailedV2b * 100 ) );	
	printf( "%.2f (%d%%)\t", wholeDatasetTransportTime, ( int )(wholeDatasetTransportTime / sumWholeTime_DetailedV2b * 100 ) );
	printf( "%.2f (%d%%)\t", memoryAllocForDatasetTime, ( int )(memoryAllocForDatasetTime / sumWholeTime_DetailedV2b * 100 ) );
	printf( "%.2f (%d%%)\n", sendDatasetToGPUTime, ( int )(sendDatasetToGPUTime / sumWholeTime_DetailedV2b * 100 ) );
}

void CCudaTimeStats::ShowTimeStats_DetailedIndivV2b(){
	printf( "DIV2b-\t" );
	printf( "%.2f/%.3f(%.2f)(%.2f)\t", wholeTime, sumWholeTime_DetailedIndivV2b, ( sumSeqTime / sumWholeTime_DetailedIndivV2b ), ( sumSeqTime / ( sumWholeTime_DetailedIndivV2b - wholeDatasetTransportTime ) ) );
	printf( "%.2f/%.3f (%.2f%%)\t", calcTime, sumCalcTime_DetailedIndivV2b, sumCalcTime_DetailedIndivV2b / sumWholeTime_DetailedIndivV2b * 100 );	
	printf( "%.2f/%.3f (%.2f%%)\t", dataTransferToGPUTime, sumDataTransferToGPUTime_DetailedIndivV2b, sumDataTransferToGPUTime_DetailedIndivV2b / sumWholeTime_DetailedIndivV2b * 100 );
	printf( "%.2f/%.3f (%.2f%%)\t", dataTransferFromGPUTime, sumDataTransferFromGPUTime_DetailedIndivV2b, sumDataTransferFromGPUTime_DetailedIndivV2b / sumWholeTime_DetailedIndivV2b * 100 );	
	printf( "%.2f/%.3f (%.2f%%)\t", dataReorganizationTime, sumDataReorganizationTime_DetailedIndivV2b, sumDataReorganizationTime_DetailedIndivV2b / sumWholeTime_DetailedIndivV2b * 100 );
	printf( "%.2f/%.3f (%.2f%%)\t", memoryAllocDeallocTime, sumMemoryAllocDeallocTime_DetailedIndivV2b, sumMemoryAllocDeallocTime_DetailedIndivV2b / sumWholeTime_DetailedIndivV2b * 100 );	
	printf( "%.2f (%.2f%%)\t", wholeDatasetTransportTime, wholeDatasetTransportTime / sumWholeTime_DetailedIndivV2b * 100 );
	printf( "%.2f (%.2f%%)\t", memoryAllocForDatasetTime, memoryAllocForDatasetTime / sumWholeTime_DetailedIndivV2b * 100 );
	printf( "%.2f (%.2f%%)\t", sendDatasetToGPUTime, sendDatasetToGPUTime / sumWholeTime_DetailedIndivV2b * 100 );
	printf( "%.2f/%.3f (%.2f%%)\t", treeRepoSearchTime, sumTreeRepoSearchTime_DetailedIndivV2b, sumTreeRepoSearchTime_DetailedIndivV2b / sumWholeTime_DetailedIndivV2b * 100 );
	printf( "%.2f/%.3f (%.2f%%)\t", treeRepoApplyTime, sumTreeRepoApplyTime_DetailedIndivV2b, sumTreeRepoApplyTime_DetailedIndivV2b / sumWholeTime_DetailedIndivV2b * 100 );
	printf( "%.2f/%.3f (%.2f%%)\n", treeRepoInsertTime, sumTreeRepoInsertTime_DetailedIndivV2b, sumTreeRepoInsertTime_DetailedIndivV2b / sumWholeTime_DetailedIndivV2b * 100 );
}

void CCudaTimeStats::ShowTimeStats_Seq(){
	printf("Seq-\t");
	printf("%f/%f\n", seqTime, sumSeqTime);
}

void CCudaTimeStats::ShowNodeStats_MLRCalc(){
	printf("MLR nodes: %d / %d / %d (MLR/RT/ALL)\n", nReCalcMLRNodes, nReCalcRTNodes, nMLRNodes);
}

#ifdef __linux__
#if CLOCK_GETTIME
void CCudaTimeStats::SaveTimeBegin( struct timespec* timeBegin ){
	clock_gettime( CLOCK_REALTIME, timeBegin );
}
#else
void CCudaTimeStats::SaveTimeBegin( timeval* timeBegin ){	
	gettimeofday( timeBegin, NULL );
}
#endif
#else
void CCudaTimeStats::SaveTimeBegin( time_t* timeBegin ){	
	time( timeBegin );
}
#endif


#ifdef __linux__
#if CLOCK_GETTIME
void CCudaTimeStats::SaveTimeEnd( struct timespec* timeBegin, struct timespec* timeEnd, double* timeSum ){
	clock_gettime( CLOCK_REALTIME, timeEnd );
	(*timeSum) += ((*timeEnd).tv_sec - (*timeBegin).tv_sec) + ((*timeEnd).tv_nsec - (*timeBegin).tv_nsec) / BILLION;
}
#else
void CCudaTimeStats::SaveTimeEnd( timeval* timeBegin, timeval* timeEnd, double* timeSum ){
	gettimeofday( timeEnd, NULL );
	(*timeSum) += ((*timeEnd).tv_sec + ((*timeEnd).tv_usec / MILLION)) - ((*timeBegin).tv_sec + ((*timeBegin).tv_usec / MILLION));
}
#endif
#else
void CCudaTimeStats::SaveTimeEnd( time_t* timeBegin, time_t* timeEnd, double* timeSum ){
	time( timeEnd );
	*timeSum += difftime( *timeEnd, *timeBegin );
}
#endif	

#ifdef __linux__
#if CLOCK_GETTIME
double CCudaTimeStats::GetWholeTimeLastDiff() {
	return (wholeTimeEnd.tv_sec - wholeTimeBegin.tv_sec) + (wholeTimeEnd.tv_nsec - wholeTimeBegin.tv_nsec) / BILLION;
}
#else
double CCudaTimeStats::GetWholeTimeLastDiff() {
	return (wholeTimeEnd.tv_sec + (wholeTimeEnd.tv_usec / MILLION)) - (wholeTimeBegin.tv_sec + (wholeTimeBegin.tv_usec / MILLION));
}
#endif
#else
double CCudaTimeStats::GetWholeTimeLastDiff() {
	return difftime(wholeTimeEnd, wholeTimeBegin);
}
#endif	



void CCudaTimeStats::WholeTimeBegin(){
	SaveTimeBegin( &wholeTimeBegin );	
}

void CCudaTimeStats::WholeTimeEnd(){
	SaveTimeEnd( &wholeTimeBegin, &wholeTimeEnd, &wholeTime );	
}	

void CCudaTimeStats::CalcTimeBegin(){
	SaveTimeBegin( &calcTimeBegin );	
}

void CCudaTimeStats::CalcTimeEnd(){
	SaveTimeEnd( &calcTimeBegin, &calcTimeEnd, &calcTime );	
}

void CCudaTimeStats::WholeDatasetTransportTimeBegin(){
	SaveTimeBegin( &wholeDatasetTransportTimeBegin );	
}

void CCudaTimeStats::WholeDatasetTransportTimeEnd(){
	SaveTimeEnd( &wholeDatasetTransportTimeBegin, &wholeDatasetTransportTimeEnd, &wholeDatasetTransportTime );		
}

void CCudaTimeStats::MemoryAllocForDatasetTimeBegin(){
	SaveTimeBegin( &memoryAllocForDatasetTimeBegin );	
}

void CCudaTimeStats::MemoryAllocForDatasetTimeEnd(){
	SaveTimeEnd( &memoryAllocForDatasetTimeBegin, &memoryAllocForDatasetTimeEnd, &memoryAllocForDatasetTime );
}

void CCudaTimeStats::SendDatasetToGPUTimeBegin(){
	SaveTimeBegin( &sendDatasetToGPUTimeBegin );	
}

void CCudaTimeStats::SendDatasetToGPUTimeEnd(){
	SaveTimeEnd( &sendDatasetToGPUTimeBegin, &sendDatasetToGPUTimeEnd, &sendDatasetToGPUTime );
}
                     
void CCudaTimeStats::DataTransferFromGPUTimeBegin(){
	SaveTimeBegin( &dataTransferFromGPUTimeBegin );	
}

void CCudaTimeStats::DataTransferFromGPUTimeEnd(){
	SaveTimeEnd( &dataTransferFromGPUTimeBegin, &dataTransferFromGPUTimeEnd, &dataTransferFromGPUTime );
}

void CCudaTimeStats::DataTransferToGPUTimeBegin(){
	SaveTimeBegin( &dataTransferToGPUTimeBegin );
}

void CCudaTimeStats::DataTransferToGPUTimeEnd(){
	SaveTimeEnd( &dataTransferToGPUTimeBegin, &dataTransferToGPUTimeEnd, &dataTransferToGPUTime );
}

void CCudaTimeStats::DataReorganizationTimeBegin(){
	SaveTimeBegin( &dataReorganizationTimeBegin );	
}

void CCudaTimeStats::DataReorganizationTimeEnd(){
	SaveTimeEnd( &dataReorganizationTimeBegin, &dataReorganizationTimeEnd, &dataReorganizationTime );	
}

void CCudaTimeStats::MemoryAllocDeallocTimeBegin(){
	SaveTimeBegin( &memoryAllocDeallocTimeBegin );	
}

void CCudaTimeStats::MemoryAllocDeallocTimeEnd(){
	SaveTimeEnd( &memoryAllocDeallocTimeBegin, &memoryAllocDeallocTimeEnd, &memoryAllocDeallocTime );	
}

void CCudaTimeStats::TreeRepoSearchTimeBegin(){
	SaveTimeBegin( &treeRepoSearchTimeBegin );
}

void CCudaTimeStats::TreeRepoSearchTimeEnd(){
	SaveTimeEnd( &treeRepoSearchTimeBegin, &treeRepoSearchTimeEnd, &treeRepoSearchTime );
}

void CCudaTimeStats::TreeRepoApplyTimeBegin(){
	SaveTimeBegin( &treeRepoApplyTimeBegin );
}

void CCudaTimeStats::TreeRepoApplyTimeEnd(){
	SaveTimeEnd( &treeRepoApplyTimeBegin, &treeRepoApplyTimeEnd, &treeRepoApplyTime );
}

void CCudaTimeStats::TreeRepoInsertTimeBegin(){
	SaveTimeBegin( &treeRepoInsertTimeBegin );
}

void CCudaTimeStats::TreeRepoInsertTimeEnd(){
	SaveTimeEnd( &treeRepoInsertTimeBegin, &treeRepoInsertTimeEnd, &treeRepoInsertTime );
}

void CCudaTimeStats::MT_Kernel1TimeBegin(){
	SaveTimeBegin( &MT_kernel1TimeBegin );	
}

void CCudaTimeStats::MT_Kernel1TimeEnd(){
	SaveTimeEnd( &MT_kernel1TimeBegin, &MT_kernel1TimeEnd, &MT_kernel1Time );	
}

void CCudaTimeStats::MT_Sort1TimeBegin(){
	SaveTimeBegin( &MT_sort1TimeBegin );	
}

void CCudaTimeStats::MT_Sort1TimeEnd(){
	SaveTimeEnd( &MT_sort1TimeBegin, &MT_sort1TimeEnd, &MT_sort1Time );	
}

void CCudaTimeStats::MT_Kernel2TimeBegin(){
	SaveTimeBegin( &MT_kernel2TimeBegin );	
}

void CCudaTimeStats::MT_Kernel2TimeEnd(){
	SaveTimeEnd( &MT_kernel2TimeBegin, &MT_kernel2TimeEnd, &MT_kernel2Time );		
}

void CCudaTimeStats::MT_Kernel3TimeBegin(){
	SaveTimeBegin( &MT_kernel3TimeBegin );	
}

void CCudaTimeStats::MT_Kernel3TimeEnd(){
	SaveTimeEnd( &MT_kernel3TimeBegin, &MT_kernel3TimeEnd, &MT_kernel3Time );	
}

void CCudaTimeStats::MT_ModelCalcTimeBegin(){
	SaveTimeBegin( &MT_modelCalcTimeBegin );
	
}

void CCudaTimeStats::MT_ModelCalcTimeEnd(){
	SaveTimeEnd( &MT_modelCalcTimeBegin, &MT_modelCalcTimeEnd, &MT_modelCalcTime );
}

void CCudaTimeStats::MT_FullMLRCalcTimeBegin(){
	SaveTimeBegin( &MT_fullMLRCalcTimeBegin );
}

void CCudaTimeStats::MT_FullMLRCalcTimeEnd(){
	SaveTimeEnd( &MT_fullMLRCalcTimeBegin, &MT_fullMLRCalcTimeEnd, &MT_fullMLRCalcTime );	
}

void CCudaTimeStats::MT_MeanRTCalcTimeBegin(){
	SaveTimeBegin( &MT_meanRTCalcTimeBegin );	
}

void CCudaTimeStats::MT_MeanRTCalcTimeEnd(){
	SaveTimeEnd( &MT_meanRTCalcTimeBegin, &MT_meanRTCalcTimeEnd, &MT_meanRTCalcTime );
}

void CCudaTimeStats::MT_Kernel4TimeBegin(){
	SaveTimeBegin( &MT_kernel4TimeBegin );	
}

void CCudaTimeStats::MT_Kernel4TimeEnd(){
	SaveTimeEnd( &MT_kernel4TimeBegin, &MT_kernel4TimeEnd, &MT_kernel4Time );	
}

void CCudaTimeStats::MT_Kernel5TimeBegin(){
	SaveTimeBegin( &MT_kernel5TimeBegin );	
}

void CCudaTimeStats::MT_Kernel5TimeEnd(){
	SaveTimeEnd( &MT_kernel5TimeBegin, &MT_kernel5TimeEnd, &MT_kernel5Time );	
}

void CCudaTimeStats::SeqTimeBegin(){
	SaveTimeBegin( &seqTimeBegin );	
}

void CCudaTimeStats::SeqTimeEnd(){
	SaveTimeEnd( &seqTimeBegin, &seqTimeEnd, &seqTime );	
	sumSeqTime += seqTime;
}