#pragma once

//TEST LAUNCH
//evolutionary aspects like crossover and mutation operations are switched off.
#define TEST 1

#if !TEST
#include "IEvolutionaryAlg.h"
#include "ExternalLibs.h"

#define CUDA_EA_ON 1

#if CUDA_EA_ON
#include <vector>
using namespace std;

#define N_DIPOL_OBJECTS			2
#define RT_N_DIPOLS				1
#define RT_N_MODEL_VALUES		2
#define MT_N_DIPOL_TMP			RT_N_DIPOLS

#define FULL_BINARY_TREE_REP			0			
#if	!FULL_BINARY_TREE_REP
 #define ADDAPTIVE_TREE_REP				0			
#endif
#define N_INFO_CORR_ERR				2			

#define DEBUG						0
#define DEBUG_COMPACT_TREE			1

//double/float for data, results, etc.
#define USE_DOUBLE_FOR_RT_RESULTS	0
#if USE_DOUBLE_FOR_RT_RESULTS == 1
 #define RT_REAL double
#else
 #define RT_REAL float
#endif
#define DS_REAL float
#define RT_TREE_TRESHOLD_REAL float

#define MT_RECALC_ALL_MODELS		0
#define MT_ATTRIBUTES_ONOFF_TYPE char

//#define TREE_REPO 1

class CWorker{
protected:
	int nAttrs;
	int nClasses;

public:
	int nThreads;
	int nBlocks;
	
	#if !FULL_BINARY_TREE_REP	
	bool bCompactOrFullTreeRep;
public:
	float adaptiveTreeRepSwitch;
	#endif	

	#if DEBUG_COMPACT_TREE
public:
	float popHostDeviceSendBytesRun;
	float resultDeviceHostSendBytesRun;
	float popHostDeviceSendBytesSum;
	float resultDeviceHostSendBytesSum;
	int nCompactRepDTsRun;
	int nFullRepDTsRun;
	int nCompactRepDTsSum;
	int nFullRepDTsSum;
	int nRuns;
	int nIters;
	#endif

public:
	CWorker();
	void WriteDebugInfo_Iter( int datasetSize );
	void WriteDebugInfo_Run( int datasetSize, int incRuns, int incIters );
	void ClearDebugInfo_Iter();
	void ClearDebugInfo_Run();
	void ClearDebugInfo();
	void ReFillTrainObjsInBranches( CDTreeNode *node, vector<CTrainObjsInNode>& vqDiv);

	protected:
	#if FULL_BINARY_TREE_REP
	void CopyBinaryTreeToTable(const CDTreeNode* node, int* attrTab, float* valueTab, unsigned int index);
	#else
	int CopyBinaryTreeToTable(const CDTreeNode* node, int* attrTab, float* valueTab, unsigned int index, int* leftChildPosTab, int* rightChildPosTab, int* parentPosTab);
	void CopyBinaryTreeToTable_FULL_BINARY_TREE_REP(const CDTreeNode* node, int* attrTab, float* valueTab, unsigned int index);
	#endif
	void CleanBinaryTreeInTable(const CDTreeNode* node, int* attrTab, float* valueTab, unsigned int index);
	int CopyBinaryTreePartToTable(const CDTreeNode* startNode, CDTreeNode* node, int* attrTab, float* valueTab, unsigned int index);
};
#endif
#endif