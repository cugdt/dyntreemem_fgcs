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

#define N_DIPOL_OBJECTS			2				//number of dipoles objects per node
#define RT_N_DIPOLS				1				//the same but for regression trees
#define RT_N_MODEL_VALUES		2
#define MT_N_DIPOL_TMP			RT_N_DIPOLS		//the same byt for model trees

#define FULL_BINARY_TREE_REP			0		//if complete representation
#if	!FULL_BINARY_TREE_REP
 #define ADDAPTIVE_TREE_REP				0		//if adaptive representatio	
#endif
#define N_INFO_CORR_ERR				2			

#define DEBUG						0			//general debug
#define DEBUG_COMPACT_TREE			1			//stats for compact rep trees

//double/float for data, results, etc.
#define USE_DOUBLE_FOR_RT_RESULTS	0			//float/doble precision
#if USE_DOUBLE_FOR_RT_RESULTS == 1
 #define RT_REAL double
#else
 #define RT_REAL float
#endif
#define DS_REAL float
#define RT_TREE_TRESHOLD_REAL float

#define MT_RECALC_ALL_MODELS		0			//if all models have to be recals
#define MT_ATTRIBUTES_ONOFF_TYPE char			//

//#define TREE_REPO 1

class CWorker{
protected:
	int nAttrs;				//number of attributes
	int nClasses;			//number of classes

public:
	int nThreads;			//number of threads
	int nBlocks;			//number of blocks
	
	#if !FULL_BINARY_TREE_REP	
	bool bCompactOrFullTreeRep;
public:
	float adaptiveTreeRepSwitch;	//switching factor - complete/compact/adaptive rep
	#endif	

	//stats when different in-memory rep of DTs are used
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

	//copy the tree from C++ pointers to 1D table
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