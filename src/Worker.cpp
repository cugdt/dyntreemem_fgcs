#include "Worker.h"

#if CUDA_EA_ON || SPARK_EA_ON

#include <stdio.h>
#include <math.h>
#include <fstream>
#include<iomanip>

//set default options
CWorker::CWorker(){
	ClearDebugInfo();	//DEBUG CT//
	nThreads = 256;		//can change later based on input file
	nBlocks = 256;		//can change later based on input file
	#if ADDAPTIVE_TREE_REP
	bCompactOrFullTreeRep = false;
	adaptiveTreeRepSwitch = 0.8;	//can change later based on input file
	#endif
}

//set again dipole objects
void CWorker::ReFillTrainObjsInBranches(CDTreeNode *node, vector<CTrainObjsInNode>& vqDiv){
	size_t nOutcomes = vqDiv.size();

	for (size_t t = 0; t < nOutcomes; t++){
		vqDiv[t].m_uSize = node->m_vBranch[t]->m_uTrain;

		for( int i = 0; i < nClasses; i++ )
			for( int j = 0; j < N_DIPOL_OBJECTS; j++ ){
			vqDiv[ t ].SetObject4Dipoles( i, j, node -> m_vBranch[ t ] -> m_TrainObjs.GetObject4Dipoles( i , j ) );
			#if TREE_REPO
			vqDiv[ t ].SetObject4Dipoles_TreeRepo( i, j, node -> m_vBranch[ t ] -> m_TrainObjs.GetObject4Dipoles_TreeRepo( i , j ) );
			#endif
		}

		vqDiv[ t ].SetNotCuttedMixedDipoleObject( node -> m_vBranch[ t ] -> m_TrainObjs.GetNotCuttedMixedDipoleObject( 0 ), 0 );
		vqDiv[ t ].SetNotCuttedMixedDipoleObject( node -> m_vBranch[ t ] -> m_TrainObjs.GetNotCuttedMixedDipoleObject( 1 ), 1 );
		vqDiv[ t ].SetCuttedPureDipoleObject( node -> m_vBranch[ t ] -> m_TrainObjs.GetCuttedPureDipoleObject( 0 ), 0 );
		vqDiv[ t ].SetCuttedPureDipoleObject( node -> m_vBranch[ t ] -> m_TrainObjs.GetCuttedPureDipoleObject( 1 ), 1 );
		vqDiv[ t ].SetCuttedMixedDipoleObject( node -> m_vBranch[ t ] -> m_TrainObjs.GetCuttedMixedDipoleObject( 0 ), 0 );
		vqDiv[ t ].SetCuttedMixedDipoleObject( node -> m_vBranch[ t ] -> m_TrainObjs.GetCuttedMixedDipoleObject( 1 ), 1 );

	}
}

//move the tree from C++ pointers to 1D table, versions for different in-memory DT rep
#if FULL_BINARY_TREE_REP
void CWorker::CopyBinaryTreeToTable( const CDTreeNode* node, int* attrTab, float* valueTab, unsigned int index ){
	if( node -> m_bLeaf ){
		attrTab[ index ] = -1;
		return;
	}

	attrTab[ index ] = ((CRealTest*)(node->m_pITest)) -> GetAttrNum();
	valueTab[ index ] = ((CRealTest*)(node->m_pITest)) -> m_dThreshold;

	if( node->m_vBranch.size() > 0 ) CopyBinaryTreeToTable( node -> m_vBranch[0], attrTab, valueTab, 2 * index + 1 );
	if( node->m_vBranch.size() > 1 ) CopyBinaryTreeToTable( node -> m_vBranch[1], attrTab, valueTab, 2 * index + 2 );
}
#else
int CWorker::CopyBinaryTreeToTable( const CDTreeNode* node, int* attrTab, float* valueTab, unsigned int index, int* leftChildPosTab, int* rightChildPosTab, int* parentPosTab ){
	if( node -> m_bLeaf ){
		attrTab[ index ] = -1;
		leftChildPosTab[ index ] = -1;
		rightChildPosTab[ index ] = -1;
		return index;
	}

	attrTab[ index ] = ((CRealTest*)(node->m_pITest)) -> GetAttrNum();
	valueTab[ index ] = ((CRealTest*)(node->m_pITest)) -> m_dThreshold;
	
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
void CWorker::CopyBinaryTreeToTable_FULL_BINARY_TREE_REP( const CDTreeNode* node, int* attrTab, float* valueTab, unsigned int index ){
	if( node -> m_bLeaf ){
		attrTab[ index ] = -1;
		return;
	}

	attrTab[ index ] = ((CRealTest*)(node->m_pITest)) -> GetAttrNum();
	valueTab[ index ] = ((CRealTest*)(node->m_pITest)) -> m_dThreshold;

	if( node->m_vBranch.size() > 0 ) CopyBinaryTreeToTable_FULL_BINARY_TREE_REP( node -> m_vBranch[0], attrTab, valueTab, 2 * index + 1 );
	if( node->m_vBranch.size() > 1 ) CopyBinaryTreeToTable_FULL_BINARY_TREE_REP( node -> m_vBranch[1], attrTab, valueTab, 2 * index + 2 );
}
#endif

//clean memory of DTs
void CWorker::CleanBinaryTreeInTable( const CDTreeNode* node, int* attrTab, float* valueTab, unsigned int index ){
	attrTab[index] = -2;
	valueTab[index] = 0.0;

	if (node->m_vBranch.size() > 0) CleanBinaryTreeInTable(node->m_vBranch[0], attrTab, valueTab, 2 * index + 1);
	if (node->m_vBranch.size() > 1) CleanBinaryTreeInTable(node->m_vBranch[1], attrTab, valueTab, 2 * index + 2);
}

int CWorker::CopyBinaryTreePartToTable( const CDTreeNode* startNode, CDTreeNode* node, int* attrTab, float* valueTab, unsigned int index ){
	unsigned int indexPom;
	if (node->m_bLeaf){
		attrTab[index] = -1;
	}
	else{
		attrTab[index] = ((CRealTest*)(node->m_pITest))->GetAttrNum();
		valueTab[index] = ((CRealTest*)(node->m_pITest))->m_dThreshold;

		if (node->m_vBranch.size() > 0){
			indexPom = CopyBinaryTreePartToTable(startNode, node->m_vBranch[0], attrTab, valueTab, 2 * index + 1);
			if (indexPom) return indexPom;
		}
		if (node->m_vBranch.size() > 1){
			indexPom = CopyBinaryTreePartToTable(startNode, node->m_vBranch[1], attrTab, valueTab, 2 * index + 2);
			if (indexPom){
				CleanBinaryTreeInTable(node->m_vBranch[0], attrTab, valueTab, 2 * index + 1);
				return indexPom;
			}
		}
	}

	if (startNode == node){
		return index;
	}
	else
		return false;
}

//clean info after the iteration of evolutionary loop
void CWorker::ClearDebugInfo_Iter(){

	#if DEBUG
	nUpdatedPartTreesIter = 0;			//DEBUG CT//
	nUpdatedTreesIter = 0;				//DEBUG CT//
	nUpdatedMutationTreesIter = 0;		//DEBUG CT//
	nUpdatedCrossTreesIter = 0;			//DEBUG CT//
	nUpdatedPostTreesIter = 0;			//DEBUG CT//
	#endif

	#if TREE_REPO
	nFoundTrees0_0Iter_TreeRepo = 0;	//DEBUG CT//
	nFoundTrees1_0Iter_TreeRepo = 0;	//DEBUG CT//
	nFoundTrees1_1Iter_TreeRepo = 0;	//DEBUG CT//	

	nFoundWholeTreesIter_TreeRepo = 0;	//DEBUG CT//
	nFoundPartTreesIter_TreeRepo = 0;	//DEBUG CT//

	nFoundAllDiffRootsIter_TreeRepo = 0;	//DEBUG CT//
	nFoundRootChangedIter_TreeRepo = 0;		//DEBUG CT//
	nFoundLeafRootChangedIter_TreeRepo = 0;	//DEBUG CT//

	nRepoInsertsIter = 0;					//DEBUG CT//
	nRepoInsertsCrossIter = 0;				//DEBUG CT//
	nRepoInsertsMutIter = 0;				//DEBUG CT//
	#endif
}

void CWorker::ClearDebugInfo_Run() {
	#if DEBUG_COMPACT_TREE
	popHostDeviceSendBytesRun = 0.0;
	resultDeviceHostSendBytesRun = 0.0;
	nCompactRepDTsRun = 0;
	nFullRepDTsRun = 0;	
	#endif
}

void CWorker::ClearDebugInfo(){

	ClearDebugInfo_Iter();
	ClearDebugInfo_Run();

	#if DEBUG
	nUpdatedPartTrees  = 0;														//DEBUG CT//
	nUpdatedTrees = 0;															//DEBUG CT//
	nUpdatedMutationTrees = 0;													//DEBUG CT//
	nUpdatedCrossTrees = 0;														//DEBUG CT//
	nUpdatedPostTrees = 0;														//DEBUG CT//
	#endif

	#if TREE_REPO
	nFoundTrees0_0_TreeRepo = 0;												//DEBUG CT//
	nFoundTrees1_0_TreeRepo = 0;												//DEBUG CT//
	nFoundTrees1_1_TreeRepo = 0;												//DEBUG CT//

	nFoundWholeTrees_TreeRepo = 0;												//DEBUG CT//
	nFoundPartTrees_TreeRepo = 0;												//DEBUG CT//
	
	nFoundAllDiffRoots_TreeRepo = 0;											//DEBUG CT//
	nFoundRootChanged_TreeRepo = 0;												//DEBUG CT//
	nFoundLeafRootChanged_TreeRepo = 0;											//DEBUG CT//

	nRepoInserts = 0;															//DEBUG CT//
	nRepoInsertsCross = 0;														//DEBUG CT//
	nRepoInsertsMut = 0;														//DEBUG CT//
	#endif

	#if DEBUG_COMPACT_TREE
	popHostDeviceSendBytesSum = 0.0;
	resultDeviceHostSendBytesSum = 0.0;	
	nCompactRepDTsSum = 0;
	nFullRepDTsSum = 0;
	nRuns = 0;
	nIters = 0;
	#endif
}

//write debug info, by default is not called - call after each iteration of the evolution
void CWorker::WriteDebugInfo_Iter( int datasetSize ){
	//DEBUG MT//std::cout << m_uMaxIterations - n + 1 << " loop mean time: " << (double)loopTime / (double)(m_uMaxIterations - n + 1) << " (last loop time:" << stop - start << "\t\tupdated trees:" << nUpdatedTrees / (double)(m_uMaxIterations - n + 1) << "(" << nUpdatedTrees << ":" << nUpdatedTreesIter << ")" << "\t\tupdated full trees:" << nUpdatedFullTrees / (double)(m_uMaxIterations - n + 1) << "(" << nUpdatedFullTrees << ":" << nUpdatedFullTreesIter << ")" << endl;//DEBUG MT//
	//DEBUG MT//std::cout << "avg tree size:" << AvgTreeSize(m_dqPopulation) << endl;//DEBUG MT//
	//DEBUG MT//nUpdatedTreesIter = 0;//DEBUG MT//
	//DEBUG MT//nUpdatedFullTreesIter = 0;//DEBUG MT//
	//DEBUG MT//nUpdatedTreesIter = 0;//DEBUG MT//
	//DEBUG MT//nUpdatedFullTreesIter = 0;//DEBUG MT//

	#if DEBUG
	std::cout << "TREES: partlyUpdated / updated /  updatedMut / updatedCross / updatedPost " << nUpdatedPartTrees << "/" << nUpdatedTrees << "/" << nUpdatedMutationTrees << "/" << nUpdatedCrossTrees << "/" << nUpdatedPostTrees;//DEBUG CT//
	std::cout << "iter: " << nUpdatedPartTreesIter << "/" << nUpdatedTreesIter << "/" << nUpdatedMutationTreesIter << "/" << nUpdatedCrossTreesIter << "/" << nUpdatedPostTreesIter << " sum:" << nUpdatedMutationTreesIter + nUpdatedCrossTreesIter + nUpdatedPostTreesIter << endl;//DEBUG CT//
	#endif

	#if TREE_REPO_MASK_DEPTH == 1
	std::cout << "trees0_0 / trees1_0 / trees1_1: " << nFoundTrees0_0_TreeRepo << "/" << nFoundTrees1_0_TreeRepo << "/" << nFoundTrees1_1_TreeRepo;//DEBUG CT//
	std::cout << " iter: " << nFoundTrees0_0Iter_TreeRepo << "/" << nFoundTrees1_0Iter_TreeRepo << "/" << nFoundTrees1_1Iter_TreeRepo;//DEBUG CT//
	std::cout << fixed << setprecision(2) << " frac:(" << nFoundTrees0_0_TreeRepo      / (double)nUpdatedTrees     * 100 << "%/" << nFoundTrees1_0_TreeRepo     / (double)nUpdatedTrees     * 100 << "%/" << nFoundTrees1_1_TreeRepo     / (double)nUpdatedTrees     * 100 << "%)";//DEBUG CT//
	std::cout << fixed << setprecision(2) << " ("       << nFoundTrees0_0Iter_TreeRepo / (double)nUpdatedTreesIter * 100 << "%/" << nFoundTrees1_0Iter_TreeRepo / (double)nUpdatedTreesIter * 100 << "%/" << nFoundTrees1_1Iter_TreeRepo / (double)nUpdatedTreesIter * 100 << "%)" << endl;//DEBUG CT//	
	#endif

	#if TREE_REPO_MASK_DEPTH == 2
	std::cout << "whole trees / part trees: " << nFoundWholeTrees_TreeRepo << "/" << nFoundPartTrees_TreeRepo;//DEBUG CT//
	std::cout << " iter: " << nFoundWholeTreesIter_TreeRepo << "/" << nFoundPartTreesIter_TreeRepo;//DEBUG CT//
	std::cout << fixed << setprecision(2) << " frac:(" << nFoundWholeTrees_TreeRepo / (double)nUpdatedTrees * 100 << "%/" << nFoundPartTrees_TreeRepo / (double)nUpdatedTrees * 100 << "%)";//DEBUG CT//
	std::cout << fixed << setprecision(2) << " (" << nFoundWholeTreesIter_TreeRepo / (double)nUpdatedTreesIter * 100 << "%/" << nFoundPartTreesIter_TreeRepo / (double)nUpdatedTreesIter * 100 << "%)" << endl;//DEBUG CT//
	#endif

	#if TREE_REPO
	std::cout << "repo inserts: " << nRepoInserts	  << " = " << nRepoInsertsMut     << "+" << nRepoInsertsCross;
	std::cout << " iter:"		  << nRepoInsertsIter << " = " << nRepoInsertsMutIter << "+" << nRepoInsertsCrossIter << endl;
	std::cout << fixed << setprecision(2) << "all diff roots: " << nFoundAllDiffRoots_TreeRepo << "/" << nUpdatedTrees << " (" << nFoundAllDiffRoots_TreeRepo / (double)nUpdatedTrees << ")";//DEBUG CT//
	std::cout << fixed << setprecision(2) << " iter: " << nFoundAllDiffRootsIter_TreeRepo << "/" << nUpdatedTreesIter << " (" << nFoundAllDiffRootsIter_TreeRepo / (double)nUpdatedTreesIter << ")" << endl;//DEBUG CT//
	std::cout << fixed << setprecision(2) << "modif root: " << nFoundRootChanged_TreeRepo << "/" << nUpdatedTrees << " (" << nFoundRootChanged_TreeRepo / (double)nUpdatedTrees * 100 << "%)";//DEBUG CT//
	std::cout << fixed << setprecision(2) << " iter: " << nFoundRootChangedIter_TreeRepo << "/" << nUpdatedTreesIter << " (" << nFoundRootChangedIter_TreeRepo / (double)nUpdatedTreesIter * 100 << "%)" << endl;//DEBUG CT//
	std::cout << fixed << setprecision(2) << "modif leaf root: " << nFoundLeafRootChanged_TreeRepo << "/" << nUpdatedTrees << " (" << nFoundLeafRootChanged_TreeRepo / (double)nUpdatedTrees * 100 << "%)";//DEBUG CT//
	std::cout << fixed << setprecision(2) << " iter: " << nFoundLeafRootChangedIter_TreeRepo << "/" << nUpdatedTreesIter << " (" << nFoundLeafRootChangedIter_TreeRepo / (double)nUpdatedTreesIter * 100 << "%)" << endl;//DEBUG CT//	
	#endif

	#if TREE_REPO_MASK_DEPTH == 1
	std::cout << fixed << setprecision(2) << "summary: " << "mod_root/found_repo/all " << nFoundRootChanged_TreeRepo << "/" << nFoundTrees0_0_TreeRepo + nFoundTrees1_0_TreeRepo + nFoundTrees1_1_TreeRepo << "/" << nUpdatedTrees;
	std::cout << fixed << setprecision(2) << " iter: " << nFoundRootChangedIter_TreeRepo << "/" << nFoundTrees0_0Iter_TreeRepo + nFoundTrees1_0Iter_TreeRepo + nFoundTrees1_1Iter_TreeRepo << "/" << nUpdatedTreesIter;
	#endif
	std::cout << endl;

	ofstream myfile;//DEBUG CT//
	string path = "tree_part_stats_";
	path += std::to_string( datasetSize );

	#if TREE_REPO
	path += "_";
	path += std::to_string( TREE_REPO_N_TREES );
	#endif

	path += ".txt";//DEBUG CT//
	myfile.open( path, ios::app);//DEBUG CT//

	#if DEBUG
	myfile << nUpdatedPartTrees << "\t" << nUpdatedTrees << "\t" << nUpdatedPartTreesIter << "\t" << nUpdatedTreesIter << "\t";//DEBUG CT//
	#endif

	#if TREE_REPO_MASK_DEPTH == 1
	myfile << nFoundTrees0_0_TreeRepo << "\t" << nFoundTrees1_0_TreeRepo << "\t" << nFoundTrees1_1_TreeRepo << "\t";//DEBUG CT//
	myfile << nFoundTrees0_0Iter_TreeRepo << "\t" << nFoundTrees1_0Iter_TreeRepo << "\t" << nFoundTrees1_0Iter_TreeRepo << "\t";//DEBUG CT//
	#endif

	#if TREE_REPO_MASK_DEPTH == 2
	myfile << nFoundWholeTrees_TreeRepo << "\t" << nFoundWholeTreesIter_TreeRepo << "\t";
	myfile << nFoundPartTrees_TreeRepo << "\t" << nFoundPartTreesIter_TreeRepo << "\t";
	myfile << fixed << setprecision(2) << nFoundWholeTrees_TreeRepo / (double)nUpdatedTrees * 100 << "\t" << nFoundWholeTreesIter_TreeRepo / (double)nUpdatedTreesIter * 100 << "\t";
	myfile << fixed << setprecision(2) << nFoundPartTrees_TreeRepo / (double)nUpdatedTrees * 100 << "\t" << nFoundPartTreesIter_TreeRepo / (double)nUpdatedTreesIter * 100 << "\t";
	#endif

	#if TREE_REPO
	myfile << nFoundAllDiffRoots_TreeRepo << "\t";//DEBUG CT//
	myfile << nFoundAllDiffRootsIter_TreeRepo << "\t";//DEBUG CT//
	myfile << nFoundRootChanged_TreeRepo << "\t";//DEBUG CT//
	myfile << nFoundRootChangedIter_TreeRepo << "\t";//DEBUG CT//
	myfile << endl;
	myfile.close();//DEBUG CT//			
	#endif	
}

//debug info - write after each Run
void CWorker::WriteDebugInfo_Run( int datasetSize, int incRuns, int incIters ){
	#if DEBUG_COMPACT_TREE
	nRuns  += incRuns;
	nIters += incIters;
	std::cout << "Host  -> Device:\t";
	std::cout << " last/total/meanR/meanI\t";
	std::cout << popHostDeviceSendBytesRun / 1000000 << " / ";
	std::cout << popHostDeviceSendBytesSum / 1000000 << " / ";
	std::cout << popHostDeviceSendBytesSum / 1000000 / nRuns << " / ";
	std::cout << popHostDeviceSendBytesSum / 1000000 / nIters;
	std::cout << " [MB]" << endl;

	std::cout << "Device  -> Host:\t";
	std::cout << " last/total/meanR/meanI\t";
	std::cout << resultDeviceHostSendBytesRun / 1000000 << " / ";
	std::cout << resultDeviceHostSendBytesSum / 1000000 << " / ";
	std::cout << resultDeviceHostSendBytesSum / 1000000 / nRuns << " / ";
	std::cout << resultDeviceHostSendBytesSum / 1000000 / nIters;
	std::cout << " [MB]" << endl;	

	double factorFullRun = (double)nFullRepDTsRun / (nFullRepDTsRun+nCompactRepDTsRun);
	double factorFullSum = (double)nFullRepDTsSum / (nFullRepDTsSum + nCompactRepDTsSum);
	double factorCompactRun = (double)nCompactRepDTsRun / (nFullRepDTsRun + nCompactRepDTsRun);
	double factorCompactSum = (double)nCompactRepDTsSum / (nFullRepDTsSum + nCompactRepDTsSum);

	std::cout << fixed << setprecision(2) << "Full:    " << nFullRepDTsRun << " / " << nFullRepDTsSum;
	std::cout << fixed << setprecision(2) << " (" << factorFullRun << " / " << factorFullSum << ")" << endl;
	std::cout << fixed << setprecision(2) << "Compact: " << nCompactRepDTsRun << " / " << nCompactRepDTsSum;
	std::cout << fixed << setprecision(2) << " (" << factorCompactRun << " / " << factorCompactSum << ")" << endl;

	std:cout << "nRuns: " << nRuns << " nIters: " << nIters << endl;
	#endif
}
#endif