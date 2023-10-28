#pragma once

#include "ITest.h"
#include "AttrValue.h"
#include <string.h>
#include <vector>

// Decion tree node (simple version)
class CDTreeNodeSim
{
public:
	
	// number of samples from the training datasets in the node (leaf)
	// Liczba przykladow ze zbioru uczacego w wezle (lisciu)
	unsigned int m_uTrain;

	//numer of misclassified samples - only for classification trees
	unsigned int m_uErrors;

	bool m_bLeaf;

	//test
	ITest* m_pITest;

	//child nodes
	std::vector<CDTreeNodeSim*> m_vBranch;

	// get number of nodes in the subtree
	unsigned GetNodeCount() const;

	// get the number of leaves in the subnode
	unsigned GetLeavesCount() const;
	
	// get the tree depth
	unsigned GetDepth();	

public:
	CDTreeNodeSim();
	CDTreeNodeSim(Real threshold, unsigned short attrNum);
	~CDTreeNodeSim();
};