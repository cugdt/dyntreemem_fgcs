#pragma once

#include "DTreeNodeSim.h"
#include "RealTest.h"
//#include "RealAttr.h"

#include <iostream>
#include <cassert>
#include <typeinfo>
#include <algorithm>
#include <stdio.h>
using namespace std;

CDTreeNodeSim::CDTreeNodeSim()
{
	m_bLeaf = false;
	m_pITest = NULL;
	m_uErrors = 0;
}

CDTreeNodeSim::CDTreeNodeSim(Real threshold, unsigned short attrNum)
{
	m_pITest = new CRealTest(threshold, attrNum);
	m_bLeaf = false;
}

CDTreeNodeSim::~CDTreeNodeSim()
{
	if (m_pITest && !m_bLeaf)
		delete m_pITest;
	for (unsigned i=0; i<m_vBranch.size(); i++)
		delete m_vBranch[i];
	m_vBranch.clear();
}

unsigned CDTreeNodeSim::GetNodeCount() const
{
	// Wliczenie wezla biezacego
	unsigned n=1;
	// Policzenie synow wezla
	for (unsigned i=0; i<m_vBranch.size(); i++)
		n+=m_vBranch[i]->GetNodeCount();

	return n;
}


unsigned CDTreeNodeSim::GetLeavesCount() const
{
	// Wliczenie wezla biezacego, tylko wtedy kiedy jest on lisciem
	if (m_bLeaf) return 1;

	// Policzenie synow wezla
	unsigned n=0;
	for (unsigned i=0; i<m_vBranch.size(); i++)
		n+=m_vBranch[i]->GetLeavesCount();
	return n;
}

unsigned CDTreeNodeSim::GetDepth()
{
	unsigned n=1;
	unsigned max=0;
	if (!m_vBranch.empty())
		max = m_vBranch.front()->GetDepth();

	for (unsigned i=1; i<m_vBranch.size(); i++)
	{
		unsigned t = m_vBranch[i]->GetDepth();
		if (t > max) max = t;
	}
	n+=max;
	return n;
}