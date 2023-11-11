#include "RealTest.h"


CRealTest::CRealTest(Real threshold, unsigned short attrNum)
{
	m_iAttrNum = attrNum;
	m_dThreshold = threshold;
}


CRealTest::CRealTest(const CRealTest& templ)
{
	m_iAttrNum = templ.m_iAttrNum;
	m_dThreshold = templ.m_dThreshold;
}


ITest* CRealTest::CloneTest() const
{
	CRealTest* temp = new CRealTest(*this);
	return (ITest*) temp;
}

bool CRealTest::operator==(const ITest& it) const
{
	assert(it.GetType()==REAL_TEST);
	CRealTest &rRT=(CRealTest&)it;

	if (m_iAttrNum==rRT.m_iAttrNum && m_dThreshold==rRT.m_dThreshold)
		return true;
	return false;
}


bool CRealTest::operator<(const ITest& it) const
{
	assert(it.GetType()==REAL_TEST);
	CRealTest &rRT=(CRealTest&)it;

	if (m_iAttrNum == rRT.m_iAttrNum)
		if (m_dThreshold < rRT.m_dThreshold)
			return true;
	return false;
}


bool CRealTest::operator>(const ITest& it) const 
{
	assert(it.GetType()==REAL_TEST);
	CRealTest &rRT=(CRealTest&)it;

	if (m_iAttrNum == rRT.m_iAttrNum)
		if (m_dThreshold > rRT.m_dThreshold)
			return true;
	return false;
}