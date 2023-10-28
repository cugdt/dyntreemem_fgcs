#pragma once
#include "ITest.h"
#include "AttrValue.h"

class CRealTest : public ITest
{
public:
	virtual eTestType GetType() const		{ return REAL_TEST; }

	Real m_dThreshold;
	
	virtual int GetAttrNum() const			{ return m_iAttrNum; }
	virtual unsigned GetComplexity() const	{ return 1; }
		
	virtual unsigned GetSubSetCount() const	{ return 2; }
	virtual bool operator==(const ITest& it) const;
	virtual bool operator<(const ITest& it) const;
	virtual bool operator>(const ITest& it) const;
	
	virtual ITest* CloneTest() const;
	CRealTest(const CRealTest& templ);
	CRealTest(Real threshold, unsigned short attrNum);
	virtual ~CRealTest(void)				{}

	CRealTest(void)							{}

protected:	
	// Numer atrybutu, ktorego dotyczy test
	unsigned short m_iAttrNum;
};
