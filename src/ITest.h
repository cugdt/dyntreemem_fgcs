#pragma once

#include <string>
#include <deque>

//type of test: real, nominal, etc.
class ITest
{
public:
	enum eTestType { REAL_TEST, NOMINAL_TEST, NOM_INNER_DIS_TEST, OBLIQUE_TEST, MULTI_TEST };
	
	// get the test type
	virtual eTestType GetType() const = 0;
	virtual ~ITest(void) {}
};
