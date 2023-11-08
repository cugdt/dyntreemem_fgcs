#pragma once

#include <iostream>
#include <cassert>

#ifdef __linux__
typedef char eAttrType;
#define NOMINAL 0
#define CONTINUOUS 1
#else
enum eAttrType : char { NOMINAL=0, CONTINUOUS=1};
#endif

// to control the memory complexity
typedef float Real;

//the attribute value, nominal or real
class IAttrValue
{
public:
	union Val
	{
		unsigned int u_iNom;
		Real u_dReal;
	} m_AttrValue;			// attribute value

	bool m_bMissedValue;	// if the value is set

public:
	IAttrValue(eAttrType Type);
	IAttrValue();

	eAttrType GetAttrType() const { return m_Type; }
	void SetAttrType(eAttrType Type)
	{
		m_Type = Type;
	}
	bool operator==(const IAttrValue &ia) const;
	bool operator!=(const IAttrValue &ia) const;
	bool operator>(const IAttrValue &ia) const;
	bool operator<(const IAttrValue &ia) const;
	IAttrValue* operator->() { return this; }

private:
	eAttrType m_Type;		// attribute type
};