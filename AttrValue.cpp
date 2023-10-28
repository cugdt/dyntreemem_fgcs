#include "AttrValue.h"

IAttrValue::IAttrValue(eAttrType Type)
{
	m_Type=Type;
	m_bMissedValue=false;
}

IAttrValue::IAttrValue()
{
	m_bMissedValue = false;
}

bool IAttrValue::operator==(const IAttrValue &ia) const
{
	assert(GetAttrType() == ia.GetAttrType());
	if (GetAttrType()==NOMINAL)
		return (m_AttrValue.u_iNom == ia.m_AttrValue.u_iNom);
	//if (GetAttrType()==CONTINUOUS)
	return (m_AttrValue.u_dReal == ia.m_AttrValue.u_dReal);
}


bool IAttrValue::operator!=(const IAttrValue &ia) const 
{ 
	return !((*this)==ia); 
}


bool IAttrValue::operator>(const IAttrValue &ia) const
{
	assert(GetAttrType() == ia.GetAttrType());
	if (GetAttrType()==NOMINAL)
		return (m_AttrValue.u_iNom > ia.m_AttrValue.u_iNom);
//	if (GetAttrType()==CONTINUOUS)
	return (m_AttrValue.u_dReal > ia.m_AttrValue.u_dReal);
}


bool IAttrValue::operator<(const IAttrValue &ia) const
{
	assert(GetAttrType() == ia.GetAttrType());
	if (GetAttrType()==NOMINAL)
		return (m_AttrValue.u_iNom < ia.m_AttrValue.u_iNom);
//	if (GetAttrType()==CONTINUOUS)
	return (m_AttrValue.u_dReal < ia.m_AttrValue.u_dReal);
}