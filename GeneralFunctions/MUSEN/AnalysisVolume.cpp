/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "AnalysisVolume.h"
CVector3 CAnalysisVolume::GetCenter(double _time) const
{
	return Center() + GetShift(_time);
}

void CAnalysisVolume::SetCenter(const CVector3& _center)
{
	Move(_center - GetCenter(0));
}

double CAnalysisVolume::Volume() const
{
	switch (nVolumeType)
	{
	case EVolumeType::VOLUME_SPHERE:        return 4.0 * PI * std::pow(vProps[0], 3) / 3.0;
	case EVolumeType::VOLUME_BOX:           return vProps[0] * vProps[1] * vProps[2];
	case EVolumeType::VOLUME_CYLINDER:      return PI * vProps[0] * vProps[0] * vProps[1];
	case EVolumeType::VOLUME_HOLLOW_SPHERE: return 4.0 * PI / 3.0 * (std::pow(vProps[0], 3) - std::pow(vProps[1], 3));
	case EVolumeType::VOLUME_STL:	        return CTriangularMesh::Volume();
	default:								return 0;
	}
}

SVolumeType CAnalysisVolume::BoundingBox(double _time) const
{
	SVolumeType bb = CTriangularMesh::BoundingBox();
	const CVector3 shift = GetShift(_time);
	bb.coordBeg += shift;
	bb.coordEnd += shift;
	return bb;
}


double CAnalysisVolume::MaxInscribedDiameter() const
{
	switch (nVolumeType)
	{
	case EVolumeType::VOLUME_SPHERE:		return vProps[0];
	case EVolumeType::VOLUME_HOLLOW_SPHERE:	return vProps[0] - vProps[1];
	case EVolumeType::VOLUME_CYLINDER:		return std::min({ vProps[0], vProps[1] });
	case EVolumeType::VOLUME_BOX:			return std::min({ vProps[0], vProps[1], vProps[2] });
	case EVolumeType::VOLUME_STL:			return std::pow(Volume() / PI * 3 / 4, 1 / 3.) * 2;
	}
	return 0;
}

void CAnalysisVolume::Scale(double _factor)
{
	const CVector3 center = GetCenter(0);
	for (auto& t : vTriangles)
	{
		t.p1 = center + (t.p1 - center) * _factor;
		t.p2 = center + (t.p2 - center) * _factor;
		t.p3 = center + (t.p3 - center) * _factor;
	}
	for (auto& p : vProps)
		p *= _factor;
}

void CAnalysisVolume::Rotate(const CMatrix3& _rotation)
{
	const CVector3 center = GetCenter(0);
	for (auto& t : vTriangles)
	{
		t.p1 = center + _rotation * (t.p1 - center);
		t.p2 = center + _rotation * (t.p2 - center);
		t.p3 = center + _rotation * (t.p3 - center);
	}
	mRotation = _rotation * mRotation;
}

void CAnalysisVolume::AddTimePoint()
{
	if (!m_vIntervals.empty())
		m_vIntervals.push_back(m_vIntervals.back());
	else
		m_vIntervals.push_back(SVelInterval{ 1, CVector3{ 0 } });
}

CVector3 CAnalysisVolume::GetShift(double _time) const
{
	CVector3 shift{ 0 };
	for (size_t i = 0; i < m_vIntervals.size(); ++i)
		if (_time > m_vIntervals[i].dTime) // consider whole interval
		{
			if (i == 0)
				shift += m_vIntervals[i].dTime * m_vIntervals[i].vVel;
			else
				shift += (m_vIntervals[i].dTime - m_vIntervals[i - 1].dTime) * m_vIntervals[i].vVel;
		}
		else // consider interval partially
		{
			if (i == 0)
				shift += _time * m_vIntervals[i].vVel;
			else
				shift += (_time - m_vIntervals[i - 1].dTime) * m_vIntervals[i].vVel;
			break;
		}
	return shift;
}