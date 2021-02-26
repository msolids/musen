/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "InsideVolumeChecker.h"

CInsideVolumeChecker::CInsideVolumeChecker(const CAnalysisVolume* _volume, double _time)
{
	SetTriangles(_volume->Mesh(_time).Triangles());
}

CInsideVolumeChecker::CInsideVolumeChecker(const std::vector<CTriangle>& _triangles)
{
	SetTriangles(_triangles);
}

std::pair<int, int> CInsideVolumeChecker::GetIndexByPoint(const CVector3& point) const
{
	int y = static_cast<int>((point.y - m_vMin.y) / (m_vMax.y - m_vMin.y) * MAX_INDEXES);
	int z = static_cast<int>((point.z - m_vMin.z) / (m_vMax.z - m_vMin.z) * MAX_INDEXES);

	y = std::clamp(y, 0, MAX_INDEXES - 1);
	z = std::clamp(z, 0, MAX_INDEXES - 1);
	return std::make_pair(y, z);
}

void CInsideVolumeChecker::SetTriangles(const std::vector<CTriangle>& _triangles)
{
	if ( _triangles.empty() ) return;
	m_Triangles = _triangles;

	for (size_t i = 0; i < m_Triangles.size(); i++)
	{
		if (i == 0)
			m_vMin = m_vMax = m_Triangles[0].p1;
		m_vMin = Min(m_vMin, m_Triangles[i].p1, m_Triangles[i].p2, m_Triangles[i].p3);
		m_vMax = Max(m_vMax, m_Triangles[i].p1, m_Triangles[i].p2, m_Triangles[i].p3);
	}

	for (size_t i1 = 0; i1 < MAX_INDEXES; ++i1)
		for (size_t i2 = 0; i2 < MAX_INDEXES; ++i2)
			m_vIndexes[i1][i2].clear();
	for (size_t i = 0; i < m_Triangles.size(); i++)
	{
		std::pair<int, int> index1 = GetIndexByPoint(m_Triangles[i].p1);
		std::pair<int, int> index2 = GetIndexByPoint(m_Triangles[i].p2);
		std::pair<int, int> index3 = GetIndexByPoint(m_Triangles[i].p3);

		if (index1.first > index2.first) std::swap(index1.first, index2.first);
		if (index1.first > index3.first) std::swap(index1.first, index3.first);
		if (index2.first > index3.first) std::swap(index2.first, index3.first);

		if (index1.second > index2.second) std::swap(index1.second, index2.second);
		if (index1.second > index3.second) std::swap(index1.second, index3.second);
		if (index2.second > index3.second) std::swap(index2.second, index3.second);

		for (int i1 = index1.first; i1 <= index3.first; i1++)
			for (int i2 = index1.second; i2 <= index3.second; i2++)
				m_vIndexes[i1][i2].push_back(i);
	}
}

void CInsideVolumeChecker::IsSphereInside(const CVector3& pos, double radius, bool &isInside, bool &isPartInside, bool &isUndefined) const
{
	CVector3 particlePos[15];
	for (size_t i = 0; i < 7; i++) particlePos[i] = pos;

	particlePos[1].x += radius;
	particlePos[2].x -= radius;
	particlePos[3].y += radius;
	particlePos[4].y -= radius;
	particlePos[5].z += radius;
	particlePos[6].z -= radius;

	double dScaleFactor = radius / 1.73205081; // square root of 3
	particlePos[7] = pos + CVector3(1, 1, 1) * dScaleFactor;
	particlePos[8] = pos + CVector3(1, 1, -1) * dScaleFactor;
	particlePos[9] = pos + CVector3(1, -1, 1) * dScaleFactor;
	particlePos[10] = pos + CVector3(1, -1, -1) * dScaleFactor;
	particlePos[11] = pos + CVector3(-1, 1, 1) * dScaleFactor;
	particlePos[12] = pos + CVector3(-1, 1, -1) * dScaleFactor;
	particlePos[13] = pos + CVector3(-1, -1, 1) * dScaleFactor;
	particlePos[14] = pos + CVector3(-1, -1, -1) * dScaleFactor;

	int isInsideCounter = 0;
	isUndefined = false;

	for (size_t i = 0; i < 15; i++)
	{
		bool isInsideCurrent = false;
		bool isUndefinedCurrent = false;

		IsPointInside(particlePos[i], isInsideCurrent, isUndefinedCurrent);

		isInsideCounter += isUndefinedCurrent ? 0 : isInsideCurrent;
		isUndefined |= isUndefinedCurrent;

		if (isUndefinedCurrent == true || isInsideCurrent == false) break;
	}

	isInside = isInsideCounter == 15;
	isPartInside = isInsideCounter > 0;
}

void CInsideVolumeChecker::IsPointInside(const CVector3& point, bool &isInside, bool &isUndefined) const
{
	int totalIntersections = 0;
	std::pair<int, int> index = GetIndexByPoint(point);
	const std::vector<size_t> &indexes = m_vIndexes[index.first][index.second];

	for (unsigned j = 0; j < indexes.size(); j++)
	{
		CTriangle triangle = m_Triangles[ indexes[ j ] ];
		triangle.p1 -= point;
		triangle.p2 -= point;
		triangle.p3 -= point;

		bool isIntersecting(false), isOnEdge(false);
		TryIntersectWithXAxis(triangle, isIntersecting, isOnEdge);

		if (isOnEdge)
		{
			isUndefined = true;
			break;
		}
		if (isIntersecting)
			totalIntersections++;
	}

	isInside = totalIntersections % 2 == 1;
}

std::vector<size_t> CInsideVolumeChecker::GetObjectsInside(const std::vector<CVector3>& _vCoords, const std::vector<size_t>& _vIDs /*= std::vector<size_t>()*/) const
{
	return p_GetObjectsInside(_vCoords, _vIDs, std::vector<double>(), true);
}

std::vector<size_t> CInsideVolumeChecker::GetSpheresTotallyInside(const std::vector<CVector3>& _vCoords, const std::vector<double>& _vRadiuses, const std::vector<size_t>& _vIDs /*= std::vector<size_t>()*/) const
{
	return p_GetObjectsInside(_vCoords, _vIDs, _vRadiuses, true);
}

std::vector<size_t> CInsideVolumeChecker::GetSpheresPartiallyInside(const std::vector<CVector3>& _vCoords, const std::vector<double>& _vRadiuses, const std::vector<size_t>& _vIDs /*= std::vector<size_t>()*/) const
{
	return p_GetObjectsInside(_vCoords, _vIDs, _vRadiuses, false);
}

std::vector<size_t> CInsideVolumeChecker::GetSpheresTotallyOutside(const std::vector<CVector3>& _vCoords, const std::vector<double>& _vRadiuses, const std::vector<size_t>& _vIDs /*= std::vector<size_t>()*/) const
{
	std::vector<size_t> vOutside;
	std::vector<size_t> vInside = p_GetObjectsInside(_vCoords, _vIDs, _vRadiuses, false);
	if (_vIDs.empty())
	{
		std::vector<size_t> vAll(_vCoords.size());
		for (size_t i = 0; i < _vCoords.size(); ++i)
			vAll[i] = i;
		vOutside = VectorDifference(vAll, vInside);
	}
	else
		vOutside = VectorDifference(_vIDs, vInside);
	return vOutside;
}

std::vector<size_t> CInsideVolumeChecker::GetSpheresPartiallyOutside(const std::vector<CVector3>& _vCoords, const std::vector<double>& _vRadiuses, const std::vector<size_t>& _vIDs /*= std::vector<size_t>()*/) const
{
	std::vector<size_t> vOutside;
	std::vector<size_t> vInside = p_GetObjectsInside(_vCoords, _vIDs, _vRadiuses, true);
	if (_vIDs.empty())
	{
		std::vector<size_t> vAll(_vCoords.size());
		for (size_t i = 0; i < _vCoords.size(); ++i)
			vAll[i] = i;
		vOutside = VectorDifference(vAll, vInside);
	}
	else
		vOutside = VectorDifference(_vIDs, vInside);
	return vOutside;
}

std::vector<size_t> CInsideVolumeChecker::p_GetObjectsInside(const std::vector<CVector3>& _vCoords, const std::vector<size_t>& _vIDs, const std::vector<double>& _vRadiuses, bool _bTotally) const
{
	bool bExternalID = !_vIDs.empty();
	bool bSpheres = !_vRadiuses.empty();

	std::vector<size_t> vIDsToProcess(_vCoords.size());
	for (size_t i = 0; i < _vCoords.size(); ++i)
		vIDsToProcess[i] = i;

	CVector3 rotation(0, 0, 0);

	std::vector<size_t> vIDsInside;
	for (int iterationNumber = 0; !vIDsToProcess.empty() && iterationNumber < 100; iterationNumber++)
	{
		CMatrix3 rotatationMatrix = CQuaternion(rotation).ToRotmat();

		std::vector<size_t> vFailedToProcessIDs;
		std::vector<unsigned char> vParticleInsideFlag(vIDsToProcess.size(), 0);
		std::vector<unsigned char> vFailedToProcessFlag(vIDsToProcess.size(), 0);

		ParallelFor(vIDsToProcess.size(), [&](size_t i)
		{
			bool isInside = false;
			bool isUndefined = false;
			bool isPartInside = false;
			if (!bSpheres)
				IsPointInside(rotatationMatrix * _vCoords[vIDsToProcess[i]], isInside, isUndefined);
			else
				IsSphereInside(rotatationMatrix * _vCoords[vIDsToProcess[i]], _vRadiuses[vIDsToProcess[i]], isInside, isPartInside, isUndefined);

			if (bSpheres && !_bTotally && isPartInside)
				vParticleInsideFlag[i] = 1;
			else if (isUndefined)
				vFailedToProcessFlag[i] = 1;
			else if (isInside)
				vParticleInsideFlag[i] = 1;
		});

		// gather all data
		for (unsigned i = 0; i < vIDsToProcess.size(); i++)
		{
			if (vFailedToProcessFlag[i] == 1)
				vFailedToProcessIDs.push_back(vIDsToProcess[i]);
			else if (vParticleInsideFlag[i] == 1)
				vIDsInside.push_back(vIDsToProcess[i]);
		}

		std::swap(vIDsToProcess, vFailedToProcessIDs);
		vFailedToProcessIDs.clear();

		rotation.y += 0.01 * PI;
		rotation.z += 0.01 * PI;
	}

	if (!bExternalID)
	{
		return vIDsInside;
	}
	else
	{
		std::vector<size_t> vRes(vIDsInside.size());
		for (size_t i = 0; i < vIDsInside.size(); ++i)
			vRes[i] = _vIDs[vIDsInside[i]];
		return vRes;
	}
}

void CInsideVolumeChecker::TryIntersectWithXAxis( const CTriangle &triangle, bool &isIntersecting, bool &isOnEdge ) const
{
	double e = 1e-20;
	CVector3 p1, p2, p3;
	p1 = triangle.p1;
	p2 = triangle.p2;
	p3 = triangle.p3;

	double c1 = p1.z * p2.y - p1.y * p2.z;
	double c2 = p2.z * p3.y - p2.y * p3.z;
	double c3 = p3.z * p1.y - p3.y * p1.z;

	isOnEdge = true;
	if (fabs(p1.z) < e && fabs(p1.y) < e) return;
	if (fabs(p2.z) < e && fabs(p2.y) < e) return;
	if (fabs(p3.z) < e && fabs(p3.y) < e) return;

	if (fabs(c1) < e)
	{
		if (fabs(p1.y) < e	&& p1.z < 0 == p2.z > 0
			|| p1.y < 0 == p2.y > 0) return;
	}
	if (fabs(c2) < e)
	{
		if (fabs(p2.y) < e	&& p2.z < 0 == p3.z > 0
			|| p2.y < 0 == p3.y > 0) return;
	}
	if (fabs(c3) < e)
	{
		if (fabs(p3.y) < e	&& p3.z < 0 == p1.z > 0
			|| p3.y < 0 == p1.y > 0) return;
	}

	isOnEdge = false;

	bool anyC0 = c1 == 0 || c2 == 0 || c3 == 0;
	if (anyC0) isIntersecting = false;
	else
		isIntersecting = !anyC0 && ((c1 < 0) == (c2 < 0) && (c1 < 0) == (c3 < 0));

	if (isIntersecting)
	{
		double A = p1.y * (p2.z - p3.z) + p2.y * (p3.z - p1.z) + p3.y  * (p1.z - p2.z);
		double D = -(p1.x * (p2.y * p3.z - p3.y * p2.z) + p2.x * (p3.y * p1.z - p1.y * p3.z) + p3.x * (p1.y * p2.z - p2.y * p1.z));

		isIntersecting = A <= 0 == D <= 0;
	}
}
