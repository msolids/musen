/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ThreadPool.h"
#include "AnalysisVolume.h"
#include "MUSENVectorFunctions.h"

class CInsideVolumeChecker
{
private:
	static const int MAX_INDEXES = 20;
	std::vector<CTriangle> m_Triangles;
	std::vector<size_t> m_vIndexes[MAX_INDEXES][MAX_INDEXES];
	CVector3 m_vMax, m_vMin;

public:
	CInsideVolumeChecker() = default;
	CInsideVolumeChecker(const CAnalysisVolume* _volume, double _time);
	CInsideVolumeChecker(const std::vector<CTriangle>& _triangles);
	void SetTriangles(const std::vector<CTriangle>& _triangles);

	/// Returns spheres, which are totally or partially inside the volume.
	std::vector<size_t> GetObjectsInside(const std::vector<CVector3>& _vCoords, const std::vector<size_t>& _vIDs = std::vector<size_t>()) const;
	std::vector<size_t> GetSpheresTotallyInside(const std::vector<CVector3>& _vCoords, const std::vector<double>& _vRadiuses, const std::vector<size_t>& _vIDs = std::vector<size_t>()) const;
	std::vector<size_t> GetSpheresPartiallyInside(const std::vector<CVector3>& _vCoords, const std::vector<double>& _vRadiuses, const std::vector<size_t>& _vIDs = std::vector<size_t>()) const;
	std::vector<size_t> GetSpheresTotallyOutside(const std::vector<CVector3>& _vCoords, const std::vector<double>& _vRadiuses, const std::vector<size_t>& _vIDs = std::vector<size_t>()) const;
	std::vector<size_t> GetSpheresPartiallyOutside(const std::vector<CVector3>& _vCoords, const std::vector<double>& _vRadiuses, const std::vector<size_t>& _vIDs = std::vector<size_t>()) const;

private:
	/// If radii are specified, objects will be considered as spheres; if flag _bTotally = 1, spheres, which are only partially in the volume will be considered as outside volume.
	std::vector<size_t> p_GetObjectsInside(const std::vector<CVector3>& _vCoords, const std::vector<size_t>& _vIDs, const std::vector<double>& _vRadiuses, bool _bTotally) const;
	void IsSphereInside(const CVector3& pos, double radius, bool &isInside, bool &isPartInside, bool &isUndefined) const;
	void IsPointInside(const CVector3& point, bool &isInside, bool &isUndefined) const;
	inline std::pair<int, int> GetIndexByPoint(const CVector3& point) const;
	void TryIntersectWithXAxis( const CTriangle &triangle, bool &isIntersecting, bool &isOnEdge ) const;
};