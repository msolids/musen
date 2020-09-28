/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "TriangularWall.h"
#include "../SimResultsStorage/DemStorage.h"

CTriangularWall::CTriangularWall(unsigned _id, CDemStorage* _storage) :CPhysicalObject(_id, _storage)
{
	m_storage->Object(m_lObjectID)->set_type(ProtoParticleInfo::kTriangularWall);
}

void CTriangularWall::SetPlaneCoord(double _time, const CVector3& _vert1, const CVector3& _vert2, const CVector3& _vert3) const
{
	SetCoordinates(_time, _vert1);
	SetOrientation(_time, CQuaternion{ _vert2.x, _vert2.y, _vert2.z, 0 });
	SetAngleVelocity(_time, _vert3);
}

void CTriangularWall::SetPlaneCoord(double _time, const CTriangle& _triangle) const
{
	SetPlaneCoord(_time, _triangle.p1, _triangle.p2, _triangle.p3);
}

void CTriangularWall::SetPlaneCoord(const CVector3& _vert1, const CVector3& _vert2, const CVector3& _vert3) const
{
	SetCoordinates(_vert1);
	SetOrientation(CQuaternion{ _vert2.x, _vert2.y, _vert2.z, 0 });
	SetAngleVelocity(_vert3);
}

void CTriangularWall::SetPlaneCoord(const CTriangle& _triangle) const
{
	SetPlaneCoord(_triangle.p1, _triangle.p2, _triangle.p3);
}

CVector3 CTriangularWall::GetNormalVector(double _time) const
{
	return Normalized((GetCoordVertex2(_time) - GetCoordVertex1(_time)) * (GetCoordVertex3(_time) - GetCoordVertex1(_time)));
}

CVector3 CTriangularWall::GetNormalVector() const
{
	return Normalized((GetCoordVertex2() - GetCoordVertex1()) * (GetCoordVertex3() - GetCoordVertex1()));
}
