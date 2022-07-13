/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "RealGeometry.h"
#include "MeshGenerator.h"
#include "SystemStructure.h"

CRealGeometry::CRealGeometry(CSystemStructure* _systemStructure) :
	m_systemStructure{ _systemStructure }
{
	SetColor(CColor::DefaultRealGeometryColor());
}

size_t CRealGeometry::TrianglesNumber() const
{
	return m_planes.size();
}

std::vector<size_t> CRealGeometry::Planes() const
{
	return m_planes;
}

CBasicVector3<bool> CRealGeometry::FreeMotion() const
{
	return m_freeMotion;
}

double CRealGeometry::Mass() const
{
	return m_mass;
}

bool CRealGeometry::RotateAroundCenter() const
{
	return m_rotateAroundCenter;
}

CVector3 CRealGeometry::Center(double _time) const
{
	m_systemStructure->PrepareTimePointForRead(_time);
	double dTotalSurface = 0;
	CVector3 center{ 0.0 };
	for (const auto& wall : Walls())
	{
		double dSurface = 0.5* Length(((wall->GetCoordVertex2() - wall->GetCoordVertex1())*(wall->GetCoordVertex3() - wall->GetCoordVertex1())));
		center += dSurface/3.0*(wall->GetCoordVertex1() + wall->GetCoordVertex2() + wall->GetCoordVertex3());
		dTotalSurface += dSurface;
	}
	if (dTotalSurface != 0) center = center / dTotalSurface;
	return center;
}

std::string CRealGeometry::Material() const
{
	if (m_planes.empty()) return {};
	return m_systemStructure->GetObjectByIndex(m_planes.front())->GetCompoundKey();
}

SVolumeType CRealGeometry::BoundingBox(double _time) const
{
	SVolumeType bb{ CVector3{ 0 }, CVector3{ 0 } };
	if (TrianglesNumber() == 0) return bb;
	m_systemStructure->PrepareTimePointForRead(_time);
	const auto walls = Walls();
	bb.coordEnd = bb.coordBeg = walls.front()->GetCoordVertex1();
	for (const auto& w : walls)
	{
		const auto t = w->GetPlaneCoords();
		bb.coordBeg = Min(bb.coordBeg, t.p1, t.p2, t.p3);
		bb.coordEnd = Max(bb.coordEnd, t.p1, t.p2, t.p3);
	}
	return bb;
}

void CRealGeometry::SetMesh(const CTriangularMesh& _mesh)
{
	const auto& triangles = _mesh.Triangles();
	m_systemStructure->DeleteObjects(m_planes);
	m_planes.clear();
	m_planes.reserve(triangles.size());
	auto walls = m_systemStructure->AddSeveralObjects(TRIANGULAR_WALL, triangles.size());
	m_systemStructure->PrepareTimePointForWrite(0.0);
	for (size_t i = 0; i < walls.size(); ++i)
	{
		auto* wall = dynamic_cast<CTriangularWall*>(walls[i]);
		wall->SetPlaneCoord(triangles[i]);
		m_planes.push_back(wall->m_lObjectID);
	}

}

void CRealGeometry::SetPlanesIndices(const std::vector<size_t>& _planes)
{
	m_planes = _planes;
}

void CRealGeometry::SetFreeMotion(const CBasicVector3<bool>& _flags)
{
	m_freeMotion = _flags;
}

void CRealGeometry::SetMass(double _mass)
{
	m_mass = _mass;
}

void CRealGeometry::SetRotateAroundCenter(bool _flag)
{
	m_rotateAroundCenter = _flag;
}

void CRealGeometry::SetAccuracy(size_t _value)
{
	if (Shape() == EVolumeShape::VOLUME_STL) return;
	if (_value == Accuracy()) return;
	const auto mesh = CMeshGenerator::GenerateMesh(Shape(), Sizes(), Center(), RotationMatrix(), _value);
	SetMesh(mesh);
}

void CRealGeometry::Shift(const CVector3& _offset)
{
	m_systemStructure->PrepareTimePointForRead(0.0);
	m_systemStructure->PrepareTimePointForWrite(0.0);
	for (auto& wall : Walls())
		wall->SetPlaneCoord(wall->GetPlaneCoords().Shifted(_offset));
}

void CRealGeometry::SetCenter(const CVector3& _coord)
{
	Shift(_coord - Center());
}

void CRealGeometry::SetMaterial(const std::string& _compoundKey)
{
	for (auto& wall : Walls())
		wall->SetCompoundKey(_compoundKey);
}

void CRealGeometry::Scale(double _factor)
{
	m_systemStructure->PrepareTimePointForRead(0.0);
	m_systemStructure->PrepareTimePointForWrite(0.0);
	const CVector3 center = Center();
	for (auto& wall : Walls())
		wall->SetPlaneCoord(wall->GetPlaneCoords().Scaled(center, _factor / ScalingFactor()));
	CBaseGeometry::Scale(_factor);
}

void CRealGeometry::DeformSTL(const CVector3& _factors)
{
	CBaseGeometry::DeformSTL(_factors);
	if (Shape() != EVolumeShape::VOLUME_STL) return;
	m_systemStructure->PrepareTimePointForRead(0.0);
	m_systemStructure->PrepareTimePointForWrite(0.0);
	const CVector3 center = Center();
	for (auto& wall : Walls())
		wall->SetPlaneCoord(wall->GetPlaneCoords().Scaled(center, _factors));
}

void CRealGeometry::Rotate(const CMatrix3& _rotation)
{
	CBaseGeometry::Rotate(_rotation);
	m_systemStructure->PrepareTimePointForRead(0.0);
	m_systemStructure->PrepareTimePointForWrite(0.0);
	const CVector3 center = Center();
	for (auto& wall : Walls())
		wall->SetPlaneCoord(wall->GetPlaneCoords().Rotated(center, _rotation));
}

void CRealGeometry::UpdateMotionInfo(double _dependentValue)
{
	Motion()->UpdateMotionInfo(_dependentValue);
}

CVector3 CRealGeometry::GetCurrentVelocity() const
{
	return Motion()->GetCurrentMotion().velocity;
}

CVector3 CRealGeometry::GetCurrentRotVelocity() const
{
	return Motion()->GetCurrentMotion().rotationVelocity;
}

CVector3 CRealGeometry::GetCurrentRotCenter() const
{
	return Motion()->GetCurrentMotion().rotationCenter;
}

void CRealGeometry::SaveToProto(ProtoRealGeometry& _proto) const
{
	_proto.set_version(0);
	CBaseGeometry::SaveToProto(*_proto.mutable_base_geometry());
	for (const auto& plane : m_planes)
		_proto.add_planes(plane);
	Val2Proto(_proto.mutable_free_motion(), m_freeMotion);
	_proto.set_mass(m_mass);
	_proto.set_rotate_around_center(m_rotateAroundCenter);
}

void CRealGeometry::LoadFromProto(const ProtoRealGeometry& _proto)
{
	CBaseGeometry::LoadFromProto(_proto.base_geometry());
	m_planes = Proto2Val<size_t>(_proto.planes());
	m_freeMotion = Proto2Val(_proto.free_motion());
	m_mass = _proto.mass();
	m_rotateAroundCenter = _proto.rotate_around_center();
}

void CRealGeometry::LoadFromProto_v0(const ProtoRealGeometry_v0& _proto)
{
	SetName(_proto.name());
	SetKey(_proto.key());
	m_mass = _proto.mass();
	m_freeMotion = Proto2Val(_proto.vfreemotion());
	m_rotateAroundCenter = _proto.rotate_aroundmasscenter();
	if (_proto.tdval_size() == 0)
		Motion()->SetMotionType(CGeometryMotion::EMotionType::NONE);
	else if (_proto.forcedependentvel())
		Motion()->SetMotionType(CGeometryMotion::EMotionType::FORCE_DEPENDENT);
	else
		Motion()->SetMotionType(CGeometryMotion::EMotionType::TIME_DEPENDENT);
	m_planes = Proto2Val<size_t>(_proto.planes());
	Motion()->Clear();
	switch (Motion()->MotionType())
	{
	case CGeometryMotion::EMotionType::TIME_DEPENDENT:
	{
		double prev = 0.0;
		for (const auto& td : _proto.tdval())
		{
			Motion()->AddTimeInterval(CGeometryMotion::STimeMotionInterval{ prev, td.time(),
				CGeometryMotion::SMotionInfo{Proto2Val(td.velocity()), Proto2Val(td.rotvelocity()), Proto2Val(td.rotcenter())} });
			prev = td.time();
		}
		if (Motion()->GetTimeIntervals().size() == 1 && prev == 0.0)
		{
			const CGeometryMotion::SMotionInfo motion = Motion()->GetTimeIntervals().front().motion;
			Motion()->ChangeTimeInterval(0, { 0.0, 99.0, motion });
		}
		break;
	}
	case CGeometryMotion::EMotionType::FORCE_DEPENDENT:
	{
		bool prev = true;
		for (const auto& td : _proto.tdval())
		{
			Motion()->AddForceInterval(CGeometryMotion::SForceMotionInterval{ td.time(), prev ? CGeometryMotion::SForceMotionInterval::ELimitType::MAX : CGeometryMotion::SForceMotionInterval::ELimitType::MIN,
				CGeometryMotion::SMotionInfo{Proto2Val(td.velocity()), Proto2Val(td.rotvelocity()), Proto2Val(td.rotcenter())} });
			prev = !prev;
		}
		break;
	}
	case CGeometryMotion::EMotionType::CONSTANT_FORCE:
	case CGeometryMotion::EMotionType::NONE: break;
	}

	SetShape(static_cast<EVolumeShape>(_proto.type()));
	SetSizesFromVector(Proto2Val<double>(_proto.props()));
	SetRotationMatrix(Proto2Val(_proto.rotation()));
	SetColor(_proto.has_color() ? Proto2Val(_proto.color()) : CColor::DefaultRealGeometryColor());
}

std::vector<CTriangularWall*> CRealGeometry::Walls()
{
	std::vector<CTriangularWall*> walls;
	walls.reserve(m_planes.size());
	for (auto iPlane : m_planes)
		if (auto* wall = dynamic_cast<CTriangularWall*>(m_systemStructure->GetObjectByIndex(iPlane)))
			walls.push_back(wall);
	return walls;
}

std::vector<const CTriangularWall*> CRealGeometry::Walls() const
{
	std::vector<const CTriangularWall*> walls;
	walls.reserve(m_planes.size());
	for (auto iPlane : m_planes)
		if (const auto* wall = dynamic_cast<const CTriangularWall*>(m_systemStructure->GetObjectByIndex(iPlane)))
			walls.push_back(wall);
	return walls;
}
