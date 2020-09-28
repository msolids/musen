/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "AnalysisVolume.h"
#include "SystemStructure.h"
#include "MeshGenerator.h"

CAnalysisVolume::CAnalysisVolume(const CSystemStructure* _systemStructure) :
	m_systemStructure{ _systemStructure }
{
	SetColor(CColor{ 0.6f, 1.0f, 0.6f, 1.0f });
}

size_t CAnalysisVolume::TrianglesNumber() const
{
	return m_mesh.TrianglesNumber();
}

CTriangularMesh CAnalysisVolume::Mesh(double _time) const
{
	return m_mesh.Shifted(Motion()->TimeDependentShift(_time));
}

CVector3 CAnalysisVolume::Center(double _time) const
{
	return m_mesh.Center() + Motion()->TimeDependentShift(_time);
}

double CAnalysisVolume::Volume() const
{
	switch (Shape())
	{
	case EVolumeShape::VOLUME_SPHERE:			return 4.0 * PI * std::pow(m_sizes.Radius(), 3) / 3.0;
	case EVolumeShape::VOLUME_BOX:				return m_sizes.Width() * m_sizes.Depth() * m_sizes.Height();
	case EVolumeShape::VOLUME_CYLINDER:			return PI * m_sizes.Radius() * m_sizes.Radius() * m_sizes.Height();
	case EVolumeShape::VOLUME_HOLLOW_SPHERE:	return 4.0 * PI / 3.0 * (std::pow(m_sizes.Radius(), 3) - std::pow(m_sizes.InnerRadius(), 3));
	case EVolumeShape::VOLUME_STL:				return m_mesh.Volume();
	}
	return 0;
}

SVolumeType CAnalysisVolume::BoundingBox(double _time) const
{
	SVolumeType bb = m_mesh.BoundingBox();
	const CVector3 shift = Motion()->TimeDependentShift(_time);
	bb.coordBeg += shift;
	bb.coordEnd += shift;
	return bb;
}

double CAnalysisVolume::MaxInscribedDiameter() const
{
	switch (Shape())
	{
	case EVolumeShape::VOLUME_SPHERE:			return m_sizes.Radius();
	case EVolumeShape::VOLUME_HOLLOW_SPHERE:	return m_sizes.Radius() - m_sizes.InnerRadius();
	case EVolumeShape::VOLUME_CYLINDER:			return std::min({ m_sizes.Radius(), m_sizes.Height() });
	case EVolumeShape::VOLUME_BOX:				return std::min({ m_sizes.Width(), m_sizes.Depth(), m_sizes.Height() });
	case EVolumeShape::VOLUME_STL:				return std::pow(Volume() / PI * 3 / 4, 1 / 3.) * 2;
	}
	return 0;
}

void CAnalysisVolume::SetMesh(const CTriangularMesh& _mesh)
{
	m_mesh = _mesh;
}

void CAnalysisVolume::SetCenter(const CVector3& _center)
{
	m_mesh.SetCenter(_center);
}

void CAnalysisVolume::SetSizes(const CGeometrySizes& _sizes)
{
	if (m_sizes == _sizes) return;
	if (Shape() != EVolumeShape::VOLUME_STL)
	{
		CBaseGeometry::SetSizes(_sizes);
		const auto mesh = CMeshGenerator::GenerateMesh(Shape(), _sizes, Center(), RotationMatrix(), Accuracy());
		SetMesh(mesh);
	}
	else
	{
		// get current bounding box
		const auto bb = BoundingBox();
		// divide entry-wise new sizes by old sizes to get an elongation factor in each direction
		const CVector3 factors = CVector3{ _sizes.Width(), _sizes.Depth(), _sizes.Height() } / (bb.coordEnd - bb.coordBeg);
		// resize
		ScaleSTL(factors);
	}
}

void CAnalysisVolume::SetAccuracy(size_t _value)
{
	if (Shape() == EVolumeShape::VOLUME_STL) return;
	if (_value == Accuracy()) return;
	m_mesh = CMeshGenerator::GenerateMesh(Shape(), Sizes(), Center(), RotationMatrix(), _value);
}

void CAnalysisVolume::Shift(const CVector3& _offset)
{
	m_mesh.Shift(_offset);
}

void CAnalysisVolume::Scale(double _factor)
{
	m_mesh.Scale(_factor / ScalingFactor());
	CBaseGeometry::Scale(_factor);
}

void CAnalysisVolume::ScaleSTL(const CVector3& _factors)
{
	CBaseGeometry::ScaleSTL(_factors);
	if (Shape() != EVolumeShape::VOLUME_STL) return;
	m_mesh.Scale(_factors);
}

void CAnalysisVolume::Rotate(const CMatrix3& _rotation)
{
	CBaseGeometry::Rotate(_rotation);
	m_mesh.Rotate(_rotation);
}

void CAnalysisVolume::SaveToProto(ProtoAnalysisVolume& _proto) const
{
	_proto.set_version(0);
	CBaseGeometry::SaveToProto(*_proto.mutable_base_geometry());
	for (const auto& t : m_mesh.Triangles())
		Val2Proto(_proto.add_triangles(), t);
}

void CAnalysisVolume::LoadFromProto(const ProtoAnalysisVolume& _proto)
{
	CBaseGeometry::LoadFromProto(_proto.base_geometry());
	m_mesh.SetTriangles(Proto2Val<CTriangle>(_proto.triangles()));
}

void CAnalysisVolume::LoadFromProto_v0(const ProtoAnalysisVolume_v0& _proto)
{
	SetName(_proto.name());
	SetKey(_proto.key());
	StoreShape(static_cast<EVolumeShape>(_proto.type()));
	m_sizes.SetRelevantSizes(Proto2Val<double>(_proto.vprops()), Shape());
	m_mesh.SetTriangles(Proto2Val<CTriangle>(_proto.triangles()));
	if (_proto.has_rotation())
		StoreRotationMatrix(Proto2Val(_proto.rotation()));
	else if (Shape() != EVolumeShape::VOLUME_STL) // compatibility with older versions
		m_mesh.SetTriangles(CMeshGenerator::GenerateMesh(Shape(), Sizes(), Proto2Val(_proto.vcenter()), CMatrix3::Diagonal()).Triangles());
	if (_proto.has_color())
		SetColor(Proto2Val(_proto.color()));

	double prev = 0.0;
	Motion()->Clear();
	for (const auto& td : _proto.tdval())
	{
		Motion()->AddTimeInterval(CGeometryMotion::STimeMotionInterval{ prev, td.time(), CGeometryMotion::SMotionInfo{Proto2Val(td.velocity()), CVector3{ 0.0 }, CVector3{ 0.0 }} });
		prev = td.time();
	}
	if (Motion()->HasMotion())
		Motion()->SetMotionType(CGeometryMotion::EMotionType::TIME_DEPENDENT);
}

std::vector<size_t> CAnalysisVolume::GetParticleIndicesInside(double _time, bool _totallyInside /*= true*/) const
{
	const auto particles = m_systemStructure->GetAllSpheres(_time);
	auto IDs    = ReservedVector<size_t>(particles.size());
	auto radii  = ReservedVector<double>(particles.size());
	auto coords = ReservedVector<CVector3>(particles.size());
	for (const auto& p : particles)
	{
		IDs.push_back(p->m_lObjectID);
		if (_totallyInside)
			radii.push_back(p->GetRadius());
		coords.push_back(p->GetCoordinates(_time));
	}

	return CInsideVolumeChecker{ this, _time }.GetSpheresTotallyInside(coords, radii, IDs);
}

std::vector<const CSphere*> CAnalysisVolume::GetParticlesInside(double _time, bool _totallyInside /*= true*/) const
{
	const std::vector<size_t> indices = GetParticleIndicesInside(_time, _totallyInside);
	auto res = ReservedVector<const CSphere*>(indices.size());
	for (size_t i : indices)
		res.push_back(dynamic_cast<const CSphere*>(m_systemStructure->GetObjectByIndex(i)));
	return res;
}

std::vector<size_t> CAnalysisVolume::GetSolidBondIndicesInside(double _time) const
{
	const auto bonds = m_systemStructure->GetAllSolidBonds(_time);
	auto IDs    = ReservedVector<size_t>(bonds.size());
	auto coords = ReservedVector<CVector3>(bonds.size());
	for (const auto& b : bonds)
	{
		IDs.push_back(b->m_lObjectID);
		coords.push_back(m_systemStructure->GetBondCoordinate(_time, b->m_lObjectID));
	}

	return CInsideVolumeChecker{ this, _time }.GetObjectsInside(coords, IDs);
}

std::vector<const CSolidBond*> CAnalysisVolume::GetSolidBondsInside(double _time) const
{
	const std::vector<size_t> indices = GetSolidBondIndicesInside(_time);
	auto res = ReservedVector<const CSolidBond*>(indices.size());
	for (size_t i : indices)
		res.push_back(dynamic_cast<const CSolidBond*>(m_systemStructure->GetObjectByIndex(i)));
	return res;
}

std::vector<size_t> CAnalysisVolume::GetLiquidBondIndicesInside(double _time) const
{
	const auto bonds = m_systemStructure->GetAllLiquidBonds(_time);
	auto IDs    = ReservedVector<size_t>(bonds.size());
	auto coords = ReservedVector<CVector3>(bonds.size());
	for (const auto& b : bonds)
	{
		IDs.push_back(b->m_lObjectID);
		coords.push_back(m_systemStructure->GetBondCoordinate(_time, b->m_lObjectID));
	}

	return CInsideVolumeChecker{ this, _time }.GetObjectsInside(coords, IDs);
}

std::vector<const CLiquidBond*> CAnalysisVolume::GetLiquidBondsInside(double _time) const
{
	const std::vector<size_t> indices = GetLiquidBondIndicesInside(_time);
	auto res = ReservedVector<const CLiquidBond*>(indices.size());
	res.reserve(indices.size());
	for (size_t i : indices)
		res.push_back(dynamic_cast<const CLiquidBond*>(m_systemStructure->GetObjectByIndex(i)));
	return res;
}

std::vector<size_t> CAnalysisVolume::GetBondIndicesInside(double _time) const
{
	const auto bonds = m_systemStructure->GetAllBonds(_time);
	auto IDs    = ReservedVector<size_t>(bonds.size());
	auto coords = ReservedVector<CVector3>(bonds.size());
	for (const auto& b : bonds)
	{
		IDs.push_back(b->m_lObjectID);
		coords.push_back(m_systemStructure->GetBondCoordinate(_time, b->m_lObjectID));
	}

	return CInsideVolumeChecker{ this, _time }.GetObjectsInside(coords, IDs);
}

std::vector<const CBond*> CAnalysisVolume::GetBondsInside(double _time) const
{
	const std::vector<size_t> indices = GetBondIndicesInside(_time);
	auto res = ReservedVector<const CBond*>(indices.size());
	for (size_t i : indices)
		res.push_back(dynamic_cast<const CBond*>(m_systemStructure->GetObjectByIndex(i)));
	return res;
}

std::vector<size_t> CAnalysisVolume::GetWallIndicesInside(double _time) const
{
	const auto walls = m_systemStructure->GetAllWalls(_time);
	auto wallsID     = ReservedVector<size_t>(walls.size());
	auto wallsCoordX = ReservedVector<CVector3>(walls.size());
	auto wallsCoordY = ReservedVector<CVector3>(walls.size());
	auto wallsCoordZ = ReservedVector<CVector3>(walls.size());
	for (const auto& w : walls)
	{
		wallsID.push_back(w->m_lObjectID);
		const CTriangle t = w->GetCoords(_time);
		wallsCoordX.push_back(t.p1);
		wallsCoordY.push_back(t.p2);
		wallsCoordZ.push_back(t.p3);
	}
	const CInsideVolumeChecker checher(this, _time);
	const std::vector<size_t> resultsIDx = checher.GetObjectsInside(wallsCoordX, wallsID);
	const std::vector<size_t> resultsIDy = checher.GetObjectsInside(wallsCoordY, wallsID);
	const std::vector<size_t> resultsIDz = checher.GetObjectsInside(wallsCoordZ, wallsID);
	return VectorIntersection({ resultsIDx, resultsIDy, resultsIDz });
}

std::vector<const CTriangularWall*> CAnalysisVolume::GetWallsInside(double _time) const
{
	const std::vector<size_t> indices = GetWallIndicesInside(_time);
	auto res = ReservedVector<const CTriangularWall*>(indices.size());
	for (size_t i : indices)
		res.push_back(dynamic_cast<const CTriangularWall*>(m_systemStructure->GetObjectByIndex(i)));
	return res;
}

std::vector<size_t> CAnalysisVolume::GetObjectIndicesInside(double _time, const std::vector<CVector3>& _coords) const
{
	return CInsideVolumeChecker{ this, _time }.GetObjectsInside(_coords);
}
