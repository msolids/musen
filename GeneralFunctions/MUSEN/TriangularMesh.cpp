/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "TriangularMesh.h"
#include <algorithm>
#include <utility>
#include <array>

CTriangularMesh::CTriangularMesh(std::string _name, std::vector<CTriangle> _triangles) :
	m_name(std::move(_name)), m_triangles(std::move(_triangles))
{}

std::string CTriangularMesh::Name() const
{
	return m_name;
}

void CTriangularMesh::SetName(const std::string& _name)
{
	m_name = _name;
}

std::vector<CTriangle> CTriangularMesh::Triangles() const
{
	return m_triangles;
}

void CTriangularMesh::SetTriangles(const std::vector<CTriangle>& _triangles)
{
	m_triangles = _triangles;
}

void CTriangularMesh::AddTriangle(const CTriangle& _triangle)
{
	m_triangles.push_back(_triangle);
}

size_t CTriangularMesh::TrianglesNumber() const
{
	return m_triangles.size();
}

bool CTriangularMesh::IsEmpty() const
{
	return m_triangles.empty();
}

void CTriangularMesh::SetCenter(const CVector3& _center)
{
	Shift(_center - Center());
}

void CTriangularMesh::Shift(const CVector3& _offs)
{
	for (auto& t : m_triangles)
		t.Shift(_offs);
}

CTriangularMesh CTriangularMesh::Shifted(const CVector3& _offs) const
{
	CTriangularMesh copy{ m_name, m_triangles };
	copy.Shift(_offs);
	return copy;
}

void CTriangularMesh::Scale(double _factor)
{
	const CVector3 center = Center();
	for (auto& t : m_triangles)
		t.Scale(center, _factor);
}

void CTriangularMesh::Scale(const CVector3& _factors)
{
	const CVector3 center = Center();
	for (auto& t : m_triangles)
		t.Scale(center, _factors);
}

void CTriangularMesh::Rotate(const CMatrix3& _rotation)
{
	const CVector3 center = Center();
	for (auto& t : m_triangles)
		t.Rotate(center, _rotation);
}

void CTriangularMesh::InvertFaceNormals()
{
	for (auto & t : m_triangles)
		t.InvertOrientation();
}

CVector3 CTriangularMesh::Center() const
{
	CVector3 center(0.0);
	for (size_t i = 0; i < m_triangles.size(); ++i)
		center = (center * static_cast<double>(i) + (m_triangles[i].p1 + m_triangles[i].p2 + m_triangles[i].p3) / 3.) / (static_cast<double>(i) + 1.);
	return center;
}

double CTriangularMesh::Volume() const
{
	double volume = 0;
	for (const auto& t : m_triangles)
	{
		const double v321 = t.p3.x * t.p2.y * t.p1.z;
		const double v231 = t.p2.x * t.p3.y * t.p1.z;
		const double v132 = t.p1.x * t.p3.y * t.p2.z;
		const double v312 = t.p3.x * t.p1.y * t.p2.z;
		const double v213 = t.p2.x * t.p1.y * t.p3.z;
		const double v123 = t.p1.x * t.p2.y * t.p3.z;
		volume += (-v321 + v231 + v312 - v132 - v213 + v123) / 6.;
	}
	return std::fabs(volume);
}

SVolumeType CTriangularMesh::BoundingBox() const
{
	SVolumeType bb{ CVector3{ 0 }, CVector3{ 0 } };
	if (m_triangles.empty()) return bb;
	bb.coordEnd = bb.coordBeg = m_triangles.front().p1;
	for (const auto& t : m_triangles)
	{
		bb.coordBeg = Min(bb.coordBeg, t.p1, t.p2, t.p3);
		bb.coordEnd = Max(bb.coordEnd, t.p1, t.p2, t.p3);
	}
	return bb;
}

CTriangularMesh CTriangularMesh::CreateInvertedMesh() const
{
	CTriangularMesh out(*this);
	out.InvertFaceNormals();
	return out;
}

namespace MeshCheck
{
	double tol = 0.0;

	struct SEdge
	{
		CVector3 v1, v2;	// two vertices
		bool revDir;		// reversed direction of vertices
		SEdge(const CVector3& _vertex1, const CVector3& _vertex2, bool _reversedDirection)
			: v1(_vertex1), v2(_vertex2), revDir(_reversedDirection) {};
	};

	// Used to sort edges (sorting purpose: faster finding of faces with common edges in same direction).
	bool CompareEdge(const SEdge& e1, const SEdge& e2)
	{
		if (std::fabs(e1.v1.x - e2.v1.x) > tol) return e1.v1.x > e2.v1.x;
		if (std::fabs(e1.v1.y - e2.v1.y) > tol) return e1.v1.y > e2.v1.y;
		if (std::fabs(e1.v1.z - e2.v1.z) > tol) return e1.v1.z > e2.v1.z;
		if (std::fabs(e1.v2.x - e2.v2.x) > tol) return e1.v2.x > e2.v2.x;
		if (std::fabs(e1.v2.y - e2.v2.y) > tol) return e1.v2.y > e2.v2.y;
		if (std::fabs(e1.v2.z - e2.v2.z) > tol) return e1.v2.z > e2.v2.z;
		return e1.revDir > e2.revDir;
	}

	bool EqualEdge(const SEdge& e1, const SEdge& e2)
	{
		return std::fabs(e1.v1.x - e2.v1.x) < tol &&
			   std::fabs(e1.v1.y - e2.v1.y) < tol &&
			   std::fabs(e1.v1.z - e2.v1.z) < tol &&
			   std::fabs(e1.v2.x - e2.v2.x) < tol &&
			   std::fabs(e1.v2.y - e2.v2.y) < tol &&
			   std::fabs(e1.v2.z - e2.v2.z) < tol;
	}

	// Checks if edges are common and have same direction.
	bool InvalidEdge(const SEdge& e1, const SEdge& e2)
	{
		return EqualEdge(e1, e2) && e1.revDir == e2.revDir;
	}
}

bool CTriangularMesh::IsFaceNormalsConsistent() const
{
	// Checks if two faces share a common edge with the same direction, ie.e. that adjacent faces have normals in opposite direction.
	MeshCheck::tol = 1e-8; // update tolerance

	// Generate edges, using both directions
	std::vector<MeshCheck::SEdge> edges;
	edges.reserve(m_triangles.size() * 6);
	for (const auto& triangle : m_triangles)
	{
		edges.emplace_back(triangle.p1, triangle.p2, false);
		edges.emplace_back(triangle.p2, triangle.p3, false);
		edges.emplace_back(triangle.p3, triangle.p1, false);
		edges.emplace_back(triangle.p2, triangle.p1, true);
		edges.emplace_back(triangle.p3, triangle.p2, true);
		edges.emplace_back(triangle.p1, triangle.p3, true);
	}
	// Remark: using both directions increases needed memory, but should result in higher speed.
	// Otherwise (I think) two nested loops would be necessary

	// sort edges
	std::sort(edges.begin(), edges.end(), MeshCheck::CompareEdge);

	// look for any equal edges which traversed in the same direction --> face normals are in opposite directions
	return std::adjacent_find(edges.begin(), edges.end(), MeshCheck::InvalidEdge) == edges.end();
}
