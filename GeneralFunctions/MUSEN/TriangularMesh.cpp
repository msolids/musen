/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "TriangularMesh.h"
#include <algorithm>
#include <utility>
#include <map>
#include <array>

CTriangularMesh::CTriangularMesh(std::string _name, std::vector<STriangleType> _triangles) :
	sName(std::move(_name)), vTriangles(std::move(_triangles))
{};

void CTriangularMesh::Move(const CVector3& _offs)
{
	for (auto& t : vTriangles)
		t.Shift(_offs);
}

void CTriangularMesh::Rotate(const CMatrix3& _rotation)
{
	const CVector3 center = Center();
	for (auto& t : vTriangles)
		t.Rotate(center, _rotation);
}

void CTriangularMesh::InvertFaceNormals()
{
	for (auto & t : vTriangles)
		t.InvertOrientation();
}

CVector3 CTriangularMesh::Center() const
{
	CVector3 center(0.);
	for (size_t i = 0; i < vTriangles.size(); ++i)
		center = (center*double(i) + (vTriangles[i].p1 + vTriangles[i].p2 + vTriangles[i].p3) / 3.) / (double(i) + 1.);
	return center;
}

double CTriangularMesh::Volume() const
{
	double volume = 0;
	for (const auto& t : vTriangles)
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
	if (vTriangles.empty()) return bb;
	bb.coordEnd = bb.coordBeg = vTriangles.front().p1;
	for (const auto& t : vTriangles)
	{
		bb.coordBeg = Min(bb.coordBeg, t.p1);
		bb.coordBeg = Min(bb.coordBeg, t.p2);
		bb.coordBeg = Min(bb.coordBeg, t.p3);
		bb.coordEnd = Max(bb.coordEnd, t.p1);
		bb.coordEnd = Max(bb.coordEnd, t.p2);
		bb.coordEnd = Max(bb.coordEnd, t.p3);
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
		CVector3 vertex1;
		CVector3 vertex2;
		bool reversedDirection;
		SEdge(const CVector3& _vertex1, const CVector3& _vertex2, bool _reversedDirection)
			: vertex1(_vertex1), vertex2(_vertex2), reversedDirection(_reversedDirection) {};
	};

	// Used to sort edges (sorting purpose: faster finding of faces with common edges in same direction).
	bool CompareEdge(const SEdge& e1, const SEdge& e2)
	{
		if (std::fabs(e1.vertex1.x - e2.vertex1.x) > tol)
			return e1.vertex1.x > e2.vertex1.x;
		if (std::fabs(e1.vertex1.y - e2.vertex1.y) > tol)
			return  e1.vertex1.y > e2.vertex1.y;
		if (std::fabs(e1.vertex1.z - e2.vertex1.z) > tol)
			return e1.vertex1.z > e2.vertex1.z;
		if (std::fabs(e1.vertex2.x - e2.vertex2.x) > tol)
			return e1.vertex2.x > e2.vertex2.x;
		if (std::fabs(e1.vertex2.y - e2.vertex2.y) > tol)
			return  e1.vertex2.y > e2.vertex2.y;
		if (std::fabs(e1.vertex2.z - e2.vertex2.z) > tol)
			return e1.vertex2.z > e2.vertex2.z;
		return e1.reversedDirection > e2.reversedDirection;
	}

	bool EqualEdge(const SEdge& e1, const SEdge& e2)
	{
		return std::fabs(e1.vertex1.x - e2.vertex1.x) < tol &&
			   std::fabs(e1.vertex1.y - e2.vertex1.y) < tol &&
			   std::fabs(e1.vertex1.z - e2.vertex1.z) < tol &&
			   std::fabs(e1.vertex2.x - e2.vertex2.x) < tol &&
			   std::fabs(e1.vertex2.y - e2.vertex2.y) < tol &&
			   std::fabs(e1.vertex2.z - e2.vertex2.z) < tol;
	}

	// Checks if edges are common and have same direction.
	bool InvalidEdge(const SEdge& e1, const SEdge& e2)
	{
		return EqualEdge(e1, e2) && e1.reversedDirection == e2.reversedDirection;
	}
}

bool CTriangularMesh::IsFaceNormalsConsistent() const
{
	// Checks if two faces share a common edge with the same direction, ie.e. that adjacent faces have normals in opposite direction.
	MeshCheck::tol = tol; // update tolerance

	// Generate edges, using both directions
	std::vector<MeshCheck::SEdge> edges;
	edges.reserve(vTriangles.size() * 6);
	for (const auto& triangle : vTriangles)
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

CTriangularMesh CTriangularMesh::GenerateMesh(const EVolumeType& _type, const std::vector<double>& _params, const CVector3& _center, const CMatrix3& _rotation, size_t _accuracy)
{
	CTriangularMesh mesh;
	switch (_type)
	{
	case EVolumeType::VOLUME_BOX:
		if (_params.size() < 3) return {};
		mesh = GenerateBoxMesh(_params[0], _params[1], _params[2]);
		break;
	case EVolumeType::VOLUME_CYLINDER:
		if (_params.size() < 2) return {};
		mesh = GenerateCylinderMesh(_params[0], _params[1], _accuracy);
		break;
	case EVolumeType::VOLUME_HOLLOW_SPHERE:
		if (_params.size() < 2) return {};
		mesh = GenerateHollowSphereMesh(_params[0], _params[1], _accuracy);
		break;
	case EVolumeType::VOLUME_SPHERE:
		if (_params.empty()) return {};
		mesh = GenerateSphereMesh(_params[0], _accuracy);
		break;
	case EVolumeType::VOLUME_STL:
		break;
	}

	// move vertices to the center
	mesh.Move(_center);
	// rotate figure
	mesh.Rotate(_rotation);

	return mesh;
}

CTriangularMesh CTriangularMesh::GenerateBoxMesh(double _length, double _width, double _height)
{
	// generate 8 vertices
	const std::vector<CVector3> vertices =
	{
		// 4 vertices on the top plane
		CVector3(-_length / 2, -_width / 2, _height / 2),
		CVector3(-_length / 2,  _width / 2, _height / 2),
		CVector3(_length / 2,  _width / 2, _height / 2),
		CVector3(_length / 2, -_width / 2, _height / 2),
		// 4 vertices on the bottom plane
		CVector3(-_length / 2, -_width / 2, -_height / 2),
		CVector3(-_length / 2,  _width / 2, -_height / 2),
		CVector3(_length / 2,  _width / 2, -_height / 2),
		CVector3(_length / 2, -_width / 2, -_height / 2)
	};

	// generate 12 triangles
	const std::vector<std::vector<size_t>> triangles =
	{
		{ 0, 1, 5 },{ 0, 5, 4 },{ 2, 3, 6 },{ 3, 7, 6 },
		{ 0, 4, 3 },{ 3, 4, 7 },{ 1, 2, 5 },{ 2, 6, 5 },
		{ 0, 3, 1 },{ 1, 3, 2 },{ 4, 5, 7 },{ 5, 6, 7 }
	};

	// assemble mesh
	return AssembleMesh(vertices, triangles, "Box");
}

CTriangularMesh CTriangularMesh::GenerateCylinderMesh(double _radius, double _height, size_t _accuracy)
{
	const size_t q = _accuracy != 0 ? _accuracy : 64; // determines number of points used to generate each circle of the cylinder
	const double a = 360. / q * PI / 180;	          // offset in radians between each point on circles

	// generate vertices
	std::vector<CVector3> vertices(2 * q + 2); // additional two points: in the center of each circle
	vertices[0] = CVector3{ 0, 0, _height / 2 };			// point in the center of the the top circle
	vertices[2 * q + 1] = CVector3{ 0, 0, -_height / 2 };	// point in the center of the the bottom circle
	const double x = -_radius; // initial coordinates on the circle to rotate
	const double y = 0;	       // initial coordinates on the circle to rotate
	for (size_t i = 0; i < q; ++i) // rotate initial point...
	{
		vertices[i + 1] = CVector3{ x*cos(i*a) - y * sin(i*a),  x*sin(i*a) + y * cos(i*a),  _height / 2 };			// ... on the top circle counterclockwise
		vertices[q + i + 1] = CVector3{ x*cos(i*a) + y * sin(i*a), -x * sin(i*a) + y * cos(i*a), -_height / 2 };	// ... on the bottom circle clockwise
	}

	// generate triangles
	std::vector<std::vector<size_t>> triangles(4 * q, { 0, 0, 0 });
	for (size_t i = 0; i < q; ++i) // q triangles for each circle: 2 points on the circumference + 1 point in the center
	{
		triangles[i] = { 0,			i + 1,		(i + 1) % q + 1 };	// for the top circle
		triangles[3 * q + i] = { 2 * q + 1, q + i + 1,	q + (i + 1) % q + 1 };	// for the bottom circle
	}
	for (size_t i = 0; i < q; ++i) // 2q triangles for the side surface
	{
		triangles[q + i] = { i + 1, 2 * q - i,				(i + 1) % q + 1 };	// \/-directed triangles
		triangles[2 * q + i] = { i + 1, (q - i) % q + q + 1,	2 * q - i };	// /\-directed triangles
	}

	// assemble mesh
	return AssembleMesh(vertices, triangles, "Cylinder");
}

CTriangularMesh CTriangularMesh::GenerateSphereMesh(double _radius, size_t _accuracy)
{
	const size_t subdivs = _accuracy != 0 ? _accuracy : 4; // number of successive subdivisions of the sphere surface
	// generate vertices of initial icosahedron
	std::vector<CVector3> vertices(12);
	vertices[0]  = CVector3{ 0., 0.,  _radius };	// north pole
	vertices[11] = CVector3{ 0., 0., -_radius };	// south pole
	const double latitude = atan(0.5);				// latitude of the rest 10 vertices, 5 with +, 5 with -
	const double offs36 = 36.*PI / 180.;			// offset in 36 degree
	const double offs72 = 72.*PI / 180.;			// offset in 72 degrees
	for (size_t i = 0; i < 5; ++i)
	{
		vertices[i + 1] = CVector3{ _radius * std::cos(i * offs72) * std::cos(latitude),
									_radius * std::sin(i * offs72) * std::cos(latitude),
									_radius * std::sin(latitude) };	// top 5 points
		vertices[i + 6] = CVector3{ _radius * std::cos(offs36 + i * offs72) * std::cos(-latitude),
									_radius * std::sin(offs36 + i * offs72) * std::cos(-latitude),
									_radius * std::sin(-latitude) };	// bottom 5 points
	}

	// generate triangles
	std::vector<std::vector<size_t>> triangles =
	{
		{ 0, 1,  2 }, { 0, 2,  3 }, { 0, 3,  4 }, {  0,  4,  5 }, {  0,  5,  1 },
		{ 6, 2,  1 }, { 7, 3,  2 }, { 8, 4,  3 }, {  9,  5,  4 }, { 10,  1,  5 },
		{ 6, 7,  2 }, { 7, 8,  3 }, { 8, 9,  4 }, {  9, 10,  5 }, { 10,  6,  1 },
		{ 7, 6, 11 }, { 8, 7, 11 }, { 9, 8, 11 }, { 10,  9, 11 }, {  6, 10, 11 }
	};

	// perform subdivision replacing each triangle with 4 new triangles
	for (size_t i = 0; i < subdivs; ++i)
	{
		std::vector<std::vector<size_t>> newTriangles;
		std::map<std::pair<size_t, size_t>, size_t> lookup;
		for (auto&& t : triangles)
		{
			std::array<size_t, 3> mid{};
			for (size_t edge = 0; edge < 3; ++edge) // find new vertex on each edge
			{
				std::map<std::pair<size_t, size_t>, size_t>::key_type key(t[edge], t[(edge + 1) % 3]);
				if (key.first > key.second)
					std::swap(key.first, key.second);
				const auto inserted = lookup.insert({ key, vertices.size() });
				if (inserted.second)
				{
					auto& edge0 = vertices[t[edge]];
					auto& edge1 = vertices[t[(edge + 1) % 3]];
					auto point = Normalized(edge0 + edge1) * _radius;
					vertices.push_back(point);
				}
				mid[edge] = inserted.first->second;
			}
			newTriangles.push_back({   t[0], mid[0], mid[2] });
			newTriangles.push_back({   t[1], mid[1], mid[0] });
			newTriangles.push_back({   t[2], mid[2], mid[1] });
			newTriangles.push_back({ mid[0], mid[1], mid[2] });
		}
		triangles = newTriangles;
	}

	// assemble mesh
	return AssembleMesh(vertices, triangles, "Sphere");
}

CTriangularMesh CTriangularMesh::GenerateHollowSphereMesh(double _outerRadius, double _innerRadius, size_t _accuracy)
{
	// generate outer mesh
	CTriangularMesh outer = GenerateSphereMesh(_outerRadius, _accuracy);
	// generate inner mesh
	CTriangularMesh inner = GenerateSphereMesh(_innerRadius, _accuracy);
	// invert normals of each triangle in the inner mesh
	for (auto& t : inner.vTriangles)
		std::swap(t.p1, t.p3);
	// combine meshes
	CTriangularMesh hollowSphere;
	hollowSphere.sName = "Hollow sphere";
	hollowSphere.vTriangles.reserve(outer.vTriangles.size() + inner.vTriangles.size());
	hollowSphere.vTriangles.insert(hollowSphere.vTriangles.end(), outer.vTriangles.begin(), outer.vTriangles.end());
	hollowSphere.vTriangles.insert(hollowSphere.vTriangles.end(), inner.vTriangles.begin(), inner.vTriangles.end());
	return hollowSphere;
}

CTriangularMesh CTriangularMesh::AssembleMesh(const std::vector<CVector3>& _vertices, const std::vector<std::vector<size_t>>& _triangles, const std::string& _name)
{
	CTriangularMesh mesh;
	mesh.sName = _name;
	mesh.vTriangles.resize(_triangles.size());
	for (size_t i = 0; i < _triangles.size(); ++i)
		mesh.vTriangles[i] = { _vertices[_triangles[i][0]], _vertices[_triangles[i][1]], _vertices[_triangles[i][2]] };
	return mesh;
}