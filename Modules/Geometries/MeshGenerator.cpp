/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "MeshGenerator.h"
#include "TriangularMesh.h"
#include "GeometrySizes.h"
#include <array>

CTriangularMesh CMeshGenerator::GenerateMesh(EVolumeShape _shape, const CGeometrySizes& _sizes, const CVector3& _center, const CMatrix3& _rotation, size_t _accuracy/* = 0*/)
{
	CTriangularMesh mesh;
	switch (_shape)
	{
	case EVolumeShape::VOLUME_BOX:
		mesh = Box(_sizes.Width(), _sizes.Depth(), _sizes.Height());
		break;
	case EVolumeShape::VOLUME_CYLINDER:
		mesh = Cylinder(_sizes.Radius(), _sizes.Height(), _accuracy);
		break;
	case EVolumeShape::VOLUME_HOLLOW_SPHERE:
		mesh = HollowSphere(_sizes.Radius(), _sizes.InnerRadius(), _accuracy);
		break;
	case EVolumeShape::VOLUME_SPHERE:
		mesh = Sphere(_sizes.Radius(), _accuracy);
		break;
	case EVolumeShape::VOLUME_STL:
		break;
	}

	// move vertices to the center
	mesh.Shift(_center);
	// rotate figure
	mesh.Rotate(_rotation);

	return mesh;
}

CTriangularMesh CMeshGenerator::Box(double _width, double _depth, double _height)
{
	// generate 8 vertices
	const std::vector<CVector3> vertices =
	{
		// 4 vertices on the top plane
		CVector3{-_width / 2, -_depth / 2,  _height / 2},
		CVector3{-_width / 2,  _depth / 2,  _height / 2},
		CVector3{ _width / 2,  _depth / 2,  _height / 2},
		CVector3{ _width / 2, -_depth / 2,  _height / 2},
		// 4 vertices on the bottom plane
		CVector3{-_width / 2, -_depth / 2, -_height / 2},
		CVector3{-_width / 2,  _depth / 2, -_height / 2},
		CVector3{ _width / 2,  _depth / 2, -_height / 2},
		CVector3{ _width / 2, -_depth / 2, -_height / 2}
	};

	// generate 12 triangles
	const std::vector<std::vector<size_t>> triangles
	{
		{ 0, 1, 5 },{ 0, 5, 4 },{ 2, 3, 6 },{ 3, 7, 6 },
		{ 0, 4, 3 },{ 3, 4, 7 },{ 1, 2, 5 },{ 2, 6, 5 },
		{ 0, 3, 1 },{ 1, 3, 2 },{ 4, 5, 7 },{ 5, 6, 7 }
	};

	// assemble mesh
	return AssembleMesh(vertices, triangles, "Box");
}

CTriangularMesh CMeshGenerator::Cylinder(double _radius, double _height, size_t _accuracy)
{
	const size_t q = _accuracy != 0 ? _accuracy : 64; // determines number of points used to generate each circle of the cylinder
	const double a = 360. / q * PI / 180;	          // offset in radians between each point on circles

	// generate vertices
	std::vector<CVector3> vertices(2 * q + 2); // additional two points: in the center of each circle
	vertices[0]         = CVector3{ 0, 0,  _height / 2 };	// point in the center of the the top circle
	vertices[2 * q + 1] = CVector3{ 0, 0, -_height / 2 };	// point in the center of the the bottom circle
	const double x = -_radius; // initial coordinates on the circle to rotate
	const double y = 0;	       // initial coordinates on the circle to rotate
	for (size_t i = 0; i < q; ++i) // rotate initial point...
	{
		vertices[i + 1]     = CVector3{ x*std::cos(i*a) - y*std::sin(i*a),  x*std::sin(i*a) + y*std::cos(i*a),  _height / 2 };	// ... on the top circle counterclockwise
		vertices[q + i + 1] = CVector3{ x*std::cos(i*a) + y*std::sin(i*a), -x*std::sin(i*a) + y*std::cos(i*a), -_height / 2 };	// ... on the bottom circle clockwise
	}

	// generate triangles
	std::vector<std::vector<size_t>> triangles(4 * q, { 0, 0, 0 });
	for (size_t i = 0; i < q; ++i) // q triangles for each circle: 2 points on the circumference + 1 point in the center
	{
		triangles[i]         = { 0,			i + 1,		(i + 1) % q + 1     };	// for the top circle
		triangles[3 * q + i] = { 2 * q + 1, q + i + 1,	q + (i + 1) % q + 1 };	// for the bottom circle
	}
	for (size_t i = 0; i < q; ++i) // 2q triangles for the side surface
	{
		triangles[q + i]     = { i + 1, 2 * q - i,			 (i + 1) % q + 1 };	// \/-directed triangles
		triangles[2 * q + i] = { i + 1, (q - i) % q + q + 1, 2 * q - i       };	// /\-directed triangles
	}

	// assemble mesh
	return AssembleMesh(vertices, triangles, "Cylinder");
}

CTriangularMesh CMeshGenerator::Sphere(double _radius, size_t _accuracy)
{
	const size_t subdivs = _accuracy != 0 ? _accuracy : 4; // number of successive subdivisions of the sphere surface

	// generate vertices of initial icosahedron
	std::vector<CVector3> vertices(12);
	vertices[0]  = CVector3{ 0., 0.,  _radius };	// north pole
	vertices[11] = CVector3{ 0., 0., -_radius };	// south pole
	const double latitude = std::atan(0.5);			// latitude of the rest 10 vertices, 5 with '+', 5 with '-'
	const double offs36 = 36.*PI / 180.;			// offset in 36 degrees
	const double offs72 = 72.*PI / 180.;			// offset in 72 degrees
	for (size_t i = 0; i < 5; ++i)
	{
		vertices[i + 1] = CVector3{ _radius * std::cos(i * offs72) * std::cos(latitude),
									_radius * std::sin(i * offs72) * std::cos(latitude),
									_radius * std::sin(latitude) };		// top 5 points
		vertices[i + 6] = CVector3{ _radius * std::cos(offs36 + i * offs72) * std::cos(-latitude),
									_radius * std::sin(offs36 + i * offs72) * std::cos(-latitude),
									_radius * std::sin(-latitude) };	// bottom 5 points
	}

	// generate triangles
	std::vector<std::vector<size_t>> triangles
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
			newTriangles.push_back({ t[0], mid[0], mid[2] });
			newTriangles.push_back({ t[1], mid[1], mid[0] });
			newTriangles.push_back({ t[2], mid[2], mid[1] });
			newTriangles.push_back({ mid[0], mid[1], mid[2] });
		}
		triangles = newTriangles;
	}

	// assemble mesh
	return AssembleMesh(vertices, triangles, "Sphere");
}

CTriangularMesh CMeshGenerator::HollowSphere(double _outerRadius, double _innerRadius, size_t _accuracy)
{
	// generate outer mesh
	const CTriangularMesh outer = Sphere(_outerRadius, _accuracy);
	// generate inner mesh
	const CTriangularMesh inner = Sphere(_innerRadius, _accuracy).CreateInvertedMesh();
	// combine meshes
	std::vector<CTriangle> outerTriangles = outer.Triangles();
	std::vector<CTriangle> innerTriangles = inner.Triangles();
	std::vector<CTriangle> triangles;
	triangles.insert(triangles.begin(), outerTriangles.begin(), outerTriangles.end());
	triangles.insert(triangles.begin(), innerTriangles.begin(), innerTriangles.end());
	return CTriangularMesh{ "Hollow sphere", triangles };
}

size_t CMeshGenerator::TrianglesToAccuracy(EVolumeShape _shape, size_t _trianglesNumber)
{
	switch (_shape)
	{
	case EVolumeShape::VOLUME_SPHERE:
		return static_cast<size_t>(std::log(_trianglesNumber / 20) / std::log(4));
	case EVolumeShape::VOLUME_BOX:
		return 0;
	case EVolumeShape::VOLUME_CYLINDER:
		return _trianglesNumber / 4;
	case EVolumeShape::VOLUME_HOLLOW_SPHERE:
		return static_cast<size_t>(std::log(_trianglesNumber / 40) / std::log(4));
	case EVolumeShape::VOLUME_STL:
		return 0;
	}

	return 0;
}

size_t CMeshGenerator::MinAccuracy(EVolumeShape _shape)
{
	switch (_shape)
	{
	case EVolumeShape::VOLUME_SPHERE:			return 1;
	case EVolumeShape::VOLUME_BOX:				return 0;
	case EVolumeShape::VOLUME_CYLINDER:			return 4;
	case EVolumeShape::VOLUME_HOLLOW_SPHERE:	return 1;
	case EVolumeShape::VOLUME_STL:				return 0;
	}

	return 0;
}

size_t CMeshGenerator::MaxAccuracy(EVolumeShape _shape)
{
	switch (_shape)
	{
	case EVolumeShape::VOLUME_SPHERE:			return 6;
	case EVolumeShape::VOLUME_BOX:				return 0;
	case EVolumeShape::VOLUME_CYLINDER:			return 512;
	case EVolumeShape::VOLUME_HOLLOW_SPHERE:	return 6;
	case EVolumeShape::VOLUME_STL:				return 0;
	}

	return 0;
}

size_t CMeshGenerator::AccuracyToTriangles(EVolumeShape _shape, size_t _accuracy)
{
	switch (_shape)
	{
	case EVolumeShape::VOLUME_SPHERE:			return static_cast<size_t>(std::round(20.0 * std::exp(static_cast<double>(_accuracy) * std::log(4))));
	case EVolumeShape::VOLUME_BOX:				return 12;
	case EVolumeShape::VOLUME_CYLINDER:			return 4 * _accuracy;
	case EVolumeShape::VOLUME_HOLLOW_SPHERE:	return static_cast<size_t>(std::round(40.0 * std::exp(static_cast<double>(_accuracy) * std::log(4))));
	case EVolumeShape::VOLUME_STL:				return 0;
	}

	return 0;
}

std::vector<size_t> CMeshGenerator::AllowedTrianglesNumber(EVolumeShape _shape)
{
	if (_shape == EVolumeShape::VOLUME_STL) return {};
	std::vector<size_t> res;
	for (size_t accuracy = MinAccuracy(_shape); accuracy <= MaxAccuracy(_shape); ++accuracy)
		res.push_back(AccuracyToTriangles(_shape, accuracy));
	return res;
}

CTriangularMesh CMeshGenerator::AssembleMesh(const std::vector<CVector3>& _vertices, const std::vector<std::vector<size_t>>& _indices, const std::string& _name)
{
	std::vector<CTriangle> triangles(_indices.size());
	for (size_t i = 0; i < _indices.size(); ++i)
		triangles[i] = { _vertices[_indices[i][0]], _vertices[_indices[i][1]], _vertices[_indices[i][2]] };
	return CTriangularMesh{ _name, triangles };
}
