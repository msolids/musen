/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "Matrix3.h"
#include <vector>

class CTriangularMesh;
class CGeometrySizes;
enum class EVolumeShape : unsigned;

class CMeshGenerator
{
public:
	// Returns a triangulated mesh of the given type with the specified parameters.
	static CTriangularMesh GenerateMesh(EVolumeShape _shape, const CGeometrySizes& _sizes, const CVector3& _center, const CMatrix3& _rotation, size_t _accuracy = 0);

	// Returns a mesh of a cuboid with the specified parameters centered at point (0,0,0).
	static CTriangularMesh Box(double _width, double _depth, double _height);
	// Returns a mesh of a cylinder with the specified parameters centered at point (0,0,0).
	static CTriangularMesh Cylinder(double _radius, double _height, size_t _accuracy = 0);
	// Returns a mesh of a sphere with the specified parameters centered at point (0,0,0).
	static CTriangularMesh Sphere(double _radius, size_t _accuracy = 0);
	// Returns a mesh of a hollow sphere with the specified parameters centered at point (0,0,0).
	static CTriangularMesh HollowSphere(double _outerRadius, double _innerRadius, size_t _accuracy = 0);

	// Calculates accuracy from triangles number for the given shape.
	static size_t TrianglesToAccuracy(EVolumeShape _shape, size_t _trianglesNumber);
	// Returns minimum allowed accuracy for the given shape.
	static size_t MinAccuracy(EVolumeShape _shape);
	// Returns maximum allowed accuracy for the given shape.
	static size_t MaxAccuracy(EVolumeShape _shape);
	// Returns number of triangles that corresponds to the selected accuracy of the given shape.
	static size_t AccuracyToTriangles(EVolumeShape _shape, size_t _accuracy);
	// Returns the list of triangles number, which are allowed for the given shape.
	static std::vector<size_t> AllowedTrianglesNumber(EVolumeShape _shape);

private:
	// Assembles a mesh from the list of points and the list of indices.
	static CTriangularMesh AssembleMesh(const std::vector<CVector3>& _vertices, const std::vector<std::vector<size_t>>& _indices, const std::string& _name = "");
};

