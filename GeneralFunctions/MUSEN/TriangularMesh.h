/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "BasicTypes.h"

enum class EVolumeType : unsigned
{
	VOLUME_SPHERE        = 0,
	VOLUME_BOX           = 1,
	VOLUME_CYLINDER      = 2,
	VOLUME_HOLLOW_SPHERE = 3,
	VOLUME_STL           = 5
};

class CTriangularMesh
{
public:
	static constexpr double tol = 1e-8;		// Tolerance for checking for same vertices/points.

	std::string sName;						// Name of the geometry.
	std::vector<STriangleType> vTriangles;	// Three vertices per triangle.

	CTriangularMesh() = default;
	CTriangularMesh(std::string _name, std::vector<STriangleType> _triangles);
	virtual ~CTriangularMesh() = default;

	void Move(const CVector3& _offs);				// Moves all points of the mesh to the given offset.
	virtual void Rotate(const CMatrix3& _rotation);	// Rotates mesh according to the given rotation matrix.
	void InvertFaceNormals();						// Inverts the face normals of each triangle by reordering the vertices.

	CVector3 Center() const;					// Returns center of the mesh.
	virtual double Volume() const;				// Returns volume of the object.
	SVolumeType BoundingBox() const;			// Return bounding box of the object.

	CTriangularMesh CreateInvertedMesh() const;	// Creates a mesh with inverted face normals.
	bool IsFaceNormalsConsistent() const;		// Checks if adjacent face have normals in opposite direction.

	// Returns triangulates mesh of the given type with specified parameters.
	static CTriangularMesh GenerateMesh(const EVolumeType& _type, const std::vector<double>& _params, const CVector3& _center, const CMatrix3& _rotation, size_t _accuracy = 0);
	// Generates cuboid with specified parameters, triangulates it and returns its mesh.
	static CTriangularMesh GenerateBoxMesh(double _length, double _width, double _height);
	// Generates cylinder with specified parameters, triangulates it and returns its mesh.
	static CTriangularMesh GenerateCylinderMesh(double _radius, double _height, size_t _accuracy = 0);
	// Generates sphere with specified parameters, triangulates it and returns its mesh.
	static CTriangularMesh GenerateSphereMesh(double _radius, size_t _accuracy = 0);
	// Generates hollow sphere with specified parameters, triangulates it and returns its mesh.
	static CTriangularMesh GenerateHollowSphereMesh(double _outerRadius, double _innerRadius, size_t _accuracy = 0);

private:
	// Assembles mesh from the list of points and the list of indices.
	static CTriangularMesh AssembleMesh(const std::vector<CVector3>& _vertices, const std::vector<std::vector<size_t>>& _triangles, const std::string& _name = "");
};

