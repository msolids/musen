/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BasicTypes.h"
#include "Triangle.h"

class CTriangularMesh
{
protected:
	std::string m_name;					// Name of the geometry.
	std::vector<CTriangle> m_triangles;	// Three vertices per triangle.

public:
	CTriangularMesh() = default;
	CTriangularMesh(std::string _name, std::vector<CTriangle> _triangles);
	CTriangularMesh(const CTriangularMesh& _other)		          = default;
	CTriangularMesh(CTriangularMesh&& _other) noexcept	          = default;
	CTriangularMesh& operator=(const CTriangularMesh& _other)     = default;
	CTriangularMesh& operator=(CTriangularMesh&& _other) noexcept = default;
	~CTriangularMesh()											  = default;

	std::string Name() const;										// Returns the name of the mesh.
	void SetName(const std::string& _name);							// Sets new name of the mesh.
	std::vector<CTriangle> Triangles() const;						// Returns all triangles defined in the mesh.
	void SetTriangles(const std::vector<CTriangle>& _triangles);	// Sets triangles to the mesh.
	void AddTriangle(const CTriangle& _triangle);					// Adds new triangle to the mesh.
	size_t TrianglesNumber() const;									// Returns the number of defined triangles.
	bool IsEmpty() const;											// Returns true if the mesh contains no triangles.

	void SetCenter(const CVector3& _center);				// Sets new center of the geometry.
	void Shift(const CVector3& _offs);						// Shifts all points of the mesh to the given offset.
	CTriangularMesh Shifted(const CVector3& _offs) const;	// Returns a copy of the mesh shifted to the given offset.
	void Scale(double _factor);								// Scales the size of the geometry.
	void Scale(const CVector3& _factors);					// Scales the size of the geometry different in each dimension.
	void Rotate(const CMatrix3& _rotation);					// Rotates mesh according to the given rotation matrix.
	void InvertFaceNormals();								// Inverts the face normals of each triangle by reordering the vertices.

	CVector3 Center() const;			// Returns center of the mesh.
	double Volume() const;				// Returns volume of the object.
	SVolumeType BoundingBox() const;	// Return bounding box of the object.

	CTriangularMesh CreateInvertedMesh() const;	// Creates a mesh with inverted face normals.
	bool IsFaceNormalsConsistent() const;		// Checks if adjacent face have normals in opposite direction.

	friend std::ostream& operator<<(std::ostream& _s, const CTriangularMesh& _obj);
	friend std::istream& operator>>(std::istream& _s, CTriangularMesh& _obj);
};

