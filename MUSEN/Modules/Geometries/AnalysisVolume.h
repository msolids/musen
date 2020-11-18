/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BaseGeometry.h"
#include "TriangularMesh.h"

class ProtoAnalysisVolume;
class ProtoAnalysisVolume_v0;
class CSystemStructure;
class CSphere;
class CSolidBond;
class CLiquidBond;
class CBond;
class CTriangularWall;
class CGeometrySizes;

class CAnalysisVolume : public CBaseGeometry
{
	CTriangularMesh m_mesh;	// Triangular mesh.

	const CSystemStructure* m_systemStructure; // Pointer to a system structure.

public:
	CAnalysisVolume(const CSystemStructure* _systemStructure);

	size_t TrianglesNumber() const override;					// Returns the number of triangles in the volume.
	CTriangularMesh Mesh(double _time = 0.0) const;				// Returns triangular mesh at the specific time point.
	CVector3 Center(double _time = 0.0) const override;			// Returns center of the volume.
	double Volume() const;										// Returns volume of the figure.
	SVolumeType BoundingBox(double _time = 0.0) const override;	// Returns bounding box of the volume.
	double MaxInscribedDiameter() const;						// Returns maximum diameter of a sphere that can be placed in this volume. For STL, is calculated via volume.

	void SetMesh(const CTriangularMesh& _mesh) override;	// Sets new mesh.
	void SetCenter(const CVector3& _center) override;		// Sets new center of the volume at time point 0.
	void SetAccuracy(size_t _value) override;				// Sets new accuracy of a non-STL shape.
	void Shift(const CVector3& _offset) override;			// Shifts the volume at time point 0.
	void Scale(double _factor) override;					// Scales sizes of the volume by the given factor at time point 0.
	void DeformSTL(const CVector3& _factors) override;		// Scales sizes of the STL volume by the given factors different in each dimension at time point 0.
	void Rotate(const CMatrix3& _rotation) override;		// Rotates volume by the specified rotation matrix at time point 0.

	// TODO: import/export as text as internal methods
	void SaveToProto(ProtoAnalysisVolume& _proto) const;			// Save to protobuf message.
	void LoadFromProto(const ProtoAnalysisVolume& _proto);			// Load from protobuf message.
	void LoadFromProto_v0(const ProtoAnalysisVolume_v0& _proto);	// Load from protobuf message. Compatibility version.

	// Returns indices of all particles placed in the volume.
	std::vector<size_t> GetParticleIndicesInside(double _time, bool _totallyInside = true) const;
	// Returns all particles placed in the volume.
	std::vector<const CSphere*> GetParticlesInside(double _time, bool _totallyInside = true) const;
	// Returns indices of solid bonds placed in the volume. Middle point between two aligned particles is used as bond's coordinate.
	std::vector<size_t> GetSolidBondIndicesInside(double _time) const;
	// Returns solid bonds placed in the volume. Middle point between two aligned particles is used as bond's coordinate.
	std::vector<const CSolidBond*> GetSolidBondsInside(double _time) const;
	// Returns indices of liquid bonds placed in the volume. Middle point between two aligned particles is used as bond's coordinate.
	std::vector<size_t> GetLiquidBondIndicesInside(double _time) const;
	// Returns liquid bonds placed in the volume. Middle point between two aligned particles is used as bond's coordinate.
	std::vector<const CLiquidBond*> GetLiquidBondsInside(double _time) const;
	// Returns indices of bonds placed in the volume. Middle point between two aligned particles is used as bond's coordinate.
	std::vector<size_t> GetBondIndicesInside(double _time) const;
	// Returns bonds placed in the volume. Middle point between two aligned particles is used as bond's coordinate.
	std::vector<const CBond*> GetBondsInside(double _time) const;
	// Returns indices of walls placed in the volume. A wall is in the volume only of all its edges are inside.
	std::vector<size_t> GetWallIndicesInside(double _time) const;
	// Returns walls placed in the volume. A wall is in the volume only of all its edges are inside.
	std::vector<const CTriangularWall*> GetWallsInside(double _time) const;
	// Finds all coordinates situated in the volume and returns their indexes.
	std::vector<size_t> GetObjectIndicesInside(double _time, const std::vector<CVector3>& _coords) const;
};

