/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BaseGeometry.h"

class CSystemStructure;
class CTriangularWall;
class CTriangularMesh;
class ProtoRealGeometry;
class ProtoRealGeometry_v0;

class CRealGeometry : public CBaseGeometry
{
	std::vector<size_t> m_planes;				// Indexes of triangular planes which belong to this geometry.
	CBasicVector3<bool> m_freeMotion{ false };	// Directions in which free motion is allowed.
	double m_mass{ 0.0 };						// Mass of the geometry if free motion is enabled.
	bool m_rotateAroundCenter{ false };			// Whether the rotation is around center.

	CSystemStructure* m_systemStructure{ nullptr };	// Pointer to a system structure.

public:
	CRealGeometry(CSystemStructure* _systemStructure);

	size_t TrianglesNumber() const override;					// Returns the number of triangles in the geometry.
	std::vector<size_t> Planes() const;							// Returns indexes of triangular planes which belong to this geometry.
	CBasicVector3<bool> FreeMotion() const;						// Returns directions in which free motion is allowed.
	double Mass() const;										// Returns mass of the geometry.
	bool RotateAroundCenter() const;							// Returns whether the rotation is performed around center.
	CVector3 Center(double _time = 0.0) const override;			// Returns center of the geometry.
	std::string Material() const;								// Returns material of the geometry.
	SVolumeType BoundingBox(double _time = 0.0) const override;	// Returns bounding box of the geometry at the given time point.

	void SetMesh(const CTriangularMesh& _mesh) override;	// Creates a new geometry form the mesh.
	// TODO: remove
	void SetPlanesIndices(const std::vector<size_t>& _planes);	// Sets indexes of triangular planes which belong to this geometry.
	void SetFreeMotion(const CBasicVector3<bool>& _flags);		// Sets directions in which free motion is allowed.
	void SetMass(double _mass);									// Sets mass of the geometry.
	void SetRotateAroundCenter(bool _flag);						// Sets whether the rotation is around center.
	void SetAccuracy(size_t _value) override;					// Sets new accuracy of a non-STL shape.
	void Shift(const CVector3& _offset) override; 				// Shifts the geometry by the specified coordinates at time point 0.
	void SetCenter(const CVector3& _coord) override; 			// Moves geometry to a point with specified coordinates at time point 0.
	void SetMaterial(const std::string& _compoundKey);			// Sets material for all related planes of the geometry.
	void Scale(double _factor) override;						// Scales sizes of the geometry by the given factor at time point 0.
	void DeformSTL(const CVector3& _factors) override;			// Scales sizes of the STL geometry by the given factors different in each dimension at time point 0.
	void Rotate(const CMatrix3& _rotation) override;			// Rotates the geometry according to the given rotational matrix at time point 0.

	void UpdateMotionInfo(double _dependentValue);	// Updates current motion characteristics according to the current time or force.
	CVector3 GetCurrentVelocity() const;			// Returns current translational velocity.
	CVector3 GetCurrentRotVelocity() const;			// Returns current rotational velocity.
	CVector3 GetCurrentRotCenter() const;			// Returns current center of rotation.

	// TODO: import/export text as internal methods
	void SaveToProto(ProtoRealGeometry& _proto) const;			// Save to protobuf message.
	void LoadFromProto(const ProtoRealGeometry& _proto);		// Load from protobuf message.
	void LoadFromProto_v0(const ProtoRealGeometry_v0& _proto);	// Load from protobuf message. Compatibility version.

	std::vector<CTriangularWall*> Walls();				// Returns a list of valid triangular walls belonging to this geometry.
	std::vector<const CTriangularWall*> Walls() const;	// Returns a list of valid triangular walls belonging to this geometry.
};
