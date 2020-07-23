/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "GeometricFunctions.h"
#include "TriangularMesh.h"

class CAnalysisVolume : public CTriangularMesh
{
	/// Time dependent velocity.
	struct SVelInterval
	{
		double dTime{};
		CVector3 vVel;
	};

public:
	EVolumeType nVolumeType;	// Type of volume: sphere, box, etc.
	std::string sKey;			// Unique key.
	std::vector<double> vProps; // Geometry specific parameters.
	CMatrix3 mRotation;			// Rotation matrix.
	CColor color;				// User defined color of the geometry.

	std::vector<SVelInterval> m_vIntervals;	// Time or force dependent velocities of planes.

public:
	CVector3 GetCenter(double _time) const;
	void SetCenter(const CVector3& _center);

	double Volume() const override;					// Returns volume of the figure.
	SVolumeType BoundingBox(double _time) const;	// Return bounding box of the object.
	double MaxInscribedDiameter() const;			// Returns maximal diameter of sphere which can be placed in this volume.

	void Scale(double _factor);							// Scales volume by the given factor.
	void Rotate(const CMatrix3& _rotation) override;	// Rotates volume by specified rotation matrix.

	void AddTimePoint();
	CVector3 GetShift(double _time) const;	// Returns translational shift.

	static std::vector<EVolumeType> AllVolumeTypes()
	{
		return { EVolumeType::VOLUME_BOX, EVolumeType::VOLUME_CYLINDER, EVolumeType::VOLUME_HOLLOW_SPHERE, EVolumeType::VOLUME_SPHERE, EVolumeType::VOLUME_STL };
	}
	static std::vector<std::string> AllVolumeTypesNames()
	{
		return { "Box",                   "Cylinder",                   "Hollow sphere",                   "Sphere",                   "STL" };
	}
};

