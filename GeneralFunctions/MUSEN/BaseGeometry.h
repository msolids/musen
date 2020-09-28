/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BasicTypes.h"
#include "MixedFunctions.h"
#include "GeometryMotion.h"
#include "GeometrySizes.h"

class ProtoBaseGeometry;

// TODO: make pure virtual
class CBaseGeometry
{
	// TODO: check that name can be with spaces
	std::string m_name{ "Unspecified" };				// Name of the geometry.
	std::string m_key;									// Unique key of the geometry.
	EVolumeShape m_shape{ EVolumeShape::VOLUME_STL };	// Shape of the geometry: sphere, box, etc.
	CColor m_color;										// Color of the geometry.
	CMatrix3 m_rotation{ CMatrix3::Diagonal() };		// Rotation matrix.
	CGeometryMotion m_motion;							// Description of the motion of the geometry.
	double m_scaling{ 1.0 };							// Current scaling factor applied to the geometry.

protected:
	// TODO: make private
	CGeometrySizes m_sizes;	// Geometry specific size parameters.

public:
	// TODO: rename store
	std::string Name() const;							// Returns name of the geometry.
	void SetName(const std::string& _name);				// Sets name of the geometry.
	std::string Key() const;							// Returns unique key of the geometry.
	void SetKey(const std::string& _key);				// Sets unique key of the geometry.
	EVolumeShape Shape() const;							// Returns shape type of the geometry.
	void StoreShape(const EVolumeShape& _shape);		// Sets shape type of the geometry.
	CColor Color() const;								// Returns color of the geometry.
	void SetColor(const CColor& _color);				// Sets color of the geometry.
	CMatrix3 RotationMatrix() const;					// Returns current rotation matrix.
	// TODO: maybe remove
	void StoreRotationMatrix(const CMatrix3& _matrix);	// Sets new rotation matrix without application.

	virtual SVolumeType BoundingBox(double _time = 0.0) const = 0;	// Returns bounding box of the geometry.
	virtual size_t TrianglesNumber() const = 0;						// Returns the number of triangles in the geometry.

	virtual CVector3 Center(double _time = 0.0) const = 0;	// Returns center of the geometry.
	virtual void SetCenter(const CVector3& _center) = 0;	// Sets new center of the geometry.

	size_t Accuracy() const;								// Returns current accuracy of a non-STL shape.
	virtual void SetAccuracy(size_t _value) = 0;			// Sets new accuracy of a non-STL shape.

	virtual void SetSizes(const CGeometrySizes& _sizes);	// Sets new geometry specific size parameters.
	CGeometrySizes Sizes() const;							// Returns current geometry specific size parameters.

	virtual void Shift(const CVector3& _offset) = 0;		// Shifts the geometry by the specified coordinates.
	virtual void Rotate(const CMatrix3& _rotation);			// Rotates the geometry according to the given rotational matrix.

	double ScalingFactor() const;							// Returns current scaling factor.
	virtual void Scale(double _factor);						// Scales sizes of the geometry by the given factor.
	virtual void ScaleSTL(const CVector3& _factors);		// Scales sizes of the STL geometry by the given factors different in each dimension.

	CGeometryMotion* Motion();				// Returns pointer to a geometry motion descriptor.
	const CGeometryMotion* Motion() const;	// Returns pointer to a geometry motion descriptor.

protected:
	void SaveToProto(ProtoBaseGeometry& _proto) const;		// Save to protobuf message.
	void LoadFromProto(const ProtoBaseGeometry& _proto);	// Load from protobuf message.
};

