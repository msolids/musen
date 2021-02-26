/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "Matrix3.h"

/// A triangle described by three vertices.
class CTriangle
{
public:
	CVector3 p1, p2, p3;

	CTriangle() = default;
	explicit CTriangle(double _val);
	CTriangle(const CVector3& _v1, const CVector3& _v2, const CVector3& _v3);

	void Shift(const CVector3& _offs);
	CTriangle Shifted(const CVector3& _offs) const;

	// Scales the triangle by the given factor.
	void Scale(const CVector3& _center, double _factor);
	// Scales the triangle by the given factors in each spatial direction.
	void Scale(const CVector3& _center, const CVector3& _factors);
	// Returns a triangle scaled by the given factor.
	CTriangle Scaled(const CVector3& _center, double _factor) const;
	// Returns a triangle scaled by the given factors in each spatial direction.
	CTriangle Scaled(const CVector3& _center, const CVector3& _factors) const;

	void Rotate(const CVector3& _center, const CMatrix3& _rot);
	CTriangle Rotated(const CVector3& _center, const CMatrix3& _rot) const;

	void InvertOrientation();

	CVector3 Normal() const;

	// Returns number of points.
	size_t Size() const;

	CVector3 operator[](size_t _i) const;
	CVector3& operator[](size_t _i);


	friend std::ostream& operator<<(std::ostream& _s, const CTriangle& _obj);
	friend std::istream& operator>>(std::istream& _s, CTriangle& _obj);
};

