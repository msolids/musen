/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BasicTypes.h"

// Holds the dimensions of geometric objects.
class CGeometrySizes
{
	double m_width      { 0.0 };
	double m_depth      { 0.0 };
	double m_height     { 0.0 };
	double m_radius     { 0.0 };
	double m_innerRadius{ 0.0 };

	std::vector<double*> m_sizes{ &m_width, &m_depth, &m_height, &m_radius, &m_innerRadius }; // Pointers to all sizes to simplify massive operations.

public:
	CGeometrySizes() = default;
	~CGeometrySizes() = default;
	CGeometrySizes(const CGeometrySizes& _other);
	CGeometrySizes(CGeometrySizes&& _other) noexcept;
	CGeometrySizes& operator=(const CGeometrySizes& _other);
	CGeometrySizes& operator=(CGeometrySizes&& _other) noexcept;

	double Width() const { return m_width; }							// Returns width of the geometry.
	void SetWidth(double _width) { m_width = _width; }					// Sets width of the geometry.
	double Depth() const { return m_depth; }							// Returns depth of the geometry.
	void SetDepth(double _depth) { m_depth = _depth; }					// Sets depth of the geometry.
	double Height() const { return m_height; }							// Returns height of the geometry.
	void SetHeight(double _height) { m_height = _height; }				// Sets height of the geometry.
	double Radius() const { return m_radius; }							// Returns radius of the geometry.
	void SetRadius(double _radius) { m_radius = _radius; }				// Sets radius of the geometry.
	double InnerRadius() const { return m_innerRadius; }				// Returns inner radius of the geometry.
	void SetInnerRadius(double _radius) { m_innerRadius = _radius; }	// Sets inner radius of the geometry.

	void Scale(double _factor);						// Scales all sizes by the given factor.

	void ResetToDefaults(double _reference = 0.0);				// Calculates and sets default values of all sizes using reference size as a base for scaling.
	static CGeometrySizes Defaults(double _reference = 0.0);	// Returns default values of all sizes using reference size as a base for scaling.

	std::vector<double> RelevantSizes(EVolumeShape _shape) const;					// Returns a vector of sizes required for the specified shape.
	void SetRelevantSizes(const std::vector<double>& _sizes, EVolumeShape _shape);	// Sets sizes required for the specified shape from the vector.

	friend bool operator==(const CGeometrySizes& _lhs, const CGeometrySizes& _rhs);
	friend bool operator!=(const CGeometrySizes& _lhs, const CGeometrySizes& _rhs);

	friend std::ostream& operator<<(std::ostream& _s, const CGeometrySizes& _obj);
	friend std::istream& operator>>(std::istream& _s, CGeometrySizes& _obj);
};

