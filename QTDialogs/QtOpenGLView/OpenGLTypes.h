/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <QVector3D>

enum class ERenderType : unsigned
{
	GLU = 0,
	MIXED = 1,
	SHADER = 2,
	NONE = 31
};

enum class EColoring : unsigned
{
	NONE,
	AGGL_SIZE,
	BOND_TOTAL_FORCE, // can be negative
	CONTACT_DIAMETER,
	COORDINATE,
	COORD_NUMBER,
	DIAMETER,
	FORCE,
	MATERIAL,
	OVERLAP,
	ANGLE_VELOCITY,
	STRESS,
	VELOCITY,
	BOND_STRAIN, // positive - compression, negative - tension
	BOND_NORMAL_STRESS,
	TEMPERATURE,
	DISPLACEMENT
};

enum class EColorComponent : unsigned
{
	TOTAL = 0,
	X = 1,
	Y = 2,
	Z = 3
};

enum class ESlicePlane : unsigned
{
	NONE = 0,
	X = 1,
	Y = 2,
	Z = 3,
};

struct SBox
{
	QVector3D minCoord{};
	QVector3D maxCoord{};
};

/* Description of the viewport. */
struct SViewport
{
	float fovy{ 15.0f };	// Field of view angle, in degrees, in the Y direction.
	float aspect{ 1.0f };	// The aspect ratio (width/height), determines the field of view in the X direction.
	float zNear{ 0.002f };  // The distance from the viewer to the near clipping plane.
	float zFar{ 200.0f };   // The distance from the viewer to the far clipping plane.
};

/* All settings needed to describe camera. */
struct SCameraSettings
{
	SViewport viewport;
	QVector3D translation;
	QVector3D rotation;
};
