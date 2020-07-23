/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
class IPackageGeneratorHelper
{
public:
	virtual ~IPackageGeneratorHelper() = default;
	virtual void LimitVelocities() const = 0;
	virtual void ScaleVelocitiesToRadius(double _minRadius) const = 0;
	virtual double MaxRelativeVelocity() const = 0;
	virtual void ResetMovement() = 0;
	virtual void SaveVelocities() = 0;
};