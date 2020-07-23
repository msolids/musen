/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "GeneratedFiles/SimulationDescription.pb.h"
#include "Quaternion.h"
#include "Vector3.h"

/* A wrapper for MutableTimePoint to write into it.
 * It is used to set time-dependent data of objects at specific time.
 * Contains functions for setting values of specified time-dependent properties. */
class CTimePointW
{
	friend class CDemStorage;
	double time{ -1 };							// Current time of the time point.
	int count{ 0 };								// Objects number currently available in the time point.
	ProtoTimePoint* protoTimePoint{ nullptr };	// Pointer to a protocol buffer time point.

public:
	void SetCoord(int _objectID, const CVector3& _coord) const;
	void SetVel(int _objectID, const CVector3& _vel) const;
	void SetAngleVel(int _objectID, const CVector3& _angleVel) const;
	void SetForce(int _objectID, const CVector3& _force) const;
	void SetTemperature(int _objectID, const double& _temperature) const;
	void SetTotalTorque(int _objectID, const double& _totalTorque) const;
	void SetOrientation(int _objectID, const CQuaternion& _orientation) const;
	void SetStressTensor(int _objectID, const CMatrix3& _stressTensor) const;
	void ClearAllTDData(int _objectID) const;
};

