/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "Quaternion.h"
#include "Vector3.h"

class ProtoMatrix;
class ProtoQuaternion;
class ProtoVector;
class ProtoTimePoint;

/* A wrapper for TimePoint to read from it.
 * It is used to get time-dependent data of objects at specific time.
 * Contains functions for getting values of specified time-dependent properties using linear interpolation. */
class CTimePointR
{
	friend class CDemStorage;
	double timeRequest{ -1 };					// Requested time point.
	double timeL{ -1 };							// Time point of current left block.
	double timeR{ -1 };							// Time point of current right block.
	int count{ 0 };								// Objects number currently available in the time point.
	ProtoTimePoint* protoTimePointL{ nullptr };	// Pointer to a protocol buffer time point.
	ProtoTimePoint* protoTimePointR{ nullptr };	// Pointer to a protocol buffer time point.

public:
	CVector3 GetCoord(int _objectID) const;
	CVector3 GetAngles(int _objectID) const;
	CVector3 GetVel(int _objectID) const;
	CVector3 GetAngleVel(int _objectID) const;
	CVector3 GetAngleAccl(int _objectID) const;
	CVector3 GetForce(int _objectID) const;
	double GetTemperature(int _objectID) const;
	double GetTotalTorque(int _objectID) const;
	CQuaternion GetOrientation(int _objectID) const;
	CMatrix3 GetStressTensor(int _objectID) const;

	bool IsQuaternionSet(int _objectID) const;

private:
	double PairToVal(const double& _valL, const double& _valR) const;
	CVector3 PairToVal(const ProtoVector& _valL, const ProtoVector& _valR) const;
	CQuaternion PairToVal(const ProtoQuaternion& _valL, const ProtoQuaternion& _valR) const;
	CMatrix3 PairToVal(const ProtoMatrix& _valL, const ProtoMatrix& _valR) const;

	// Performs linear interpolation between two values.
	double Interpolate(double _valL, double _valR) const;
};

