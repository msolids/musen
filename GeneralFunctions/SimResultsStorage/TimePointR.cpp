/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "TimePointR.h"

CVector3 CTimePointR::GetCoord(int _objectID) const
{
	return PairToVal(protoTimePointL->particles(_objectID).coord(), protoTimePointR->particles(_objectID).coord());
}

CVector3 CTimePointR::GetAngles(int _objectID) const
{
	return PairToVal(protoTimePointL->particles(_objectID).angles(), protoTimePointR->particles(_objectID).angles());
}

CVector3 CTimePointR::GetVel(int _objectID) const
{
	return PairToVal(protoTimePointL->particles(_objectID).vel(), protoTimePointR->particles(_objectID).vel());
}

CVector3 CTimePointR::GetAngleVel(int _objectID) const
{
	return PairToVal(protoTimePointL->particles(_objectID).angle_vel(), protoTimePointR->particles(_objectID).angle_vel());
}

CVector3 CTimePointR::GetAngleAccl(int _objectID) const
{
	return PairToVal(protoTimePointL->particles(_objectID).angle_accl(), protoTimePointR->particles(_objectID).angle_accl());
}

CVector3 CTimePointR::GetForce(int _objectID) const
{
	return PairToVal(protoTimePointL->particles(_objectID).force(), protoTimePointR->particles(_objectID).force());
}

double CTimePointR::GetTemperature(int _objectID) const
{
	return PairToVal(protoTimePointL->particles(_objectID).temperature(), protoTimePointR->particles(_objectID).temperature());
}

double CTimePointR::GetTotalTorque(int _objectID) const
{
	return PairToVal(protoTimePointL->particles(_objectID).total_torque(), protoTimePointR->particles(_objectID).total_torque());
}

CQuaternion CTimePointR::GetOrientation(int _objectID) const
{
	return PairToVal(protoTimePointL->particles(_objectID).quaternion(), protoTimePointR->particles(_objectID).quaternion());
}

CMatrix3 CTimePointR::GetStressTensor(int _objectID) const
{
	return PairToVal(protoTimePointL->particles(_objectID).stress_tensor(), protoTimePointR->particles(_objectID).stress_tensor());
}

bool CTimePointR::IsQuaternionSet(int _objectID) const
{
	return protoTimePointL->particles(_objectID).has_quaternion() && protoTimePointR->particles(_objectID).has_quaternion();
}

double CTimePointR::PairToVal(const double& _valL, const double& _valR) const
{
	if (protoTimePointL->time() == timeRequest)
		return _valL;
	return Interpolate(_valL, _valR);
}

CVector3 CTimePointR::PairToVal(const ProtoVector& _valL, const ProtoVector& _valR) const
{
	if (protoTimePointL->time() == timeRequest)
		return CVector3{ _valL.x(), _valL.y(), _valL.z() };
	return CVector3{
		Interpolate(_valL.x(), _valR.x()),
		Interpolate(_valL.y(), _valR.y()),
		Interpolate(_valL.z(), _valR.z()) };
}

CQuaternion CTimePointR::PairToVal(const ProtoQuaternion& _valL, const ProtoQuaternion& _valR) const
{
	if (protoTimePointL->time() == timeRequest)
		return CQuaternion{ _valL.q0(), _valL.q1(), _valL.q2(), _valL.q3() };
	return CQuaternion{
		Interpolate(_valL.q0(), _valR.q0()),
		Interpolate(_valL.q1(), _valR.q1()),
		Interpolate(_valL.q2(), _valR.q2()),
		Interpolate(_valL.q3(), _valR.q3()) };
}

CMatrix3 CTimePointR::PairToVal(const ProtoMatrix& _valL, const ProtoMatrix& _valR) const
{
	if (protoTimePointL->time() == timeRequest)
		return CMatrix3{
			_valL.v1().x(), _valL.v1().y(), _valL.v1().z(),
			_valL.v2().x(), _valL.v2().y(), _valL.v2().z(),
			_valL.v3().x(), _valL.v3().y(), _valL.v3().z() };
	return CMatrix3{
		Interpolate(_valL.v1().x(), _valR.v1().x()),
		Interpolate(_valL.v1().y(), _valR.v1().y()),
		Interpolate(_valL.v1().z(), _valR.v1().z()),
		Interpolate(_valL.v2().x(), _valR.v2().x()),
		Interpolate(_valL.v2().y(), _valR.v2().y()),
		Interpolate(_valL.v2().z(), _valR.v2().z()),
		Interpolate(_valL.v3().x(), _valR.v3().x()),
		Interpolate(_valL.v3().y(), _valR.v3().y()),
		Interpolate(_valL.v3().z(), _valR.v3().z()) };
}

double CTimePointR::Interpolate(double _valL, double _valR) const
{
	return _valL + (_valR - _valL) / (protoTimePointR->time() - protoTimePointL->time()) * (timeRequest - protoTimePointL->time());
}
