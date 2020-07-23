/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "TimePointW.h"

void CTimePointW::SetCoord(int _objectID, const CVector3& _coord) const
{
	protoTimePoint->mutable_particles(_objectID)->mutable_coord()->set_x(_coord.x);
	protoTimePoint->mutable_particles(_objectID)->mutable_coord()->set_y(_coord.y);
	protoTimePoint->mutable_particles(_objectID)->mutable_coord()->set_z(_coord.z);
}

void CTimePointW::SetVel(int _objectID, const CVector3& _vel) const
{
	protoTimePoint->mutable_particles(_objectID)->mutable_vel()->set_x(_vel.x);
	protoTimePoint->mutable_particles(_objectID)->mutable_vel()->set_y(_vel.y);
	protoTimePoint->mutable_particles(_objectID)->mutable_vel()->set_z(_vel.z);
}

void CTimePointW::SetAngleVel(int _objectID, const CVector3& _angleVel) const
{
	protoTimePoint->mutable_particles(_objectID)->mutable_angle_vel()->set_x(_angleVel.x);
	protoTimePoint->mutable_particles(_objectID)->mutable_angle_vel()->set_y(_angleVel.y);
	protoTimePoint->mutable_particles(_objectID)->mutable_angle_vel()->set_z(_angleVel.z);
}

void CTimePointW::SetForce(int _objectID, const CVector3& _force) const
{
	protoTimePoint->mutable_particles(_objectID)->mutable_force()->set_x(_force.x);
	protoTimePoint->mutable_particles(_objectID)->mutable_force()->set_y(_force.y);
	protoTimePoint->mutable_particles(_objectID)->mutable_force()->set_z(_force.z);
}

void CTimePointW::SetTemperature(int _objectID, const double& _temperature) const
{
	protoTimePoint->mutable_particles(_objectID)->set_temperature(_temperature);
}

void CTimePointW::SetTotalTorque(int _objectID, const double& _totalTorque) const
{
	protoTimePoint->mutable_particles(_objectID)->set_total_torque(_totalTorque);
}

void CTimePointW::SetOrientation(int _objectID, const CQuaternion& _orientation) const
{
	protoTimePoint->mutable_particles(_objectID)->mutable_quaternion()->set_q0(_orientation.q0);
	protoTimePoint->mutable_particles(_objectID)->mutable_quaternion()->set_q1(_orientation.q1);
	protoTimePoint->mutable_particles(_objectID)->mutable_quaternion()->set_q2(_orientation.q2);
	protoTimePoint->mutable_particles(_objectID)->mutable_quaternion()->set_q3(_orientation.q3);
}

void CTimePointW::SetStressTensor(int _objectID, const CMatrix3& _stressTensor) const
{
	protoTimePoint->mutable_particles(_objectID)->mutable_stress_tensor()->mutable_v1()->set_x(_stressTensor.values[0][0]);
	protoTimePoint->mutable_particles(_objectID)->mutable_stress_tensor()->mutable_v1()->set_y(_stressTensor.values[0][1]);
	protoTimePoint->mutable_particles(_objectID)->mutable_stress_tensor()->mutable_v1()->set_z(_stressTensor.values[0][2]);

	protoTimePoint->mutable_particles(_objectID)->mutable_stress_tensor()->mutable_v2()->set_x(_stressTensor.values[1][0]);
	protoTimePoint->mutable_particles(_objectID)->mutable_stress_tensor()->mutable_v2()->set_y(_stressTensor.values[1][1]);
	protoTimePoint->mutable_particles(_objectID)->mutable_stress_tensor()->mutable_v2()->set_z(_stressTensor.values[1][2]);

	protoTimePoint->mutable_particles(_objectID)->mutable_stress_tensor()->mutable_v3()->set_x(_stressTensor.values[2][0]);
	protoTimePoint->mutable_particles(_objectID)->mutable_stress_tensor()->mutable_v3()->set_y(_stressTensor.values[2][1]);
	protoTimePoint->mutable_particles(_objectID)->mutable_stress_tensor()->mutable_v3()->set_z(_stressTensor.values[2][2]);
}

void CTimePointW::ClearAllTDData(int _objectID) const
{
	protoTimePoint->mutable_particles(_objectID)->clear_coord();
	protoTimePoint->mutable_particles(_objectID)->clear_angles();
	protoTimePoint->mutable_particles(_objectID)->clear_vel();
	protoTimePoint->mutable_particles(_objectID)->clear_angle_vel();
	protoTimePoint->mutable_particles(_objectID)->clear_angle_accl();
	protoTimePoint->mutable_particles(_objectID)->clear_total_torque();
	protoTimePoint->mutable_particles(_objectID)->clear_force();
	protoTimePoint->mutable_particles(_objectID)->clear_quaternion();
}
