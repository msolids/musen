/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "PhysicalObject.h"
#include "../SimResultsStorage/DemStorage.h"

CPhysicalObject::CPhysicalObject(unsigned _id, CDemStorage* _storage) : m_lObjectID(_id), m_storage(_storage)
{
	m_storage->Object(m_lObjectID)->set_id(_id);
	m_dMass = 1; // [kg]
	m_ObjectColor = CColor{ 0, 0, 1, 1 };
}

void CPhysicalObject::Save() const
{
	const std::vector<uint8_t> data = GetObjectGeometryBin();
	m_storage->Object(m_lObjectID)->set_encoded_properties(data.data(), data.size());
}

void CPhysicalObject::Load()
{
	if (m_storage->GetFileVersion() < 2) // compatibility mode for older file versions
	{
		std::stringstream ss(m_storage->Object(m_lObjectID)->encoded_properties());
		SetObjectGeometryText(ss);
	}
	else // current version
	{
		const std::string data = m_storage->Object(m_lObjectID)->encoded_properties();
		SetObjectGeometryBin({ data.begin(), data.end() });
	}
}

void CPhysicalObject::CloneData(const CPhysicalObject& _other)
{
	m_lObjectID        = _other.m_lObjectID;
	m_dMass            = _other.m_dMass;
	m_ObjectColor      = _other.m_ObjectColor;
	m_ConnectedBondsID = _other.m_ConnectedBondsID;
}

unsigned CPhysicalObject::GetObjectType() const
{
	return m_storage->Object(m_lObjectID)->type();
}

std::string CPhysicalObject::GetCompoundKey() const
{
	return m_storage->Object(m_lObjectID)->compound();
}

void CPhysicalObject::SetCompound(const CCompound* _pCompound)
{
	m_storage->Object(m_lObjectID)->set_compound(_pCompound->GetKey());
	UpdateCompoundProperties(_pCompound);
}

void CPhysicalObject::SetCompoundKey(const std::string& _compoundKey) const
{
	m_storage->Object(m_lObjectID)->set_compound(_compoundKey);
}

CVector3 CPhysicalObject::GetCoordinates(double _time) const
{
	return m_storage->GetTimePointR(_time, static_cast<int>(m_lObjectID))->GetCoord(static_cast<int>(m_lObjectID));
}

CVector3 CPhysicalObject::GetVelocity(double _time) const
{
	return m_storage->GetTimePointR(_time, static_cast<int>(m_lObjectID))->GetVel(static_cast<int>(m_lObjectID));
}

CVector3 CPhysicalObject::GetAngles(double _time) const
{
	return m_storage->GetTimePointR(_time, static_cast<int>(m_lObjectID))->GetAngles(static_cast<int>(m_lObjectID));
}

CVector3 CPhysicalObject::GetAngleVelocity(double _time) const
{
	return m_storage->GetTimePointR(_time, static_cast<int>(m_lObjectID))->GetAngleVel(static_cast<int>(m_lObjectID));
}

CVector3 CPhysicalObject::GetAngleAcceleration(double _time) const
{
	return m_storage->GetTimePointR(_time, static_cast<int>(m_lObjectID))->GetAngleAccl(static_cast<int>(m_lObjectID));
}

CVector3 CPhysicalObject::GetForce(double _time) const
{
	return m_storage->GetTimePointR(_time, static_cast<int>(m_lObjectID))->GetForce(static_cast<int>(m_lObjectID));
}

double CPhysicalObject::GetTemperature(double _time) const
{
	return m_storage->GetTimePointR(_time, static_cast<int>(m_lObjectID))->GetTemperature(static_cast<int>(m_lObjectID));
}

double CPhysicalObject::GetTotalTorque(double _time) const
{
	return m_storage->GetTimePointR(_time, static_cast<int>(m_lObjectID))->GetTotalTorque(static_cast<int>(m_lObjectID));
}

CQuaternion CPhysicalObject::GetOrientation(double _time) const
{
	return m_storage->GetTimePointR(_time, static_cast<int>(m_lObjectID))->GetOrientation(static_cast<int>(m_lObjectID));
}

CMatrix3 CPhysicalObject::GetStressTensor(double _time) const
{
	return m_storage->GetTimePointR(_time, static_cast<int>(m_lObjectID))->GetStressTensor(static_cast<int>(m_lObjectID));
}

CVector3 CPhysicalObject::GetNormalStress(double _time) const
{
	CMatrix3 st = m_storage->GetTimePointR(_time, static_cast<int>(m_lObjectID))->GetStressTensor(static_cast<int>(m_lObjectID));
	return CVector3(st.values[0][0], st.values[1][1], st.values[2][2]);
}

CVector3 CPhysicalObject::GetCoordinates() const
{
	return m_storage->GetTimePointR()->GetCoord(static_cast<int>(m_lObjectID));
}

CVector3 CPhysicalObject::GetVelocity() const
{
	return m_storage->GetTimePointR()->GetVel(static_cast<int>(m_lObjectID));
}

CVector3 CPhysicalObject::GetAngles() const
{
	return m_storage->GetTimePointR()->GetAngles(static_cast<int>(m_lObjectID));
}

CVector3 CPhysicalObject::GetAngleVelocity() const
{
	return m_storage->GetTimePointR()->GetAngleVel(static_cast<int>(m_lObjectID));
}

CVector3 CPhysicalObject::GetAngleAcceleration() const
{
	return m_storage->GetTimePointR()->GetAngleAccl(static_cast<int>(m_lObjectID));
}

CVector3 CPhysicalObject::GetForce() const
{
	return m_storage->GetTimePointR()->GetForce(static_cast<int>(m_lObjectID));
}

double CPhysicalObject::GetTemperature() const
{
	return m_storage->GetTimePointR()->GetTemperature(static_cast<int>(m_lObjectID));
}

double CPhysicalObject::GetTotalTorque() const
{
	return m_storage->GetTimePointR()->GetTotalTorque(static_cast<int>(m_lObjectID));
}

CQuaternion CPhysicalObject::GetOrientation() const
{
	return m_storage->GetTimePointR()->GetOrientation(static_cast<int>(m_lObjectID));
}

CMatrix3 CPhysicalObject::GetStressTensor() const
{
	return m_storage->GetTimePointR()->GetStressTensor(static_cast<int>(m_lObjectID));
}

CVector3 CPhysicalObject::GetNormalStress() const
{
	CMatrix3 st = m_storage->GetTimePointR()->GetStressTensor(static_cast<int>(m_lObjectID));
	return CVector3(st.values[0][0], st.values[1][1], st.values[2][2]);
}

void CPhysicalObject::SetCoordinates(double _time, const CVector3& _coordinates) const
{
	m_storage->GetTimePointW(_time, static_cast<int>(m_lObjectID))->SetCoord(static_cast<int>(m_lObjectID), _coordinates);
}

void CPhysicalObject::SetVelocity(double _time, const CVector3& _velocity) const
{
	m_storage->GetTimePointW(_time, static_cast<int>(m_lObjectID))->SetVel(static_cast<int>(m_lObjectID), _velocity);
}

void CPhysicalObject::SetAngleVelocity(double _time, const CVector3& _angleVelocity) const
{
	m_storage->GetTimePointW(_time, static_cast<int>(m_lObjectID))->SetAngleVel(static_cast<int>(m_lObjectID), _angleVelocity);
}

void CPhysicalObject::SetForce(double _time, const CVector3& _force) const
{
	m_storage->GetTimePointW(_time, static_cast<int>(m_lObjectID))->SetForce(static_cast<int>(m_lObjectID), _force);
}

void CPhysicalObject::SetTemperature(double _time, const double& _temperature) const
{
	m_storage->GetTimePointW(_time, static_cast<int>(m_lObjectID))->SetTemperature(static_cast<int>(m_lObjectID), _temperature);
}

void CPhysicalObject::SetTotalTorque(double _time, const double& _totalTorque) const
{
	m_storage->GetTimePointW(_time, static_cast<int>(m_lObjectID))->SetTotalTorque(static_cast<int>(m_lObjectID), _totalTorque);
}

void CPhysicalObject::SetOrientation(double _time, const CQuaternion& _orientation) const
{
	m_storage->GetTimePointW(_time, static_cast<int>(m_lObjectID))->SetOrientation(static_cast<int>(m_lObjectID), _orientation);
}

void CPhysicalObject::SetStressTensor(double _time, const CMatrix3& _stressTensor) const
{
	m_storage->GetTimePointW(_time, static_cast<int>(m_lObjectID))->SetStressTensor(static_cast<int>(m_lObjectID), _stressTensor);
}

void CPhysicalObject::SetCoordinates(const CVector3& _coordinates) const
{
	m_storage->GetTimePointW()->SetCoord(static_cast<int>(m_lObjectID), _coordinates);
}

void CPhysicalObject::SetVelocity(const CVector3& _velocity) const
{
	m_storage->GetTimePointW()->SetVel(static_cast<int>(m_lObjectID), _velocity);
}

void CPhysicalObject::SetAngleVelocity(const CVector3& _angleVelocity) const
{
	m_storage->GetTimePointW()->SetAngleVel(static_cast<int>(m_lObjectID), _angleVelocity);
}

void CPhysicalObject::SetForce(const CVector3& _force) const
{
	m_storage->GetTimePointW()->SetForce(static_cast<int>(m_lObjectID), _force);
}

void CPhysicalObject::SetTemperature(const double& _temperature) const
{
	m_storage->GetTimePointW()->SetTemperature(static_cast<int>(m_lObjectID), _temperature);
}

void CPhysicalObject::SetTotalTorque(const double& _totalTorque) const
{
	m_storage->GetTimePointW()->SetTotalTorque(static_cast<int>(m_lObjectID), _totalTorque);
}

void CPhysicalObject::SetOrientation(const CQuaternion& _orientation) const
{
	m_storage->GetTimePointW()->SetOrientation(static_cast<int>(m_lObjectID), _orientation);
}

void CPhysicalObject::SetStressTensor(const CMatrix3& _stressTensor) const
{
	m_storage->GetTimePointW()->SetStressTensor(static_cast<int>(m_lObjectID), _stressTensor);
}

bool CPhysicalObject::IsActive(const double& _dTime) const
{
	auto p = m_storage->Object(m_lObjectID);
	if (p->activity_end() == p->activity_start()) return false;
	return p->activity_start() <= _dTime && _dTime <= p->activity_end();
}

void CPhysicalObject::GetActivityTimeInterval(double* _pStartTime, double* _pEndTime) const
{
	auto p = m_storage->Object(m_lObjectID);
	*_pStartTime = p->activity_start();
	*_pEndTime = p->activity_end();
}

std::pair<double, double> CPhysicalObject::GetActivityTimeInterval() const
{
	const auto* p = m_storage->Object(m_lObjectID);
	return { p->activity_start(), p->activity_end() };
}

double CPhysicalObject::GetActivityStart() const
{
	auto p = m_storage->Object(m_lObjectID);
	return p->activity_start();
}

double CPhysicalObject::GetActivityEnd() const
{
	auto p = m_storage->Object(m_lObjectID);
	return p->activity_end();
}

void CPhysicalObject::SetStartActivityTime(double _dTime)
{
	auto p = m_storage->Object(m_lObjectID);
	p->set_activity_start(_dTime);
	if (_dTime > p->activity_end())
		p->set_activity_end(_dTime);
}

void CPhysicalObject::SetEndActivityTime(double _dTime)
{
	auto p = m_storage->Object(m_lObjectID);
	p->set_activity_end(_dTime);
	if (_dTime < p->activity_start())
		p->set_activity_start(_dTime);
}

void CPhysicalObject::SetObjectActivity(double _dTime, bool _bActive)
{
	auto p = m_storage->Object(m_lObjectID);
	if (_bActive)
	{
		if (p->activity_start() > _dTime)
			p->set_activity_start(_dTime);
		if (p->activity_end() < _dTime)
			p->set_activity_end(_dTime);
	}
	else if (p->activity_start() <= _dTime && _dTime <= p->activity_end())
		p->set_activity_end(_dTime);
}

void CPhysicalObject::AddBond(const unsigned& _nObjectID)
{
	unsigned i = 0;
	while (i < m_ConnectedBondsID.size())
	{
		// avoid double considering of the same object
		if (m_ConnectedBondsID[i] == _nObjectID)
			return;
		else
			i++;
	}
	m_ConnectedBondsID.push_back(_nObjectID);
}

void CPhysicalObject::DeleteAllBonds()
{
	m_ConnectedBondsID.clear();
}

bool CPhysicalObject::IsQuaternionSet(const double _dTime) const
{
	return m_storage->GetTimePointR(_dTime, m_lObjectID)->IsQuaternionSet(m_lObjectID);
}

void CPhysicalObject::ClearAllTDData(double _dTime)
{
	m_storage->GetTimePointW(_dTime, m_lObjectID)->ClearAllTDData(m_lObjectID);
}