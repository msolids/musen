/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "Correlation.h"

CCorrelation::CCorrelation(void) :
	m_dT1(0),
	m_dT2(0),
	m_dP1(0),
	m_dP2(0),
	m_nCorType(ECorrelationTypes::UNDEFINED)
{
}

CCorrelation::CCorrelation(double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nType, const std::vector<double>& _vParams) :
	m_dT1(_dT1),
	m_dT2(_dT2),
	m_dP1(_dP1),
	m_dP2(_dP2),
	m_nCorType(_nType),
	m_vParameters(_vParams)
{
}

double CCorrelation::GetT1() const
{
	return m_dT1;
}

double CCorrelation::GetT2() const
{
	return m_dT2;
}

double CCorrelation::GetP1() const
{
	return m_dP1;
}

double CCorrelation::GetP2() const
{
	return m_dP2;
}

ECorrelationTypes CCorrelation::GetType() const
{
	return m_nCorType;
}

std::vector<double> CCorrelation::GetParameters() const
{
	return m_vParameters;
}

void CCorrelation::SetT1(double _dT)
{
	m_dT1 = _dT;
}

void CCorrelation::SetT2(double _dT)
{
	m_dT2 = _dT;
}

void CCorrelation::SetP1(double _dP)
{
	m_dP1 = _dP;
}

void CCorrelation::SetP2(double _dP)
{
	m_dP2 = _dP;
}

void CCorrelation::SetType(ECorrelationTypes _nType)
{
	m_nCorType = _nType;
	switch (_nType)
	{
	case ECorrelationTypes::CONSTANT:	m_vParameters.resize(1, 1.0);	break;
	case ECorrelationTypes::LINEAR:		m_vParameters.resize(3, 1.0);	break;
	case ECorrelationTypes::EXPONENT_1:	m_vParameters.resize(3, 1.0);	break;
	case ECorrelationTypes::POW_1:		m_vParameters.resize(2, 1.0);	break;
	default: { m_nCorType = ECorrelationTypes::UNDEFINED; m_vParameters.resize(0); }
	}
}

bool CCorrelation::SetParameters(const std::vector<double>& _vParams)
{
	switch (m_nCorType)
	{
	case ECorrelationTypes::CONSTANT:	if (_vParams.size() != 1) { m_nCorType = ECorrelationTypes::UNDEFINED; m_vParameters.resize(0); return false; }	break;
	case ECorrelationTypes::LINEAR:		if (_vParams.size() != 3) { m_nCorType = ECorrelationTypes::UNDEFINED; m_vParameters.resize(0); return false; }	break;
	case ECorrelationTypes::EXPONENT_1:	if (_vParams.size() != 3) { m_nCorType = ECorrelationTypes::UNDEFINED; m_vParameters.resize(0); return false; }	break;
	case ECorrelationTypes::POW_1:		if (_vParams.size() != 2) { m_nCorType = ECorrelationTypes::UNDEFINED; m_vParameters.resize(0); return false; }	break;
	default: { m_nCorType = ECorrelationTypes::UNDEFINED; m_vParameters.resize(0); return false; }
	}
	m_vParameters = _vParams;
	return true;
}

bool CCorrelation::SetCorrelation(double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nType, const std::vector<double>& _vParams)
{
	switch (_nType)
	{
	case ECorrelationTypes::CONSTANT:	if (_vParams.size() != 1) { m_nCorType = ECorrelationTypes::UNDEFINED; return false; }	break;
	case ECorrelationTypes::LINEAR:		if (_vParams.size() != 3) { m_nCorType = ECorrelationTypes::UNDEFINED; return false; }	break;
	case ECorrelationTypes::EXPONENT_1:	if (_vParams.size() != 3) { m_nCorType = ECorrelationTypes::UNDEFINED; return false; }	break;
	case ECorrelationTypes::POW_1:		if (_vParams.size() != 2) { m_nCorType = ECorrelationTypes::UNDEFINED; return false; }	break;
	default: { m_nCorType = ECorrelationTypes::UNDEFINED; return false; }
	}
	m_dT1 = _dT1;
	m_dT2 = _dT2;
	m_dP1 = _dP1;
	m_dP2 = _dP2;
	m_nCorType = _nType;
	m_vParameters = _vParams;
	return true;
}

double CCorrelation::GetCorrelationValue(double _dT, double _dP) const
{
	if (!IsInInterval(_dT, _dP))
		return _NOT_A_NUMBER;

	switch (m_nCorType)
	{
	case ECorrelationTypes::UNDEFINED:	return _NOT_A_NUMBER;
	case ECorrelationTypes::CONSTANT:	return m_vParameters[0];
	case ECorrelationTypes::LINEAR:		return _dT*m_vParameters[2] + _dP*m_vParameters[1] + m_vParameters[0];
	case ECorrelationTypes::EXPONENT_1:	return exp(m_vParameters[0] + m_vParameters[1] / (_dT + m_vParameters[2]));
	case ECorrelationTypes::POW_1:		return m_vParameters[0] * pow(_dT, m_vParameters[1]);
	}
	return _NOT_A_NUMBER;
}

bool CCorrelation::IsTInInterval(double _dT) const
{
	return ((_dT >= m_dT1) && (_dT <= m_dT2));
}

bool CCorrelation::IsPInInterval(double _dP) const
{
	return ((_dP >= m_dP1) && (_dP <= m_dP2));
}

bool CCorrelation::IsInInterval(double _dT, double _dP) const
{
	return (IsTInInterval(_dT) && IsPInInterval(_dP));
}

void CCorrelation::SaveToProtobuf(ProtoCorrelation& _protoCorrelation)
{
	_protoCorrelation.set_t1(m_dT1);
	_protoCorrelation.set_t2(m_dT2);
	_protoCorrelation.set_p1(m_dP1);
	_protoCorrelation.set_p2(m_dP2);
	_protoCorrelation.set_cortype(static_cast<unsigned>(m_nCorType));
	for (size_t i = 0; i < m_vParameters.size(); ++i)
		_protoCorrelation.add_parameter(m_vParameters[i]);
}

void CCorrelation::LoadFromProtobuf(const ProtoCorrelation& _protoCorrelation)
{
	m_dT1 = _protoCorrelation.t1();
	m_dT2 = _protoCorrelation.t2();
	m_dP1 = _protoCorrelation.p1();
	m_dP2 = _protoCorrelation.p2();
	m_nCorType = static_cast<ECorrelationTypes>(_protoCorrelation.cortype());
	m_vParameters.clear();
	for (int i = 0; i < _protoCorrelation.parameter_size(); ++i)
		m_vParameters.push_back(_protoCorrelation.parameter(i));
}