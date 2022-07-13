/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "Sphere.h"
#include <array>
#include "../SimResultsStorage/DemStorage.h"

CSphere::CSphere(unsigned _id, CDemStorage* _storage) : CPhysicalObject(_id, _storage)
{
	m_storage->Object(m_lObjectID)->set_type(ProtoParticleInfo::kSphere);
	m_dRadius = 0;
	m_dContactRadius = 0;
	m_dHeatCapacity = _HEAT_CAPACITY_DEFAULT;
	UpdatePrecalculatedValues();
}

std::string CSphere::GetObjectGeometryText() const
{
	std::ostringstream outputStream;
	outputStream << std::setprecision(std::numeric_limits<double>::digits10 + 1) << m_dRadius << " " << m_dContactRadius;
	return outputStream.str();
}

void CSphere::SetObjectGeometryText(std::stringstream& _inputStream)
{
	_inputStream >> m_dRadius >> m_dContactRadius;
	if (m_dContactRadius == 0)
		m_dContactRadius = m_dRadius;
	UpdatePrecalculatedValues();
}

std::vector<uint8_t> CSphere::GetObjectGeometryBin() const
{
	USaveHelper saver{ USaveHelper::SSaveData{ m_dRadius, m_dContactRadius } };
	return { std::make_move_iterator(std::begin(saver.binary)), std::make_move_iterator(std::end(saver.binary)) };
}

void CSphere::SetObjectGeometryBin(const std::vector<uint8_t>& _data)
{
	const USaveHelper saver{ _data };
	m_dRadius = saver.data.radius;
	m_dContactRadius = saver.data.contactRadius;
}

void CSphere::SetRadius(const double& _radius)
{
	m_dRadius = (_radius >= 0 ? _radius : 0);
	if (m_dContactRadius == 0)
		m_dContactRadius = _radius;
	UpdatePrecalculatedValues();
}

void CSphere::SetContactRadius(const double& _radius)
{
	m_dContactRadius = (_radius >= 0 ? _radius : 0);
	UpdatePrecalculatedValues();
}

void CSphere::UpdateCompoundProperties(const CCompound* _pCompound)
{
	m_dMass = GetVolume() * _pCompound->GetPropertyValue(PROPERTY_DENSITY);
	m_dHeatCapacity = _pCompound->GetPropertyValue(PROPERTY_HEAT_CAPACITY);
	UpdatePrecalculatedValues();
}

void CSphere::UpdatePrecalculatedValues()
{
	m_dInertiaMoment = 2.0 / 5 * m_dMass*m_dRadius*m_dRadius;
	m_dRadiusSqrt = sqrt(m_dRadius);
	if (m_dContactRadius == 0)
		m_dContactRadius = m_dRadius;
}
