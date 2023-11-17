/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "Bond.h"
#include "../SimResultsStorage/DemStorage.h"

CBond::CBond(unsigned _id, CDemStorage* _storage) :CPhysicalObject(_id, _storage)
{
	m_nLeftObjectID = 0;
	m_nRightObjectID = 0;
	m_dDiameter = 1e-3;
	m_dInitialLength = 0.01; // initial length [m]
	m_dViscosity = 1e+20;
}

std::string CBond::GetObjectGeometryText() const
{
	std::ostringstream outputStream;
	outputStream << m_nLeftObjectID << " " << m_nRightObjectID << " " << std::setprecision(25) << m_dDiameter << " " << m_dInitialLength;
	return outputStream.str();
}

void CBond::SetObjectGeometryText(std::stringstream& _inputStream)
{
	_inputStream >> m_nLeftObjectID >> m_nRightObjectID >> m_dDiameter >> m_dInitialLength;
	UpdatePrecalculatedValues();
}

std::vector<uint8_t> CBond::GetObjectGeometryBin() const
{
	USaveHelper saver{ USaveHelper::SSaveData{ m_nLeftObjectID, m_nRightObjectID, m_dDiameter, m_dInitialLength } };
	return { std::make_move_iterator(std::begin(saver.binary)), std::make_move_iterator(std::end(saver.binary)) };
}

void CBond::SetObjectGeometryBin(const std::vector<uint8_t>& _data)
{
	const USaveHelper info{ _data };
	m_nLeftObjectID  = static_cast<unsigned>(info.data.idLeft);		// explicit cast for cross-platform compatibility
	m_nRightObjectID = static_cast<unsigned>(info.data.idRight);	// explicit cast for cross-platform compatibility
	m_dDiameter      = info.data.diameter;
	m_dInitialLength = info.data.initialLength;
}
