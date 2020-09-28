/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "GeometriesAnalyzer.h"
#include <vector>

CGeometriesAnalyzer::CGeometriesAnalyzer()
{
	SetPropertyType(EPropertyType::ForceTotal);
	m_pObject = nullptr;
}

bool CGeometriesAnalyzer::Export()
{
	if (!InitAnalyzer(GetProperties()))
		return false;

	// loop time points
	for (size_t i = 0; i < m_vTimePoints.size(); ++i)
	{
		if (CheckTerminationFlag())
			return false;
		m_sStatusDescr = "Processing " + std::to_string(i) + "/" + std::to_string(m_vTimePoints.size());
		m_nProgress = static_cast<unsigned>((i + 1.) / static_cast<double>(m_vTimePoints.size()) * 100);
		WriteTimePoint(m_vTimePoints[i]);
	}

	// close file if m_pout is a filestream
	std::shared_ptr<std::ofstream> fileoutstream(std::dynamic_pointer_cast<std::ofstream>(m_pout));
	if (fileoutstream)
		fileoutstream->close();

	return m_pout->good();
}

bool CGeometriesAnalyzer::InitAnalyzer(std::vector<EPropertyType> _properties)
{
	m_vCalcFunctions.clear();
	m_pObject = m_pSystemStructure->Geometry(m_nGeometryIndex);
	if (!m_pObject) return false;

	m_vInitPosition = m_pObject->Center(0);

	m_bCustomFileWriter = true; // to omit calling of WriteResultsToFile() from CResultsAnalyzer::StartExport
	m_sStatusDescr = "Starting export.";
	*m_pout << "Time[s]";

	std::sort(_properties.begin(), _properties.end(), [](const auto& l, const auto& r){	return E2I(l) < E2I(r); }); // to have a consistent output order

	for (auto const& i : _properties)
		switch (i)
		{
		case EPropertyType::ForceTotal:
			// init calculation
			m_vCalcFunctions.emplace_back([&](const double& _timePoint)
			{
				const CVector3 currForce = this->CalculateForce(_timePoint);
				*m_pout << "; " << currForce.x << "; " << currForce.y << "; " << currForce.z << "; " << currForce.Length();
			});
			*m_pout << "; X[N]; Y[N]; Z[N]; Total[N] ";
			break;
		case EPropertyType::Distance:
			// init calculation
			m_vInitPosition = m_pObject->Center(0);
			m_vCalcFunctions.emplace_back([&](const double& _timePoint)
			{
				const CVector3 currDistance = this->CalculateDistance(_timePoint);
				*m_pout << "; " << currDistance.x << "; " << currDistance.y << "; " << currDistance.z << "; " << currDistance.Length();
			});

			*m_pout << "; X[m]; Y[m]; Z[m]; Total[m] ";
			break;
		default:
			break;
		}
	*m_pout << std::endl;
	return true;
}

void CGeometriesAnalyzer::WriteTimePoint(const double& _timePoint)
{
	*m_pout << _timePoint;
	for (const calcFunction_type& calcFunction : m_vCalcFunctions)
		calcFunction(_timePoint);
	*m_pout << "\n"; // note: no flush here, otherwise constant writing would be too slow
}

void CGeometriesAnalyzer::FlushStream() const
{
	m_pout->flush();
}

CVector3 CGeometriesAnalyzer::CalculateDistance(const double& _timePoint) const
{
	return m_pObject->Center(_timePoint) - m_vInitPosition;
}

CVector3 CGeometriesAnalyzer::CalculateForce(const double& _timePoint) const
{
	CVector3 tmp(0.0);
	for (auto plane : m_pObject->Planes())
		tmp += m_pSystemStructure->GetObjectByIndex(plane)->GetForce(_timePoint);
	return tmp;
}
