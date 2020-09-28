/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ParticlesAnalyzer.h"
#include "GeometricFunctions.h"

CParticlesAnalyzer::CParticlesAnalyzer()
{
	SetPropertyType(CResultsAnalyzer::EPropertyType::Coordinate);
}

CParticlesAnalyzer::~CParticlesAnalyzer()
{
}

bool CParticlesAnalyzer::Export()
{
	// request proper material database if needed
	if ((GetProperty() == CResultsAnalyzer::EPropertyType::KineticEnergy) || (GetProperty() == CResultsAnalyzer::EPropertyType::PotentialEnergy))
		if (!CheckMaterialsDatabase())
			return false;

	if (GetProperty() == CResultsAnalyzer::EPropertyType::ResidenceTime)
		m_nResultsType = CResultsAnalyzer::EResultType::Distribution;
	else if ((m_bConcParam) || (m_nResultsType != CResultsAnalyzer::EResultType::Distribution))
		m_nPropSteps = 1;

	if (GetProperty() == CResultsAnalyzer::EPropertyType::ResidenceTime)
	{
		ResidenceTimeAnalysis();
		return true;
	}

	for (size_t iTime = 0; iTime < m_vDistrResults.size(); ++iTime)
	{
		if (CheckTerminationFlag()) return false;

		double dTime = m_vTimePoints[iTime];

		if (GetProperty() == CResultsAnalyzer::EPropertyType::CoordinationNumber)
			CoordinationNumberAnalysis(dTime, iTime);
		else
		{
			// status description
			m_sStatusDescr = "Time = " + std::to_string(dTime) + " [s]. Applying constraints. ";
			std::vector<size_t> vParticles = m_Constraints.FilteredParticles(dTime);

			// for calculation of max overlap
			std::vector<double> vOverlaps;
			if (GetProperty() == CResultsAnalyzer::EPropertyType::MaxOverlap)
				vOverlaps = m_pSystemStructure->GetMaxOverlaps(dTime, vParticles);

			// status description
			m_sStatusDescr = "Time = " + std::to_string(dTime) + " [s]. Processing " + std::to_string(vParticles.size()) + " particles";

			for (size_t j = 0; j < vParticles.size(); ++j)
			{
				if (CheckTerminationFlag()) return false;

				CSphere* pSphere = dynamic_cast<CSphere*>(m_pSystemStructure->GetObjectByIndex(vParticles[j]));
				switch (GetProperty())
				{
				case CResultsAnalyzer::EPropertyType::Coordinate:
					WriteComponentToResults(pSphere->GetCoordinates(dTime), iTime);
					break;
				case CResultsAnalyzer::EPropertyType::Distance:
					if (m_nDistance == CResultsAnalyzer::EDistanceType::ToPoint)
						WriteComponentToResults(pSphere->GetCoordinates(dTime) - m_Point1, iTime);
					else
						WriteValueToResults(DistanceFromPointToSegment(pSphere->GetCoordinates(dTime), m_Point1, m_Point2), iTime);
					break;
				case CResultsAnalyzer::EPropertyType::ForceTotal:
					WriteComponentToResults(pSphere->GetForce(dTime), iTime);
					break;
				case CResultsAnalyzer::EPropertyType::MaxOverlap:
					WriteValueToResults(vOverlaps[vParticles[j]], iTime);
					break;
				case CResultsAnalyzer::EPropertyType::Number:
					m_vConcResults[iTime]++;
					break;
				case CResultsAnalyzer::EPropertyType::TotalVolume:
					m_vConcResults[iTime] += pSphere->GetVolume();
					break;
				case CResultsAnalyzer::EPropertyType::Temperature:
					WriteValueToResults(pSphere->GetTemperature(dTime), iTime);
					break;
				case CResultsAnalyzer::EPropertyType::VelocityTotal:
					WriteComponentToResults(pSphere->GetVelocity(dTime), iTime);
					break;
				case CResultsAnalyzer::EPropertyType::VelocityRotational:
					WriteComponentToResults(pSphere->GetAngleVelocity(dTime), iTime);
					break;
				case CResultsAnalyzer::EPropertyType::Stress:
					WriteComponentToResults(pSphere->GetNormalStress(dTime), iTime);
					break;
				case CResultsAnalyzer::EPropertyType::KineticEnergy:
					WriteValueToResults(0.5 * pSphere->GetMass() * pSphere->GetVelocity(dTime).SquaredLength(), iTime);
					break;
				case CResultsAnalyzer::EPropertyType::PotentialEnergy:
					WriteValueToResults(pSphere->GetMass()* GRAVITY_CONSTANT * pSphere->GetCoordinates(dTime).z, iTime);
					break;
				default:
					break;
				}
			}
		}
		m_nProgress = (unsigned)((iTime + 1.) / (double)m_vDistrResults.size() * 100);
	}

	return true;
}

void CParticlesAnalyzer::ResidenceTimeAnalysis()
{
	std::string sTime;
	std::set<size_t> vInitParticles;
	std::map<size_t, double> vResTimes;
	std::set<size_t> vCurrParticles;	// filtered particles in volume

	for (size_t iTime = 0; iTime < m_vDistrResults.size(); ++iTime)
	{

		if (CheckTerminationFlag()) return;

		double dTime = m_vTimePoints[iTime];

		// status description
		sTime = "Time = " + std::to_string(dTime) + " [s]. ";
		m_sStatusDescr = sTime + "Applying constraints. ";

		std::vector<size_t> vParticles = m_Constraints.FilteredParticles(dTime);
		vCurrParticles.clear();
		std::copy(vParticles.begin(), vParticles.end(), std::inserter(vCurrParticles, vCurrParticles.begin()));

		// status description
		m_sStatusDescr = sTime + "Processing " + std::to_string(vParticles.size()) + " particles";

		if (iTime == 0)
			vInitParticles = vCurrParticles;
		else
		{
			std::set<size_t> vConciderable = SetDifference(vCurrParticles, vInitParticles);
			vInitParticles = SetIntersection(vInitParticles, vCurrParticles);
			double dTimeStep = (m_vTimePoints.size() == 1) ? 0 : ((iTime != m_vTimePoints.size() - 1) ? (m_vTimePoints[iTime + 1] - m_vTimePoints[iTime]) : (m_vTimePoints[iTime] - m_vTimePoints[iTime - 1]));
			for (std::set<size_t>::iterator it = vConciderable.begin(); it != vConciderable.end(); ++it)
				if (vResTimes.find(*it) == vResTimes.end())	// new particle
					vResTimes.insert(std::pair<size_t, double>(*it, dTimeStep));
				else // already exits
					vResTimes[*it] += dTimeStep;
		}

		m_nProgress = (unsigned)((iTime + 1.) / (double)m_vDistrResults.size() * 100);
	}

	// remove particles that are left in the analysis volume
	for (std::set<size_t>::iterator it = vCurrParticles.begin(); it != vCurrParticles.end(); ++it)
		if (vResTimes.find(*it) != vResTimes.end())
			vResTimes.erase(*it);

	double dMid = 0;
	double dMax = vResTimes.empty() ? 0 : vResTimes.begin()->second;
	double dMin = 0;
	m_vConcResults.resize(m_nPropSteps, 0);
	for (std::map<size_t, double>::iterator it = vResTimes.begin(); it != vResTimes.end(); ++it)
	{
		m_vConcResults[CalculatePropIndex(it->second)]++;
		dMid += it->second;
		if (it->second > dMax)
			dMax = it->second;
		if (it->second < dMin)
			dMin = it->second;
	}
	if (!vResTimes.empty())
		dMid /= vResTimes.size();

	m_bCustomFileWriter = true; // to omit calling of WriteResultsToFile() from CResultsAnalyzer::StartExport
	m_sStatusDescr = "Writing results into file. ";

	*m_pout << "ResidenseTime[s]; ";
	for (unsigned i = 0; i < m_nPropSteps; i++)
		*m_pout << i*m_dPropStep + m_dPropMin << "; ";
	*m_pout << std::endl;
	*m_pout << "ParticlesNumber[-]; ";
	for (size_t i = 0; i < m_vConcResults.size(); ++i)
		*m_pout << m_vConcResults[i] << "; ";
	*m_pout << std::endl << std::endl;
	*m_pout << "MinValue[s]: " << dMin << std::endl;
	*m_pout << "MaxValue[s]: " << dMax << std::endl;
	*m_pout << "MidValue[s]: " << dMid << std::endl;
	std::dynamic_pointer_cast<std::ofstream>(m_pout)->close();
}

void CParticlesAnalyzer::CoordinationNumberAnalysis(double _dTime, size_t _iTime)
{
	m_sStatusDescr = "Time = " + std::to_string(_dTime) + " [s]. " + "Applying constraints. ";
	std::vector<size_t> vParticles = m_Constraints.FilteredParticles(_dTime);

	m_sStatusDescr = "Time = " + std::to_string(_dTime) + " [s]. " + "Processing " + std::to_string(vParticles.size()) + " particles";
	std::vector<unsigned> vAllCoordNums = m_pSystemStructure->GetCoordinationNumbers(_dTime);

	for (size_t i = 0; i < vParticles.size(); ++i)
		WriteValueToResults(vAllCoordNums[vParticles[i]], _iTime);
}