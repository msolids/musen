/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ResultsAnalyzer.h"
#include <fstream>
#include <utility>

CResultsAnalyzer::CResultsAnalyzer() :
	m_pout(&std::cout, [](void*) {}) // Sets m_pout out to cout without the destructor
{
	m_pSystemStructure = nullptr;
	m_nProgress = 0;
	m_sStatusDescr = "";
	m_bError = false;
	m_bCustomFileWriter = false;
	m_CurrentStatus = EStatus::Idle;
	SetPropertyType(static_cast<EPropertyType>(0));
	SetDistanceType(static_cast<EDistanceType>(0));
	SetVectorComponent(static_cast<EVectorComponent>(0));
	SetRelatonType(static_cast<ERelationType>(0));
	SetCollisionType(static_cast<ECollisionType>(0));
	SetResultsType(static_cast<EResultType>(0));
	SetGeometryIndex(0);
	SetPoint1(CVector3{ 0 });
	SetPoint2(CVector3{ 0 });
	SetTime(0, 0.1, 0.01);
	SetProperty(0, 1, 10);
}

CResultsAnalyzer::~CResultsAnalyzer()
{
	// ensure that if m_pout is a filestream it is closed if the ResultsAnalyzer is finished, since 
	// an open m_pout makes no sense after completion. Note that the closing would also happen automatically if no further shared_ptr to m_pout existed. 
	std::shared_ptr<std::ofstream> fileoutstream(std::dynamic_pointer_cast<std::ofstream>(m_pout));
	if (fileoutstream)
		fileoutstream->close();
}

void CResultsAnalyzer::UpdateSettings()
{
	m_nProgress = 0;
	m_sStatusDescr = "";
	m_bError = false;
	if(m_nGeometryIndex >= m_pSystemStructure->GeometriesNumber())
		SetGeometryIndex(0);
}

void CResultsAnalyzer::SetSystemStructure(CSystemStructure* _pSystemStructure)
{
	m_pSystemStructure = _pSystemStructure;
}

const CSystemStructure* CResultsAnalyzer::GetSystemStructure() const
{
	return m_pSystemStructure;
}

CConstraints* CResultsAnalyzer::GetConstraintsPtr()
{
	return &m_Constraints;
}

CResultsAnalyzer::EPropertyType CResultsAnalyzer::GetProperty()
{
	return m_vProperties.front();
}

std::vector<CResultsAnalyzer::EPropertyType> CResultsAnalyzer::GetProperties() const
{
	return m_vProperties;
}

void CResultsAnalyzer::SetPropertyType(VPropertyType _property)
{
	m_vProperties = std::move(_property);
	m_bConcParam = m_vProperties.front() == EPropertyType::Number || m_vProperties.front() == EPropertyType::TotalVolume; // TODO: update structure, makes no sense at the moment
}

void CResultsAnalyzer::SetPropertyType(EPropertyType _property)
{
	SetPropertyType(VPropertyType{ _property });
}

void CResultsAnalyzer::SetOutputStream(std::shared_ptr<std::ostream> _out)
{
	if (_out)
		m_pout = _out;
}

void CResultsAnalyzer::SetDistanceType(EDistanceType _distance)
{
	m_nDistance = _distance;
}

void CResultsAnalyzer::SetVectorComponent(EVectorComponent _component)
{
	m_nComponent = _component;
}

void CResultsAnalyzer::SetRelatonType(ERelationType _relation)
{
	m_nRelation = _relation;
}

void CResultsAnalyzer::SetCollisionType(ECollisionType _type)
{
	m_nCollisionType = _type;
}

void CResultsAnalyzer::SetResultsType(EResultType _type)
{
	m_nResultsType = _type;
}

void CResultsAnalyzer::SetGeometryIndex(unsigned _index)
{
	m_nGeometryIndex = _index;
}

void CResultsAnalyzer::SetPoint1(const CVector3& _point)
{
	m_Point1 = _point;
}

void CResultsAnalyzer::SetPoint2(const CVector3& _point)
{
	m_Point2 = _point;
}

void CResultsAnalyzer::SetTime(double _timeMin, double _timeMax, double _timeStep, bool _bOnlySaved /*= false*/)
{
	m_dTimeMin = _timeMin;
	m_dTimeMax = _timeMax;
	m_dTimeStep = _timeStep;
	m_bOnlySavedTP = _bOnlySaved;
	m_vTimePoints.clear();

	if (_timeMin > _timeMax) return;

	if (_bOnlySaved)
	{
		auto allTP = m_pSystemStructure->GetAllTimePoints();
		std::copy_if(allTP.begin(), allTP.end(), std::back_inserter(m_vTimePoints), [&_timeMin, &_timeMax](double tp) { return tp >= _timeMin && std::round(tp * 1e+20) / 1e+20 <= _timeMax; });
	}
	else
	{
		size_t num = size_t((_timeMax - _timeMin) / _timeStep);
		for (size_t i = 0; i <= num; ++i)
			m_vTimePoints.push_back(_timeMin + i*_timeStep);
	}
}

void CResultsAnalyzer::SetProperty(double _propMin, double _propMax, unsigned _propSteps)
{
	m_dPropMin = _propMin;
	m_dPropMax = _propMax;
	m_nPropSteps = _propSteps;
}

unsigned CResultsAnalyzer::GetExportProgress() const
{
	return m_nProgress;
}

std::string CResultsAnalyzer::GetStatusDescription() const
{
	return m_sStatusDescr;
}

CResultsAnalyzer::EStatus CResultsAnalyzer::GetCurrentStatus() const
{
	return m_CurrentStatus;
}

bool CResultsAnalyzer::IsError() const
{
	return m_bError;
}

void CResultsAnalyzer::SetCurrentStatus(CResultsAnalyzer::EStatus _Status)
{
	m_CurrentStatus = _Status;
}

void CResultsAnalyzer::StartExport()
{
	StartExport(m_sOutputFileName);
}

void CResultsAnalyzer::StartExport(const std::string& _sFileName)
{
	m_bError = false;
	m_bCustomFileWriter = false;
	if (!CheckTimePoints()) return;
	if (!PrepareOutFile(_sFileName)) return;
	m_CurrentStatus = CResultsAnalyzer::EStatus::Runned;
	m_nProgress = 0;
	m_sStatusDescr = "Initializing. ";
	CalculateSteps();
	PrepareResultVectors();
	if (Export() && !m_bError && !m_bCustomFileWriter)
	{
		m_sStatusDescr = "Writing results into file. ";
		WriteResultsToFile();
	}
	m_CurrentStatus = CResultsAnalyzer::EStatus::Idle;
}

bool CResultsAnalyzer::PrepareOutFile(const std::string& _sFileName)
{

	std::shared_ptr<std::ofstream> outFile = std::make_shared<std::ofstream>();
	m_pout = outFile;
	outFile->close();
	outFile->open(UnicodePath(_sFileName));
	if (outFile->fail())
	{
		m_bError = true;
		m_sStatusDescr = "Error: Unable to open output file for writing. ";
		return false;
	}
	return true;
}

void CResultsAnalyzer::CalculateSteps()
{
	m_dPropStep = (m_dPropMax - m_dPropMin) / m_nPropSteps;
}

void CResultsAnalyzer::PrepareResultVectors()
{
	m_vDistrResults.clear();
	m_vConcResults.clear();
	m_vValueResults.clear();

	size_t nSize = m_vTimePoints.size();
	if (m_nRelation == CResultsAnalyzer::ERelationType::Appeared)
		nSize--; 	// -1 because values here are calculated for the center of intervals, not for each time point: [0:1:5] -> 5 time points

	m_vDistrResults.resize(nSize, std::vector<size_t>(m_nPropSteps, 0));
	m_vValueResults.resize(nSize);
	m_vConcResults.resize(nSize, 0.0);
}

int CResultsAnalyzer::CalculatePropIndex(double _dValue) const
{
	int index = static_cast<int>((_dValue - m_dPropMin) / m_dPropStep);
	if (index < 0) index = 0;
	if (index >= static_cast<int>(m_nPropSteps)) index = m_nPropSteps - 1;
	return index;
}

bool CResultsAnalyzer::CheckTimePoints()
{
	if(m_vTimePoints.empty())
	{
		m_bError = true;
		m_sStatusDescr = "Error: No data will be exported with current time settings. ";
		return false;
	}
	return true;
}

bool CResultsAnalyzer::CheckMaterialsDatabase()
{
	m_pSystemStructure->UpdateAllObjectsCompoundsProperties();
	if (!m_pSystemStructure->IsAllCompoundsDefined().empty())
	{
		m_bError = true;
		m_sStatusDescr = m_pSystemStructure->IsAllCompoundsDefined();
		return false;
	}
	return true;
}

bool CResultsAnalyzer::CheckTerminationFlag()
{
	if (m_CurrentStatus == CResultsAnalyzer::EStatus::ShouldBeStopped)
	{
		m_bError = true;
		m_sStatusDescr = "Analysis was terminated by user. ";
		return true;
	}
	return false;
}

void CResultsAnalyzer::WriteValueToResults(double _dResult, size_t _nTimeIndex)
{
	switch (m_nResultsType)
	{
	case CResultsAnalyzer::EResultType::Distribution:
	{
		if ((_dResult < m_dPropMin) || (_dResult > m_dPropMax))	// value is not in interval
			return;
		int iProperty = CalculatePropIndex(_dResult);
		m_vDistrResults[_nTimeIndex][iProperty]++;
		break;
	}
	case CResultsAnalyzer::EResultType::Average:
		m_vConcResults[_nTimeIndex] += _dResult;
		m_vDistrResults[_nTimeIndex][0]++;
		m_vValueResults[_nTimeIndex].push_back(_dResult);
		break;
	case CResultsAnalyzer::EResultType::Maximum:
		if (m_vDistrResults[_nTimeIndex][0] == 0)
		{
			m_vConcResults[_nTimeIndex] = _dResult;
			m_vDistrResults[_nTimeIndex][0]++;
		}
		else
			if (_dResult > m_vConcResults[_nTimeIndex])
				m_vConcResults[_nTimeIndex] = _dResult;
		break;
	case CResultsAnalyzer::EResultType::Minimum:
		if (m_vDistrResults[_nTimeIndex][0] == 0)
		{
			m_vConcResults[_nTimeIndex] = _dResult;
			m_vDistrResults[_nTimeIndex][0]++;
		}
		else
			if (_dResult < m_vConcResults[_nTimeIndex])
				m_vConcResults[_nTimeIndex] = _dResult;
		break;
	default:
		break;
	}

}

void CResultsAnalyzer::WriteComponentToResults(const CVector3& _vResult, size_t _nTimeIndex)
{
	double dResult;
	switch (m_nComponent)
	{
	case CResultsAnalyzer::EVectorComponent::Total:
		dResult = _vResult.Length();
		break;
	case CResultsAnalyzer::EVectorComponent::X:
		dResult = _vResult.x;
		break;
	case CResultsAnalyzer::EVectorComponent::Y:
		dResult = _vResult.y;
		break;
	case CResultsAnalyzer::EVectorComponent::Z:
		dResult = _vResult.z;
		break;
	default:
		dResult = 0;
		break;
	}

	WriteValueToResults(dResult, _nTimeIndex);
}

void CResultsAnalyzer::WriteDistrResultsToFile()
{
	if (m_nRelation == CResultsAnalyzer::ERelationType::Existing)	// time points are analyzed
		*m_pout << "TimePoint[s]; ";
	else // intervals between time points are analyzed
		*m_pout << "TimeStart[s]:TimeEnd[s]; ";

	for (unsigned i = 0; i < m_nPropSteps; ++i)
		*m_pout << i*m_dPropStep + m_dPropMin + m_dPropStep/2 << "; "; // middle point of each size class
	*m_pout << std::endl;
	for (size_t i = 0; i < m_vDistrResults.size(); ++i)
	{
		if (m_nRelation == CResultsAnalyzer::ERelationType::Existing)	// time points are analyzed
			*m_pout << m_vTimePoints[i] << "; ";
		else // intervals between time points are analyzed
			*m_pout << m_vTimePoints[i] << ":" << m_vTimePoints[i + 1] << "; ";
		for (size_t j = 0; j < m_vDistrResults[i].size(); ++j)
			*m_pout << m_vDistrResults[i][j] << "; ";
		*m_pout << std::endl;
	}
}

void CResultsAnalyzer::WriteConcResultsToFile()
{
	if (m_nRelation == CResultsAnalyzer::ERelationType::Existing)	// time points are analyzed
		*m_pout << "TimePoint[s]; ";
	else // intervals between time points are analyzed
		*m_pout << "TimeStart[s]:TimeEnd[s]; ";

	if (m_bConcParam)
		*m_pout << "Value" << std::endl;
	else
		switch (m_nResultsType)
	{
		case CResultsAnalyzer::EResultType::Average:
			*m_pout << "Average; Number; Deviation" << std::endl;
			break;
		case CResultsAnalyzer::EResultType::Maximum:
			*m_pout << "Max" << std::endl;
			break;
		case CResultsAnalyzer::EResultType::Minimum:
			*m_pout << "Min" << std::endl;
			break;
		case CResultsAnalyzer::EResultType::Distribution: break;
	}

	for (size_t i = 0; i < m_vConcResults.size(); ++i)
	{
		if (m_nRelation == CResultsAnalyzer::ERelationType::Existing)	// time points are analyzed
			*m_pout << m_vTimePoints[i] << "; ";
		else // intervals between time points are analyzed
			*m_pout << m_vTimePoints[i] << ":" << m_vTimePoints[i + 1] << "; ";

		*m_pout << m_vConcResults[i];
		if (!m_bConcParam && m_nResultsType == CResultsAnalyzer::EResultType::Average)
			*m_pout << "; " << m_vDistrResults[i][0] << "; " << m_vValueResults[i][0];	// for average values
		*m_pout << std::endl;
	}

}

void CResultsAnalyzer::WriteResultsToFile()
{
	if (m_bConcParam)
		WriteConcResultsToFile();
	else
	{
		switch (m_nResultsType)
		{
		case CResultsAnalyzer::EResultType::Distribution:
			WriteDistrResultsToFile();
			break;
		case CResultsAnalyzer::EResultType::Average:
			for (size_t i = 0; i < m_vConcResults.size(); ++i)
			{
				double dDeviation = 0;
				if (m_vDistrResults[i][0] != 0)
				{
					m_vConcResults[i] /= m_vDistrResults[i][0];	// calculate average
					for (size_t j = 0; j < m_vValueResults[i].size(); ++j)	// calculate standard deviation
						dDeviation += pow(m_vValueResults[i][j] - m_vConcResults[i], 2.);
					dDeviation = sqrt(dDeviation / m_vDistrResults[i][0]);
				}
				if (!m_vValueResults[i].empty())
					m_vValueResults[i][0] = dDeviation;
				else
					m_vValueResults[i].push_back(dDeviation);
			}
		case CResultsAnalyzer::EResultType::Maximum: [[fallthrough]];
		case CResultsAnalyzer::EResultType::Minimum:
			WriteConcResultsToFile();
			break;
		default:
			break;
		}
	}
	std::dynamic_pointer_cast<std::ofstream>(m_pout)->close();
}
