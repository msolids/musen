/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ConsoleResultsAnalyzer.h"
#include "GeometriesAnalyzer.h"
#include "BondsAnalyzer.h"
#include "ParticlesAnalyzer.h"
#include "AgglomeratesAnalyzer.h"

CConsoleResultsAnalyzer::CConsoleResultsAnalyzer(std::ostream& _out/* = std::cout*/, std::ostream& _err/* = std::cerr*/) :
	m_out(_out),
	m_err(_err),
	m_sFileExt{ "csv" },
	m_sOutputPrefix{ "" }
{
}

CConsoleResultsAnalyzer::EProcessResultType CConsoleResultsAnalyzer::SetupAnalyzer(const std::vector<std::string>& _commandSet,
	CSystemStructure& _systemStructure, std::shared_ptr<CResultsAnalyzer>& _outAnalyzer) const
{
	// Init Analyzer
	EAnalyzerTypes analyzerType;
	if (_outAnalyzer)
		m_out << "CConsoleResultsAnalyzer: non-empty pointer to analyzer was passed. Pointer will be overwritten.";
	if (_commandSet[0] == "GeometriesAnalyzer")
	{
		_outAnalyzer = std::make_shared<CGeometriesAnalyzer>();
		analyzerType = EAnalyzerTypes::GeometryAnalyzer;
	}
	else if (_commandSet[0] == "BondsAnalyzer")
	{
		_outAnalyzer = std::make_shared<CBondsAnalyzer>();
		analyzerType = EAnalyzerTypes::BondsAnalyzer;
	}
	else if (_commandSet[0] == "ParticlesAnalyzer")
	{
		_outAnalyzer = std::make_shared<CParticlesAnalyzer>();
		analyzerType = EAnalyzerTypes::ParticlesAnalyzer;
	}
	else if (_commandSet[0] == "AgglomeratesAnalyzer")
	{
		_outAnalyzer = std::make_shared<CAgglomeratesAnalyzer>();
		analyzerType = EAnalyzerTypes::AgglomeratesAnalyzer;
	}
	else
		return WriteError(EProcessResultType::WrongInput, "Unknown analyzer");

	_outAnalyzer->SetSystemStructure(&_systemStructure);

	// check for correct number of parameters
	const size_t nParameters = _commandSet.size();
	bool bCorrectNumberOfInputs = false;
	switch (analyzerType)
	{
	case EAnalyzerTypes::GeometryAnalyzer:
		bCorrectNumberOfInputs = nParameters == 3;
		break;
	case EAnalyzerTypes::BondsAnalyzer:
		bCorrectNumberOfInputs = nParameters == 3;
		break;
	case EAnalyzerTypes::ParticlesAnalyzer:
		bCorrectNumberOfInputs = nParameters == 3;
		break;
	case EAnalyzerTypes::AgglomeratesAnalyzer:
		bCorrectNumberOfInputs = nParameters == 3 || nParameters == 4;
		break;
	}
	if (!bCorrectNumberOfInputs)
		return WriteError(EProcessResultType::WrongNumberArguments);

	// start processing inputs
	size_t iParameter = 1;

	// set property
	std::string sProperty;
	if (analyzerType == EAnalyzerTypes::GeometryAnalyzer || analyzerType == EAnalyzerTypes::BondsAnalyzer || analyzerType == EAnalyzerTypes::ParticlesAnalyzer || analyzerType == EAnalyzerTypes::AgglomeratesAnalyzer)
	{
		sProperty = _commandSet[iParameter++];
		const EProcessResultType state = SetProperties(sProperty, analyzerType, *_outAnalyzer);
		if (state != EProcessResultType::Success)
			return state;
	}

	// set geometry
	std::string sGeometry;
	if (analyzerType == EAnalyzerTypes::GeometryAnalyzer)
	{
		sGeometry = _commandSet[iParameter++];
		const EProcessResultType state = SetGeometry(sGeometry, analyzerType, *_outAnalyzer);
		if (state != EProcessResultType::Success)
			return state;
	}

	// set result type
	std::string sResultType;
	if (analyzerType == EAnalyzerTypes::BondsAnalyzer || analyzerType == EAnalyzerTypes::ParticlesAnalyzer || analyzerType == EAnalyzerTypes::AgglomeratesAnalyzer)
	{
		sResultType = _commandSet[iParameter++];
		const EProcessResultType state = SetResultType(sResultType, analyzerType, *_outAnalyzer);
		if (state != EProcessResultType::Success)
			return state;
	}

	// get output filename if changed
	m_out << "Set up ";
	if (!m_job.resultFileName.empty())
		_outAnalyzer->m_sOutputFileName = MUSENFileFunctions::removeFileExt(m_job.resultFileName);
	else
		_outAnalyzer->m_sOutputFileName = MUSENFileFunctions::removeFileExt(_systemStructure.GetFileName());

	if (m_job.resultFileName.empty() || m_job.vPostProcessCommands.size() > 1)
	{
		switch (analyzerType)
		{
		case EAnalyzerTypes::GeometryAnalyzer:
			_outAnalyzer->m_sOutputFileName += "_" + sProperty + "_" + sGeometry; // default name scheme
			m_out << "GeometryAnalyzer " << sProperty << " of " << sGeometry;
			break;
		case EAnalyzerTypes::BondsAnalyzer:
			_outAnalyzer->m_sOutputFileName += "_" + sResultType + "_" + sProperty;
			m_out << "BondsAnalyzer " << sResultType << " of " << sProperty;
			break;
		case EAnalyzerTypes::ParticlesAnalyzer:
			_outAnalyzer->m_sOutputFileName += "_" + sResultType + "_" + sProperty;
			m_out << "ParticlesAnalyzer " << sResultType << " of " << sProperty;
			break;
		case EAnalyzerTypes::AgglomeratesAnalyzer:
			_outAnalyzer->m_sOutputFileName += "_" + sResultType + "_" + sProperty;
			m_out << "AgglomeratesAnalyzer " << sResultType << " of " << sProperty;
			break;
		}
	}

	_outAnalyzer->m_sOutputFileName += "." + m_sFileExt;

	m_out << std::endl;

	// set time
	_outAnalyzer->SetTime(_systemStructure.GetMinTime(), _systemStructure.GetMaxTime(), 0, true); // TODO: allow input for requested time steps

	// set constraints
	_outAnalyzer->GetConstraintsPtr()->SetPointers(&_systemStructure, &_systemStructure.m_MaterialDatabase);

	return EProcessResultType::Success;
}

void CConsoleResultsAnalyzer::EvaluateResults(const SJob& _job, CSystemStructure& _systemStructure)
{
	m_job = _job;
	std::vector<std::shared_ptr<CResultsAnalyzer>> analyzers(GetAnalyzers(_systemStructure));
	for (auto const& analyzer : analyzers)
	{
		analyzer->StartExport();
			if (analyzer->IsError())
				(void)WriteError(EProcessResultType::CantWriteFile, analyzer->GetStatusDescription());
		m_out << m_sOutputPrefix << "Export finished" << std::endl;
	}
}

std::vector<std::shared_ptr<CResultsAnalyzer>> CConsoleResultsAnalyzer::GetAnalyzers(CSystemStructure& _systemStructure) const
{
	std::vector<std::shared_ptr<CResultsAnalyzer>> outAnalyzer;
	for (auto const& command : m_job.vPostProcessCommands)
	{
		std::istringstream iss(command);
		std::vector<std::string> vCommandSet(std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>());

		m_out << m_sOutputPrefix << "Processing: ";
		for (auto const& s : vCommandSet)
			m_out << s << " ";
		m_out << std::endl;
		std::shared_ptr<CResultsAnalyzer> tmpAnalyzer;
		const EProcessResultType res = SetupAnalyzer(vCommandSet, _systemStructure, tmpAnalyzer);
		if (res == EProcessResultType::Success)
			outAnalyzer.push_back(std::move(tmpAnalyzer));
		// errors were already printed;
	}
	return outAnalyzer;
}

CConsoleResultsAnalyzer::EProcessResultType CConsoleResultsAnalyzer::WriteError(EProcessResultType _errorType, const std::string& _comment/* = ""*/) const
{
	m_err << m_sOutputPrefix << "  ERROR:";
	switch (_errorType)
	{
	case EProcessResultType::WrongNumberArguments:
		m_err << m_sOutputPrefix << "Wrong number of arguments " << _comment << std::endl;
		break;
	case EProcessResultType::WrongInput:
		m_err << m_sOutputPrefix << "Wrong input: " << _comment << std::endl;
		break;
	case EProcessResultType::CantWriteFile:
		m_err << m_sOutputPrefix << "Problem with writing file: " << _comment << std::endl;
		break;
	default: break;
	}
	return _errorType;
}

CConsoleResultsAnalyzer::EProcessResultType CConsoleResultsAnalyzer::SetProperties(const std::string& _inputString, const EAnalyzerTypes& _analyzerType, CResultsAnalyzer& _analyzer) const
{
	std::map<std::string, CResultsAnalyzer::EPropertyType> lookupTable;
	if (_analyzerType == EAnalyzerTypes::GeometryAnalyzer)
	{
		lookupTable["Displacement"] = CResultsAnalyzer::EPropertyType::Distance;
		lookupTable["Distance"]     = CResultsAnalyzer::EPropertyType::Distance;
	}
	if (_analyzerType == EAnalyzerTypes::GeometryAnalyzer || _analyzerType == EAnalyzerTypes::BondsAnalyzer)
	{
		lookupTable["Force"]      = CResultsAnalyzer::EPropertyType::ForceTotal;
		lookupTable["ForceTotal"] = CResultsAnalyzer::EPropertyType::ForceTotal;
	}
	if (_analyzerType == EAnalyzerTypes::BondsAnalyzer)
	{
		lookupTable["BondForce"]     = CResultsAnalyzer::EPropertyType::BondForce;
		lookupTable["Diameter"]      = CResultsAnalyzer::EPropertyType::Diameter;
		lookupTable["Length"]        = CResultsAnalyzer::EPropertyType::Length;
		lookupTable["Number"]        = CResultsAnalyzer::EPropertyType::Number;
		lookupTable["VelocityTotal"] = CResultsAnalyzer::EPropertyType::VelocityTotal;
		lookupTable["Deformation"]   = CResultsAnalyzer::EPropertyType::Deformation;
		lookupTable["Strain"]        = CResultsAnalyzer::EPropertyType::Strain;
	}
	if (_analyzerType == EAnalyzerTypes::ParticlesAnalyzer)
	{
		lookupTable["Coordinate"]         = CResultsAnalyzer::EPropertyType::Coordinate;
		lookupTable["CoordinationNumber"] = CResultsAnalyzer::EPropertyType::CoordinationNumber;
		lookupTable["Distance"]           = CResultsAnalyzer::EPropertyType::Distance;
		lookupTable["ForceTotal"]         = CResultsAnalyzer::EPropertyType::ForceTotal;
		lookupTable["KineticEnergy"]      = CResultsAnalyzer::EPropertyType::KineticEnergy;
		lookupTable["MaxOverlap"]         = CResultsAnalyzer::EPropertyType::MaxOverlap;
		lookupTable["Number"]             = CResultsAnalyzer::EPropertyType::Number;
		lookupTable["PotentialEnergy"]    = CResultsAnalyzer::EPropertyType::PotentialEnergy;
		lookupTable["ResidenceTime"]      = CResultsAnalyzer::EPropertyType::ResidenceTime;
		lookupTable["TotalVolume"]        = CResultsAnalyzer::EPropertyType::TotalVolume;
		lookupTable["VelocityTotal"]      = CResultsAnalyzer::EPropertyType::VelocityTotal;
		lookupTable["VelocityRotational"] = CResultsAnalyzer::EPropertyType::VelocityRotational;
		lookupTable["Stress"]             = CResultsAnalyzer::EPropertyType::Stress;
	}
	if (_analyzerType == EAnalyzerTypes::AgglomeratesAnalyzer)
	{
		lookupTable["Coordinate"]    = CResultsAnalyzer::EPropertyType::Coordinate;
		lookupTable["Diameter"]      = CResultsAnalyzer::EPropertyType::Diameter;
		lookupTable["Number"]        = CResultsAnalyzer::EPropertyType::Number;
		lookupTable["BondNumber"]    = CResultsAnalyzer::EPropertyType::BondNumber;
		lookupTable["PartNumber"]    = CResultsAnalyzer::EPropertyType::PartNumber;
		lookupTable["Orientation"]   = CResultsAnalyzer::EPropertyType::Orientation;
		lookupTable["VelocityTotal"] = CResultsAnalyzer::EPropertyType::VelocityTotal;
	}

	CResultsAnalyzer::VPropertyType properties;
	for (const std::string& _property : SplitString(_inputString, ','))
	{
		auto it = lookupTable.find(_property);
		if (it != lookupTable.end())
			properties.push_back(it->second);
		else
			WriteError(EProcessResultType::WrongInput, "Unknown or unavailable variable: " + _property + ". Trying to proceed.");
	}

	if ((_analyzerType == EAnalyzerTypes::BondsAnalyzer || _analyzerType == EAnalyzerTypes::ParticlesAnalyzer || _analyzerType == EAnalyzerTypes::AgglomeratesAnalyzer) && properties.size() > 1)
	{
		(void)WriteError(EProcessResultType::WrongInput, "Multiple properties for this analyzer are currently not supported. Using only the first input.");
		properties = CResultsAnalyzer::VPropertyType{ properties.front() };
	}
	if (!properties.empty())
		_analyzer.SetPropertyType(properties);
	else
		return WriteError(EProcessResultType::WrongInput, "No valid property to proceed.");

	return EProcessResultType::Success;
}

CConsoleResultsAnalyzer::EProcessResultType CConsoleResultsAnalyzer::SetResultType(const std::string& _inputString, const EAnalyzerTypes& _analyzerType, CResultsAnalyzer& _analyzer) const
{
	std::vector<std::string> vInputs = SplitString(_inputString, ',');
	if (vInputs.size() == 1)
	{
		if (vInputs.front() == "Average")
		{
			_analyzer.SetResultsType(CResultsAnalyzer::EResultType::Average);
			return EProcessResultType::Success;
		}
		if (vInputs.front() == "Maximum")
		{
			_analyzer.SetResultsType(CResultsAnalyzer::EResultType::Maximum);
			return EProcessResultType::Success;
		}
		if (vInputs.front() == "Minimum")
		{
			_analyzer.SetResultsType(CResultsAnalyzer::EResultType::Minimum);
			return EProcessResultType::Success;
		}
	}
	else if (vInputs.front() == "Distribution" && vInputs.size() == 4)
	{
		_analyzer.SetResultsType(CResultsAnalyzer::EResultType::Distribution);
		_analyzer.SetProperty(std::stod(vInputs[1]), std::stod(vInputs[2]), std::stoi(vInputs[3]));
		return EProcessResultType::Success;
	}

	// reached here: unknown type
	return WriteError(EProcessResultType::WrongInput, "Unknown result types: " + _inputString);
}

CConsoleResultsAnalyzer::EProcessResultType CConsoleResultsAnalyzer::SetGeometry(const std::string& _inputString, const EAnalyzerTypes& _analyzerType, CResultsAnalyzer& _analyzer) const
{
	const CSystemStructure* systemStructure = _analyzer.GetSystemStructure();
	// try to find a geometry by key
	const SGeometryObject* geometry = systemStructure->GetGeometry(_inputString);
	// try to find a geometry by name
	if (!geometry)
		geometry = systemStructure->GetGeometryByName(_inputString);
	// check that something found
	if (!geometry)
		return WriteError(EProcessResultType::WrongInput, "Unknown geometry name or key: " + _inputString);
	_analyzer.m_nGeometryIndex = systemStructure->GetGeometryIndex(geometry->sKey);
	return EProcessResultType::Success;
}
