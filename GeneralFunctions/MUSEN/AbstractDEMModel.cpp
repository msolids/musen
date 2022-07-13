/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "AbstractDEMModel.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
////////// CAbstractDEMModel

CAbstractDEMModel::CAbstractDEMModel() :
	m_type{ EMusenModelType::UNSPECIFIED },
	m_name{ "Unspecified" },
	m_uniqueKey{ "" },
	m_helpFileName{ "" },
	m_hasGPUSupport{ false },
	m_PBC{}
{
}

EMusenModelType CAbstractDEMModel::GetType() const
{
	return m_type;
}

std::string CAbstractDEMModel::GetName() const
{
	return m_name;
}

std::string CAbstractDEMModel::GetUniqueKey() const
{
	return m_uniqueKey;
}

std::string CAbstractDEMModel::GetHelpFileName() const
{
	return m_helpFileName;
}

size_t CAbstractDEMModel::GetParametersNumber() const
{
	return m_parameters.size();
}

std::vector<SModelParameter> CAbstractDEMModel::GetAllParameters() const
{
	return m_parameters;
}

std::string CAbstractDEMModel::GetParametersStr() const
{
	std::stringstream ss;
	for (const auto& p : m_parameters)
		ss << p.uniqueName << " " << p.value << " ";
	return ss.str();
}

void CAbstractDEMModel::SetParametersStr(const std::string& _parameters)
{
	if (_parameters.empty()) return;
	std::string name;
	double value;
	std::stringstream ss(_parameters);
	while (ss.good())
	{
		ss >> name >> value;
		SetParameterValue(name, value); // set specified parameter to the new value
	}
}

double CAbstractDEMModel::GetParameterValue(const std::string& _name) const
{
	for (const auto& parameter : m_parameters)
		if (parameter.uniqueName == _name)
			return parameter.value;
	return 0;
}

void CAbstractDEMModel::SetParameterValue(const std::string& _name, double _value)
{
	for (auto& p : m_parameters)
		if (p.uniqueName == _name)
		{
			p.value = _value;
			return;
		}
}

void CAbstractDEMModel::SetDefaultValues()
{
	for (auto& p : m_parameters)
		p.value = p.defaultValue;
}

void CAbstractDEMModel::SetPBC(SPBC _pbc)
{
	m_PBC = _pbc;
}

SOptionalVariables CAbstractDEMModel::GetUtilizedVariables() const
{
	return m_requieredVariables;
}

bool CAbstractDEMModel::HasGPUSupport() const
{
	return m_hasGPUSupport;
}

bool CAbstractDEMModel::AddParameter(const SModelParameter& _parameter)
{
	if (_parameter.uniqueName.find_first_of("\t\n ") != std::string::npos) // name contains spaces
		return false;
	// check the uniqueness of the parameter name
	for (auto& p : m_parameters)
		if (p.uniqueName == _parameter.uniqueName)
			return false;
	m_parameters.push_back(_parameter);
	return true;
}

bool CAbstractDEMModel::AddParameter(const std::string& _name, const std::string& _description, double _defaultValue)
{
	return AddParameter(SModelParameter{ _name, _description, _defaultValue });
}

void CAbstractDEMModel::InitializeGPU(const CCUDADefines* _cudaDefines)
{
	m_cudaDefines = _cudaDefines;
	std::vector<double> params(m_parameters.size());
	for (size_t i = 0; i < m_parameters.size(); ++i)
		params[i] = m_parameters[i].value;
	SetParametersGPU(params, m_PBC);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
////////// CParticleParticleModel

CParticleParticleModel::CParticleParticleModel()
{
	m_type = EMusenModelType::PP;
}

bool CParticleParticleModel::Initialize(SParticleStruct* _particles, SWallStruct* _walls, SSolidBondStruct* _solidBinds, SLiquidBondStruct* _liquidBonds, std::vector<SInteractProps>* _interactProps)
{
	m_particles = _particles;
	m_interactProps = _interactProps;
	if (!m_particles || !m_interactProps) return false;
	if (m_requieredVariables.bThermals && !m_particles->ThermalsExist()) return false;
	return true;
}

void CParticleParticleModel::Precalculate(double _time, double _timeStep)
{
	PrecalculatePPModel(_time, _timeStep, m_particles);
}

void CParticleParticleModel::Calculate(double _time, double _timeStep, SCollision* _collision) const
{
	CalculatePPForce(_time, _timeStep, _collision->nSrcID, _collision->nDstID, InteractionProperty(_collision->nInteractProp), _collision);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
////////// CParticleWallModel

CParticleWallModel::CParticleWallModel()
{
	m_type = EMusenModelType::PW;
}

bool CParticleWallModel::Initialize(SParticleStruct* _particles, SWallStruct* _walls, SSolidBondStruct* _solidBinds, SLiquidBondStruct* _liquidBonds, std::vector<SInteractProps>* _interactProps)
{
	m_particles = _particles;
	m_walls = _walls;
	m_interactProps = _interactProps;
	if (!m_particles || !m_interactProps) return false;
	return true;
}

void CParticleWallModel::Precalculate(double _time, double _timeStep)
{
	PrecalculatePWModel(_time, _timeStep, m_particles, m_walls);
}

void CParticleWallModel::Calculate(double _time, double _timeStep, SCollision* _collision) const
{
	CalculatePWForce(_time, _timeStep, _collision->nSrcID, _collision->nDstID, InteractionProperty(_collision->nInteractProp), _collision);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
////////// CSolidBondModel

CSolidBondModel::CSolidBondModel()
{
	m_type = EMusenModelType::SB;
}

bool CSolidBondModel::Initialize(SParticleStruct* _particles, SWallStruct* _walls, SSolidBondStruct* _solidBinds, SLiquidBondStruct* _liquidBonds, std::vector<SInteractProps>* _interactProps)
{
	m_particles = _particles;
	m_bonds = _solidBinds;
	return m_particles && m_bonds;
}

void CSolidBondModel::Precalculate(double _time, double _timeStep)
{
	PrecalculateSBModel(_time, _timeStep, m_particles, m_bonds);
}

void CSolidBondModel::Calculate(double _time, double _timeStep, size_t _iBond, SSolidBondStruct& _bonds, unsigned* _pBrokenBondsNum) const
{
	CalculateSBForce(_time, _timeStep, _bonds.LeftID(_iBond), _bonds.RightID(_iBond), _iBond, _bonds, _pBrokenBondsNum);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
////////// CLiquidBondModel

CLiquidBondModel::CLiquidBondModel()
{
	m_type = EMusenModelType::LB;
}

bool CLiquidBondModel::Initialize(SParticleStruct* _particles, SWallStruct* _walls, SSolidBondStruct* _solidBinds, SLiquidBondStruct* _liquidBonds, std::vector<SInteractProps>* _interactProps)
{
	m_particles = _particles;
	m_bonds = _liquidBonds;
	return m_particles && m_bonds;
}

void CLiquidBondModel::Precalculate(double _time, double _timeStep)
{
	PrecalculateLBModel(_time, _timeStep, m_particles, m_bonds);
}

void CLiquidBondModel::Calculate(double _time, double _timeStep, size_t _iBond, SLiquidBondStruct& _bonds, unsigned* _pBrokenBondsNum) const
{
	CalculateLBForce(_time, _timeStep, _bonds.LeftID(_iBond), _bonds.RightID(_iBond), _iBond, _bonds, _pBrokenBondsNum);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
////////// CExternalForceModel

CExternalForceModel::CExternalForceModel()
{
	m_type = EMusenModelType::EF;
}

bool CExternalForceModel::Initialize(SParticleStruct* _particles, SWallStruct* _walls, SSolidBondStruct* _solidBinds, SLiquidBondStruct* _liquidBonds, std::vector<SInteractProps>* _interactProps)
{
	m_particles = _particles;
	return m_particles;
}

void CExternalForceModel::Precalculate(double _time, double _timeStep)
{
	PrecalculateEFModel(_time, _timeStep, m_particles);
}

void CExternalForceModel::Calculate(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles) const
{
	CalculateEFForce(_time, _timeStep, _iPart, _particles);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////// CHeatTransferPPModel

CPPHeatTransferModel::CPPHeatTransferModel()
{
	m_type = EMusenModelType::PPHT;
	m_requieredVariables.bThermals = true;
}

bool CPPHeatTransferModel::Initialize(SParticleStruct* _particles, SWallStruct* _walls, SSolidBondStruct* _solidBinds, SLiquidBondStruct* _liquidBonds, std::vector<SInteractProps>* _interactProps)
{
	m_particles = _particles;
	m_interactProps = _interactProps;
	if (!m_particles || !m_interactProps) return false;
	if (!m_particles->ThermalsExist()) return false;
	return true;
}

void CPPHeatTransferModel::Precalculate(double _time, double _timeStep)
{
	PrecalculatePPHTModel(_time, _timeStep, m_particles);
}

void CPPHeatTransferModel::Calculate(double _time, double _timeStep, SCollision* _collision) const
{
	CalculatePPHeatTransfer(_time, _timeStep, _collision->nSrcID, _collision->nDstID, InteractionProperty(_collision->nInteractProp), _collision);
}