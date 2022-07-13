/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelManager.h"
#include "MUSENFileFunctions.h"

#include "../MUSEN/Models/ParticleParticle/Hertz/ModelPPHertz.h"
#include "../MUSEN/Models/ParticleParticle/HertzMindlin/ModelPPHertzMindlin.h"
#include "../MUSEN/Models/ParticleParticle/ChealNess/ModelPPChealNess.h"
#include "../MUSEN/Models/ParticleParticle/HertzMindlinLiquid/ModelPPHertzMindlinLiquid.h"
#include "../MUSEN/Models/ParticleParticle/JKR/ModelPPJKR.h"
#include "../MUSEN/Models/ParticleParticle/LinearElastic/ModelPPLinearElastic.h"
#include "../MUSEN/Models/ParticleParticle/PopovJKR/ModelPPPopovJKR.h"
#include "../MUSEN/Models/ParticleParticle/SimpleViscoElastic/ModelPPSimpleViscoElastic.h"
#include "../MUSEN/Models/ParticleParticle/TestSinteringModel/ModelPPSintering.h"
#include "../MUSEN/Models/ParticleParticle/SinteringTemperature/ModelPPSinteringTemperature.h"

#include "../MUSEN/Models/ParticleWall/PWHertzMindlin/ModelPWHertzMindlin.h"
#include "../MUSEN/Models/ParticleWall/PWHertzMindlinLiquid/ModelPWHertzMindlinLiquid.h"
#include "../MUSEN/Models/ParticleWall/PWJKR/ModelPWJKR.h"
#include "../MUSEN/Models/ParticleWall/PWPopovJKR/ModelPWPopovJKR.h"
#include "../MUSEN/Models/ParticleWall/PWSimpleViscoElastic/ModelPWSimpleViscoElastic.h"

#include "../MUSEN/Models/SolidBonds/BondModelAerogel/ModelSBAerogel.h"
#include "../MUSEN/Models/SolidBonds/BondModelElastic/ModelSBElastic.h"
#include "../MUSEN/Models/SolidBonds/BondModelElasticPerfectlyPlastic/ModelSBElasticPerfectlyPlastic.h"
#include "../MUSEN/Models/SolidBonds/BondModelCreep/ModelSBCreep.h"
#include "../MUSEN/Models/SolidBonds/BondModelKelvin/ModelSBKelvin.h"
#include "../MUSEN/Models/SolidBonds/BondModelLinearPlastic/ModelSBLinearPlastic.h"
#include "../MUSEN/Models/SolidBonds/BondModelThermal/ModelSBThermal.h"
#include "../MUSEN/Models/SolidBonds/BondModelWeakening/ModelSBWeakening.h"

#include "../MUSEN/Models/LiquidBonds/CapilaryViscous/ModelLBCapilarViscous.h"

#include "../MUSEN/Models/ExternalForce/CentrifugalCasting/ModelEFCentrifugalCasting.h"
#include "../MUSEN/Models/ExternalForce/ViscousField/ModelEFViscousField.h"
#include "../MUSEN/Models/ExternalForce/HeatTransfer/ModelEFHeatTransfer.h"

#include "../MUSEN/Models/HeatTransfer/PPHeatConduction/ModelPPHeatConduction.h"


namespace StaticLibs
{
	struct SModule
	{
		std::string name;
		CreateModelFunction function;
	};

	template<class T>
	class Constructor
	{
		static CAbstractDEMModel* alloc()
		{
			return new T();
		}

	public:
		static SModule get()
		{
			SModule m;
			m.function = alloc;
			m.name = typeid(T).name();
			while (m.name.compare(0, 5, "Model"))
				m.name = m.name.substr(1, m.name.size() - 1);
			return m;
		}
	};

	SModule all_modules[] =
	{
		Constructor<CModelPPHertz>::get(),
		Constructor<CModelPPHertzMindlin>::get(),
		Constructor<CModelPPHertzMindlinLiquid>::get(),
		Constructor<CModelPPChealNess>::get(),
		Constructor<CModelPPJKR>::get(),
		Constructor<CModelPPLinearElastic>::get(),
		Constructor<CModelPPPopovJKR>::get(),
		Constructor<CModelPPSimpleViscoElastic>::get(),
		Constructor<CModelPPSintering>::get(),
		Constructor<CModelPPSinteringTemperature>::get(),

		Constructor<CModelPWHertzMindlin>::get(),
		Constructor<CModelPWHertzMindlinLiquid>::get(),
		Constructor<CModelPWJKR>::get(),
		Constructor<CModelPWPopovJKR>::get(),
		Constructor<CModelPWSimpleViscoElastic>::get(),

		Constructor<CModelSBAerogel>::get(),
		Constructor<CModelSBElastic>::get(),
		Constructor<CModelSBElasticPerfectlyPlastic>::get(),
		Constructor<CModelSBCreep>::get(),
		Constructor<CModelSBKelvin>::get(),
		Constructor<CModelSBLinearPlastic>::get(),
		Constructor<CModelSBThermal>::get(),
		Constructor<CModelSBWeakening>::get(),

		Constructor<CModelLBCapilarViscous>::get(),

		Constructor<CModelEFCentrifugalCasting>::get(),
		Constructor<CModelEFViscousField>::get(),
		Constructor<CModelEFHeatTransfer>::get(),

		Constructor<CModelPPHeatConduction>::get(),
	};

	SModule* LoadLibrary_static(const std::string& _sName)
	{
		for (size_t i = 0; i < sizeof(all_modules) / sizeof(SModule); ++i)
			if(_sName == all_modules[i].name)
				return &all_modules[i];
		return nullptr;
	}
}


CModelManager::CModelManager()
{
	m_vCurrentModels[EMusenModelType::PP]   = SModelInfo();
	m_vCurrentModels[EMusenModelType::PW]   = SModelInfo();
	m_vCurrentModels[EMusenModelType::SB]   = SModelInfo();
	m_vCurrentModels[EMusenModelType::LB]   = SModelInfo();
	m_vCurrentModels[EMusenModelType::EF]   = SModelInfo();
	m_vCurrentModels[EMusenModelType::PPHT] = SModelInfo();
	UpdateAvailableModels();
}

CModelManager::~CModelManager()
{
	ClearAllModels();

	for (size_t i = 0; i < m_vAvailableModels.size(); ++i)
		delete m_vAvailableModels[i].pModel;
}

std::vector<std::string> CModelManager::GetDirs() const
{
	return m_vDirList;
}

void CModelManager::SetDirs(const std::vector<std::string>& _vDirs)
{
	m_vDirList.clear();

	for (size_t i = 0; i < _vDirs.size(); ++i)
		m_vDirList.push_back(unifyPath(_vDirs[i]));
	UpdateAvailableModels();
}

void CModelManager::AddDir(const std::string& _sDir)
{
	if (_sDir.empty()) return;
	bool bNewDir = true;
	for (size_t i = 0; i < m_vDirList.size(); ++i)
		if (std::find(m_vDirList.begin(), m_vDirList.end(), _sDir) != m_vDirList.end()) // already in the list
		{
			bNewDir = false;
			break;
		}
	if(bNewDir)
	{
		m_vDirList.push_back(unifyPath(_sDir));
		UpdateAvailableModels();
	}
}

void CModelManager::RemoveDir(size_t _index)
{
	if (_index >= m_vDirList.size()) return;
	m_vDirList.erase(m_vDirList.begin() + _index);
	UpdateAvailableModels();
}

void CModelManager::UpDir(size_t _index)
{
	if ((_index < m_vDirList.size()) && (_index != 0))
		std::iter_swap(m_vDirList.begin() + _index, m_vDirList.begin() + _index - 1);
}

void CModelManager::DownDir(size_t _index)
{
	if ((_index < m_vDirList.size()) && (_index != m_vDirList.size() - 1))
		std::iter_swap(m_vDirList.begin() + _index, m_vDirList.begin() + _index + 1);
}

std::vector<CModelManager::SModelInfo> CModelManager::GetAllAvailableModels() const
{
	return m_vAvailableModels;
}

bool CModelManager::IsModelDefined(const EMusenModelType& _modelType) const
{
	return m_vCurrentModels.at(_modelType).pModel != nullptr;
}

bool CModelManager::IsModelGPUCompatible(const EMusenModelType& _modelType) const
{
	if (m_vCurrentModels.at(_modelType).pModel)
		return m_vCurrentModels.at(_modelType).pModel->HasGPUSupport();
	else
		return false;
}

CAbstractDEMModel* CModelManager::GetModel(const EMusenModelType& _modelType)
{
	return m_vCurrentModels.at(_modelType).pModel;
}

const CAbstractDEMModel* CModelManager::GetModel(const EMusenModelType& _modelType) const
{
	return m_vCurrentModels.at(_modelType).pModel;
}

std::string CModelManager::GetModelPath(const EMusenModelType& _modelType) const
{
	if (m_vCurrentModels.at(_modelType).pModel)
		return m_vCurrentModels.at(_modelType).sPath;
	else
		return "";
}

void CModelManager::SetModelPath(const EMusenModelType& _modelType, const std::string& _sPath)
{
	std::string sNewPath = unifyPath(_sPath);
	if (m_vCurrentModels[_modelType].sPath == sNewPath)	// this model is already loaded
		return;

	ClearModel(_modelType);	// delete old model

	m_vCurrentModels[_modelType] = LoadModelByPath(sNewPath, _modelType);
}
const SOptionalVariables CModelManager::GetUtilizedVariables()
{
	SOptionalVariables sActiveVariables; // result variable
	for (auto element : m_vCurrentModels)
	{
		if (element.second.pModel)
			sActiveVariables |= element.second.pModel->GetUtilizedVariables();
	}
	return std::move(sActiveVariables);
}

std::string CModelManager::GetModelParameters(const EMusenModelType& _modelType) const
{
	if (m_vCurrentModels.at(_modelType).pModel)
		return m_vCurrentModels.at(_modelType).pModel->GetParametersStr();
	else
		return "";
}

void CModelManager::SetModelParameters(const EMusenModelType& _modelType, const std::string& _sParams)
{
	if (m_vCurrentModels[_modelType].pModel)
		m_vCurrentModels[_modelType].pModel->SetParametersStr(_sParams);
}

void CModelManager::SetModelDefaultParameters(const EMusenModelType& _modelType)
{
	if (m_vCurrentModels[_modelType].pModel)
		m_vCurrentModels[_modelType].pModel->SetDefaultValues();
}

std::string CModelManager::GetModelError(const EMusenModelType& _modelType) const
{
	return m_vCurrentModels.at(_modelType).sError;
}

void CModelManager::LoadConfiguration()
{
	ClearAllModels();

	ProtoModulesData& _pProtoMessage = *m_pSystemStructure->GetProtoModulesData();
	if (!_pProtoMessage.has_model_manager()) return; // for old versions of file

	const ProtoModuleModelManager& MM = _pProtoMessage.model_manager();

	LoadModelConfiguration(MM.pp_model(),    EMusenModelType::PP,   &m_vCurrentModels[EMusenModelType::PP]);
	LoadModelConfiguration(MM.pw_model(),    EMusenModelType::PW,   &m_vCurrentModels[EMusenModelType::PW]);
	LoadModelConfiguration(MM.sb_model(),    EMusenModelType::SB,   &m_vCurrentModels[EMusenModelType::SB]);
	LoadModelConfiguration(MM.lb_model(),    EMusenModelType::LB,   &m_vCurrentModels[EMusenModelType::LB]);
	LoadModelConfiguration(MM.ef_model(),    EMusenModelType::EF,   &m_vCurrentModels[EMusenModelType::EF]);
	LoadModelConfiguration(MM.ht_pp_model(), EMusenModelType::PPHT, &m_vCurrentModels[EMusenModelType::PPHT]);
	m_bConnectedPPContact = MM.connected_pp_contact();
}

void CModelManager::SaveConfiguration()
{
	ProtoModulesData& _pProtoMessage = *m_pSystemStructure->GetProtoModulesData();
	ProtoModuleModelManager* pMM = _pProtoMessage.mutable_model_manager();

	SaveModelConfiguration(pMM->mutable_pp_model(),    m_vCurrentModels[EMusenModelType::PP]);
	SaveModelConfiguration(pMM->mutable_pw_model(),    m_vCurrentModels[EMusenModelType::PW]);
	SaveModelConfiguration(pMM->mutable_sb_model(),    m_vCurrentModels[EMusenModelType::SB]);
	SaveModelConfiguration(pMM->mutable_lb_model(),    m_vCurrentModels[EMusenModelType::LB]);
	SaveModelConfiguration(pMM->mutable_ef_model(),    m_vCurrentModels[EMusenModelType::EF]);
	SaveModelConfiguration(pMM->mutable_ht_pp_model(), m_vCurrentModels[EMusenModelType::PPHT]);
	pMM->set_connected_pp_contact( m_bConnectedPPContact );
}

void CModelManager::LoadModelConfiguration(const ProtoMusenModel& _protoModel, const EMusenModelType& _modelType, SModelInfo* _pModelInfo)
{
	if (_protoModel.key().empty()) // no model was previously saved
		return;

	*_pModelInfo = LoadModelByPath(_protoModel.path(), _modelType); // try to load by path
	if (!_pModelInfo->pModel)
		*_pModelInfo = LoadModelByKey(_protoModel.key(), _modelType); // try to load by key

	if (_pModelInfo->pModel) // model loaded
		_pModelInfo->pModel->SetParametersStr(_protoModel.params());
}

void CModelManager::SaveModelConfiguration(ProtoMusenModel* _pProtoModel, SModelInfo& _model)
{
	if (_model.pModel)
	{
		_pProtoModel->set_key(_model.pModel->GetUniqueKey());
		_pProtoModel->set_path(_model.sPath);
		_pProtoModel->set_params(_model.pModel->GetParametersStr());
	}
	else
	{
		_pProtoModel->set_key("");
		_pProtoModel->set_path("");
		_pProtoModel->set_params("");
	}
}

void CModelManager::UpdateAvailableModels()
{
	// clear old models
	for (size_t i = 0; i < m_vAvailableModels.size(); ++i)
		delete m_vAvailableModels[i].pModel;
	m_vAvailableModels.clear();

	// static models
	for (size_t i = 0; i < sizeof(StaticLibs::all_modules) / sizeof(StaticLibs::SModule); ++i)
		m_vAvailableModels.push_back(LoadStaticModel(StaticLibs::all_modules[i].name));

	// dynamic models
	for (size_t i = 0; i < m_vDirList.size(); ++i)
	{
		std::vector<std::string> vDLLs = MUSENFileFunctions::filesList(m_vDirList[i], "*.dll");
		for (size_t j = 0; j < vDLLs.size(); ++j)
		{
			CModelManager::SModelInfo model = LoadDynamicModel(vDLLs[j]);
			if (model.pModel)
				m_vAvailableModels.push_back(model);
		}
	}
}

CModelManager::SModelInfo CModelManager::LoadDynamicModel(const std::string& _sModelPath, const EMusenModelType& _modelType /*= EMusenModelType::UNSPECIFIED*/)
{
#ifdef _WIN32

	std::string sPath = unifyPath(_sModelPath);

	if (sPath.empty())	// just an empty model (will not be used during the simulation)
		return SModelInfo();

	// try to load library
	HINSTANCE hInstLibrary = LoadLibraryA(windowsPath(sPath).c_str());
	if (!hInstLibrary)
		return SModelInfo(nullptr, "", BuildErrorDescription(_modelType, EErrorType::WRONG_PATH), ELibType::DYNAMIC);

	// try to get constructor
	CreateModelFunction createModelFunc = (CreateModelFunction)GetProcAddress(hInstLibrary, MUSEN_CREATE_MODEL_FUN_NAME);
	if (!createModelFunc)
	{
		FreeLibrary(hInstLibrary);
		return SModelInfo(nullptr, "", BuildErrorDescription(_modelType, EErrorType::WRONG_DLL_INTERFACE), ELibType::DYNAMIC);
	}

	// try to create model
	CAbstractDEMModel* pLoadedModel;
	try
	{
		pLoadedModel = createModelFunc();
	}
	catch (...)
	{
		FreeLibrary(hInstLibrary);
		return SModelInfo(nullptr, "", BuildErrorDescription(_modelType, EErrorType::WRONG_DLL_VERSION), ELibType::DYNAMIC);
	}

	// check model type
	if(pLoadedModel->GetType() == EMusenModelType::UNSPECIFIED)
	{
		delete pLoadedModel;
		FreeLibrary(hInstLibrary);
		return SModelInfo(nullptr, "", BuildErrorDescription(_modelType, EErrorType::UNSPEC_MODEL_TYPE), ELibType::DYNAMIC);
	}
	if (_modelType != EMusenModelType::UNSPECIFIED)
		if (pLoadedModel->GetType() != _modelType)
		{
			delete pLoadedModel;
			FreeLibrary(hInstLibrary);
			return SModelInfo(nullptr, "", BuildErrorDescription(_modelType, EErrorType::WRONG_MODEL_TYPE), ELibType::DYNAMIC);
		}

	return SModelInfo(pLoadedModel, sPath, "", ELibType::DYNAMIC);

#else
	return SModelInfo(nullptr, "", "_WIN32", ELibType::DYNAMIC);
#endif
}

CModelManager::SModelInfo CModelManager::LoadStaticModel(const std::string& _sModelName, const EMusenModelType& _modelType /*= EMusenModelType::UNSPECIFIED*/)
{
	if (_sModelName.empty())	// just an empty model (will not be used during the simulation)
		return SModelInfo();

	// try to load library
	StaticLibs::SModule* hInstLibrary = StaticLibs::LoadLibrary_static(_sModelName);
	if (!hInstLibrary)
		return SModelInfo(nullptr, "", BuildErrorDescription(_modelType, EErrorType::WRONG_PATH), ELibType::STATIC);

	// create model
	CAbstractDEMModel* pLoadedModel = hInstLibrary->function();

	// check model type
	if (pLoadedModel->GetType() == EMusenModelType::UNSPECIFIED)
	{
		delete pLoadedModel;
		return SModelInfo(nullptr, "", BuildErrorDescription(_modelType, EErrorType::UNSPEC_MODEL_TYPE), ELibType::STATIC);
	}
	if (_modelType != EMusenModelType::UNSPECIFIED)
		if (pLoadedModel->GetType() != _modelType)
		{
			delete pLoadedModel;
			return SModelInfo(nullptr, "", BuildErrorDescription(_modelType, EErrorType::WRONG_MODEL_TYPE), ELibType::STATIC);
		}

	return SModelInfo(pLoadedModel, _sModelName, "", ELibType::STATIC);
}

CModelManager::SModelInfo CModelManager::LoadModelByPath(const std::string& _sModelPath, const EMusenModelType& _modelType /*= EMusenModelType::UNSPECIFIED*/)
{
	CModelManager::SModelInfo model;
	// try to load static model
	model = LoadStaticModel(_sModelPath, _modelType);
	if (model.pModel) return model; // found static
	// try to load dynamic model
	model = LoadDynamicModel(_sModelPath, _modelType);
	if (model.pModel) return model; // found dynamic

	return SModelInfo(); // not found
}

CModelManager::SModelInfo CModelManager::LoadModelByKey(const std::string& _sModelKey, const EMusenModelType& _modelType /*= EMusenModelType::UNSPECIFIED*/)
{
	for (size_t i = 0; i < m_vAvailableModels.size(); ++i)
		if (m_vAvailableModels[i].pModel->GetUniqueKey() == _sModelKey)
		{
			CModelManager::SModelInfo model = LoadModelByPath(m_vAvailableModels[i].sPath, _modelType);
			if (model.pModel) return model;
		}
	return SModelInfo();
}

std::string CModelManager::BuildErrorDescription(const EMusenModelType& _modelType, const EErrorType& _type) const
{
	std::string sError;

	std::string sModelTypeDescr;
	switch (_modelType)
	{
	case EMusenModelType::PP:
		sModelTypeDescr = "particle-particle contact "; break;
	case EMusenModelType::PW:
		sModelTypeDescr = "particle-wall contact "; break;
	case EMusenModelType::SB:
		sModelTypeDescr = "solid bond "; break;
	case EMusenModelType::LB:
		sModelTypeDescr = "liquid bond "; break;
	case EMusenModelType::EF:
		sModelTypeDescr = "external force "; break;
	case EMusenModelType::PPHT:
		sModelTypeDescr = "heat transfer PP "; break;
	default: break;
	}

	switch (_type)
	{
	case EErrorType::WRONG_PATH:
		sError = "Cannot find " + sModelTypeDescr + "model";
		break;
	case EErrorType::WRONG_DLL_INTERFACE:
		sError = "Wrong format of " + sModelTypeDescr + "model";
		break;
	case EErrorType::WRONG_DLL_VERSION:
		sError = "Unsupported version of " + sModelTypeDescr + "model";
		break;
	case EErrorType::UNSPEC_MODEL_TYPE:
		sError = "Unspecified model type of " + sModelTypeDescr + "model";
		break;
	case EErrorType::WRONG_MODEL_TYPE:
		sError = "Wrong model type of " + sModelTypeDescr + "model";
		break;
	default:
		break;
	}

	return sError;
}

void CModelManager::ClearModel(const EMusenModelType& _modelType)
{
	if (m_vCurrentModels[_modelType].pModel)
	{
		delete m_vCurrentModels[_modelType].pModel;
		m_vCurrentModels[_modelType].pModel = nullptr;
		m_vCurrentModels[_modelType].sPath.clear();
	}
}

void CModelManager::ClearAllModels()
{
	for (auto it = m_vCurrentModels.begin(); it != m_vCurrentModels.end(); ++it)
		if (it->second.pModel)
		{
			delete it->second.pModel;
			it->second.pModel = nullptr;
			it->second.sPath.clear();
			it->second.sError.clear();
		}
}

